"""xECG (Lunelli et al. 2025, "BenchECG and xECG", arXiv:2509.10151).

Reference: https://github.com/dlaskalab/bench-xecg (release branch)

Architecture: a single xLSTM stack of `(s,s,m,m,s,s,m,m,s)` blocks at embed=1024,
processed bidirectionally by alternating the sequence direction between blocks
(block 0 forward, block 1 reversed, etc.). With 9 blocks the final flip count
is even, so the output is in the forward direction. This is a single shared
stack — not parallel forward+reverse — which is what gives the paper's 57M
parameter count.

The patch embedding is a linear Conv1d (kernel=stride=patch_size). Positional
embeddings are sinusoidal so the same model handles global views (40 patches at
seg_len=2500, patch_size=50) and local views (20 patches) without retraining a
length-dependent table.

Pretraining is SimDINOv2-style with two globals and four locals: a slow EMA
teacher sees the unmasked globals; the student sees masked globals plus the
locals; loss is `compression + 0.1*expansion + patch_loss` where patch_loss is
a masked cosine on patch-level features (see neural_networks/xecg/loss.py).
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch import nn

from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig
from xlstm.blocks.mlstm.block import mLSTMBlockConfig
from xlstm.blocks.mlstm.layer import mLSTMLayerConfig
from xlstm.blocks.slstm.block import sLSTMBlockConfig
from xlstm.blocks.slstm.layer import sLSTMLayerConfig
from xlstm.components.feedforward import FeedForwardConfig

from .loss import SimDINOv2Loss, masked_cosine_loss


@dataclass
class xECGConfig:
    seq_len: int = 2500
    patch_size: int = 50
    num_leads: int = 12
    embedding_size: int = 1024
    num_heads: int = 4
    num_blocks: int = 9
    block_pattern: List[str] = field(default_factory=lambda: ["s", "s", "m", "m", "s", "s", "m", "m", "s"])
    conv1d_kernel_size: int = 4
    proj_factor_mlstm: float = 2.0
    proj_factor_slstm_ff: float = 1.3
    qkv_proj_blocksize: int = 4
    mask_ratio: float = 0.3
    masking_type: str = "block"
    bidirectional: bool = True
    dropout: float = 0.0
    slstm_backend: str = "cuda"
    activation_fn: str = "gelu"
    context_length: int = 8000
    post_encoder_norm: bool = False

    def __post_init__(self):
        assert self.seq_len % self.patch_size == 0, "seq_len must be divisible by patch_size"
        assert len(self.block_pattern) == self.num_blocks, "block_pattern length must equal num_blocks"
        assert self.embedding_size % self.num_heads == 0, "embedding_size must be divisible by num_heads"
        self.num_patches = self.seq_len // self.patch_size
        self.slstm_at = [i for i, b in enumerate(self.block_pattern) if b == "s"]


@dataclass
class xECGOutput:
    patches: torch.Tensor
    cls: torch.Tensor
    mask: Optional[torch.Tensor] = None


@dataclass
class xECGPretrainOutput:
    loss: torch.Tensor
    compression: torch.Tensor
    expansion: torch.Tensor
    patch: torch.Tensor


class AttentionPooling(nn.Module):
    """Perception-Encoder attention probe (Bolya et al.), as in bench-xecg pooling.py.

    A single learnable query cross-attends over the patch tokens, followed by a
    residual MLP block: `x = attn(q, x, x); x = x + mlp(layernorm(x))`. Pools
    (B, N, D) -> (B, D).
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.probe = nn.Parameter(torch.randn(1, 1, dim))
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(dim)
        mlp_width = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_width),
            nn.GELU(),
            nn.Linear(mlp_width, dim),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.probe.expand(x.size(0), -1, -1).to(x.dtype)
        out, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        out = out + self.mlp(self.layernorm(out))
        return out.squeeze(1)


def _build_block_stack(cfg: xECGConfig, batch_size_hint: int) -> xLSTMBlockStack:
    mlstm_cfg = mLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(
            conv1d_kernel_size=cfg.conv1d_kernel_size,
            qkv_proj_blocksize=cfg.qkv_proj_blocksize,
            num_heads=cfg.num_heads,
            proj_factor=cfg.proj_factor_mlstm,
        )
    )
    slstm_cfg = sLSTMBlockConfig(
        slstm=sLSTMLayerConfig(
            num_heads=cfg.num_heads,
            backend=cfg.slstm_backend,
            conv1d_kernel_size=cfg.conv1d_kernel_size,
            bias_init="powerlaw_blockdependent",
            batch_size=batch_size_hint,
        ),
        feedforward=FeedForwardConfig(proj_factor=cfg.proj_factor_slstm_ff, act_fn=cfg.activation_fn),
    ) if cfg.slstm_at else None
    stack_cfg = xLSTMBlockStackConfig(
        mlstm_block=mlstm_cfg,
        slstm_block=slstm_cfg,
        context_length=cfg.context_length,
        num_blocks=cfg.num_blocks,
        embedding_dim=cfg.embedding_size,
        slstm_at=cfg.slstm_at,
        dropout=cfg.dropout,
        add_post_blocks_norm=False,
    )
    return xLSTMBlockStack(stack_cfg)


class _AlternatingBidirStack(nn.Module):
    """Wraps xLSTMBlockStack so blocks see alternating directions.

    Block 0 sees the sequence forward; each subsequent block flips the input,
    so block 1 is reversed, block 2 forward, etc. With an odd `num_blocks` the
    output ends up in the forward direction (even number of flips). With an
    even `num_blocks` we re-flip the final output to keep it forward-aligned.
    """

    def __init__(self, stack: xLSTMBlockStack, bidirectional: bool):
        super().__init__()
        self.stack = stack
        self.bidirectional = bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flips = 0
        for i, block in enumerate(self.stack.blocks):
            if self.bidirectional and i > 0:
                x = x.flip(1).contiguous()
                flips += 1
            x = block(x)
        if self.bidirectional and flips % 2 == 1:
            x = x.flip(1).contiguous()
        return x


class xECG(nn.Module):
    def __init__(self, cfg: xECGConfig, batch_size_hint: int = 64, lambda_code_rate: float = 0.1, sim_dino_eps: float = 0.05, gather_for_expansion: bool = True):
        super().__init__()
        self.cfg = cfg
        self.lambda_code_rate = lambda_code_rate

        self.patch_embed = nn.Conv1d(cfg.num_leads, cfg.embedding_size, kernel_size=cfg.patch_size, stride=cfg.patch_size, bias=False)
        # No positional embedding: the xLSTM recurrence is inherently order-aware,
        # matching the paper and bench-xecg (neither adds positional encodings).
        self.mask_token = nn.Parameter(torch.zeros(cfg.embedding_size))

        stack = _build_block_stack(cfg, batch_size_hint)
        self.encoder = _AlternatingBidirStack(stack, bidirectional=cfg.bidirectional)
        self.norm = nn.LayerNorm(cfg.embedding_size) if cfg.post_encoder_norm else nn.Identity()
        self.attn_pool = AttentionPooling(cfg.embedding_size, cfg.num_heads)

        self.sim_dino_loss = SimDINOv2Loss(eps=sim_dino_eps, gather_for_expansion=gather_for_expansion)
        self._teacher: Optional["xECG"] = None

    def patchify(self, signal: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(signal)
        return x.transpose(1, 2)

    def _block_mask(self, num_patches: int, batch_size: int, device: torch.device) -> torch.Tensor:
        # block masking: each starting position with prob (mask_ratio/4) extends to 4 contiguous patches.
        mask = torch.rand(batch_size, num_patches, device=device) < (self.cfg.mask_ratio / 4)
        for _ in range(3):
            mask = mask | mask.roll(-1, dims=1)
        return mask

    def _random_mask(self, num_patches: int, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.rand(batch_size, num_patches, device=device) < self.cfg.mask_ratio

    def make_mask(self, num_patches: int, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.cfg.masking_type == "block":
            return self._block_mask(num_patches, batch_size, device)
        return self._random_mask(num_patches, batch_size, device)

    def encode(self, signal: torch.Tensor, mask: Optional[torch.Tensor] = None, padding_mask: Optional[torch.Tensor] = None) -> xECGOutput:
        x = self.patchify(signal)

        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), self.mask_token.to(x.dtype).expand_as(x), x)

        out = self.encoder(x)
        out = self.norm(out)
        cls = self.attn_pool(out, key_padding_mask=padding_mask)
        return xECGOutput(patches=out, cls=cls, mask=mask)

    def forward(self, signal: Optional[torch.Tensor] = None, global_signals: Optional[torch.Tensor] = None, local_signals: Optional[torch.Tensor] = None, masking: bool = True, padding_mask: Optional[torch.Tensor] = None):
        if global_signals is not None:
            return self._pretrain_forward(global_signals, local_signals)
        B = signal.size(0)
        N = signal.size(-1) // self.cfg.patch_size
        mask = self.make_mask(N, B, signal.device) if masking else None
        return self.encode(signal, mask=mask, padding_mask=padding_mask)

    def _pretrain_forward(self, global_signals: torch.Tensor, local_signals: torch.Tensor) -> xECGPretrainOutput:
        # global_signals: (B, V_g, C, T_g); local_signals: (B, V_l, C, T_l)
        assert self._teacher is not None, "call init_teacher() before pretrain forward"
        B, V_g, C, T_g = global_signals.shape
        _, V_l, _, T_l = local_signals.shape
        N_g = T_g // self.cfg.patch_size

        g_flat = global_signals.reshape(B * V_g, C, T_g)
        l_flat = local_signals.reshape(B * V_l, C, T_l)

        g_mask = self.make_mask(N_g, B * V_g, g_flat.device)
        student_g = self.encode(g_flat, mask=g_mask)
        student_l = self.encode(l_flat, mask=None)
        with torch.no_grad():
            teacher_g = self.teacher_fwd(g_flat)

        # CLS tensors regrouped to [V, B, D] as expected by SimDINOv2Loss.
        cls_g_s = student_g.cls.view(B, V_g, -1).transpose(0, 1)
        cls_l_s = student_l.cls.view(B, V_l, -1).transpose(0, 1)
        cls_g_t = teacher_g.cls.view(B, V_g, -1).transpose(0, 1)
        cls_student = torch.cat([cls_g_s, cls_l_s], dim=0)
        compression, expansion = self.sim_dino_loss(cls_student, cls_g_t)

        # Patch loss is over student-masked / teacher-unmasked global patches only.
        patch_loss = masked_cosine_loss(student_g.patches, teacher_g.patches, mask=g_mask)

        loss = compression + self.lambda_code_rate * expansion + patch_loss
        return xECGPretrainOutput(loss=loss, compression=compression.detach(), expansion=expansion.detach(), patch=patch_loss.detach())

    def get_features(self, signal: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encode(signal, mask=None, padding_mask=padding_mask).patches

    def init_teacher(self):
        teacher = deepcopy(self)
        teacher._teacher = None
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.eval()
        self._teacher = teacher

    @torch.no_grad()
    def teacher_fwd(self, signal: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> xECGOutput:
        return self._teacher.encode(signal, mask=None, padding_mask=padding_mask)

    @torch.no_grad()
    def update_teacher(self, beta: float):
        student_params = dict(self.named_parameters())
        for name, p_t in self._teacher.named_parameters():
            p_s = student_params.get(name)
            if p_s is None:
                continue
            p_t.data.mul_(beta).add_(p_s.data, alpha=1.0 - beta)

    def trainable_parameters(self):
        for name, p in self.named_parameters():
            if name.startswith("_teacher."):
                continue
            yield p

    def get_param_groups(self, base_lr: float, weight_decay: float, layerwise_lr_decay: float = 1.0):
        """Bench-xECG layer-wise LR decay: deepest block keeps base_lr, each earlier block
        scales by `decay`, patch embedding is one step below the first block. The post-encoder
        norm, attention pool, and mask_token train at base_lr (no decay).
        """
        blocks = self.encoder.stack.blocks
        num_layers = len(blocks) + 1
        groups = []
        for i, block in enumerate(blocks):
            scale = layerwise_lr_decay ** (num_layers - i - 1)
            groups.append({
                "params": [p for p in block.parameters() if p.requires_grad],
                "lr_scale": scale,
                "weight_decay": weight_decay,
                "name": f"block_{i}",
            })
        groups.append({
            "params": [p for p in self.patch_embed.parameters() if p.requires_grad],
            "lr_scale": layerwise_lr_decay ** num_layers,
            "weight_decay": weight_decay,
            "name": "patch_embed",
        })
        head_params = list(self.attn_pool.parameters()) + [self.mask_token]
        if isinstance(self.norm, nn.LayerNorm):
            head_params += list(self.norm.parameters())
        groups.append({
            "params": [p for p in head_params if p.requires_grad],
            "lr_scale": 1.0,
            "weight_decay": weight_decay,
            "name": "head",
        })
        return groups
