# xECG: xLSTM-based ECG encoder.
#
# Two pretraining strategies:
#
#   --xecg_strategy mae (default; backward-compatible with our earlier
#       xLSTM-MAE adaptation): single-signal masked patch reconstruction
#       with norm_pix_loss. Loss returned directly from forward(signal=...).
#
#   --xecg_strategy sim_dino_v2 (faithful to bench-xECG / Lunelli et al.
#       2024): multi-view DINO-v2 self-distillation with EMA teacher.
#       Compression + lambda * expansion + patch-level cosine similarity
#       against the teacher. Driven from forward(global_signals=...).
#
# Regardless of strategy the encoder always emits a sequence of length
# num_patches + 1 internally (CLS at index 0). For ELM downstream,
# get_features(signal) returns the same (B, num_patches, out_dim) it
# always did -- CLS is only used at training time for the DINO objective.

import copy
from dataclasses import dataclass, field
from typing import Optional

import torch
from einops import rearrange
from torch import nn

from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig
from xlstm.blocks.mlstm.block import mLSTMBlockConfig
from xlstm.blocks.mlstm.layer import mLSTMLayerConfig
from xlstm.blocks.slstm.block import sLSTMBlockConfig
from xlstm.blocks.slstm.layer import sLSTMLayerConfig

from neural_networks.xecg.losses import SimDINOv2Loss, masked_cosine_loss


@dataclass
class xECGConfig:
    seq_len: int = 2500
    patch_size: int = 100
    num_leads: int = 12
    embedding_size: int = 768
    num_blocks: int = 12
    num_heads: int = 4
    conv1d_kernel_size: int = 4
    qkv_proj_blocksize: int = 4
    mask_ratio: float = 0.5
    norm_pix_loss: bool = True
    dropout: float = 0.0
    bidirectional: bool = True
    slstm_at: Optional[list] = None
    pretrain_strategy: str = "mae"
    # DINO-specific knobs (only used when pretrain_strategy == "sim_dino_v2");
    # defaults mirror bench-xECG's pretrain_run_config.yaml.
    dino_lambda_code_rate: float = 0.1
    dino_eps: float = 0.05
    dino_patch_loss_weight: float = 1.0
    dino_ema_start: float = 0.99
    dino_ema_end: float = 1.0
    d_model: int = None

    def __post_init__(self):
        assert self.seq_len % self.patch_size == 0, "seq_len must be divisible by patch_size"
        assert self.pretrain_strategy in ("mae", "sim_dino_v2"), self.pretrain_strategy
        self.num_patches = self.seq_len // self.patch_size
        if self.slstm_at is None:
            self.slstm_at = list(range(3, self.num_blocks, 4))
        if self.d_model is None:
            self.d_model = self.embedding_size * (2 if self.bidirectional else 1)


@dataclass
class xECGOutput:
    loss: Optional[torch.Tensor]
    out: Optional[torch.Tensor]


def _build_stack(cfg: xECGConfig, context_length: int) -> xLSTMBlockStack:
    """Build a single xLSTMBlockStack for a given sequence length.

    Sequence length includes the CLS token, so callers pass num_patches + 1.
    """
    mlstm_cfg = mLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(
            conv1d_kernel_size=cfg.conv1d_kernel_size,
            qkv_proj_blocksize=cfg.qkv_proj_blocksize,
            num_heads=cfg.num_heads,
        )
    )
    # vanilla backend: xlstm 2.0.5's CUDA backend fails to JIT-compile against modern PyTorch.
    slstm_cfg = sLSTMBlockConfig(
        slstm=sLSTMLayerConfig(
            num_heads=cfg.num_heads,
            conv1d_kernel_size=cfg.conv1d_kernel_size,
            backend="vanilla",
        )
    ) if cfg.slstm_at else None
    stack_cfg = xLSTMBlockStackConfig(
        mlstm_block=mlstm_cfg,
        slstm_block=slstm_cfg,
        context_length=context_length,
        num_blocks=cfg.num_blocks,
        embedding_dim=cfg.embedding_size,
        slstm_at=cfg.slstm_at,
        dropout=cfg.dropout,
    )
    return xLSTMBlockStack(stack_cfg)


class xECG(nn.Module):
    def __init__(self, cfg: xECGConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg.patch_size
        self.num_patches = cfg.num_patches
        self.num_leads = cfg.num_leads
        self.norm_pix_loss = cfg.norm_pix_loss
        self.strategy = cfg.pretrain_strategy

        # Token sequence is [CLS, patch_0, patch_1, ..., patch_{n-1}] of length num_patches + 1.
        ctx_len = cfg.num_patches + 1
        patch_dim = cfg.num_leads * cfg.patch_size
        self.patch_proj = nn.Linear(patch_dim, cfg.embedding_size)
        self.pos_embed = nn.Parameter(torch.randn(1, ctx_len, cfg.embedding_size) * 0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embedding_size))
        nn.init.xavier_uniform_(self.cls_token, gain=1.0)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.embedding_size))
        nn.init.normal_(self.mask_token, std=0.02)

        self.xlstm_fwd = _build_stack(cfg, ctx_len)
        if cfg.bidirectional:
            self.xlstm_bwd = _build_stack(cfg, ctx_len)

        out_dim = cfg.embedding_size * (2 if cfg.bidirectional else 1)
        self.norm = nn.LayerNorm(out_dim)
        # Separate norm for the CLS feature (matches bench-xECG's normalization_layer pattern).
        self.cls_norm = nn.LayerNorm(out_dim)

        # Reconstruction head is only used in mae strategy. Constructing it
        # unconditionally keeps state_dict shape stable across strategies and
        # makes it trivial to flip strategies for ablation without rewriting
        # the architecture; the parameters are simply unused in dino mode.
        self.reconstruction = nn.Linear(out_dim, patch_dim)

        # DINO loss, only used at training time when strategy=sim_dino_v2.
        self.dino_loss = SimDINOv2Loss(eps=cfg.dino_eps)

        # Teacher network is created lazily by init_teacher() after the
        # student is constructed. Stored as a regular submodule but filtered
        # out of state_dict() so checkpoints stay small and ELM-compatible.
        self._teacher: Optional[nn.Module] = None

    # ----- patching utilities ---------------------------------------------

    def patchify(self, signal: torch.Tensor) -> torch.Tensor:
        """signal: (B, num_leads, seq_len) -> (B, num_patches, num_leads*patch_size)."""
        return rearrange(signal, "b c (n p) -> b n (c p)", p=self.patch_size)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "b n (c p) -> b c (n p)", c=self.num_leads)

    # ----- shared encoding path -------------------------------------------

    def _random_patch_mask(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Returns a (B, num_patches) mask where 1 = masked (replace with mask_token), 0 = visible.

        Each sample independently masks floor(num_patches * mask_ratio) positions.
        """
        n = self.num_patches
        num_mask = int(self.cfg.mask_ratio * n)
        noise = torch.rand(batch_size, n, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        mask = torch.zeros(batch_size, n, device=device, dtype=dtype)
        mask.scatter_(1, ids_shuffle[:, :num_mask], 1.0)
        return mask

    def _encode(
        self,
        signal: torch.Tensor,
        masking: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Run the encoder backbone. Always emits CLS + patches.

        Returns:
            cls: (B, out_dim)
            patches: (B, num_patches, out_dim)
            mask: (B, num_patches) or None if masking=False -- 1 means the patch
                was replaced by mask_token at the input.
        """
        B = signal.shape[0]
        x = self.patch_proj(self.patchify(signal))  # (B, n, embed)

        if masking:
            patch_mask = self._random_patch_mask(B, signal.device, x.dtype)  # (B, n)
            mask_tokens = self.mask_token.expand(B, self.num_patches, -1).to(x.dtype)
            x = torch.where(patch_mask.unsqueeze(-1).bool(), mask_tokens, x)
        else:
            patch_mask = None

        # Prepend CLS, then add positional embedding (one slot per token incl. CLS).
        cls = self.cls_token.expand(B, -1, -1).to(x.dtype)
        x = torch.cat([cls, x], dim=1)  # (B, n+1, embed)
        x = x + self.pos_embed.to(x.dtype)

        # Bidirectional: forward stack on x, backward stack on reversed x.
        out_fwd = self.xlstm_fwd(x)
        if self.cfg.bidirectional:
            out_bwd = self.xlstm_bwd(torch.flip(x, dims=[1]))
            out_bwd = torch.flip(out_bwd, dims=[1])
            out = torch.cat([out_fwd, out_bwd], dim=-1)
        else:
            out = out_fwd

        out = self.norm(out)
        cls_out = self.cls_norm(out[:, 0])
        patches_out = out[:, 1:]
        return cls_out, patches_out, patch_mask

    # ----- MAE strategy ----------------------------------------------------

    def _mae_recon_loss(
        self, signal: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor, padding_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        target = self.patchify(signal)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5

        per_elem = (pred - target) ** 2
        per_patch = per_elem.mean(dim=-1)  # (B, n)

        if padding_mask is not None:
            B = padding_mask.shape[0]
            per_patch_padded = padding_mask.view(B, self.num_patches, self.patch_size).any(dim=-1)
            mask = mask * (~per_patch_padded).to(mask.dtype)

        return (per_patch * mask).sum() / mask.sum().clamp(min=1.0)

    def _forward_mae(self, signal: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> xECGOutput:
        cls, patches, mask = self._encode(signal, masking=True)
        pred = self.reconstruction(patches)
        loss = self._mae_recon_loss(signal, pred, mask, padding_mask)
        return xECGOutput(loss=loss, out=patches)

    # ----- DINO strategy ---------------------------------------------------

    def _patch_padding_mask(self, padding_mask: Optional[torch.Tensor], B: int, device: torch.device) -> Optional[torch.Tensor]:
        """Convert per-sample (B, seq_len) bool padding mask to per-patch (B, num_patches)."""
        if padding_mask is None:
            return None
        return padding_mask.view(B, self.num_patches, self.patch_size).any(dim=-1)

    def _forward_dino(
        self,
        global_signals: torch.Tensor,
        padding_masks: Optional[torch.Tensor],
    ) -> xECGOutput:
        """global_signals: (B, n_global, num_leads, seq_len). padding_masks: (B, n_global, seq_len) bool or None.

        Implements bench-xECG's reconstruct_batch_sim_dino_v2:
            student passes globals with masking and locals without masking;
            teacher passes globals without masking, no_grad;
            loss = compression(student CLS vs teacher CLS, cross-view)
                 + lambda_code_rate * expansion(student CLS over globals only)
                 + patch_loss_weight * masked_cosine(student patches vs teacher patches, masked positions)
        """
        if self._teacher is None:
            raise RuntimeError("xECG: DINO strategy requires init_teacher() to have been called before forward().")

        B, n_global, num_leads, seq_len = global_signals.shape
        # Per-view forward, with masking. We don't currently support local views;
        # the SimDINOv2Loss math still works with student_views == teacher_views == n_global.
        student_cls_list = []
        student_patches_list = []
        student_masks_list = []
        for v in range(n_global):
            cls, patches, mask = self._encode(global_signals[:, v], masking=True)
            student_cls_list.append(cls)
            student_patches_list.append(patches)
            student_masks_list.append(mask)

        # Teacher: no masking, no grad.
        with torch.no_grad():
            teacher_cls_list = []
            teacher_patches_list = []
            for v in range(n_global):
                t_cls, t_patches, _ = self._teacher._encode(global_signals[:, v], masking=False)
                teacher_cls_list.append(t_cls)
                teacher_patches_list.append(t_patches)

        # SimDINOv2 on stacked CLS features.
        student_cls = torch.stack(student_cls_list, dim=0)  # (n_views, B, D)
        teacher_cls = torch.stack(teacher_cls_list, dim=0)
        compression, expansion = self.dino_loss(student_cls, teacher_cls)

        # Patch-level cosine: only count positions that the student saw masked AND aren't padding.
        student_patches = torch.cat(student_patches_list, dim=1)  # (B, n_global*n_patches, D)
        teacher_patches = torch.cat(teacher_patches_list, dim=1)
        masks = torch.cat(student_masks_list, dim=1).bool()  # (B, n_global*n_patches)

        if padding_masks is not None:
            # padding_masks: (B, n_global, seq_len). Convert to per-patch validity per view, concat.
            pad_per_view = []
            for v in range(n_global):
                pad_per_view.append(self._patch_padding_mask(padding_masks[:, v], B, global_signals.device))
            pad_per_patch = torch.cat(pad_per_view, dim=1)  # (B, n_global*n_patches)
            masks = masks & ~pad_per_patch

        patch_loss = masked_cosine_loss(student_patches, teacher_patches, mask=masks)

        total = compression + self.cfg.dino_lambda_code_rate * expansion + self.cfg.dino_patch_loss_weight * patch_loss

        # `out` is the student's first-view patches -- something with the right
        # downstream shape so callers that read .out for sanity logging don't choke.
        return xECGOutput(loss=total, out=student_patches_list[0])

    # ----- Public dispatch -------------------------------------------------

    def forward(
        self,
        signal: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        global_signals: Optional[torch.Tensor] = None,
        padding_masks: Optional[torch.Tensor] = None,
    ) -> xECGOutput:
        if global_signals is not None:
            assert self.strategy == "sim_dino_v2", (
                f"xECG: received global_signals (DINO-style batch) but pretrain_strategy is "
                f"{self.strategy!r}; reconstruct your dataloader or set --xecg_strategy sim_dino_v2."
            )
            return self._forward_dino(global_signals, padding_masks)

        assert signal is not None, "xECG.forward() needs either signal=... or global_signals=..."
        if self.strategy != "mae":
            # Strategy is dino but a single-signal batch came in -- this is the inference
            # path or someone called the model wrong; we fall back to encoding without loss.
            cls, patches, _ = self._encode(signal, masking=False)
            return xECGOutput(loss=None, out=patches)

        return self._forward_mae(signal, padding_mask)

    # ----- Inference path for ELM downstream -------------------------------

    def get_features(self, signal: torch.Tensor) -> torch.Tensor:
        """Return per-patch features for downstream use. Same shape as our previous xECG:
        (B, num_patches, out_dim). The CLS token is computed but not exposed here, to keep
        ELM's connector wiring backward-compatible."""
        _, patches, _ = self._encode(signal, masking=False)
        return patches

    # ----- Teacher network management --------------------------------------

    def init_teacher(self) -> None:
        """Create the EMA teacher as a deep copy of the student.

        Called externally (from build_nn) after model construction. Idempotent.
        """
        if self._teacher is not None:
            return
        teacher = copy.deepcopy(self)
        # Strip nested teacher / loss / reconstruction modules from the teacher so
        # it doesn't recurse and so deepcopy doesn't create teachers-of-teachers.
        if getattr(teacher, "_teacher", None) is not None:
            teacher._teacher = None
        # Reconstruction and dino_loss are non-essential to the teacher's forward
        # path (teacher is only used for representation); drop them so the teacher
        # is leaner and EMA updates don't waste effort tracking them.
        teacher.reconstruction = nn.Identity()
        teacher.dino_loss = nn.Identity()
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.eval()
        # Use object.__setattr__ to register as a submodule via the standard path.
        # We DO want it as an nn.Module so .to(device) and DDP propagate parameters
        # to all ranks -- but we'll filter it from state_dict() below.
        self._teacher = teacher

    @torch.no_grad()
    def update_teacher(self, beta: float) -> None:
        """In-place EMA update: teacher = beta * teacher + (1 - beta) * student.

        Buffers are copied from student each step (matches bench-xECG's behavior --
        the teacher's running stats track the student exactly, only parameters
        are EMA-blended).
        """
        if self._teacher is None:
            raise RuntimeError("update_teacher called before init_teacher")
        student_params = dict(self.named_parameters())
        for name, param_t in self._teacher.named_parameters():
            if name not in student_params:
                continue  # skipped modules (reconstruction was replaced by Identity)
            param_t.data.mul_(beta).add_(student_params[name].data, alpha=1.0 - beta)
        # Sync buffers (e.g. LayerNorm running stats are not buffers in nn.LayerNorm but
        # any future buffers should follow the student exactly, not be EMA-blended).
        student_buffers = dict(self.named_buffers())
        for name, buf_t in self._teacher.named_buffers():
            if name in student_buffers:
                buf_t.data.copy_(student_buffers[name].data)

    def ema_beta_at_step(self, global_step: int, total_steps: int) -> float:
        """Linear schedule from cfg.dino_ema_start to cfg.dino_ema_end over training."""
        if total_steps <= 0:
            return self.cfg.dino_ema_start
        frac = max(0.0, min(1.0, global_step / total_steps))
        return self.cfg.dino_ema_start + frac * (self.cfg.dino_ema_end - self.cfg.dino_ema_start)

    # ----- State dict filtering --------------------------------------------

    def state_dict(self, *args, destination=None, prefix: str = "", keep_vars: bool = False):
        # We deliberately exclude teacher params from saved state. The teacher is
        # a runtime artifact recreated from the student via init_teacher(); keeping
        # it in checkpoints would double the file size and complicate ELM downstream.
        sd = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        teacher_prefix = prefix + "_teacher."
        return type(sd)((k, v) for k, v in sd.items() if not k.startswith(teacher_prefix))

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        # The saved checkpoint won't contain teacher keys; if a teacher exists we
        # need to allow strict=False just for those, then re-init the teacher
        # from the freshly-loaded student.
        had_teacher = self._teacher is not None
        if had_teacher:
            # Drop the teacher submodule so super().load_state_dict doesn't complain
            # about missing keys, then re-init from the student afterward.
            self._teacher = None

        result = super().load_state_dict(state_dict, strict=strict, assign=assign)

        if had_teacher:
            self.init_teacher()

        return result
