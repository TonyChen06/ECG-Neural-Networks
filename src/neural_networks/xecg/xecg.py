# Loose adaptation of xECG (Lunelli et al. 2024, "BenchECG and xECG").
# Original repo: https://github.com/dlaskalab/bench-xecg
# Architecture: bidirectional xLSTM (Beck et al. 2024) over cross-lead time patches.
# Pretraining: masked patch reconstruction, MAE-style.

from dataclasses import dataclass
from typing import Optional

import torch
from einops import rearrange
from torch import nn

from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig
from xlstm.blocks.mlstm.block import mLSTMBlockConfig
from xlstm.blocks.mlstm.layer import mLSTMLayerConfig
from xlstm.blocks.slstm.block import sLSTMBlockConfig
from xlstm.blocks.slstm.layer import sLSTMLayerConfig


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
    d_model: int = None

    def __post_init__(self):
        assert self.seq_len % self.patch_size == 0, "seq_len must be divisible by patch_size"
        self.num_patches = self.seq_len // self.patch_size
        if self.slstm_at is None:
            self.slstm_at = list(range(3, self.num_blocks, 4))
        if self.d_model is None:
            self.d_model = self.embedding_size * (2 if self.bidirectional else 1)


@dataclass
class xECGOutput:
    loss: Optional[torch.Tensor]
    out: Optional[torch.Tensor]


def _build_stack(cfg: xECGConfig) -> xLSTMBlockStack:
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
        context_length=cfg.num_patches,
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

        patch_dim = cfg.num_leads * cfg.patch_size
        self.patch_proj = nn.Linear(patch_dim, cfg.embedding_size)
        self.pos_embed = nn.Parameter(torch.randn(1, cfg.num_patches, cfg.embedding_size) * 0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.embedding_size))
        nn.init.normal_(self.mask_token, std=0.02)

        self.xlstm_fwd = _build_stack(cfg)
        if cfg.bidirectional:
            self.xlstm_bwd = _build_stack(cfg)

        out_dim = cfg.embedding_size * (2 if cfg.bidirectional else 1)
        self.norm = nn.LayerNorm(out_dim)
        self.reconstruction = nn.Linear(out_dim, patch_dim)

    def patchify(self, signal: torch.Tensor) -> torch.Tensor:
        return rearrange(signal, "b c (n p) -> b n (c p)", p=self.patch_size)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "b n (c p) -> b c (n p)", c=self.num_leads)

    def random_mask(self, x: torch.Tensor):
        B, N, D = x.shape
        num_mask = int(self.cfg.mask_ratio * N)
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        mask = torch.zeros(B, N, device=x.device, dtype=x.dtype)
        mask.scatter_(1, ids_shuffle[:, :num_mask], 1.0)
        mask_tokens = self.mask_token.expand(B, N, D).to(x.dtype)
        masked_x = torch.where(mask.unsqueeze(-1).bool(), mask_tokens, x)
        return masked_x, mask

    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        out_fwd = self.xlstm_fwd(x)
        if self.cfg.bidirectional:
            out_bwd = self.xlstm_bwd(torch.flip(x, dims=[1]))
            out_bwd = torch.flip(out_bwd, dims=[1])
            out = torch.cat([out_fwd, out_bwd], dim=-1)
        else:
            out = out_fwd
        return self.norm(out)

    def forward_loss(self, signal, pred, mask, padding_mask=None):
        target = self.patchify(signal)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)

        if padding_mask is not None:
            B, _ = padding_mask.shape
            per_patch_padded = padding_mask.view(B, self.num_patches, self.patch_size).any(dim=-1)
            mask = mask * (~per_patch_padded).to(mask.dtype)

        return (loss * mask).sum() / mask.sum().clamp(min=1.0)

    def get_features(self, signal: torch.Tensor) -> torch.Tensor:
        x = self.patch_proj(self.patchify(signal)) + self.pos_embed
        return self.forward_encoder(x)

    def forward(self, signal: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> xECGOutput:
        x = self.patch_proj(self.patchify(signal))
        x = x + self.pos_embed
        masked_x, mask = self.random_mask(x)
        encoded = self.forward_encoder(masked_x)
        pred = self.reconstruction(encoded)
        loss = self.forward_loss(signal, pred, mask, padding_mask)
        return xECGOutput(loss=loss, out=encoded)
