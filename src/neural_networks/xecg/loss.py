"""SimDINOv2 loss for xECG (port of bench_xecg/utils/loss_utils.py).

L_total = L_compression + lambda_cr * L_expansion + L_patch

- L_compression: 1 - mean cosine similarity across (teacher_global, student_view)
  pairs, with the (i,i) self-comparison excluded. For unit vectors this is
  equivalent to MSE up to a positive constant, matching the paper's L_view.
- L_expansion: -coding_rate(student_globals) with the SimDINOv2 gamma weighting
  (eps=0.05).
- L_patch: 1 - cosine_similarity(student_patches, teacher_patches), evaluated
  only at masked, non-padded patches.
"""

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _all_gather_with_local_grad(feat: torch.Tensor) -> torch.Tensor:
    """Concat `feat` from every DDP rank along dim=1 (batch).

    `dist.all_gather` doesn't propagate gradients through the gathered tensors;
    we splice our own rank's tensor back in so the local samples retain their
    autograd graph. The other ranks' samples enter the covariance as
    no-grad constants, which is the standard pattern for SimCLR/SimDINO-style
    cross-rank contrastive losses.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return feat
    world_size = dist.get_world_size()
    if world_size == 1:
        return feat
    gathered = [torch.zeros_like(feat) for _ in range(world_size)]
    dist.all_gather(gathered, feat)
    gathered[dist.get_rank()] = feat
    return torch.cat(gathered, dim=1)


def masked_cosine_loss(student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    cos = F.cosine_similarity(teacher, student, dim=-1)
    sel = cos[mask]
    if sel.numel() == 0:
        return torch.zeros((), device=student.device, dtype=student.dtype)
    return 1.0 - sel.mean()


class SimDINOv2Loss(nn.Module):
    def __init__(self, eps: float = 0.05, gather_for_expansion: bool = True):
        super().__init__()
        self.eps = eps
        self.gather_for_expansion = gather_for_expansion

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor):
        student_feat = F.normalize(student_feat, p=2, dim=-1)
        teacher_feat = F.normalize(teacher_feat, p=2, dim=-1)
        compression = self._compression(student_feat, teacher_feat)
        expansion = self._expansion(student_feat[: teacher_feat.shape[0]])
        return compression, expansion

    def _compression(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        # student: [V_s, B, D], teacher: [V_t, B, D] (V_t == 2 globals; V_s == globals + locals)
        # cosine similarity across each (teacher_view, student_view) pair, averaged over batch.
        sim = F.cosine_similarity(teacher.unsqueeze(1), student.unsqueeze(0), dim=-1)  # [V_t, V_s, B]
        # zero out the (i, i) pairs that match identical views.
        v_t, v_s = sim.shape[0], sim.shape[1]
        sim.view(-1, sim.shape[-1])[:: v_s + 1, :].fill_(0)
        n_loss_terms = v_t * v_s - min(v_t, v_s)
        comp = sim.mean(-1).sum() / n_loss_terms
        return 1.0 - comp

    def _expansion(self, feat: torch.Tensor) -> torch.Tensor:
        # Gather across DDP ranks so the covariance sees the full effective
        # batch — log_det is non-linear in the batch dim, so per-rank gradient
        # averaging doesn't reconstruct what we'd see from a global Cov.
        if self.gather_for_expansion:
            feat = _all_gather_with_local_grad(feat)
        v, b, d = feat.shape
        cov = torch.einsum("nbc,nbd->ncd", feat, feat)  # [V, D, D]
        scalar = d / (b * self.eps)
        eye = torch.eye(d, device=feat.device, dtype=cov.dtype)
        loss = torch.zeros((), device=feat.device, dtype=cov.dtype)
        for i in range(v):
            l_chol = torch.linalg.cholesky_ex(eye + scalar * cov[i])[0]
            loss = loss + l_chol.diagonal().log().sum()
        loss = loss / v
        loss = loss * (self.eps * math.sqrt(b / (d * min(d, b))))
        return -loss
