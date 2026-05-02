# SimDINOv2 self-distillation losses for the xECG faithful (sim_dino_v2) recipe.
# Math copied directly from bench-xecg (Lunelli et al., 2024) at
# https://github.com/dlaskalab/bench-xecg/blob/release/bench_xecg/utils/loss_utils.py
# Specifically: SimDINOv2Loss (compression + coding-rate expansion) and masked_cosine_loss.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_cosine_loss(student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """1 - mean cosine similarity, optionally restricted to a boolean mask.

    student, teacher: (..., D). mask: (...) bool selecting positions to include.
    Returns a scalar.
    """
    cos = F.cosine_similarity(teacher, student, dim=-1)
    if mask is not None:
        cos = cos[mask]
    return 1.0 - cos.mean()


class SimDINOv2Loss(nn.Module):
    """Compression + coding-rate-expansion loss between student/teacher CLS features.

    student_feat: (n_views_student, B, D) where n_views_student = n_global + n_local.
    teacher_feat: (n_views_teacher, B, D) — teacher only sees globals, so n_views_teacher = n_global.

    Returns (compression_term, expansion_term). The trainer forms
        total_dino = compression + lambda_code_rate * expansion
    where the expansion term is already negated so that adding it minimizes
    -log_det(I + cov), i.e. encourages high-rank features (anti-collapse).

    eps controls the regularizer strength inside the coding-rate estimator.
    The bench-xecg trainer constructs this with eps=0.05.
    """

    def __init__(self, eps: float = 0.05) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        student_feat = F.normalize(student_feat, p=2, dim=-1)
        teacher_feat = F.normalize(teacher_feat, p=2, dim=-1)

        compression = self._compression(student_feat, teacher_feat)
        # Coding-rate expansion is computed only on the views the teacher saw (the globals).
        expansion = self._expansion(student_feat[: len(teacher_feat)])
        return compression, expansion

    @staticmethod
    def _compression(student_feat_list: torch.Tensor, teacher_feat_list: torch.Tensor) -> torch.Tensor:
        # sim shape: (n_teacher, n_student, B). For each (teacher_view, student_view) pair,
        # take cosine similarity averaged over the batch.
        sim = F.cosine_similarity(teacher_feat_list.unsqueeze(1), student_feat_list.unsqueeze(0), dim=-1)
        # Zero out the diagonal so we don't push a teacher view toward "the same" student view.
        # The slicing trick mirrors the upstream implementation exactly.
        sim.view(-1, sim.shape[-1])[:: (len(student_feat_list) + 1), :].fill_(0.0)
        n_loss_terms = len(teacher_feat_list) * len(student_feat_list) - min(len(teacher_feat_list), len(student_feat_list))
        comp = sim.mean(dim=2).sum() / n_loss_terms
        return 1.0 - comp

    def _expansion(self, feat_list: torch.Tensor) -> torch.Tensor:
        # feat_list: (n_views, B, D). For each view, estimate the coding rate
        # log det(I + (D/(B*eps)) * X^T X), summed over views, then averaged.
        # cholesky_ex doesn't support bf16/fp16, so the whole term is computed
        # in fp32 regardless of input dtype. Cost is small (one D x D solve per view).
        feats_fp32 = feat_list.float()
        num_views, m, p = feats_fp32.shape
        cov = torch.einsum("nbc,nbd->ncd", feats_fp32, feats_fp32)  # (n_views, D, D)
        scalar = p / (m * self.eps)
        identity = torch.eye(p, device=cov.device, dtype=cov.dtype)
        loss = feats_fp32.new_zeros(())
        for i in range(num_views):
            # Cholesky-based log-det: stable and matches upstream implementation.
            chol = torch.linalg.cholesky_ex(identity + scalar * cov[i])[0]
            loss = loss + chol.diagonal().log().sum()
        loss = loss / num_views
        # Heuristic balancing factor from the upstream code.
        loss = loss * (self.eps * np.sqrt(m / (p * float(np.min([p, m])))))
        return -loss
