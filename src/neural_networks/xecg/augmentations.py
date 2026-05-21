"""ECG-specific augmentations for xECG pretraining.

Ports of the bench_xecg augmentations that are actually wired up at the values
used in `configs/pretrain/pretrain_run_config.yaml`. Per-item ops here operate
on numpy (C, T) arrays; the per-batch baseline-wander shuffle operates on a
torch tensor batch and is meant to be called in the train loop.
"""

import numpy as np
import torch

_default_rng = np.random.default_rng()


def _rng(rng: np.random.Generator | None) -> np.random.Generator:
    return rng if rng is not None else _default_rng


def random_drop_leads(x: np.ndarray, prob: float, keep_lead_idx: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
    if prob <= 0.0:
        return x
    g = _rng(rng)
    drop = g.random(x.shape[0]) < prob
    drop[keep_lead_idx] = False
    if drop.all():
        drop[g.integers(0, x.shape[0])] = False
    out = x.copy()
    out[drop] = 0.0
    return out


def jitter(x: np.ndarray, sigma: float = 0.2, amplitude: float = 0.6, prob: float = 0.1, rng: np.random.Generator | None = None) -> np.ndarray:
    g = _rng(rng)
    if prob <= 0.0 or g.random() > prob:
        return x
    noise = g.standard_normal(x.shape).astype(x.dtype) * sigma
    return x + amplitude * x * noise


def random_amplitude_scale(x: np.ndarray, amplitude_range: float = 0.2, prob: float = 0.1, rng: np.random.Generator | None = None) -> np.ndarray:
    g = _rng(rng)
    if prob <= 0.0 or g.random() > prob:
        return x
    scale = (g.random() - 0.5) * amplitude_range + 1.0
    return x * scale


def random_crop(x: np.ndarray, target_len: int, rng: np.random.Generator | None = None) -> np.ndarray:
    g = _rng(rng)
    T = x.shape[-1]
    if target_len >= T:
        return x[..., :target_len] if x.shape[-1] >= target_len else x
    start = int(g.integers(0, T - target_len + 1))
    return x[..., start : start + target_len]


def _extract_baseline(signals: torch.Tensor, fs: float, cutoff: float) -> torch.Tensor:
    """FFT-based lowpass extraction. signals: (B, C, T) → (B, C, T) baseline."""
    T = signals.shape[-1]
    fft = torch.fft.rfft(signals, dim=-1)
    freqs = torch.fft.rfftfreq(T, d=1.0 / fs, device=signals.device)
    mask = (freqs <= cutoff).view(1, 1, -1)
    return torch.fft.irfft(fft * mask, n=T, dim=-1)


def shuffle_baseline_wander_batched(signals: torch.Tensor, fs: float = 250.0, cutoff: float = 0.5) -> torch.Tensor:
    """Replace each sample's <cutoff Hz baseline with a random other sample's baseline.

    signals: (B, C, T). Returns same shape with baselines cross-sample-shuffled.
    Leads that are fully zero (e.g. dropped by RandomDropLeads) stay zero so the
    swap doesn't reintroduce signal into a deliberately-dropped channel.
    """
    if signals.size(0) <= 1:
        return signals
    baseline = _extract_baseline(signals, fs, cutoff)
    perm = torch.randperm(signals.size(0), device=signals.device)
    zeroed_lead = (signals == 0).all(dim=-1, keepdim=True)
    out = signals - baseline + baseline[perm]
    return out.masked_fill(zeroed_lead, 0.0)
