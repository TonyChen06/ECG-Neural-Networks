# Multi-view data representation for xECG with --xecg_strategy sim_dino_v2.
#
# For each input record, produces N independently-augmented "global" views.
# Each view is augmented (per-view gain + noise + optional lead dropout),
# then normalized and padded by a shared Signal instance, so all views land
# in the same numerical regime as the rest of our codebase.
#
# Output dict keys:
#   global_signals : np.ndarray (n_views, num_leads, segment_len) float32
#   padding_masks  : np.ndarray (n_views, segment_len) bool
#
# We deliberately do NOT include local (smaller-crop) views in this first
# implementation: our pos_embed length is tied to num_patches at construction
# time and supporting variable-length views requires interpolating positional
# embeddings (which is doable but added complexity for an unclear win at our
# scale). The SimDINOv2 compression and expansion math both work cleanly with
# n_global only -- locals are an optimization, not a correctness requirement.

import numpy as np

from dataloaders.data_representation.signal import Signal


class MultiViewSignal:
    def __init__(self, args):
        self.args = args
        self.num_global_views = getattr(args, "multi_view_num_global", 2)
        self.gain_low = getattr(args, "multi_view_gain_low", 0.9)
        self.gain_high = getattr(args, "multi_view_gain_high", 1.1)
        self.noise_std_frac = getattr(args, "multi_view_noise_std_frac", 0.02)
        # Per-view lead-dropout probability: with prob p, choose a small random
        # number of leads (1 or 2) and zero them. Disabled by default since
        # bench-xECG ships random_drop_leads=0; available as a knob for ablation.
        self.lead_dropout_prob = getattr(args, "multi_view_lead_dropout_prob", 0.0)
        self._signal = Signal(args)

    def __call__(self, data: dict) -> dict:
        raw_ecg = np.asarray(data["ecg"])
        global_views = []
        padding_masks = []

        for _ in range(self.num_global_views):
            augmented = self._augment_view(raw_ecg)
            view_data = dict(data)
            view_data["ecg"] = augmented
            transformed = self._signal(view_data)
            global_views.append(transformed["transformed_data"])
            padding_masks.append(transformed["padding_mask"])

        result = {
            "global_signals": np.stack(global_views, axis=0).astype(np.float32),
            "padding_masks": np.stack(padding_masks, axis=0).astype(bool),
        }

        # Forward through report / 12-lead-gt / condition keys if present, since
        # downstream tasks may rely on them. Take the first-view normalized signal
        # as the canonical "12_lead_gt" if needed -- views are aug-equivalent so
        # the choice is arbitrary.
        if "report" in data:
            result["report"] = data["report"]
        return result

    def _augment_view(self, signal: np.ndarray) -> np.ndarray:
        """Independent per-view augmentation: gain jitter + Gaussian noise +
        optional lead dropout. Each call samples its own randomness.

        The augmentation is intentionally MILD. DINO's value comes from many
        slightly-different views encouraging the encoder to be invariant to
        small perturbations; aggressive augmentation can erase signal that the
        model needs to learn.
        """
        out = signal.astype(np.float32, copy=True)

        gain = np.random.uniform(self.gain_low, self.gain_high)
        out = out * gain

        if self.noise_std_frac > 0:
            sigma = self.noise_std_frac * float(np.std(out) + 1e-8)
            out = out + np.random.normal(0.0, sigma, out.shape).astype(np.float32)

        if self.lead_dropout_prob > 0 and np.random.rand() < self.lead_dropout_prob:
            n_leads = out.shape[0]
            n_drop = np.random.randint(1, max(2, n_leads // 6))  # at most ~2 of 12
            drop_idx = np.random.choice(n_leads, size=n_drop, replace=False)
            out[drop_idx] = 0.0

        return out
