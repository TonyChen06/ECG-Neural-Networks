import torch
import numpy as np

from neural_networks.xecg.augmentations import jitter, random_amplitude_scale, random_crop, random_drop_leads, random_resample


class Pretrain:
    def __init__(self, args):
        self.args = args

    def __call__(self, transformed_data):
        if self.args.data_representation == "bpe_symbolic":
            return self.bpe_symbolic(transformed_data,)
        elif self.args.data_representation == "signal":
            return self.signal(transformed_data,)

    def signal(self, transformed_data,):
        inputs = np.asarray(transformed_data["transformed_data"])
        if self.args.neural_network == "xecg":
            return self._xecg_multiview(inputs)
        if self.args.neural_network in ("mlae", "mtae", "st_mem"):
            out = {"signal": inputs.astype(np.float32)}
            if self.args.neural_network in ("mtae", "st_mem") and "padding_mask" in transformed_data:
                out["padding_mask"] = np.asarray(transformed_data["padding_mask"], dtype=bool)
            return out
        if self.args.objective == "autoregressive":
            out =  {
                "signal": inputs,
            }
        elif self.args.objective == "merl":
            out =  {
                "signal": inputs,
            }
            condition = transformed_data.get("condition")
            out["condition"] = condition
        elif self.args.objective in ("rectified_flow", "ddpm"):
            out =  {
                "signal": inputs,
            }
            if self.args.condition:
                condition = transformed_data.get("condition")
                out["condition"] = condition
            if self.args.task in ["reconstruction", "generation"]:
                out["report"] = transformed_data["report"]
                out["12_lead_gt"] = transformed_data["12_lead_gt"]
        elif self.args.objective == "mae":
            # Each lead is a patch and we mask out 75% of them (9 leads)
            num_masked = int(self.args.num_patches * 0.75)
            perm = np.random.permutation(self.args.num_patches)
            visible_mask = np.ones(self.args.num_patches, dtype = np.float32)
            visible_mask[perm[:num_masked]] = 0
            targets = inputs.copy()
            out = {
                "patches": inputs.astype(np.float32),
                "visible_mask": visible_mask,
                "targets": targets.astype(np.float32),
            }
        return out

    def _xecg_multiview(self, inputs: np.ndarray) -> dict:
        # Augmentation-only multi-view: every view is a crop of one recording.
        return self._xecg_views([inputs])

    def xecg_patient_multiview(self, signals: list) -> dict:
        # Patient-pair multi-view: `signals` is a list of normalized (C, T) recordings
        # of the same patient. Following bench-xecg, global view i is drawn from
        # signals[i % n], and local view i from signals[(i + n_global) % n].
        return self._xecg_views(signals)

    def _xecg_views(self, source_signals: list) -> dict:
        # source_signals: list of (C, T) recordings, each already normalized by Signal.
        T = self.args.segment_len
        ps = self.args.xecg_patch_size
        global_len = (int(self.args.xecg_global_crop * T) // ps) * ps
        local_len = (int(self.args.xecg_local_crop * T) // ps) * ps

        n_global = self.args.xecg_n_global
        n_local = self.args.xecg_n_local
        ns = len(source_signals)
        n_leads = source_signals[0].shape[0]
        globals_ = np.empty((n_global, n_leads, global_len), dtype=np.float32)
        locals_ = np.empty((n_local, n_leads, local_len), dtype=np.float32)

        for i in range(n_global):
            globals_[i] = self._augment_view(random_crop(source_signals[i % ns], global_len))
        for i in range(n_local):
            locals_[i] = self._augment_view(random_crop(source_signals[(i + n_global) % ns], local_len))

        return {"global_signals": globals_, "local_signals": locals_}

    def _augment_view(self, view: np.ndarray) -> np.ndarray:
        view = view.astype(np.float32, copy=False)
        view = random_drop_leads(view, prob=self.args.xecg_drop_leads_prob, keep_lead_idx=1)
        view = jitter(view, sigma=0.1, amplitude=0.6, prob=self.args.xecg_jitter_prob)
        view = random_resample(view, max_ratio=getattr(self.args, "xecg_resample_ratio", 0.0))
        view = random_amplitude_scale(view, amplitude_range=0.2, prob=self.args.xecg_amp_scale_prob)
        return view

    def bpe_symbolic(self, transformed_data):
        inputs = np.asarray(transformed_data["transformed_data"])
        labels = inputs.copy()
        labels[labels == self.args.pad_id] = -100
        labels[labels == self.args.bos_id] = -100
        inputs = torch.as_tensor(inputs, dtype=torch.long)
        labels = torch.as_tensor(labels, dtype=torch.long) if labels is not None else None
        out = {"tgt_ids": inputs, "labels": labels}
        if getattr(self.args, "signal_head", False):
            out["signal"] = torch.as_tensor(transformed_data["signal"], dtype=torch.float32)
        return out