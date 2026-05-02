import torch
import numpy as np


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
        if self.args.neural_network in ("mlae", "mtae", "st_mem", "xecg"):
            out = {"signal": inputs.astype(np.float32)}
            if self.args.neural_network in ("mtae", "st_mem", "xecg") and "padding_mask" in transformed_data:
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