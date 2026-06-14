from typing import Tuple
import numpy as np
from transformers import AutoTokenizer

from configs.constants import PTB_INDEPENDENT_IDX

class Signal:
    def __init__(self, args):
        self.args = args
        self.text_feature_extractor = AutoTokenizer.from_pretrained(self.args.text_feature_extractor) if self.args.text_feature_extractor else None

    def __call__(self, data):
        normalized_data = self.normalize(data["ecg"])
        padded_data, padding_mask = self.pad(normalized_data)
        report = data.get("report", "")
        result =  {
            "transformed_data": padded_data,
            "padding_mask": padding_mask,
        }

        if self.args.condition == "lead":
            result["condition"] = padded_data[self.args.condition_lead].astype(np.float32)
            other = [i for i in PTB_INDEPENDENT_IDX if i != self.args.condition_lead]
            result["transformed_data"] = padded_data[other]

        if self.text_feature_extractor:
            result["condition"] = self.encode_text(report, self.args.condition_text_max_len)
                    
        if self.args.task in ["reconstruction", "generation"]:
            result["report"] = report
            result["12_lead_gt"] = padded_data
        if self.args.neural_network == "dbeta":
            result["report"] = report
        return result

    def pad(self, signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        M, N = signals.shape
        valid_len = min(N, self.args.segment_len)
        padded = np.zeros((12, self.args.segment_len), dtype=np.float32)
        padded[:M, :valid_len] = signals[:, :valid_len]

        padding_mask = np.arange(self.args.segment_len) >= valid_len

        return padded, padding_mask

    def normalize(self, arr: np.ndarray):
        if self.args.ecg_norm == "instance_minmax":
            mn = arr.min()
            mx = arr.max()
            denom = mx - mn + self.args.norm_eps
            return np.clip((arr - mn) / denom, 0.0, 1.0)
        elif self.args.ecg_norm == "instance_zscore":
            mean = arr.mean()
            std = arr.std() + self.args.norm_eps
            return (arr - mean) / std
        elif self.args.ecg_norm == "lead_minmax":
            mn = arr.min(axis=1, keepdims=True)
            mx = arr.max(axis=1, keepdims=True)
            denom = mx - mn + self.args.norm_eps
            return np.clip((arr-mn) / denom, 0.0, 1.0)
        elif self.args.ecg_norm == "lead_zscore":
            mean = arr.mean(axis = 1, keepdims=True)
            std = arr.std(axis=1, keepdims=True) + self.args.norm_eps
            return (arr - mean) / std
        else:
            print("Please choose a normalization method")
    
    def encode_text(self, text, max_len: int = 128) -> np.ndarray:
        tokenizer_out = self.text_feature_extractor(text = text, padding = "max_length", 
                                                    max_length = max_len, truncation = True,
                                                    return_tensors= "pt")
        return {k: v.squeeze(0) for k, v in tokenizer_out.items()}