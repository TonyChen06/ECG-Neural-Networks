"""Batch collation for D-BETA pretraining.

Per-sample items carry the raw ECG and report text; all text work happens here
at batch level, mirroring bench D-BETA's `RawECGTextDataset.collator`:
  1. choose ~neg_ratio of the batch to corrupt with a non-matching report
     (in-batch random by default; a FAISS N3S retriever when provided),
  2. tokenize with the T5 tokenizer,
  3. apply BERT-style MLM masking (DataCollatorForLanguageModeling),
  4. emit is_aligned labels for the ETM/ETS losses.
"""

import numpy as np
import torch
from transformers import T5TokenizerFast, DataCollatorForLanguageModeling

MASK_SENTINEL = "<extra_id_0>"  # T5 has no [MASK]; reuse a sentinel as the MLM mask


class DBETACollator:
    def __init__(self, tokenizer_name="google/flan-t5-base", max_text_size=256,
                 mlm_prob=0.25, neg_ratio=0.5, n3s_retriever=None, seed=0):
        self.tokenizer = T5TokenizerFast.from_pretrained(tokenizer_name)
        self.tokenizer.mask_token = MASK_SENTINEL
        self.max_text_size = max_text_size
        self.mlm_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=mlm_prob)
        self.neg_ratio = neg_ratio
        self.n3s = n3s_retriever
        self.rng = np.random.default_rng(seed)

    def _negative_text(self, idx, reports):
        if self.n3s is not None:
            return self.n3s.negative(reports[idx])
        # in-batch: a random other sample's report
        others = [j for j in range(len(reports)) if j != idx]
        return reports[int(self.rng.choice(others))] if others else reports[idx]

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        ecgs = torch.stack([torch.as_tensor(b["ecg"], dtype=torch.float32) for b in batch])
        reports = [b["report"] for b in batch]
        n = len(batch)

        is_aligned = np.ones(n, dtype=np.int64)
        num_neg = int(round(n * self.neg_ratio))
        if num_neg > 0 and n > 1:
            neg_idx = self.rng.choice(n, size=min(num_neg, n), replace=False)
            texts = list(reports)
            for i in neg_idx:
                texts[i] = self._negative_text(int(i), reports)
                is_aligned[i] = 0
        else:
            texts = reports

        enc = self.tokenizer(texts, truncation=True, max_length=self.max_text_size, padding=False)
        features = [{"input_ids": ids} for ids in enc["input_ids"]]
        mlm = self.mlm_collator(features)

        return {
            "ecg": ecgs,
            "text": mlm["input_ids"],
            "text_attention_mask": mlm["attention_mask"],
            "mlm_labels": mlm["labels"],
            "is_aligned": torch.from_numpy(is_aligned),
        }
