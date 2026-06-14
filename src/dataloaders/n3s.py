"""N3S hard-negative sampling for D-BETA (port of datasets/n3s.py).

Builds a FAISS index over flan-t5-small embeddings of the (deduplicated) training
reports, then retrieves a "hard negative" by searching the NEGATED query — i.e.
the most DISSIMILAR reports. This is how D-BETA sidesteps false negatives in a
templated-report corpus: an exact/near-duplicate of the query is maximally
similar, so it is never returned among the farthest neighbours.

Build once with `python src/dataloaders/n3s.py --data mimic_iv --out data/dbeta_n3s`.
At train time, the retriever uses the precomputed embeddings (no model needed).
"""

import argparse
import glob
import json
import os

import numpy as np


def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = text.replace("ekg", "ecg")
    for ch in ("*** ", " ***", "***", "=-", "="):
        text = text.strip(ch)
    return text


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)


class N3SRetriever:
    """Lazy-loaded so it is picklable across DataLoader workers."""

    def __init__(self, prefix: str, k: int = 64, seed: int = 0):
        self.prefix = prefix
        self.k = k
        self.rng = np.random.default_rng(seed)
        self._index = None
        self._texts = None
        self._embeddings = None
        self._text_to_row = None

    def _ensure_loaded(self):
        if self._index is not None:
            return
        import faiss
        self._index = faiss.read_index(f"{self.prefix}.faiss")
        with open(f"{self.prefix}.json") as f:
            row_to_text = {int(k): v for k, v in json.load(f).items()}
        self._texts = [row_to_text[i] for i in range(len(row_to_text))]
        self._embeddings = np.load(f"{self.prefix}_emb.npy")
        self._text_to_row = {t: i for i, t in enumerate(self._texts)}

    def negative(self, query_text: str) -> str:
        self._ensure_loaded()
        row = self._text_to_row.get(normalize_text(query_text))
        if row is None:
            row = int(self.rng.integers(0, len(self._texts)))
        query = self._embeddings[row : row + 1]
        # negated query -> nearest neighbours of -q are the FARTHEST (most dissimilar) reports
        _, indices = self._index.search(-query, self.k)
        return self._texts[int(self.rng.choice(indices[0]))]


def _gather_reports(data_names, segment_len, data_dir="../data"):
    from utils.dir_file import DirFileManager
    dfm = DirFileManager()
    texts = []
    for name in data_names:
        paths = sorted(glob.glob(os.path.join(data_dir, name, f"preprocessed_{segment_len}", "*.npy")))
        for p in paths:
            t = normalize_text(dfm.open_npy(p).get("report", ""))
            if t:
                texts.append(t)
    return list(set(texts))


def build_index(data_names, out_prefix, segment_len=2500, model_name="google/flan-t5-small",
                batch_size=256, device=None, data_dir="../data"):
    import torch
    import faiss
    from transformers import AutoTokenizer, T5EncoderModel

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"gathering reports from {data_names} ...", flush=True)
    texts = _gather_reports(data_names, segment_len, data_dir)
    print(f"{len(texts)} unique reports", flush=True)

    tok = AutoTokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name).to(device).eval()

    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inp = tok(batch, max_length=256, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            h = model(**inp).last_hidden_state.mean(dim=1).cpu().numpy()
        embs.append(h)
        if i % (batch_size * 20) == 0:
            print(f"  embedded {i + len(batch)}/{len(texts)}", flush=True)
    embeddings = _l2_normalize(np.concatenate(embs, axis=0)).astype(np.float32)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    faiss.write_index(index, f"{out_prefix}.faiss")
    np.save(f"{out_prefix}_emb.npy", embeddings)
    with open(f"{out_prefix}.json", "w") as f:
        json.dump({i: t for i, t in enumerate(texts)}, f)
    print(f"saved index to {out_prefix}.faiss / _emb.npy / .json", flush=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", default="data/dbeta_n3s")
    ap.add_argument("--segment_len", type=int, default=2500)
    ap.add_argument("--model", default="google/flan-t5-small")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--data_dir", default="../data")
    args = ap.parse_args()
    build_index(args.data, args.out, args.segment_len, args.model, args.batch_size, data_dir=args.data_dir)
