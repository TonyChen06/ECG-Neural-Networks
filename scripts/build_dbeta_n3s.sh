# Precompute the D-BETA N3S hard-negative FAISS index over the pretraining reports.
# One-time job: embeds every unique report with flan-t5-small, builds an IndexFlatL2,
# and saves <out>.faiss / <out>_emb.npy / <out>.json. Pass --dbeta_n3s_index data/dbeta_n3s
# to pretrain_encoder.py afterwards to switch from in-batch to N3S hard negatives.

CUDA_VISIBLE_DEVICES=4 \
uv run python src/dataloaders/n3s.py \
--data mimic_iv \
--segment_len 2500 \
--model google/flan-t5-small \
--batch_size 512 \
--out data/dbeta_n3s
