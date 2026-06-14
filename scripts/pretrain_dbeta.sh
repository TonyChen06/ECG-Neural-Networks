# D-BETA pretraining (Pham et al. 2025, ICML, arXiv:2410.02131).
# Cross-modal masked ECG-text auto-encoder: ECG conv-transformer encoder (62M) +
# Flan-T5-base text encoder + 6 BertCrossLayer fusion blocks, trained with
# L = L_mlm + L_mem + L_etm + L_ets (equal weights). ~321M total during pretraining;
# the downstream artifact is the ~63M ECG encoder + projection/pooler.
#
# Faithful recipe (paper): Adam lr=5e-5, betas (0.9, 0.98), eps 1e-6, wd 0.01,
# mem mask 0.75, mlm mask 0.25, 50% N3S hard-negative corruption.
# We adapt to 250 Hz / 2500 samples (156 ECG tokens) instead of the paper's
# 500 Hz / 5000 (311 tokens); the conv front-end is resolution-agnostic.
#
# Negatives: in-batch by default. To use the faithful FAISS N3S hard negatives,
# first build the index (see scripts/build_dbeta_n3s.sh) and pass --dbeta_n3s_index.
#
# Per-GPU batch is small because the 321M cross-modal model + T5 is memory-heavy
# on 48GB A6000s; 4 x 16 = 64 effective. Raise if memory allows (paper used 128).

CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:$PATH \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
uv run torchrun --standalone --nproc_per_node=4 --master_port=29405 \
src/pretrain_encoder.py \
--data mimic_iv \
--data_representation "signal" \
--objective "dbeta" \
--neural_network "dbeta" \
--task "pretrain" \
--ecg_norm lead_zscore \
--dbeta_text_model google/flan-t5-base \
--dbeta_max_text_len 256 \
--dbeta_mlm_prob 0.25 \
--dbeta_mem_prob 0.75 \
--dbeta_neg_ratio 0.5 \
--batch_size 16 \
--distributed \
--ref_global_bs 64 \
--epochs 20 \
--optimizer adam \
--lr 5e-5 \
--beta1 0.9 \
--beta2 0.98 \
--eps 1e-6 \
--weight_decay 0.01 \
--lr_schedule cosine \
--warmup 5000 \
--bfloat_16 \
--patience 999 \
--patience_delta 0.0 \
--num_workers 8 \
--wandb
