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
# Batch: 64/GPU x 4 = 256 global (2x the paper's 128). epochs=49 matches the
# paper's ~38M sample-view exposure (790k samples), independent of batch size.
# If a GPU OOMs, drop --batch_size to 32 and --ref_global_bs to 128.
#
# LR (the real fix): the divergence was MLM/T5 climbing because we ran the
# PRETRAINED T5 at the same LR as the random heads. D-BETA inherits M3AE's
# (zhjohnchan/M3AE) scheme: pretrained unimodal encoders at the base LR, and the
# prediction heads + cross-modal module at 5x base (--dbeta_lr_multiplier 5).
# So base lr=1e-5 (T5 + ECG encoder), heads/cross-modal at 5e-5. AdamW, eps 1e-8,
# warmup 10000 -- all per M3AE's pretraining config. (T5 stays trainable, faithful
# to the paper; --dbeta_freeze_text is available but not needed.)
# Note: D-BETA trains in fp32 here -- the framework doesn't autocast it (the
# --bfloat_16 flag is a no-op for dbeta), which is why batch is 64/GPU.

# NCCL_P2P_DISABLE=1: this box's GPU peer-to-peer (PCIe ACS/IOMMU) deadlocks NCCL
# collectives -> multi-GPU hangs in the gradient all-reduce. Forcing shared-memory
# staging is slightly slower but works. IB disabled (no InfiniBand on this node).
TOKENIZERS_PARALLELISM=false \
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
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
--batch_size 64 \
--distributed \
--ref_global_bs 256 \
--epochs 49 \
--optimizer adamw \
--lr 1e-5 \
--dbeta_lr_multiplier 5 \
--beta1 0.9 \
--beta2 0.98 \
--eps 1e-8 \
--weight_decay 0.01 \
--lr_schedule cosine \
--warmup 10000 \
--grad_clip 1.0 \
--patience 999 \
--patience_delta 0.0 \
--num_workers 8 \
--wandb
