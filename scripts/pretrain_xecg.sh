# xECG pretraining (paper-faithful: Lunelli et al. 2025, arXiv:2509.10151).
# Recipe matches https://github.com/dlaskalab/bench-xecg release branch
# (configs/pretrain/pretrain_run_config.yaml + ssl_pretrainer.py sim_dino_v2 path):
#   - alternating-flip single-stack xLSTM, 9 blocks (s,s,m,m,s,s,m,m,s), embed=1024, 57M params.
#   - SimDINOv2 loss = compression + 0.1*expansion + masked-patch cosine.
#   - 2 globals @ 80% crop + 4 locals @ 40% crop; block masking 0.3.
#   - lead dropout 0.2 (keep II), jitter 0.1, amp scale 0.1.
#   - AdamW lr=1e-4, wd 0.04->0.4 linear, grad_clip 3.0, EMA 0.99->1.0 linear, warmup 5 epochs, 50 epochs.
# We use patch_size=50 @ 250Hz (200ms patches) instead of paper's 25 @ 100Hz to avoid resampling our datasets.

# Block 1: xECG-Base (~57M params, paper config).
CUDA_VISIBLE_DEVICES=4,5,6,7 \
uv run torchrun --standalone --nproc_per_node=4 --master_port=29403 \
src/pretrain_encoder.py \
--data mimic_iv \
--data_representation "signal" \
--objective "xecg" \
--neural_network "xecg" \
--task "pretrain" \
--xecg_size base \
--xecg_patch_size 50 \
--ecg_norm instance_zscore \
--batch_size 128 \
--distributed \
--ref_global_bs 512 \
--epochs 50 \
--lr 1e-4 \
--lr_schedule cosine \
--optimizer adamw \
--weight_decay 0.04 \
--xecg_final_wd 0.4 \
--xecg_layerwise_lr_decay 0.9 \
--xecg_shuffle_baseline_wander \
--warmup 2500 \
--grad_clip 3.0 \
--patience 999 \
--patience_delta 0.0 \
--num_workers 16 \
--wandb

# Block 2: xECG-Large (~340M params, embed=1536, 24 blocks, head_dim=256 CUDA-friendly).
# Lower per-GPU batch since the bidirectional alternating-flip + 6 student forwards is heavy.
CUDA_VISIBLE_DEVICES=4,5,6,7 \
uv run torchrun --standalone --nproc_per_node=4 --master_port=29404 \
src/pretrain_encoder.py \
--data mimic_iv \
--data_representation "signal" \
--objective "xecg" \
--neural_network "xecg" \
--task "pretrain" \
--xecg_size large \
--xecg_patch_size 50 \
--ecg_norm instance_zscore \
--batch_size 32 \
--distributed \
--ref_global_bs 128 \
--epochs 50 \
--lr 1e-4 \
--lr_schedule cosine \
--optimizer adamw \
--weight_decay 0.04 \
--xecg_final_wd 0.4 \
--xecg_layerwise_lr_decay 0.9 \
--xecg_shuffle_baseline_wander \
--warmup 2500 \
--grad_clip 3.0 \
--patience 999 \
--patience_delta 0.0 \
--num_workers 16 \
--wandb
