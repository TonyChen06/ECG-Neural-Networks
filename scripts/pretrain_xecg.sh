# xECG pretraining (loose adaptation of Lunelli et al. 2024).
# Bidirectional xLSTM over cross-lead time patches, masked patch reconstruction (MAE-style).
# Same recipe as Block 4 / 5 of pretrain_st_mem_norm.sh: experiment normalization + norm_pix_loss.

# Block 1: xECG-Base (~89M params). 4 GPUs at half batch each so eff_global_bs = 4 x 128 = 512.
CUDA_VISIBLE_DEVICES=4,5,6,7 \
uv run torchrun --standalone --nproc_per_node=4 --master_port=29403 \
src/pretrain_encoder.py \
--data mimic_iv \
--data_representation "signal" \
--objective "xecg" \
--neural_network "xecg" \
--task "pretrain" \
--xecg_size base \
--ecg_norm experiment \
--batch_size 64 \
--distributed \
--ref_global_bs 256 \
--epochs 25 \
--torch_compile \
--lr 8e-5 \
--lr_schedule cosine \
--optimizer adamw \
--augment \
--warmup 5000 \
--patience 999 \
--patience_delta 0.0 \
--grad_clip 1.0 \
--num_workers 16 \
--wandb

# Block 2: xECG-Large (~312M params). Spread across 4 GPUs at half batch each.
CUDA_VISIBLE_DEVICES=4,5,6,7 \
uv run torchrun --standalone --nproc_per_node=4 --master_port=29404 \
src/pretrain_encoder.py \
--data mimic_iv \
--data_representation "signal" \
--objective "xecg" \
--neural_network "xecg" \
--task "pretrain" \
--xecg_size large \
--ecg_norm experiment \
--batch_size 64 \
--distributed \
--ref_global_bs 256 \
--epochs 25 \
--torch_compile \
--lr 8e-5 \
--lr_schedule cosine \
--optimizer adamw \
--augment \
--warmup 5000 \
--patience 999 \
--patience_delta 0.0 \
--grad_clip 1.0 \
--num_workers 16 \
--wandb
