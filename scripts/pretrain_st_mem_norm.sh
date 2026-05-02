# ST-MEM ablation. norm_pix_loss is now True by default (matches upstream); the explicit
# --norm_pix_loss / --no-norm_pix_loss CLI flags were removed. Blocks 1 and 3 (pix=off
# variants) are kept here as historical labels — those checkpoints were produced before
# the dataclass default flip and aren't reproducible with this code anymore. Blocks 2, 4,
# 5 are runnable and all train with norm_pix_loss=True.

# Block 1 (HISTORICAL, not reproducible): A x pix=off. Existing st_mem.pt recipe.

# Block 2: A normalization, ViT-Base.
CUDA_VISIBLE_DEVICES=4,5 \
uv run torchrun --standalone --nproc_per_node=2 --master_port=29400 \
src/pretrain_encoder.py \
--data mimic_iv \
--data_representation "signal" \
--objective "st_mem" \
--neural_network "st_mem" \
--task "pretrain" \
--ecg_norm instance_minmax \
--batch_size 256 \
--distributed \
--ref_global_bs 512 \
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

# Block 3 (HISTORICAL, not reproducible): B x pix=off. Experiment norm with old loss.

# Block 4: B normalization, ViT-Base. The best base-sized recipe.
CUDA_VISIBLE_DEVICES=6,7 \
uv run torchrun --standalone --nproc_per_node=2 --master_port=29401 \
src/pretrain_encoder.py \
--data mimic_iv \
--data_representation "signal" \
--objective "st_mem" \
--neural_network "st_mem" \
--task "pretrain" \
--ecg_norm experiment \
--batch_size 256 \
--distributed \
--ref_global_bs 512 \
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

# Block 5: ViT-Large backbone (1024/24/16) with the same recipe as Block 4.
# Uses all 4 GPUs at half batch each so eff_global_bs = 4 x 128 = 512 still matches ref_global_bs.
CUDA_VISIBLE_DEVICES=4,5,6,7 \
uv run torchrun --standalone --nproc_per_node=4 --master_port=29402 \
src/pretrain_encoder.py \
--data mimic_iv \
--data_representation "signal" \
--objective "st_mem" \
--neural_network "st_mem" \
--task "pretrain" \
--st_mem_size large \
--ecg_norm experiment \
--batch_size 128 \
--distributed \
--ref_global_bs 512 \
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
