# xECG faithful pretraining: bidirectional xLSTM body + DINO-v2 self-distillation.
# Implements the sim_dino_v2 strategy from bench-xECG (Lunelli et al. 2024):
# multi-view (2 globals) per record, EMA teacher, compression + lambda*expansion +
# patch-level cosine similarity to teacher. Architecture (768/12, bidir, sLSTM at
# every-4th) keeps our scale upgrades over the published xECG (which uses a much
# smaller 128/7 model) -- only the recipe (objective + loss + multi-view data path)
# is now faithful to the paper.
#
# Memory note: each batch element produces 2 global views, the student processes
# both with masking, the teacher processes both without masking but with grad
# disabled. Roughly 3-4x the activation memory of the MAE path at the same batch
# size. If you OOM, drop --batch_size first.

# Block 1: xECG-Base + sim_dino_v2 (~89M student + same-size teacher).
CUDA_VISIBLE_DEVICES=4,5,6,7 \
uv run torchrun --standalone --nproc_per_node=4 --master_port=29405 \
src/pretrain_encoder.py \
--data mimic_iv \
--data_representation multi_view_signal \
--objective xecg \
--neural_network xecg \
--task pretrain \
--xecg_size base \
--xecg_strategy sim_dino_v2 \
--ecg_norm experiment \
--batch_size 16 \
--distributed \
--ref_global_bs 64 \
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
