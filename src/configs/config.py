import argparse
from configs.constants import Mode, ALLOWED_DATA


def get_args(mode: Mode) -> argparse.Namespace:
    if mode not in {"pretrain", "downstream_eval"}:
        raise ValueError(f"invalid mode: {mode}")

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("--seed", type=int, default=0, help="Random Seed")
    parser.add_argument("--dev", action="store_true", default=None, help="Development mode")

    if mode in {"pretrain", "downstream_eval"}:
        parser.add_argument("--task", type=str, default=None, choices=["pretrain", "forecasting", "generation", "reconstruction"])
        parser.add_argument("--forecast_ratio", type=float, default=0.5, help="Please choose the percentage you want to forecast")
        parser.add_argument(
            "--data",
            type=str,
            nargs="+",
            required=True,
            help=f"One or more datasets: {', '.join(sorted(ALLOWED_DATA))}",
        )
        parser.add_argument("--data_subset", type=float, default=1, help="Please choose the percentage of the data you want to use.")
        parser.add_argument(
            "--data_sampling_probs",
            type=float,
            nargs="+",
            default=None,
            help="If one or more datasets, specify specify the probability of sampling from each dataset.",
        )
        parser.add_argument("--augment", action="store_true", default=None, help="Choose whether you want to augment your ECG")
        parser.add_argument(
            "--data_representation",
            type=str,
            default=None,
            choices=["signal", "bpe_symbolic"],
            help="Please choose the representation of data you want to input into the neural network.",
        )
        parser.add_argument(
            "--objective",
            type=str,
            default=None,
            choices=["autoregressive", "mae", "ddpm", "rectified_flow", "merl", "mlae", "mtae", "st_mem", "xecg"],
            help="Please choose the representation of data you want to input into the neural network.",
        )
        parser.add_argument("--patch_dim", type=int, default=2500, help="Please choose a patch dim that is evenly divisible by signal_len.")
        parser.add_argument("--num_patches", type=int, default=12, help="Please choose number of patches.")
        parser.add_argument(
            "--bpe_symbolic_len", type=int, default=2048, help="Please choose the bpe symbolic len for the bpe_symbolic data representation."
        )
        parser.add_argument("--segment_len", type=int, default=2500, help="ECG Segment Length")
        parser.add_argument("--sf", type=int, default=250, help="Sampling frequency in Hz")
        parser.add_argument(
            "--bpe_tokenizer_path",
            type=str,
            default="src/dataloaders/data_representation/bpe/ecg_byte_tokenizer_10000.pkl",
            help="Please specify the path to the saved bpe tokenizer",
        )
        parser.add_argument("--num_workers", type=int, default=0, help="Please choose the num works for the dataloader")

        parser.add_argument(
            "--neural_network",
            type=str,
            default=None,
            help="Please choose the main neural network",
            choices=[
                "trans_discrete_decoder",
                "trans_continuous_nepa",
                "trans_continuous_dit",
                "mae_vit",
                "merl",
                "mlae",
                "mtae",
                "st_mem",
                "xecg",
            ],
        )
        parser.add_argument("--st_mem_size", type=str, default="base", choices=["base", "large", "huge"],
                            help="ST-MEM encoder size: base (768/12/12), large (1024/24/16), huge (1280/32/16)")
        parser.add_argument("--xecg_size", type=str, default="base", choices=["base", "large"],
                            help="xECG encoder size: base (1024/9, 57M, paper) or large (1536/24, ~340M)")
        parser.add_argument("--xecg_patch_size", type=int, default=50,
                            help="xECG patch size in samples (50 @ 250Hz = 200ms patches; paper uses 25 @ 100Hz)")
        parser.add_argument("--xecg_n_global", type=int, default=2)
        parser.add_argument("--xecg_n_local", type=int, default=4)
        parser.add_argument("--xecg_global_crop", type=float, default=0.8)
        parser.add_argument("--xecg_local_crop", type=float, default=0.4)
        parser.add_argument("--xecg_drop_leads_prob", type=float, default=0.2)
        parser.add_argument("--xecg_jitter_prob", type=float, default=0.1)
        parser.add_argument("--xecg_amp_scale_prob", type=float, default=0.1)
        parser.add_argument("--xecg_resample_ratio", type=float, default=0.03,
                            help="Random time-warp resample range (+/- ratio). 0 disables. Paper uses ~0.03.")
        parser.add_argument("--xecg_patient_pair", action="store_true", default=None,
                            help="Patient-pair multi-view: draw views from different recordings of the same "
                                 "patient (requires patient id in filename). Epoch is keyed per-patient.")
        parser.add_argument("--xecg_masking", type=str, default="block", choices=["block", "random"])
        parser.add_argument("--xecg_lambda_cr", type=float, default=0.1, help="Coding-rate loss weight")
        parser.add_argument("--xecg_sim_dino_eps", type=float, default=0.05)
        parser.add_argument("--xecg_ema_start", type=float, default=0.99)
        parser.add_argument("--xecg_ema_end", type=float, default=1.0)
        parser.add_argument("--xecg_final_wd", type=float, default=0.4, help="Final weight decay for linear schedule")
        parser.add_argument("--xecg_slstm_backend", type=str, default="cuda", choices=["cuda", "vanilla"])
        parser.add_argument("--mask_ratio", type=float, default=0.3)
        parser.add_argument("--xecg_layerwise_lr_decay", type=float, default=0.9,
                            help="Per-layer LR decay (1.0 disables; paper uses 0.9)")
        parser.add_argument("--xecg_shuffle_baseline_wander", action="store_true", default=None,
                            help="Per-batch FFT lowpass + cross-sample baseline shuffle (paper: on)")
        parser.add_argument("--xecg_post_encoder_norm", action="store_true", default=None,
                            help="LayerNorm on encoder output before attn pool. Paper has no norm; default off.")
        parser.add_argument("--xecg_gather_cls_for_expansion", action="store_true", default=True,
                            help="All-gather student global CLS across DDP ranks before the log-det. "
                                 "log_det is non-linear in batch dim, so per-rank averaging doesn't recover "
                                 "the global covariance rank. No-op single-GPU. Default on.")
        parser.add_argument("--xecg_no_gather_cls_for_expansion", dest="xecg_gather_cls_for_expansion",
                            action="store_false", help="Disable the gather (for ablation).")

        parser.add_argument("--norm_eps", type=float, default=1e-6, help="Please choose the normalization epsilon")

        parser.add_argument("--wandb", action="store_true", default=None, help="Enable logging")

        parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
        parser.add_argument("--distributed", action="store_true", default=None, help="Enable distributed training")
        parser.add_argument("--nn_ckpt", type=str, default=None, help="Path to the NN checkpoint")
        parser.add_argument("--ema", action="store_true", default=None)
        parser.add_argument("--ema_decay", type=float, default=0.999)

        parser.add_argument("--condition", type=str, default=None, choices=["text", "lead"],
                    help="Condition type for conditional diffusion (None = unconditional)")
        parser.add_argument("--condition_lead", type=int, default=0, help="Lead index for lead conditioning (0-indexed, default=1 for lead II)")
        parser.add_argument("--condition_dropout", type=float, default=0.1, help="Condition dropout probability for classifier-free guidance")
        parser.add_argument("--cfg_scale", type=float, default=1.0, help="Classifier-free guidance scale at inference (1.0 = no guidance)")
        parser.add_argument("--condition_text_max_len", type=int, default=128, help="Max text length for text conditioning (byte-level)")
        parser.add_argument("--text_feature_extractor", type=str, default=None,
                            help="HuggingFace model name for LLM text encoder")
        parser.add_argument("--ecg_norm", type = str, default = "instance_minmax",
                            choices=["instance_minmax", "instance_zscore", "lead_minmax", "lead_zscore", "experiment"], help = "choose the normalization method for the ECG")
        parser.add_argument("--bfloat_16", action = "store_true", default = None)
        parser.add_argument("--signal_head", action="store_true", default=None, help="Attach flow matching signal head to discrete decoder")
        parser.add_argument("--signal_head_layers", type=int, default=4, help="Number of transformer layers in the signal head")
        parser.add_argument("--signal_head_num_steps", type=int, default=50, help="ODE integration steps for signal head inference")
        parser.add_argument("--freeze_decoder", action="store_true", default=None, help="Freeze decoder weights during signal head training")
        parser.add_argument("--flow_loss_weight", type=float, default=1.0, help="Weight alpha for flow matching loss in combined CE + alpha*flow objective")
        parser.add_argument(
            "--torch_compile",
            action="store_true",
            default=None,
            help="Torch compile the model (should really only be used during pretraining or large finetuning.)",
        )
    if "train" in mode:
        parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
        parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
        parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "muon"], help="Optimizer type")
        parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
        parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
        parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
        parser.add_argument("--patience_delta", type=float, default=0.1, help="Delta for early stopping")
        parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for optimizer")
        parser.add_argument("--beta2", type=float, default=0.95, help="Beta2 for optimizer")
        parser.add_argument("--eps", type=float, default=1e-8, help="Epsilon for optimizer")
        parser.add_argument("--muon_momentum", type=float, default=0.95, help="Muon momentum")
        parser.add_argument("--muon_nesterov", action="store_true", default=True, help="Nesterov momentum for Muon")
        parser.add_argument("--muon_ns_steps", type=int, default=5, help="Newton-Schulz iteration steps")
        parser.add_argument("--muon_adamw_lr_ratio", type=float, default=0.015, help="AdamW LR as fraction of Muon LR")
        parser.add_argument("--lr_schedule", type=str, default="constant", choices=["constant", "cosine", "inv_sqrt"], help="LR schedule after warmup")
        parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Min LR as fraction of peak LR (for cosine schedule)")
        parser.add_argument("--warmup", type=int, default=2000, help="Warmup steps")
        parser.add_argument("--ref_global_bs", type=int, default=None)
        parser.add_argument("--grad_accum_steps", type=int, default=1)
        parser.add_argument("--grad_clip", type=float, default=0.0, help="Max gradient norm for clipping (0 to disable)")
        parser.add_argument("--scale_wd", type=str, default="none", choices=["none", "inv_sqrt", "inv_linear"])
        parser.add_argument("--save_step", action="store_true", default=None, help="Save step wise")
    return parser.parse_args()
