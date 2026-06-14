"""D-BETA config (Pham et al. 2025, "Boosting Masked ECG-Text Auto-Encoders as
Discriminative Learners", ICML 2025). Defaults mirror the released
configs/config.json from github.com/manhph2211/D-BETA.

We pretrain the architecture from scratch on our own ECG-text data (we do not
load their released weights), so values are faithful to their config rather than
chosen for checkpoint compatibility.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DBETAConfig:
    # --- ECG conv front-end (wav2vec2-style) ---
    conv_feature_layers: List = field(default_factory=lambda: [(256, 2, 2)] * 4)
    in_d: int = 12
    conv_bias: bool = False
    extractor_mode: str = "default"  # group-norm on conv layer 0 only
    feature_grad_mult: float = 1.0
    conv_pos: int = 128
    conv_pos_groups: int = 16

    # --- ECG transformer encoder ---
    encoder_layers: int = 8
    encoder_embed_dim: int = 768
    encoder_ffn_embed_dim: int = 3072
    encoder_attention_heads: int = 12
    layer_norm_first: bool = False  # post-norm
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.0
    encoder_layerdrop: float = 0.0
    dropout_input: float = 0.1

    # --- text + cross-modal fusion ---
    text_model: str = "google/flan-t5-base"
    vocab_size: int = 32128
    hidden_dim: int = 768
    num_layers: int = 6        # bert config num_hidden_layers (for cross layers)
    num_heads: int = 12
    drop_rate: float = 0.1
    num_top_layer: int = 6     # number of BertCrossLayer blocks per modality
    max_text_size: int = 256

    # --- masked-ECG-modeling (MAE) decoder ---
    mem_layer: int = 3
    mem_prob: float = 0.75
    mem_decoder_hidden_dim: int = 384
    mem_decoder_num_layers: int = 4
    mem_decoder_num_heads: int = 6
    norm_pix_loss: bool = True

    # Cross-rank all-gather of ETS text features. Off by default: at batch>=128/GPU
    # local SigLIP already has paper-equivalent negatives, and the manual collective
    # deadlocks against DDP's gradient all-reduce on the default process group.
    ets_gather: bool = False

    # --- input geometry (ours: 250 Hz x 10 s = 2500; D-BETA native is 500 Hz x 5000) ---
    input_length: int = 2500
    num_leads: int = 12

    def conv_out_length(self, length: int) -> int:
        for _, k, s in self.conv_feature_layers:
            length = (length - k) // s + 1
        return length

    @property
    def num_ecg_tokens(self) -> int:
        return self.conv_out_length(self.input_length)

    @property
    def decoded_size_per_token(self) -> int:
        # how many raw samples each ECG token reconstructs (per lead)
        return self.input_length // self.num_ecg_tokens
