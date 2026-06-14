"""D-BETA ECG encoder: conv front-end + conv positional encoding + post-norm
transformer. Mirrors ECGTransformerModel in the reference repo.

`get_embeddings` returns per-token features BEFORE the transformer and BEFORE the
CLS token / MEM masking (those are applied by the DBETA model). `get_output`
runs the transformer stack.
"""

import torch
import torch.nn as nn

from .modules import ConvFeatureExtraction, ConvPositionalEncoding, TransformerEncoder


class ECGEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        embed = cfg.conv_feature_layers[-1][0]  # conv output channels (256)

        self.feature_extractor = ConvFeatureExtraction(cfg.conv_feature_layers, cfg.in_d, cfg.conv_bias)
        self.layer_norm = nn.LayerNorm(embed)
        self.post_extract_proj = (
            nn.Linear(embed, cfg.encoder_embed_dim) if embed != cfg.encoder_embed_dim else None
        )
        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.conv_pos = ConvPositionalEncoding(cfg.encoder_embed_dim, cfg.conv_pos, cfg.conv_pos_groups)
        self.encoder = TransformerEncoder(cfg)

    def get_embeddings(self, source):  # source: (B, C, T_raw)
        x = self.feature_extractor(source)        # (B, embed, T)
        x = x.transpose(1, 2)                      # (B, T, embed)
        x = self.layer_norm(x)
        if self.post_extract_proj is not None:
            x = self.post_extract_proj(x)          # (B, T, 768)
        x = self.dropout_input(x)
        x = x + self.conv_pos(x)
        return x

    def get_output(self, x, key_padding_mask=None):
        return self.encoder(x, key_padding_mask=key_padding_mask)

    @torch.no_grad()
    def get_features(self, source):
        """Downstream feature path (no masking, no CLS-drop): full forward, return tokens."""
        x = self.get_embeddings(source)
        return self.get_output(x)
