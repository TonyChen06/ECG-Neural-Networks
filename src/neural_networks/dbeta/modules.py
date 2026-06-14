"""Building blocks for the D-BETA ECG encoder and MEM decoder.

Faithful re-implementations of the wav2vec2-style front-end and post-norm
transformer used by github.com/manhph2211/D-BETA, in clean PyTorch (we train
from scratch, so exact fairseq parameter naming is not required).
"""

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class Fp32GroupNorm(nn.GroupNorm):
    def forward(self, x):
        out = F.group_norm(
            x.float(), self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return out.type_as(x)


class SamePad(nn.Module):
    """Trim the extra column an even-kernel conv adds (wav2vec2 conv-pos)."""

    def __init__(self, kernel_size):
        super().__init__()
        self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        return x[:, :, : -self.remove] if self.remove > 0 else x


class ConvFeatureExtraction(nn.Module):
    """4x Conv1d (kernel=stride), group-norm on layer 0 only, GELU. /16 downsample."""

    def __init__(self, conv_layers, in_d, conv_bias=False):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        for i, (dim, k, stride) in enumerate(conv_layers):
            conv = nn.Conv1d(in_d, dim, k, stride=stride, bias=conv_bias)
            nn.init.kaiming_normal_(conv.weight)
            layers = [conv]
            if i == 0:
                layers.append(Fp32GroupNorm(dim, dim, affine=True))
            layers.append(nn.GELU())
            self.conv_layers.append(nn.Sequential(*layers))
            in_d = dim

    def forward(self, x):  # x: (B, C, T)
        for conv in self.conv_layers:
            x = conv(x)
        return x


class ConvPositionalEncoding(nn.Module):
    """wav2vec2 relative positional embedding: a grouped, weight-normed conv."""

    def __init__(self, dim, kernel_size, groups):
        super().__init__()
        pos_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=groups)
        std = math.sqrt(4.0 / (kernel_size * dim))
        nn.init.normal_(pos_conv.weight, mean=0.0, std=std)
        nn.init.constant_(pos_conv.bias, 0)
        pos_conv = nn.utils.weight_norm(pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(pos_conv, SamePad(kernel_size), nn.GELU())

    def forward(self, x):  # x: (B, T, D)
        return self.pos_conv(x.transpose(1, 2)).transpose(1, 2)


class PostNormTransformerLayer(nn.Module):
    """wav2vec2 TransformerEncoderLayer with layer_norm_first=False (post-norm)."""

    def __init__(self, embed_dim, ffn_dim, n_heads, dropout, attention_dropout, activation_dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=attention_dropout, batch_first=True)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        residual = x
        x, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.self_attn_layer_norm(residual + self.dropout1(x))
        residual = x
        x = self.fc2(self.dropout2(F.gelu(self.fc1(x))))
        x = self.final_layer_norm(residual + self.dropout3(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dropout = cfg.dropout
        self.layerdrop = cfg.encoder_layerdrop
        self.layers = nn.ModuleList([
            PostNormTransformerLayer(
                cfg.encoder_embed_dim, cfg.encoder_ffn_embed_dim, cfg.encoder_attention_heads,
                cfg.dropout, cfg.attention_dropout, cfg.activation_dropout,
            ) for _ in range(cfg.encoder_layers)
        ])
        self.layer_norm = nn.LayerNorm(cfg.encoder_embed_dim)

    def forward(self, x, key_padding_mask=None):
        # post-norm: normalize the input before the stack
        x = self.layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        for layer in self.layers:
            if self.training and self.layerdrop > 0 and torch.rand(1).item() < self.layerdrop:
                continue
            x = layer(x, key_padding_mask=key_padding_mask)
        return x


class Pooler(nn.Module):
    """BERT-style: take token 0, Linear, Tanh."""

    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        return self.activation(self.dense(hidden_states[:, 0]))


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """Pre-norm CLIP-style block used by the MEM decoder."""

    def __init__(self, d_model, n_head):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), need_weights=False)[0]
        x = x + self.mlp(self.ln_2(x))
        return x


class MEMDecoderTransformer(nn.Module):
    """Stack of `layers - 1` ResidualAttentionBlocks (matches D-BETA's Transformer)."""

    def __init__(self, width, layers, heads):
        super().__init__()
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers - 1)])

    def forward(self, x):
        return self.resblocks(x)
