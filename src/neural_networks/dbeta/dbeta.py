"""D-BETA model: cross-modal masked ECG-text auto-encoder (Pham et al. 2025).

Faithful re-implementation of github.com/manhph2211/D-BETA's `DBETA` + `DBETALoss`,
producing the four-term objective L = L_mlm + L_mem + L_etm + L_ets (equal weights).

The forward returns an output object with `.loss`, matching this framework's
training contract. After pretraining, `get_features(ecg)` yields the (B, 768)
downstream ECG vector (CLS -> multi_modal_ecg_proj -> unimodal_ecg_pooler).
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.t5.modeling_t5 import T5EncoderModel
from transformers.models.bert.modeling_bert import BertConfig, BertPredictionHeadTransform

from .config import DBETAConfig
from .cross_layer import BertCrossLayer
from .encoder import ECGEncoder
from .modules import Pooler, SinusoidalPositionalEncoding, MEMDecoderTransformer


def _init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


@dataclass
class DBETAOutput:
    loss: torch.Tensor
    mlm: torch.Tensor
    mem: torch.Tensor
    etm: torch.Tensor
    ets: torch.Tensor


class MLMHead(nn.Module):
    def __init__(self, bert_config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(bert_config)
        self.decoder = nn.Linear(bert_config.hidden_size, bert_config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(bert_config.vocab_size))

    def forward(self, x):
        return self.decoder(self.transform(x)) + self.bias


class MEMHead(nn.Module):
    """Asymmetric MAE decoder: re-insert mask tokens, decode, predict raw samples."""

    def __init__(self, cfg: DBETAConfig):
        super().__init__()
        d = cfg.mem_decoder_hidden_dim
        self.decoder_embed = nn.Linear(cfg.hidden_dim, d, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.mask_token, std=0.02)
        self.decoder_pos_embed = SinusoidalPositionalEncoding(d, max_len=512)
        self.decoder = MEMDecoderTransformer(d, cfg.mem_decoder_num_layers + 1, cfg.mem_decoder_num_heads)
        self.decoder_norm = nn.LayerNorm(d)
        self.decoded_size = cfg.decoded_size_per_token
        self.decoder_pred = nn.Linear(d, self.decoded_size * cfg.num_leads, bias=True)

    def forward(self, x, ids_restore):
        x = self.decoder_embed(x)
        n_masked = ids_restore.shape[1] + 1 - x.shape[1]
        mask_tokens = self.mask_token.repeat(x.shape[0], n_masked, 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = self.decoder_pos_embed(x)
        x = x.permute(1, 0, 2)
        x = self.decoder(x)
        x = x.permute(1, 0, 2)
        x = self.decoder_pred(self.decoder_norm(x))
        x = x[:, 1:, :]  # drop CLS
        return x.view(x.size(0), x.size(1), -1, self.decoded_size)  # (B, T, leads, decoded_size)


class ETMHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        return self.fc(x)


def _gather_text(text_feats):
    """All-gather text features across DDP ranks for cross-rank SigLIP negatives.

    `dist.all_gather` doesn't carry gradients, so we splice our own rank's tensor
    back in to keep its autograd graph (local-gradient variant: each text vector
    is updated by its home rank's ECGs; every rank still SEES all texts as
    negatives). Returns (all_text, rank, world_size).
    """
    if not (dist.is_available() and dist.is_initialized()):
        return text_feats, 0, 1
    world_size = dist.get_world_size()
    if world_size == 1:
        return text_feats, 0, 1
    gathered = [torch.zeros_like(text_feats) for _ in range(world_size)]
    dist.all_gather(gathered, text_feats)
    gathered[dist.get_rank()] = text_feats
    return torch.cat(gathered, dim=0), dist.get_rank(), world_size


def ets_loss(ecg_feats, text_feats, labels, logit_scale=1.0, gather=False):
    """SigLIP-style pairwise sigmoid contrastive loss.

    The only POSITIVE-eligible pairs are the local (ecg_i, text_i) diagonal, whose
    target is 2*is_aligned_i - 1 (so an N3S-swapped pair is -1). Every other pair
    -- off-diagonal local AND all cross-rank pairs -- is a negative (-1).

    With `gather=True` the text features are all-gathered across DDP ranks for
    cross-rank negatives (matches open_clip's negative_only SigLIP). Off by
    default: a manual collective here deadlocks against DDP's gradient all-reduce
    on the default process group, and at batch>=128/GPU local negatives already
    match the paper.
    """
    if gather:
        all_text, rank, world_size = _gather_text(text_feats)
    else:
        all_text, rank, world_size = text_feats, 0, 1
    logits = logit_scale * ecg_feats @ all_text.T          # (B, world_size*B)
    b = ecg_feats.size(0)
    gt = -torch.ones((b, all_text.size(0)), device=logits.device, dtype=logits.dtype)
    rows = torch.arange(b, device=logits.device)
    gt[rows, rank * b + rows] = 2.0 * labels.to(logits.dtype) - 1.0
    return -F.logsigmoid(gt * logits).mean()


class DBETA(nn.Module):
    def __init__(self, cfg: DBETAConfig):
        super().__init__()
        self.cfg = cfg
        self.mim_prob = cfg.mem_prob
        self.mim_layer = cfg.mem_layer
        self.norm_pix_loss = cfg.norm_pix_loss
        self.ets_gather = cfg.ets_gather

        self.ecg_encoder = ECGEncoder(cfg)
        self.class_embedding = nn.Parameter(torch.FloatTensor(cfg.encoder_embed_dim).uniform_())

        self.language_encoder = T5EncoderModel.from_pretrained(cfg.text_model)
        self.language_encoder.pooler = None
        self.freeze_text = cfg.freeze_text
        if cfg.freeze_text:
            for p in self.language_encoder.parameters():
                p.requires_grad = False

        self.multi_modal_language_proj = nn.Linear(cfg.encoder_embed_dim, cfg.hidden_dim)
        self.multi_modal_ecg_proj = nn.Linear(cfg.encoder_embed_dim, cfg.hidden_dim)
        self.modality_type_embeddings = nn.Embedding(2, cfg.hidden_dim)
        for m in (self.multi_modal_language_proj, self.multi_modal_ecg_proj, self.modality_type_embeddings):
            m.apply(_init_weights)

        bert_config = BertConfig(
            vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_dim, num_hidden_layers=cfg.num_layers,
            num_attention_heads=cfg.num_heads, intermediate_size=cfg.hidden_dim * 4,
            max_position_embeddings=cfg.max_text_size, hidden_dropout_prob=cfg.drop_rate,
            attention_probs_dropout_prob=cfg.drop_rate,
        )
        self.multi_modal_ecg_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(cfg.num_top_layer)])
        self.multi_modal_language_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(cfg.num_top_layer)])
        self.multi_modal_ecg_layers.apply(_init_weights)
        self.multi_modal_language_layers.apply(_init_weights)

        self.multi_modal_ecg_pooler = Pooler(cfg.hidden_dim)
        self.multi_modal_language_pooler = Pooler(cfg.hidden_dim)
        self.unimodal_ecg_pooler = Pooler(cfg.hidden_dim)
        self.unimodal_language_pooler = Pooler(cfg.hidden_dim)
        for p in (self.multi_modal_ecg_pooler, self.multi_modal_language_pooler,
                  self.unimodal_ecg_pooler, self.unimodal_language_pooler):
            p.apply(_init_weights)

        self.mlm_head = MLMHead(bert_config)
        self.mem_head = MEMHead(cfg)
        self.etm_head = ETMHead(cfg.hidden_dim * 2)
        for h in (self.mlm_head, self.mem_head, self.etm_head):
            h.apply(_init_weights)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_text:
            self.language_encoder.eval()  # frozen -> no dropout, deterministic text features
        return self

    def random_masking(self, x, mask_ratio):
        x_ = x[:, :1]
        x = x[:, 1:]
        bsz, tsz, csz = x.shape
        len_keep = int(tsz * (1 - mask_ratio))
        noise = torch.rand(bsz, tsz, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, csz))
        mask = torch.ones([bsz, tsz], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        x_masked = torch.cat((x_, x_masked), dim=1)
        return x_masked, mask, ids_restore

    def _encode(self, ecg, text, text_attention_mask, mask):
        ret = {}
        # --- text branch ---
        uni_text = self.language_encoder(input_ids=text, attention_mask=text_attention_mask)[0]
        uni_text = self.multi_modal_language_proj(uni_text)
        ret["uni_text"] = self.unimodal_language_pooler(uni_text)

        # --- ecg branch ---
        uni_ecg = self.ecg_encoder.get_embeddings(ecg)
        cls = self.class_embedding.repeat(uni_ecg.size(0), 1, 1)
        uni_ecg = torch.cat([cls, uni_ecg], dim=1)
        if mask:
            uni_ecg, mem_mask, ids_restore = self.random_masking(uni_ecg, self.mim_prob)
            uni_ecg = self.ecg_encoder.get_output(uni_ecg)
            ret["mem_mask"], ret["ids_restore"] = mem_mask, ids_restore
        else:
            uni_ecg = self.ecg_encoder.get_output(uni_ecg)
        uni_ecg = self.multi_modal_ecg_proj(uni_ecg)
        ret["uni_ecg"] = self.unimodal_ecg_pooler(uni_ecg)

        # --- cross-modal fusion ---
        ext_text = self.language_encoder.get_extended_attention_mask(text_attention_mask, text_attention_mask.size())
        ecg_attn = torch.ones(uni_ecg.shape[:2], dtype=torch.long, device=ecg.device)
        ext_ecg = self.language_encoder.get_extended_attention_mask(ecg_attn, ecg_attn.size())

        x = uni_text + self.modality_type_embeddings(torch.zeros_like(text))
        y = uni_ecg + self.modality_type_embeddings(ecg_attn)
        for idx, (tl, el) in enumerate(zip(self.multi_modal_language_layers, self.multi_modal_ecg_layers)):
            if mask and self.mim_layer == idx:
                ret["mem_ecg_feats"] = y
            x1 = tl(x, y, ext_text, ext_ecg, output_attentions=False)
            y1 = el(y, x, ext_ecg, ext_text, output_attentions=False)
            x, y = x1[0], y1[0]
        if mask and self.mim_layer == -1:
            ret["mem_ecg_feats"] = y

        ret["mm_text"] = x
        ret["mm_cls"] = torch.cat([self.multi_modal_language_pooler(x), self.multi_modal_ecg_pooler(y)], dim=-1)
        return ret

    def _mem_target(self, ecg, num_tokens):
        decoded = self.mem_head.decoded_size
        usable = num_tokens * decoded
        tgt = ecg[:, :, :usable] if ecg.size(-1) > usable else ecg
        if self.norm_pix_loss:
            mean = tgt.mean(dim=-1, keepdim=True)
            var = tgt.var(dim=-1, keepdim=True)
            tgt = (tgt - mean) / (var + 1e-6) ** 0.5
        tgt = tgt.view(tgt.size(0), tgt.size(1), num_tokens, -1)  # (B, leads, T, decoded)
        tgt = tgt.permute(0, 2, 1, 3).contiguous()                # (B, T, leads, decoded)
        return tgt.view(tgt.size(0), tgt.size(1), -1)             # (B, T, leads*decoded)

    def forward(self, ecg, text, text_attention_mask, mlm_labels, is_aligned, **kwargs) -> DBETAOutput:
        ret = self._encode(ecg, text, text_attention_mask, mask=True)

        # MLM
        mlm_logits = self.mlm_head(ret["mm_text"])
        mlm = F.cross_entropy(mlm_logits.view(-1, self.cfg.vocab_size), mlm_labels.view(-1), ignore_index=-100)

        # MEM
        mem_logits = self.mem_head(ret["mem_ecg_feats"], ret["ids_restore"])  # (B, T, leads, decoded)
        num_tokens = mem_logits.size(1)
        mem_logits = mem_logits.view(mem_logits.size(0), num_tokens, -1)       # (B, T, leads*decoded)
        mem_target = self._mem_target(ecg, num_tokens)
        mem_per_token = ((mem_logits - mem_target) ** 2).mean(dim=-1)
        mem = (mem_per_token * ret["mem_mask"]).sum() / ret["mem_mask"].sum().clamp(min=1.0)

        # ETM + ETS (labels from N3S: 1 aligned, 0 swapped negative)
        etm_logits = self.etm_head(ret["mm_cls"])
        etm = F.cross_entropy(etm_logits, is_aligned.long())
        ets = ets_loss(ret["uni_ecg"], ret["uni_text"], is_aligned, gather=self.ets_gather)

        loss = mlm + mem + etm + ets
        return DBETAOutput(loss=loss, mlm=mlm.detach(), mem=mem.detach(), etm=etm.detach(), ets=ets.detach())

    def get_param_groups(self, base_lr, weight_decay, lr_multiplier=5.0):
        """M3AE's grouping (m3ae_utils.set_schedule): the pretrained unimodal
        encoders (T5 + the ECG encoder) train at base LR; the prediction heads
        and the cross-modal module train at lr_multiplier x base. Norms/biases
        get no weight decay. lr_scale is applied on top of the LR schedule.
        """
        no_decay = ("bias", "norm.weight", "norm.bias", "LayerNorm.weight", "LayerNorm.bias",
                    "layernorm.weight", "layernorm.bias", "layer_norm.weight", "layer_norm.bias")
        scaled = ("mlm_head", "mem_head", "etm_head", "multi_modal")

        def bucket(scale_key, decay):
            params = []
            for n, p in self.named_parameters():
                if not p.requires_grad:
                    continue
                is_scaled = any(s in n for s in scaled)
                is_no_decay = any(nd in n for nd in no_decay)
                if is_scaled == scale_key and is_no_decay == (not decay):
                    params.append(p)
            return params

        groups = []
        for scale_key, lr_scale in ((False, 1.0), (True, lr_multiplier)):
            for decay in (True, False):
                params = bucket(scale_key, decay)
                if params:
                    groups.append({
                        "params": params,
                        "lr_scale": lr_scale,
                        "weight_decay": weight_decay if decay else 0.0,
                    })
        return groups

    @torch.no_grad()
    def get_features(self, ecg):
        """Downstream (B, 768) ECG vector: CLS -> ecg proj -> unimodal pooler."""
        x = self.ecg_encoder.get_embeddings(ecg)
        cls = self.class_embedding.repeat(x.size(0), 1, 1)
        x = torch.cat([cls, x], dim=1)
        x = self.ecg_encoder.get_output(x)
        x = self.multi_modal_ecg_proj(x)
        return self.unimodal_ecg_pooler(x)
