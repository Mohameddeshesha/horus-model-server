"""Custom Transformer (Track A, Trial 6) — transcribed verbatim from
Custom_Transformer.ipynb.

Architecture : word-level embedding -> sinusoidal PE -> 6 pre-LN decoupled
               encoder layers (d_model=256, heads=8, ffn=512)
               -> masked mean+max pooling -> pool_proj -> Linear(256,3)
MAX_LENGTH   : 512
Labels       : ["Credible", "Not Credible", "Undecided"]
Checkpoint   : best_model_trial6.pth (plain state_dict)

The notebook does NOT persist the word-level vocabulary inside the .pth file, so
`vocab.json` ({token: index}, including <PAD>/<UNK>/<CLS>/<SEP> at 0..3) must be
exported separately. Without it the model reports CONFIGURED — it is never
substituted with a guessed vocabulary.
"""

from __future__ import annotations

import json
import math
import re
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CHECKPOINTS, LABELS_CREDIBLE_FIRST

MAX_LENGTH = 512
D_MODEL = 256
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 6
DIM_FEEDFORWARD = 512
ATTENTION_DROPOUT = 0.15
HIDDEN_DROPOUT = 0.30
POOLING_STRATEGY = "mean_max"
NUM_LABELS = 3

PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN = "<PAD>", "<UNK>", "<CLS>", "<SEP>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN]

_ARABIC_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
_TATWEEL = re.compile(r"\u0640")
_TOKEN_PATTERN = re.compile(
    r"[\u0600-\u06FF]+|[a-zA-Z]+|\d+|[^\s\u0600-\u06FFa-zA-Z\d]"
)


def clean_arabic_text(text: str) -> str:
    text = str(text)
    text = _ARABIC_DIACRITICS.sub("", text)
    text = _TATWEEL.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def word_tokenize_arabic(text: str) -> List[str]:
    tokens = _TOKEN_PATTERN.findall(clean_arabic_text(text))
    return [t for t in tokens if t.strip()]


class Vocabulary:
    """Encoding half of the notebook's Vocabulary, restored from vocab.json."""

    def __init__(self, token2idx: Dict[str, int], max_length: int = MAX_LENGTH) -> None:
        self.token2idx = token2idx
        self.max_length = max_length
        for tok in SPECIAL_TOKENS:
            if tok not in token2idx:
                raise ValueError(f"vocab.json is missing the special token {tok}")

    def __len__(self) -> int:
        return len(self.token2idx)

    @property
    def pad_id(self) -> int:
        return self.token2idx[PAD_TOKEN]

    def encode(self, text: str) -> Tuple[List[int], List[int]]:
        tokens = word_tokenize_arabic(text)[: self.max_length - 2]
        ids = [self.token2idx[CLS_TOKEN]]
        ids += [self.token2idx.get(t, self.token2idx[UNK_TOKEN]) for t in tokens]
        ids += [self.token2idx[SEP_TOKEN]]
        mask = [1] * len(ids)
        pad_len = self.max_length - len(ids)
        if pad_len > 0:
            ids += [self.pad_id] * pad_len
            mask += [0] * pad_len
        return ids, mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = MAX_LENGTH, dropout: float = HIDDEN_DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class DecoupledTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, attn_dropout=0.2, hidden_dropout=0.3):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=attn_dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn_out_dropout = nn.Dropout(attn_dropout)
        self.ffn_mid_dropout = nn.Dropout(hidden_dropout)
        self.ffn_out_dropout = nn.Dropout(hidden_dropout)

    def forward(self, src, src_key_padding_mask=None):
        x = self.norm1(src)
        attn_out, _ = self.self_attn(
            x, x, x, key_padding_mask=src_key_padding_mask, need_weights=False
        )
        src = src + self.attn_out_dropout(attn_out)
        x = self.norm2(src)
        ff = self.linear2(self.ffn_mid_dropout(F.gelu(self.linear1(x))))
        return src + self.ffn_out_dropout(ff)


class DecoupledTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, attn_dropout=0.2, hidden_dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoupledTransformerEncoderLayer(
                    d_model, nhead, dim_feedforward, attn_dropout, hidden_dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, src, src_key_padding_mask=None):
        for layer in self.layers:
            src = layer(src, src_key_padding_mask=src_key_padding_mask)
        return self.final_norm(src)


class CustomTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = D_MODEL,
        num_heads: int = NUM_HEADS,
        num_encoder_layers: int = NUM_ENCODER_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        attn_dropout: float = ATTENTION_DROPOUT,
        hidden_dropout: float = HIDDEN_DROPOUT,
        num_labels: int = NUM_LABELS,
        max_length: int = MAX_LENGTH,
        pad_id: int = 0,
        pooling: str = POOLING_STRATEGY,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.pooling = pooling
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.embed_scale = math.sqrt(d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_length, hidden_dropout)
        self.transformer_encoder = DecoupledTransformerEncoder(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            num_layers=num_encoder_layers,
            attn_dropout=attn_dropout,
            hidden_dropout=hidden_dropout,
        )
        pooled_dim = d_model * 2 if pooling == "mean_max" else d_model
        if pooling == "mean_max":
            self.pool_proj = nn.Sequential(
                nn.Linear(pooled_dim, d_model), nn.GELU(), nn.Dropout(hidden_dropout)
            )
        else:
            self.pool_proj = nn.Identity()
        self.dropout = nn.Dropout(hidden_dropout)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids) * self.embed_scale
        x = self.positional_encoding(x)
        encoded = self.transformer_encoder(x, src_key_padding_mask=(attention_mask == 0))
        mask = attention_mask.unsqueeze(-1).type_as(encoded)
        mean_pooled = torch.sum(encoded * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        if self.pooling == "mean_max":
            masked_for_max = encoded.masked_fill(mask == 0, float("-inf"))
            max_pooled, _ = torch.max(masked_for_max, dim=1)
            max_pooled = torch.nan_to_num(max_pooled, neginf=0.0)
            pooled = torch.cat([mean_pooled, max_pooled], dim=-1)
        else:
            pooled = mean_pooled
        pooled = self.pool_proj(pooled)
        return self.classifier(self.dropout(pooled))


class CustomTransformerRunner:
    model_id = "custom_transformer"
    labels = LABELS_CREDIBLE_FIRST

    def __init__(self, device: torch.device) -> None:
        ckpt = CHECKPOINTS["custom_transformer"]
        vocab_path = CHECKPOINTS["custom_transformer_vocab"]
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        if not vocab_path.is_file():
            raise FileNotFoundError(
                "vocab.json not found next to best_model_trial6.pth. The Trial 6 "
                "notebook saves only the state_dict, so the word-level vocabulary "
                "must be exported separately (see README)."
            )
        self.device = device
        with open(vocab_path, "r", encoding="utf-8") as fh:
            token2idx = {str(k): int(v) for k, v in json.load(fh).items()}
        self.vocab = Vocabulary(token2idx, max_length=MAX_LENGTH)

        state = torch.load(ckpt, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        ckpt_vocab_size = state["embedding.weight"].shape[0]
        if ckpt_vocab_size != len(self.vocab):
            raise ValueError(
                f"vocab.json has {len(self.vocab)} tokens but the checkpoint was "
                f"trained with {ckpt_vocab_size}. Export the exact vocabulary used "
                "for Trial 6."
            )
        self.model = CustomTransformerClassifier(
            vocab_size=ckpt_vocab_size, pad_id=self.vocab.pad_id
        ).to(device)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str) -> list[float]:
        ids, mask = self.vocab.encode(text)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor([mask], dtype=torch.long, device=self.device)
        logits = self.model(input_ids, attention_mask)
        return torch.softmax(logits, dim=-1)[0].cpu().tolist()
