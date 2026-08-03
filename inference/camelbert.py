"""CAMeLBERT 5-fold ensemble — transcribed verbatim from camelbert.ipynb.

Backbone : CAMeL-Lab/bert-base-arabic-camelbert-msa
MAX_LEN  : 256
Pooling  : learned softmax-weighted sum of the last 4 hidden states, then
           masked mean pooling
Head     : DNNHead(768 -> 512 -> 256 -> 3) with LayerNorm + GELU and
           5-way multi-sample dropout averaging
Labels   : CFG.CLASS_NAMES = ["Not Credible", "Credible", "Undecided"]
           NOTE: this order differs from the other three models and must not
           be reordered — the mapping to canonical labels happens by name.
Checkpoints: models/camelbert/fold{n}_seed{s}_best.pt (state_dicts).
Inference : softmax probabilities averaged across all available folds, exactly
            as `evaluate_on_test_set` does in the notebook.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from config import CHECKPOINTS, LABELS_NOT_CREDIBLE_FIRST

MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-msa"
MAX_LEN = 256
NUM_CLASSES = 3
HIDDEN_DIMS = [512, 256]
DROPOUT_RATES = [0.3, 0.2]
NUM_DROPOUTS = 5
USE_LAYER_NORM = True
USE_BATCH_NORM = False
LAST_K = 4


class DNNHead(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        dropout_rates,
        num_classes,
        num_dropouts=NUM_DROPOUTS,
        use_batch_norm=USE_BATCH_NORM,
        use_layer_norm=USE_LAYER_NORM,
    ):
        super().__init__()
        self.num_dropouts = num_dropouts
        layers, in_dim = [], input_dim
        for h_dim, drop in zip(hidden_dims, dropout_rates):
            layers.append(nn.Linear(in_dim, h_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))
            elif use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers += [nn.GELU(), nn.Dropout(drop)]
            in_dim = h_dim
        self.feature_net = nn.Sequential(*layers)
        self.classifiers = nn.ModuleList(
            [nn.Linear(in_dim, num_classes) for _ in range(num_dropouts)]
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        feat = self.feature_net(x)
        logits = 0
        for clf in self.classifiers:
            logits = logits + clf(self.dropout(feat))
        return logits / self.num_dropouts


class CAMeLBERTDNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = AutoConfig.from_pretrained(MODEL_NAME, output_hidden_states=True)
        self.encoder = AutoModel.from_pretrained(MODEL_NAME, config=self.config)
        hidden_size = self.config.hidden_size
        self.last_k = LAST_K
        self.layer_weights = nn.Parameter(torch.ones(self.last_k) / self.last_k)
        self.head = DNNHead(
            input_dim=hidden_size,
            hidden_dims=HIDDEN_DIMS,
            dropout_rates=DROPOUT_RATES,
            num_classes=NUM_CLASSES,
        )

    def pool(self, hidden_states, attention_mask):
        weights = torch.softmax(self.layer_weights, dim=0)
        stacked = torch.stack(hidden_states[-self.last_k :], dim=1)
        weighted = (stacked * weights[None, :, None, None]).sum(dim=1)
        mask = attention_mask.unsqueeze(-1).float()
        return (weighted * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        pooled = self.pool(out.hidden_states, attention_mask)
        return self.head(pooled)


class CamelBertRunner:
    model_id = "camelbert"
    labels = LABELS_NOT_CREDIBLE_FIRST

    def __init__(self, device: torch.device) -> None:
        folder = CHECKPOINTS["camelbert"]
        ckpts: List = sorted(folder.glob("fold*_best.pt")) if folder.is_dir() else []
        if not ckpts:
            raise FileNotFoundError(
                f"No fold checkpoints (fold*_best.pt) found in: {folder}"
            )
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.fold_names = [p.name for p in ckpts]
        self.models: List[CAMeLBERTDNN] = []
        for path in ckpts:
            model = CAMeLBERTDNN().to(device)
            state = torch.load(path, map_location=device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
            model.eval()
            self.models.append(model)

    @torch.no_grad()
    def predict(self, text: str) -> list[float]:
        enc = self.tokenizer(
            str(text),
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        token_type_ids = enc.get("token_type_ids")
        token_type_ids = (
            token_type_ids.to(self.device)
            if token_type_ids is not None
            else torch.zeros_like(input_ids)
        )
        probs_sum = None
        for model in self.models:
            logits = model(input_ids, attention_mask, token_type_ids)
            probs = torch.softmax(logits, dim=-1)
            probs_sum = probs if probs_sum is None else probs_sum + probs
        return (probs_sum / len(self.models))[0].cpu().tolist()
