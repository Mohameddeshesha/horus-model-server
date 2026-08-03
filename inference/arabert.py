"""AraBERT loader — transcribed verbatim from Arabert.ipynb (Trial 5 config).

Backbone : aubmindlab/bert-base-arabertv02
MAX_LEN  : 256
Head     : LayerNorm([CLS]) -> Dropout(0.4) -> Linear(768,256) -> GELU
           -> Dropout(0.5) -> Linear(256,3)
Labels   : ['Credible', 'Not Credible', 'Undecided']
Checkpoint: best_arabert.pt  (a plain state_dict)
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from config import CHECKPOINTS, LABELS_CREDIBLE_FIRST
from utils.checkpoint_manager import ensure_checkpoint

MODEL_NAME = "aubmindlab/bert-base-arabertv02"
MAX_LEN = 256
HIDDEN_DIM = 256
NUM_CLASSES = 3
DROPOUT1 = 0.4
DROPOUT2 = 0.5


def clean_arabic(text: str) -> str:
    """Identical to `clean_arabic` in Arabert.ipynb."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^\u0600-\u06FF\u0750-\u077F\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class AraBERTClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        num_classes: int = NUM_CLASSES,
        hidden_dim: int = HIDDEN_DIM,
        dropout1: float = DROPOUT1,
        dropout2: float = DROPOUT2,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        bert_dim = self.bert.config.hidden_size
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(bert_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout1),
            nn.Linear(bert_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        if self.use_layer_norm:
            cls = self.layer_norm(cls)
        return self.classifier(cls)


class AraBertRunner:
    model_id = "arabert"
    labels = LABELS_CREDIBLE_FIRST

    def __init__(self, device: torch.device) -> None:
        path = ensure_checkpoint(
            "arabert",
            CHECKPOINTS["arabert"],
        )
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AraBERTClassifier().to(device)
        state = torch.load(path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str) -> list[float]:
        cleaned = clean_arabic(text)
        enc = self.tokenizer(
            cleaned,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_token_type_ids=False,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(enc["input_ids"], enc["attention_mask"])
        return torch.softmax(logits, dim=-1)[0].cpu().tolist()
