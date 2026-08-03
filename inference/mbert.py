"""mBERT (Trial 2) — transcribed verbatim from MBert.ipynb.

Backbone : bert-base-multilingual-cased
Head     : AutoModelForSequenceClassification, num_labels=3
Labels   : {"credible":0, "not credible":1, "undecided":2}
Max len  : data-driven in the notebook (p95, multiple of 8, capped at 512). The
           saved `final_model/` config does not store it, so it is read from
           `inference_config.json` when present, otherwise the tokenizer's
           model_max_length capped at 512 is used.
Checkpoint: models/mbert/final_model/  (HuggingFace save_pretrained directory)
"""

from __future__ import annotations

import json
import re

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import CHECKPOINTS, LABELS_CREDIBLE_FIRST

DEFAULT_MAX_LENGTH = 512


def clean_arabic_text_for_bert(text: str) -> str:
    """Identical to `clean_arabic_text_for_bert` in MBert.ipynb."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+|#", "", text)
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    text = re.sub(r"[\u064B-\u0652]", "", text)
    text = re.sub(r"[\*\(\)\<\>\|\}\{\[\]\_\+\=\/\\'\"~`]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class MBertRunner:
    model_id = "mbert"
    labels = LABELS_CREDIBLE_FIRST

    def __init__(self, device: torch.device) -> None:
        path = CHECKPOINTS["mbert"]
        if not path.is_dir():
            raise FileNotFoundError(f"final_model/ directory not found: {path}")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(path)).to(device)
        self.model.eval()

        if self.model.config.num_labels != 3:
            raise ValueError(
                f"Expected a 3-class head, found num_labels={self.model.config.num_labels}"
            )

        cfg_file = path / "inference_config.json"
        if cfg_file.is_file():
            with open(cfg_file, "r", encoding="utf-8") as fh:
                self.max_length = int(json.load(fh)["max_length"])
        else:
            tok_max = getattr(self.tokenizer, "model_max_length", DEFAULT_MAX_LENGTH)
            self.max_length = min(DEFAULT_MAX_LENGTH, int(tok_max) if tok_max else DEFAULT_MAX_LENGTH)

    @torch.no_grad()
    def predict(self, text: str) -> list[float]:
        cleaned = clean_arabic_text_for_bert(text)
        enc = self.tokenizer(
            cleaned,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**enc).logits
        return torch.softmax(logits, dim=-1)[0].cpu().tolist()
