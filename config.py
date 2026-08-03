"""Central configuration for the Ain Horus real-model inference server.

Nothing here invents model behaviour: every constant below is transcribed from
the training notebooks that produced the checkpoints.
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(os.environ.get("MODELS_DIR", BASE_DIR / "models"))

# Optional shared secret. When set, every request must send
#   Authorization: Bearer <MODEL_SERVER_TOKEN>
MODEL_SERVER_TOKEN = os.environ.get("MODEL_SERVER_TOKEN", "").strip()



# Comma separated list, or "*" for any origin.
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()
]

DEVICE = os.environ.get("DEVICE", "").strip() or None  # None -> auto (cuda if available)

# Load every checkpoint at startup instead of on first request.
EAGER_LOAD = os.environ.get("EAGER_LOAD", "1") not in {"0", "false", "False"}

# Text used for the mandatory warm-up inference that promotes a model to LIVE.
WARMUP_TEXT = "أعلنت الحكومة اليوم عن خطة جديدة لدعم قطاع التعليم في البلاد."

# ---------------------------------------------------------------------------
# Label mappings — index order is per model and MUST NOT be reordered.
# ---------------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()

HF_REPOSITORY = os.environ.get(
    "HF_REPOSITORY",
    "deshesha/ain-horus",
)

HF_FILES = {
    "arabert": "best_arabert.pt",
    "custom_transformer": "custom_transformer/best_model_trial6.pth",
}
# Arabert.ipynb   : CLASS_NAMES = ['Credible', 'Not Credible', 'Undecided']
# Custom_Transformer.ipynb : LABEL_MAP = {"Credible":0, "Not Credible":1, "Undecided":2}
# MBert.ipynb     : label_mapping = {"credible":0, "not credible":1, "undecided":2}
LABELS_CREDIBLE_FIRST = ["Credible", "Not Credible", "Undecided"]

# camelbert.ipynb : CFG.CLASS_NAMES = ["Not Credible", "Credible", "Undecided"]
LABELS_NOT_CREDIBLE_FIRST = ["Not Credible", "Credible", "Undecided"]

CANONICAL_LABELS = ["Credible", "Not Credible", "Undecided"]

MODEL_IDS = ["arabert", "custom_transformer", "mbert", "camelbert"]

CHECKPOINTS = {
    # AraBERT — Track B, Trial 5. torch.save(model.state_dict(), 'best_arabert.pt')
    "arabert": MODELS_DIR / "arabert" / "best_arabert.pt",
    # Custom Transformer — Track A, Trial 6. torch.save(model.state_dict(), ...)
    "custom_transformer": MODELS_DIR / "custom_transformer" / "best_model_trial6.pth",
    # Custom Transformer word-level vocabulary. NOT contained in the .pth file:
    # the notebook only saves the state_dict, so token2idx must be exported
    # separately (see README) or the model stays CONFIGURED.
    "custom_transformer_vocab": MODELS_DIR / "custom_transformer" / "vocab.json",
    # mBERT — HuggingFace Trainer save_pretrained() directory.
    "mbert": MODELS_DIR / "mbert" / "final_model",
    # CAMeLBERT — 5-fold checkpoints, e.g. fold1_seed42_best.pt
    "camelbert": MODELS_DIR / "camelbert",
}
