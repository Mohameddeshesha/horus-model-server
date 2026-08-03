"""Model registry: lazy/eager loading, honest per-model status, real inference.

Status contract (never inferred from configuration alone):
  live       -> checkpoint loaded AND a real warm-up inference succeeded
  configured -> the model is wired up but its artifacts are unusable/missing
  offline    -> loading raised an unexpected error
"""

from __future__ import annotations

import time
import traceback
from typing import Dict, Optional

import torch

from config import CANONICAL_LABELS, DEVICE, MODEL_IDS, WARMUP_TEXT


def resolve_device() -> torch.device:
    if DEVICE:
        return torch.device(DEVICE)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelState:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.status = "configured"
        self.detail = "Not loaded yet."
        self.runner = None
        self.loaded_at: Optional[float] = None
        self.warmup_ms: Optional[float] = None
        self.extra: Dict[str, object] = {}


class Registry:
    def __init__(self) -> None:
        self.device = resolve_device()
        self.states: Dict[str, ModelState] = {m: ModelState(m) for m in MODEL_IDS}

    # -- loading ------------------------------------------------------------
    def _build(self, model_id: str):
        if model_id == "arabert":
            from inference.arabert import AraBertRunner

            return AraBertRunner(self.device)
        if model_id == "custom_transformer":
            from inference.custom_transformer import CustomTransformerRunner

            return CustomTransformerRunner(self.device)
        if model_id == "mbert":
            from inference.mbert import MBertRunner

            return MBertRunner(self.device)
        if model_id == "camelbert":
            from inference.camelbert import CamelBertRunner

            return CamelBertRunner(self.device)
        raise KeyError(model_id)

    def load(self, model_id: str) -> ModelState:
        state = self.states[model_id]
        if state.runner is not None and state.status == "live":
            return state
        try:
            runner = self._build(model_id)
        except FileNotFoundError as exc:
            state.status = "configured"
            state.detail = str(exc)
            state.runner = None
            return state
        except Exception as exc:  # noqa: BLE001 - report honestly, never fake a model
            state.status = "offline"
            state.detail = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            state.runner = None
            return state

        # Mandatory real warm-up inference before a model may be called LIVE.
        try:
            started = time.perf_counter()
            probs = runner.predict(WARMUP_TEXT)
            elapsed = (time.perf_counter() - started) * 1000
            if len(probs) != 3:
                raise ValueError(f"Warm-up returned {len(probs)} classes, expected 3")
        except Exception as exc:  # noqa: BLE001
            state.status = "offline"
            state.detail = f"Warm-up inference failed — {type(exc).__name__}: {exc}"
            traceback.print_exc()
            state.runner = None
            return state

        state.runner = runner
        state.status = "live"
        state.detail = "Checkpoint loaded and warm-up inference verified."
        state.loaded_at = time.time()
        state.warmup_ms = round(elapsed, 2)
        if model_id == "camelbert":
            state.extra["folds"] = getattr(runner, "fold_names", [])
        return state

    def load_all(self) -> None:
        for model_id in MODEL_IDS:
            print(f"[registry] loading {model_id} ...")
            state = self.load(model_id)
            print(f"[registry] {model_id}: {state.status} — {state.detail}")

    # -- inference ----------------------------------------------------------
    def predict(self, model_id: str, text: str) -> dict:
        state = self.load(model_id)
        if state.status != "live" or state.runner is None:
            raise RuntimeError(state.detail)

        started = time.perf_counter()
        raw = state.runner.predict(text)
        latency = (time.perf_counter() - started) * 1000

        labels = state.runner.labels  # per-model index order
        by_name = {labels[i]: float(raw[i]) for i in range(len(labels))}
        probabilities = {name: round(by_name[name], 6) for name in CANONICAL_LABELS}
        top = max(probabilities, key=probabilities.get)
        print(
            f"[predict] model={model_id} chars={len(text)} label={top} "
            f"confidence={probabilities[top]:.4f} latency_ms={latency:.1f} "
            "source=real_checkpoint"
        )
        return {
            "model": model_id,
            "label": top,
            "confidence": probabilities[top],
            "probabilities": probabilities,
            "rationale": None,
            "rationale_source": None,
            "inference_source": "real_checkpoint",
            "latency_ms": round(latency, 2),
            "device": str(self.device),
        }

    # -- reporting ----------------------------------------------------------
    def health(self) -> dict:
        return {
            "models": {m: self.states[m].status for m in MODEL_IDS},
            "details": {
                m: {
                    "status": self.states[m].status,
                    "detail": self.states[m].detail,
                    "warmup_ms": self.states[m].warmup_ms,
                    "loaded_at": self.states[m].loaded_at,
                    **self.states[m].extra,
                }
                for m in MODEL_IDS
            },
            "device": str(self.device),
        }


registry = Registry()
