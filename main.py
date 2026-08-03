"""Ain Horus — real trained-model inference server.

This server ONLY serves predictions produced by the actual trained checkpoints.
It contains no LLM, no gateway fallback, and no simulated predictions. If a
checkpoint cannot be loaded, the affected model reports `configured`/`offline`
and /predict returns 503 for it.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import ALLOWED_ORIGINS, EAGER_LOAD, MODEL_IDS, MODEL_SERVER_TOKEN
from registry import registry


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=100_000)
    model: str = "arabert"
    # Accepted for contract compatibility with the Ain Horus frontend. They are
    # NOT fed to the classifier: these checkpoints score the text only.
    claims: List[Any] = Field(default_factory=list)
    evidence: List[Any] = Field(default_factory=list)


class PredictResponse(BaseModel):
    model: str
    label: str
    confidence: float
    probabilities: Dict[str, float]
    rationale: Optional[str] = None
    rationale_source: Optional[str] = None
    inference_source: str
    latency_ms: float
    device: str
    uses_claims: bool = False
    uses_evidence: bool = False


@asynccontextmanager
async def lifespan(_: FastAPI):
    if EAGER_LOAD:
        registry.load_all()
    yield


app = FastAPI(title="Ain Horus Model Server", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def authorize(authorization: Optional[str]) -> None:
    if not MODEL_SERVER_TOKEN:
        return
    expected = f"Bearer {MODEL_SERVER_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", **registry.health()}


@app.get("/models/health")
def models_health() -> dict:
    return registry.health()


@app.post("/predict", response_model=PredictResponse)
def predict(
    body: PredictRequest,
    authorization: Optional[str] = Header(default=None),
) -> PredictResponse:
    authorize(authorization)

    if body.model not in MODEL_IDS:
        raise HTTPException(status_code=404, detail=f"Unknown model '{body.model}'")
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    try:
        result = registry.predict(body.model, body.text)
    except RuntimeError as exc:
        # Model unavailable — never substitute another model or a fake score.
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return PredictResponse(**result)


@app.post("/predict/{model_id}", response_model=PredictResponse)
def predict_path(
    model_id: str,
    body: PredictRequest,
    authorization: Optional[str] = Header(default=None),
) -> PredictResponse:
    body.model = model_id
    return predict(body, authorization)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
