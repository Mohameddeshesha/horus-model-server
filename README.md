# Ain Horus — Model Server (real trained checkpoints)

FastAPI service that runs **your actual trained checkpoints**. It has no LLM, no
Lovable AI Gateway, and no simulated predictions. If a checkpoint is missing or
fails to load, the model reports `configured`/`offline` and `/predict` returns
`503` — it never falls back to anything else.

Ain Horus reaches this server from `/model-lab`. The `/detect` page and its RAG /
Truth Score pipeline are untouched and keep using the gateway.

---

## 1. Install

```bash
cd model-server
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A GPU is optional. On CPU expect a few hundred ms to a few seconds per request.

## 2. Where to put each checkpoint

```text
model-server/models/
├── arabert/
│   └── best_arabert.pt                     # torch.save(model.state_dict(), ...)
├── custom_transformer/
│   ├── best_model_trial6.pth               # torch.save(model.state_dict(), ...)
│   └── vocab.json                          # REQUIRED — see below
├── mbert/
│   └── final_model/                        # HuggingFace save_pretrained() dir
│       ├── config.json
│       ├── model.safetensors (or pytorch_model.bin)
│       ├── tokenizer.json / vocab.txt / special_tokens_map.json
│       └── inference_config.json           # optional: {"max_length": 320}
└── camelbert/
    ├── fold1_seed42_best.pt
    ├── fold2_seed123_best.pt
    ├── fold3_seed256_best.pt
    ├── fold4_seed512_best.pt
    └── fold5_seed1024_best.pt
```

### custom_transformer/vocab.json is mandatory

Trial 6 saves only `model.state_dict()`, so the word-level vocabulary is **not**
inside the `.pth`. Export it from the notebook session that trained Trial 6:

```python
import json
with open("vocab.json", "w", encoding="utf-8") as fh:
    json.dump(vocab.token2idx, fh, ensure_ascii=False)
```

The server verifies `len(vocab) == checkpoint embedding rows` and refuses to run
with a mismatched vocabulary rather than producing meaningless predictions.

### mBERT max_length

The notebook picks `MAX_LENGTH` from the p95 token length at train time and the
value is not stored in `final_model/`. Write the exact number you used into
`models/mbert/final_model/inference_config.json` as `{"max_length": <N>}`.
Without it the server falls back to the tokenizer limit capped at 512.

### What is preserved from the notebooks

| Model | Backbone | Max len | Head | Label order |
|---|---|---|---|---|
| AraBERT | `aubmindlab/bert-base-arabertv02` | 256 | LayerNorm(CLS) → Dropout .4 → 768→256 → GELU → Dropout .5 → 256→3 | Credible, Not Credible, Undecided |
| Custom Transformer | from-scratch, d_model 256 / 8 heads / 6 layers / ffn 512, mean+max pooling | 512 | pool_proj → Linear(256,3) | Credible, Not Credible, Undecided |
| mBERT | `bert-base-multilingual-cased` | data-driven (≤512) | `AutoModelForSequenceClassification`, 3 labels | Credible, Not Credible, Undecided |
| CAMeLBERT | `CAMeL-Lab/bert-base-arabic-camelbert-msa` | 256 | weighted last-4-layer pooling → DNNHead 768→512→256→3, 5-sample dropout, 5-fold probability average | **Not Credible, Credible, Undecided** |

CAMeLBERT's index order genuinely differs; the server maps by class *name*, so
responses are always keyed `Credible / Not Credible / Undecided`.

Each model's text preprocessing is the exact cleaning function from its own
notebook — they are not interchangeable and are not shared.

## 3. Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_SERVER_TOKEN` | *(empty)* | If set, every request must send `Authorization: Bearer <token>` |
| `MODELS_DIR` | `./models` | Alternative checkpoint root |
| `DEVICE` | auto | `cuda`, `cpu`, `mps` |
| `EAGER_LOAD` | `1` | Load and warm up every model at startup |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |

## 4. Start

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Startup logs one line per model, e.g.
`[registry] arabert: live — Checkpoint loaded and warm-up inference verified.`

## 5. Test /health

```bash
curl http://localhost:8000/health
curl http://localhost:8000/models/health
```

```json
{
  "models": {
    "arabert": "live",
    "custom_transformer": "configured",
    "mbert": "offline",
    "camelbert": "configured"
  },
  "details": { "arabert": { "status": "live", "warmup_ms": 412.3, "detail": "..." } },
  "device": "cuda"
}
```

`live` is only reported after the checkpoint loads **and** a real inference on a
sample Arabic sentence returns 3 valid probabilities.

## 6. Test /predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"model":"arabert","text":"أعلنت وزارة الصحة عن حملة تطعيم جديدة","claims":[],"evidence":[]}'
```

```json
{
  "model": "arabert",
  "label": "Credible",
  "confidence": 0.9234,
  "probabilities": { "Credible": 0.9234, "Not Credible": 0.0512, "Undecided": 0.0254 },
  "rationale": null,
  "rationale_source": null,
  "inference_source": "real_checkpoint",
  "latency_ms": 118.4,
  "device": "cuda"
}
```

`claims` and `evidence` are accepted for contract compatibility but are not fed
to the classifier (`uses_claims`/`uses_evidence` are `false`). `rationale` stays
`null`: these checkpoints are classifiers and do not generate explanations, and
no LLM is allowed to fill that field.

A model that is not `live` returns `503` with the reason — never a substitute
prediction.

## 7. Expose over HTTPS

The browser calls Ain Horus, and Ain Horus's server calls this service, so the
URL must be reachable from the internet over HTTPS.

- **Cloudflare Tunnel** (recommended, free, stable):
  `cloudflared tunnel --url http://localhost:8000`
- **ngrok**: `ngrok http 8000`
- **Own host**: put nginx/Caddy in front with a TLS certificate.

Always set `MODEL_SERVER_TOKEN` when the URL is public.

## 8. Connect it to Ain Horus

In the app, add these secrets (Lovable → project secrets):

- `MODEL_SERVER_URL` — e.g. `https://abc123.trycloudflare.com` (no trailing `/predict`)
- `MODEL_SERVER_TOKEN` — optional, must match the server's token

Then open `/model-lab`. The status badges come straight from `/models/health`.
While `MODEL_SERVER_URL` is unset, every model shows 🟡 CONFIGURED.

## 9. Switching a model from `lovable` to `fastapi` (do this last)

`/detect` keeps using the gateway until you deliberately flip a row. Only run
this **after** that model shows 🟢 LIVE in `/model-lab`:

```sql
UPDATE public.model_endpoints
SET provider = 'fastapi',
    endpoint_url = 'https://your-server.example.com/predict'
WHERE model_id = 'arabert';
```

Rollout order: **AraBERT → Custom Transformer → mBERT → CAMeLBERT**.

Rollback:

```sql
UPDATE public.model_endpoints
SET provider = 'lovable', endpoint_url = NULL
WHERE model_id = 'arabert';
```

## 10. Verifying the site really uses the trained checkpoint

1. `/models/health` reports `live` for the model — impossible without a
   successful checkpoint load plus warm-up inference.
2. Every request writes a server log line:
   `[predict] model=arabert chars=214 label=Credible confidence=0.9234 latency_ms=118.4 source=real_checkpoint`
   Run a prediction from `/model-lab` and watch it appear in your terminal.
3. The response carries `"inference_source": "real_checkpoint"`; `/model-lab`
   refuses to render a result without it.
4. Stop the server: `/model-lab` immediately shows `النموذج غير متاح حاليًا.`
   If a number still appeared, it would not be coming from this server.

## Research integrity

Three things are kept strictly separate:

1. **Offline research results** — AraBERT Trial 5 89.38%, Custom Transformer
   Trial 6 84.37%, mBERT Trial 2, CAMeLBERT 5-fold 93.20%. Historical test-set
   metrics only; never used as a live confidence value.
2. **Live inference** — what this server returns right now.
3. **Gateway detection** — the Gemini-backed `/detect` pipeline, which is not a
   trained-model prediction and is never labelled as one.
