# Osteofore backend

Inference API for the Osteofore knee X-ray screening tool.

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Endpoints

| Method | Path       | Returns                                             |
| ------ | ---------- | --------------------------------------------------- |
| GET    | `/health`  | `{"status": "ok", "model_loaded": bool}`             |
| POST   | `/predict` | `{"probability": float, "kl_grade": int, "confidence": float}` |

`/predict` takes a multipart form field named `image` (PNG, JPEG or WebP, max 12 MB).

## Adding the model

1. Put the trained weights at `models/oa_model.pt`, or set `OA_MODEL_PATH`.
2. Implement `_load` and `_infer` in `app/model.py`.
3. Add the runtime (torch, onnxruntime, tensorflow) to `requirements.txt`.

Until both methods are implemented, `/health` reports `model_loaded: false` and the
frontend stays in its "no model connected" state.

## Connecting the frontend

Set `API_BASE` in the frontend `index.html` to this service's URL, and set
`ALLOWED_ORIGINS` here to the frontend origin.

```bash
export ALLOWED_ORIGINS=https://your-frontend.vercel.app
```
