"""OA-SATHI inference API.

Contract used by the frontend:
  GET  /health   -> {"status": "ok", "model_loaded": bool}
  POST /predict  -> {"probability": float, "kl_grade": int, "confidence": float}
"""

from __future__ import annotations

import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .model import ModelNotLoaded, detector

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/webp"}

# Comma separated list of allowed origins, for example:
#   ALLOWED_ORIGINS=https://oa-saathi.vercel.app,http://localhost:3000
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]

app = FastAPI(title="OA-SATHI inference API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["http://localhost:8000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class PredictResponse(BaseModel):
    probability: float
    kl_grade: int
    confidence: float


@app.on_event("startup")
def startup() -> None:
    detector.load()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=detector.loaded)


@app.post("/predict", response_model=PredictResponse)
async def predict(image: UploadFile = File(...)) -> PredictResponse:
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"unsupported content type: {image.content_type}")

    raw = await image.read()
    if not raw:
        raise HTTPException(400, "empty upload")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "image exceeds 12 MB")

    try:
        result = detector.predict(raw)
    except ModelNotLoaded as exc:
        raise HTTPException(503, f"model unavailable: {exc}") from exc
    except Exception as exc:
        raise HTTPException(500, "inference failed") from exc

    return PredictResponse(
        probability=result.probability,
        kl_grade=result.kl_grade,
        confidence=result.confidence,
    )
