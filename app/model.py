"""OA detection model wrapper.

Drop the trained model in and implement `_load` and `_infer`. Everything else
(the HTTP layer, validation, error shape) already matches what the frontend expects.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

MODEL_PATH = Path(os.getenv("OA_MODEL_PATH", "models/oa_model.pt"))
INPUT_SIZE = (224, 224)


@dataclass
class Prediction:
    probability: float   # 0..1 likelihood of osteoarthritis
    kl_grade: int        # 0..4 Kellgren-Lawrence grade
    confidence: float    # 0..1 model confidence in its own output


class ModelNotLoaded(RuntimeError):
    pass


class OADetector:
    def __init__(self) -> None:
        self._model = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> bool:
        """Called once at startup. Returns True when a model is available."""
        if not MODEL_PATH.exists():
            return False
        self._model = self._load(MODEL_PATH)
        return self.loaded

    def predict(self, raw: bytes) -> Prediction:
        if not self.loaded:
            raise ModelNotLoaded("no model file loaded")
        image = self._preprocess(raw)
        return self._infer(image)

    def _preprocess(self, raw: bytes) -> Image.Image:
        image = Image.open(io.BytesIO(raw))
        image = image.convert("L").resize(INPUT_SIZE)
        return image

    def _load(self, path: Path):
        # Replace with the real load, for example:
        #   import torch; return torch.jit.load(path).eval()
        raise ModelNotLoaded(f"loading is not implemented yet for {path}")

    def _infer(self, image: Image.Image) -> Prediction:
        # Replace with the real forward pass. Must return calibrated values:
        # probability and confidence in 0..1, kl_grade an int in 0..4.
        raise ModelNotLoaded("inference is not implemented yet")


detector = OADetector()
