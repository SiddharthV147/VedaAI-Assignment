from __future__ import annotations

import gc
import io
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from app.config import trocr_model_name, use_cuda

log = logging.getLogger(__name__)

TROCR_DIRNAME = trocr_model_name()
TROCR_HUB_ID = f"microsoft/{TROCR_DIRNAME}"
HF_INFERENCE_URL = f"https://router.huggingface.co/hf-inference/models/{TROCR_HUB_ID}"


def free_cuda(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def to_pil_rgb(crop: np.ndarray) -> Image.Image:
    if crop.ndim == 2:
        rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def crop_to_png_bytes(crop: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    to_pil_rgb(crop).save(buffer, format="PNG")
    return buffer.getvalue()


def _resolve_model_dir(models_dir: Path, name: str | None = None) -> Path:
    """Directory holding the TrOCR weights.

    Falls back to any other ``trocr-*`` folder that was downloaded, so a
    container built with one size still starts if the env asks for another.
    """
    wanted = models_dir / (name or trocr_model_name())
    if (wanted / "config.json").is_file():
        return wanted
    for candidate in sorted(models_dir.glob("trocr-*")):
        if (candidate / "config.json").is_file():
            log.warning("%s missing; using %s instead", wanted.name, candidate.name)
            return candidate
    raise FileNotFoundError(
        f"TrOCR weights not found in {wanted}. "
        "Run: python download_models.py -o models"
    )


def load_trocr(models_dir: Path, name: str | None = None):
    local_dir = _resolve_model_dir(models_dir, name)
    log.info("Loading TrOCR from %s", local_dir)
    processor = TrOCRProcessor.from_pretrained(str(local_dir), local_files_only=True)
    model = VisionEncoderDecoderModel.from_pretrained(
        str(local_dir), local_files_only=True
    )
    device = torch.device("cuda" if use_cuda() else "cpu")
    model.to(device)
    model.eval()
    return processor, model, device


def recognize_batch_local(processor, model, device, crops: list[np.ndarray]) -> list[str]:
    images = [to_pil_rgb(crop) for crop in crops]
    pixel_values = processor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_new_tokens=64)
    texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return [text.strip() for text in texts]


def recognize_line_api(api_key: str, crop: np.ndarray, retries: int = 5) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "image/png",
    }
    payload = crop_to_png_bytes(crop)
    last_error = ""
    for attempt in range(retries):
        response = requests.post(
            HF_INFERENCE_URL,
            headers=headers,
            data=payload,
            timeout=120,
        )
        if response.status_code in {429, 503} or not response.ok:
            last_error = f"{response.status_code}: {response.text[:300]}"
            time.sleep(min(30, 2 ** attempt))
            continue
        data = response.json()
        if isinstance(data, list) and data:
            return str(data[0].get("generated_text", "")).strip()
        if isinstance(data, dict):
            return str(data.get("generated_text", "")).strip()
        return str(data).strip()
    raise RuntimeError(f"Hugging Face API failed after retries: {last_error}")
