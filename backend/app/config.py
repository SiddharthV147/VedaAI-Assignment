from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_TROCR_MODEL = "trocr-large-handwritten"


def load_env(base_dir: Path) -> None:
    for path in (base_dir / ".env", Path("/app/.env"), Path.cwd() / ".env"):
        if path.is_file():
            load_dotenv(path, override=False)
            break


def trocr_model_name() -> str:
    """Which TrOCR checkpoint to use.

    ``trocr-base-handwritten`` needs roughly half the memory of the large one
    and is much faster without a GPU, which is what a CPU host wants.
    """
    return (os.environ.get("TROCR_MODEL") or DEFAULT_TROCR_MODEL).strip()


@dataclass(frozen=True)
class PipelineConfig:
    """Settings for one run.

    The CRAFT thresholds are set for handwriting rather than the signage the
    model was trained on: pencil and light pen score much lower than printed
    type, so the seeds and the region growth around them are both loosened.
    Lower ``text_threshold`` finds fainter words, lower ``low_text`` grows the
    box further out from each word so ascenders and descenders stay inside,
    and lower ``link_threshold`` joins neighbouring words more readily.
    ``min_ink_ratio`` then throws away whatever came back empty.
    """

    models_dir: Path
    dpi: int = 300
    text_threshold: float = 0.30
    link_threshold: float = 0.10
    low_text: float = 0.15
    long_size: int = 2560
    min_ink_ratio: float = 0.002
    batch_size: int = 4
    trocr_model: str = DEFAULT_TROCR_MODEL
    use_api: bool = False
    huggingface_api_key: str = ""

    @classmethod
    def from_env(cls, models_dir: Path) -> "PipelineConfig":
        use_api = os.environ.get("USE_API", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        api_key = (
            os.environ.get("HUGGINGFACE_API_KEY")
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or ""
        ).strip()
        # A CPU host holds every image of the batch in memory at once, so a
        # smaller batch is the difference between running and being killed.
        default_batch = 4 if use_cuda() else 1
        try:
            batch_size = int(os.environ.get("TROCR_BATCH_SIZE", "") or default_batch)
        except ValueError:
            batch_size = default_batch
        return cls(
            models_dir=models_dir.resolve(),
            batch_size=max(1, batch_size),
            trocr_model=trocr_model_name(),
            use_api=use_api,
            huggingface_api_key=api_key,
        )


def use_cuda() -> bool:
    import torch

    flag = os.environ.get("TEXT_USE_CUDA", "auto").strip().lower()
    if flag in {"0", "false", "cpu"}:
        return False
    if flag in {"1", "true", "gpu", "cuda"}:
        return torch.cuda.is_available()
    return torch.cuda.is_available()


def resolve_models_dir(base_dir: Path) -> Path:
    env = os.environ.get("TEXT_MODELS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    for candidate in (
        base_dir / "models",
        Path("/app/models"),
        Path.cwd() / "models",
    ):
        if (candidate / "craft_mlt_25k.pth").is_file():
            return candidate.resolve()
    return (base_dir / "models").resolve()
