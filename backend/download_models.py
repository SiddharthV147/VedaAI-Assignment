#!/usr/bin/env python3
"""Download CRAFT weights into this project's models/ directory."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import gdown
from huggingface_hub import snapshot_download

CRAFT_NAME = "craft_mlt_25k.pth"
REFINER_NAME = "craft_refiner_CTW1500.pth"

CRAFT_SOURCES = [
    "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/craft_mlt_25k.pth",
    "https://huggingface.co/amitesh863/craft/resolve/main/craft_mlt_25k.pth",
    "https://drive.google.com/uc?id=1bupFXqT-VU6Jjeul13XP7yx2Sg5IHr4J",
]
REFINER_SOURCES = [
    "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/craft_refiner_CTW1500.pth",
    "https://huggingface.co/amitesh863/craft/resolve/main/craft_refiner_CTW1500.pth",
    "https://drive.google.com/uc?id=1xcE9qpJXp4ofINwXWVhhQIh9S8Z7cuGj",
]

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent / "models"
TROCR_REPO = "microsoft/trocr-large-handwritten"
TROCR_DIRNAME = "trocr-large-handwritten"


def _download_http(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, dest.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def download_if_needed(urls: list[str], dest: Path) -> None:
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        print(f"Already present: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in urls:
        try:
            print(f"Downloading {dest.name} from {url}")
            if "drive.google.com" in url:
                gdown.download(url, str(dest), quiet=False)
            else:
                _download_http(url, dest)
            if dest.is_file() and dest.stat().st_size > 100_000:
                print(f"Saved {dest}")
                return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"Failed ({exc!r}), trying next source...")
            if dest.is_file():
                dest.unlink()
    raise RuntimeError(f"Download failed for {dest.name}: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CRAFT detector weights into models/."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Directory to store weight files (default: ./models)",
    )
    parser.add_argument(
        "--skip-trocr",
        action="store_true",
        help="Only download CRAFT weights, skip TrOCR",
    )
    return parser.parse_args()


def download_trocr(models_dir: Path) -> None:
    dest = models_dir / TROCR_DIRNAME
    if (dest / "config.json").is_file() and (
        (dest / "model.safetensors").is_file() or (dest / "pytorch_model.bin").is_file()
    ):
        print(f"Already present: {dest}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {TROCR_REPO} -> {dest}")
    snapshot_download(
        repo_id=TROCR_REPO,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )
    print(f"Saved TrOCR to {dest}")


def main() -> None:
    args = parse_args()
    models_dir = args.output_dir.expanduser().resolve()
    download_if_needed(CRAFT_SOURCES, models_dir / CRAFT_NAME)
    download_if_needed(REFINER_SOURCES, models_dir / REFINER_NAME)
    if not args.skip_trocr:
        download_trocr(models_dir)
    print(f"Weights are in {models_dir}")


if __name__ == "__main__":
    main()
