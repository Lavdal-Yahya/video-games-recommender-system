"""
Phase 7.6 — one-shot helper to download the local voice models.

The voice stack is fully local: faster-whisper for STT, Piper for TTS. Their
model files are large (~200 MB total), regenerate-able, and explicitly NOT
artifacts — they live under ``./models/`` and are gitignored. This script
fetches both into the expected layout so the FastAPI lifespan can load them
without surprises.

Layout (matches src/api.py's STT_MODEL_DIR / TTS_VOICE_PATH):
    models/
        whisper/   -> faster-whisper HF cache (base.en, ~145 MB on disk)
        piper/     -> en_US-lessac-medium.onnx (~61 MB) + .onnx.json

Run from the repo root:
    conda run -n ds python -m src.fetch_voice_models
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WHISPER_DIR = ROOT / "models" / "whisper"
PIPER_DIR = ROOT / "models" / "piper"

WHISPER_MODEL = "base.en"  # ~145 MB int8 — see src/api.py for the rationale
PIPER_VOICE = "en_US-lessac-medium"  # ~61 MB .onnx — see src/api.py


def fetch_whisper() -> None:
    # faster-whisper auto-downloads from HF on first WhisperModel(...). Passing
    # download_root pins the cache under ./models/whisper instead of
    # ~/.cache/huggingface, so the model travels with the repo checkout.
    from faster_whisper import WhisperModel

    WHISPER_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[whisper] loading {WHISPER_MODEL!r} into {WHISPER_DIR} ...")
    WhisperModel(
        WHISPER_MODEL,
        device="cpu",
        compute_type="int8",
        download_root=str(WHISPER_DIR),
    )
    print("[whisper] done.")


def fetch_piper() -> None:
    # Piper ships a CLI; we shell out to it so we don't duplicate its download
    # logic. `python -m piper.download_voices <name> --data-dir <dir>` writes
    # both the .onnx and the .onnx.json into <dir>.
    PIPER_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[piper] downloading voice {PIPER_VOICE!r} into {PIPER_DIR} ...")
    subprocess.run(
        [sys.executable, "-m", "piper.download_voices", PIPER_VOICE,
         "--data-dir", str(PIPER_DIR)],
        check=True,
    )
    print("[piper] done.")


def main() -> None:
    fetch_whisper()
    fetch_piper()
    print("\nAll voice models are in place. Start the API with:")
    print("    conda run -n ds uvicorn src.api:app --host 127.0.0.1 --port 8765")


if __name__ == "__main__":
    main()
