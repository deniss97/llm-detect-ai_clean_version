from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

from huggingface_hub import snapshot_download

from app.config import settings


def main() -> None:
    if not settings.model_base_path:
        raise SystemExit("MODEL_BASE_PATH is not set.")

    token = settings.hugging_face_hub_token or settings.hf_token or None
    cache_dir = (
        os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.environ.get("HF_HUB_CACHE")
        or str(Path(os.environ.get("HF_HOME", "/root/.cache/huggingface")) / "hub")
    )

    print(f"Downloading {settings.model_base_path} into {cache_dir}...", flush=True)
    print(f"Interactive progress: {'yes' if sys.stderr.isatty() else 'no'}", flush=True)
    snapshot_download(
        repo_id=settings.model_base_path,
        cache_dir=cache_dir,
        token=token,
        resume_download=True,
    )
    print("Model files are cached.", flush=True)


if __name__ == "__main__":
    main()
