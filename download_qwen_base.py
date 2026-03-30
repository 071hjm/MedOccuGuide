from __future__ import annotations

import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


BASE_DIR = Path(r"D:\codex")
MODEL_DIR = BASE_DIR / "models" / "Qwen2.5-7B-Instruct"
ADAPTER_CONFIG = BASE_DIR / "LLaMA-Factory" / "saves" / "Qwen2.5-7B-Instruct" / "lora" / "train_2025-11-12-18-23-54" / "adapter_config.json"
HF_ENDPOINT = "https://hf-mirror.com"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MAX_WORKERS = 6
MODELSCOPE_DOWNLOAD_PARALLELS = 8


def required_model_files() -> list[str]:
    index_path = MODEL_DIR / "model.safetensors.index.json"
    if not index_path.exists():
        return []
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map", {})
    return sorted(set(weight_map.values()))


def download_complete() -> bool:
    required = required_model_files()
    if not required:
        return False
    return all((MODEL_DIR / name).exists() for name in required)


def update_adapter_path() -> None:
    if not ADAPTER_CONFIG.exists():
        return
    payload = json.loads(ADAPTER_CONFIG.read_text(encoding="utf-8"))
    payload["base_model_name_or_path"] = str(MODEL_DIR)
    ADAPTER_CONFIG.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def download_with_modelscope() -> None:
    print("Downloading with ModelScope...")
    os.environ["MODELSCOPE_DOWNLOAD_PARALLELS"] = str(MODELSCOPE_DOWNLOAD_PARALLELS)
    from modelscope import snapshot_download as ms_snapshot_download
    ms_snapshot_download(
        model_id=MODEL_ID,
        local_dir=str(MODEL_DIR),
        max_workers=MAX_WORKERS,
    )


def download_with_huggingface() -> None:
    print("Downloading with Hugging Face mirror...")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ["HF_ENDPOINT"] = HF_ENDPOINT
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(MODEL_DIR),
        max_workers=MAX_WORKERS,
    )


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not download_complete():
        try:
            download_with_modelscope()
        except Exception as exc:
            print(f"ModelScope download failed: {exc}")
    if not download_complete():
        download_with_huggingface()
    if not download_complete():
        raise RuntimeError("Base model download did not complete.")
    update_adapter_path()
    print(f"Base model ready: {MODEL_DIR}")


if __name__ == "__main__":
    main()
