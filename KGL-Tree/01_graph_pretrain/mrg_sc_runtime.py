from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = REPO_ROOT / "Data"
SUPPORTED_DATASETS = ("daily", "hug", "mul", "ultratool")
SUPPORTED_SPLITS = ("train", "test")


def _normalize(name: str) -> str:
    return name.strip().lower()


def get_data_root() -> Path:
    raw = os.getenv("MRG_SC_DATA_ROOT", str(DEFAULT_DATA_ROOT))
    return Path(raw).expanduser().resolve()


def get_dataset_name(dataset: str | None = None) -> str:
    chosen = dataset or os.getenv("MRG_SC_DATASET", "ultratool")
    normalized = _normalize(chosen)
    if normalized not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset '{chosen}'. Expected one of: {', '.join(SUPPORTED_DATASETS)}."
        )
    return normalized


def get_split_dir(split: str, dataset: str | None = None) -> Path:
    normalized_split = _normalize(split)
    if normalized_split not in SUPPORTED_SPLITS:
        raise ValueError(
            f"Unsupported split '{split}'. Expected one of: {', '.join(SUPPORTED_SPLITS)}."
        )
    return get_data_root() / normalized_split / get_dataset_name(dataset)


def build_data_path(split: str, filename: str, dataset: str | None = None) -> str:
    return str(get_split_dir(split, dataset) / filename)


def get_embedding_url() -> str:
    return os.getenv("MRG_SC_EMBEDDING_URL", "http://localhost:11434/v1/embeddings").strip()


def get_embedding_model() -> str:
    return os.getenv("MRG_SC_EMBEDDING_MODEL", "bge-m3:latest").strip()


def get_embedding_timeout() -> float:
    raw = os.getenv("MRG_SC_EMBEDDING_TIMEOUT", "300").strip()
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid MRG_SC_EMBEDDING_TIMEOUT='{raw}'. Please provide a numeric value."
        ) from exc


def get_device() -> str:
    configured = os.getenv("MRG_SC_DEVICE")
    if configured:
        return configured.strip()

    try:
        import torch
    except ImportError:
        return "cpu"

    return "cuda" if torch.cuda.is_available() else "cpu"
