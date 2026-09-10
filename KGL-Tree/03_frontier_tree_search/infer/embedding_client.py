from pathlib import Path
import sys

import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mrg_sc_runtime import get_embedding_model, get_embedding_timeout, get_embedding_url


class EmbeddingClient:
    def __init__(self, url: str | None = None, model: str | None = None, timeout: float | None = None):
        self.url = url or get_embedding_url()
        self.model = model or get_embedding_model()
        self.timeout = float(timeout if timeout is not None else get_embedding_timeout())

    def get_embedding(self, text: str):
        payload = {"model": self.model, "input": text or ""}
        response = requests.post(self.url, json=payload, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()
        if "data" not in data or not data["data"]:
            raise RuntimeError(f"Unexpected embedding response: {data}")
        return data["data"][0]["embedding"]
