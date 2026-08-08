"""Central settings, loaded from .env (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    llama_bin_dir: Path
    qwen_model_path: Path
    llama_server_host: str
    llama_server_port: int
    ml_pipeline_path: Path
    sessions_path: Path

    @property
    def llama_server_exe(self) -> Path:
        return self.llama_bin_dir / "llama-server.exe"

    @property
    def llama_server_base_url(self) -> str:
        return f"http://{self.llama_server_host}:{self.llama_server_port}"


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (REPO_ROOT / path)


def load_settings() -> Settings:
    return Settings(
        llama_bin_dir=_resolve(os.environ.get("LLAMA_BIN_DIR", r"E:\llm\bin")),
        qwen_model_path=_resolve(
            os.environ.get("QWEN_MODEL_PATH", r"E:\llm\models\qwen2.5-coder-3b-instruct-q2_k.gguf")
        ),
        llama_server_host=os.environ.get("LLAMA_SERVER_HOST", "127.0.0.1"),
        llama_server_port=int(os.environ.get("LLAMA_SERVER_PORT", "8081")),
        ml_pipeline_path=_resolve(os.environ.get("ML_PIPELINE_PATH", "model/final_pipeline.joblib")),
        sessions_path=_resolve(os.environ.get("SESSIONS_PATH", "data/sessions.json")),
    )


settings = load_settings()
