from pathlib import Path
import os
import subprocess
import sys


# ==================== 启动配置区域 ====================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = Path(__file__).resolve().parent

DATA_FOLDER_NAME = "Llama3.1-70b"
DATABASE_NAME = "mul"
DEVICE = "cuda"

EMBEDDING_URL = "http://127.0.0.1:11434/v1/embeddings"
EMBEDDING_MODEL = "bge-m3:latest"
EMBEDDING_TIMEOUT = "300"

RUN_KG_BUILD = True
RUN_PRETRAIN = True
# =====================================================


def build_env() -> dict:
    env = os.environ.copy()
    env["MRG_SC_DATA_ROOT"] = str(PROJECT_ROOT / "Data" / DATA_FOLDER_NAME)
    env["MRG_SC_DATASET"] = DATABASE_NAME
    env["MRG_SC_DEVICE"] = DEVICE
    env["MRG_SC_EMBEDDING_URL"] = EMBEDDING_URL
    env["MRG_SC_EMBEDDING_MODEL"] = EMBEDDING_MODEL
    env["MRG_SC_EMBEDDING_TIMEOUT"] = EMBEDDING_TIMEOUT
    return env


def run_script(relative_path: str, env: dict) -> None:
    script_path = MODULE_ROOT / relative_path
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    print(f"[MRG-Tree][graph_pretrain] running: {script_path}")
    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(script_path.parent),
        env=env,
        check=True,
    )


def main() -> None:
    env = build_env()
    print("=== MODULE 1: GRAPH PRETRAIN ===")
    print(f"PROJECT_ROOT    = {PROJECT_ROOT}")
    print(f"DATA_FOLDER_NAME = {DATA_FOLDER_NAME}")
    print(f"DATABASE_NAME   = {DATABASE_NAME}")
    print(f"TRAIN_DB_DIR    = {PROJECT_ROOT / 'Data' / DATA_FOLDER_NAME / 'train' / DATABASE_NAME}")
    print(f"DEVICE          = {DEVICE}")
    print(f"EMBEDDING_URL   = {EMBEDDING_URL}")
    print(f"EMBEDDING_MODEL = {EMBEDDING_MODEL}")

    if RUN_KG_BUILD:
        run_script("graph_build/KG_gen_multi.py", env)
    if RUN_PRETRAIN:
        run_script("train/train_rgcn.py", env)


if __name__ == "__main__":
    main()
