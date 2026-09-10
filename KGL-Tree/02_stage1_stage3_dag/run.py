from pathlib import Path
import os
import subprocess
import sys


# ==================== 启动配置区 ====================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = Path(__file__).resolve().parent

# 这里主要是给 stage 内部脚本提供统一环境；真正的数据路径仍按各自 main.py 顶部相对路径写法走
EMBEDDING_URL = "http://127.0.0.1:11434/v1/embeddings"
EMBEDDING_MODEL = "bge-m3:latest"
EMBEDDING_TIMEOUT = "300"
LLM_BASE_URL = "http://127.0.0.1:11434/v1/"
LLM_MODEL = "mixtral:8x7b"
DEVICE = "cuda"

RUN_STAGE1 = True
RUN_STAGE2 = True
RUN_STAGE3 = True
# ================================================


def build_env() -> dict:
    env = os.environ.copy()
    env["MRG_SC_DEVICE"] = DEVICE
    env["MRG_SC_EMBEDDING_URL"] = EMBEDDING_URL
    env["MRG_SC_EMBEDDING_MODEL"] = EMBEDDING_MODEL
    env["MRG_SC_EMBEDDING_TIMEOUT"] = EMBEDDING_TIMEOUT
    env["MRG_SC_LLM_BASE_URL"] = LLM_BASE_URL
    env["MRG_SC_LLM_MODEL"] = LLM_MODEL
    return env


def run_script(relative_path: str, env: dict) -> None:
    script_path = MODULE_ROOT / relative_path
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    print(f"[MRG-Tree][stage1_stage3_dag] running: {script_path}")
    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(MODULE_ROOT),
        env=env,
        check=True,
    )


def main() -> None:
    env = build_env()
    print("=== MODULE 2: STAGE1-STAGE3 DAG ===")
    print(f"PROJECT_ROOT    = {PROJECT_ROOT}")
    print(f"MODULE_ROOT     = {MODULE_ROOT}")
    print(f"DEVICE          = {DEVICE}")
    print(f"EMBEDDING_URL   = {EMBEDDING_URL}")
    print(f"EMBEDDING_MODEL = {EMBEDDING_MODEL}")
    print(f"LLM_BASE_URL    = {LLM_BASE_URL}")
    print(f"LLM_MODEL       = {LLM_MODEL}")
    print("Path style note : stage1/stage2/stage3 各自的 main.py 仍按顶部相对路径配置运行")

    if RUN_STAGE1:
        run_script("stage1/main.py", env)
    if RUN_STAGE2:
        run_script("stage2/main.py", env)
    if RUN_STAGE3:
        run_script("stage3/main.py", env)


if __name__ == "__main__":
    main()
