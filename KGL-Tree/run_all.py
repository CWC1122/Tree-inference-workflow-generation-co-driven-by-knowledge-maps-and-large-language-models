from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def run_launcher(relative_path: str) -> None:
    launcher_path = ROOT / relative_path
    if not launcher_path.exists():
        raise FileNotFoundError(f"Launcher not found: {launcher_path}")

    print(f"[MRG-Tree] running launcher: {launcher_path}")
    subprocess.run(
        [sys.executable, str(launcher_path)],
        cwd=str(launcher_path.parent),
        check=True,
    )


def main() -> None:
    run_launcher("01_graph_pretrain/run.py")
    run_launcher("02_stage1_stage3_dag/run.py")
    run_launcher("03_frontier_tree_search/run.py")
    run_launcher("Evaluate/evaluate_soft.py")


if __name__ == "__main__":
    main()
