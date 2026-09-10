import argparse
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from pipeline_eval import evaluate_predictions, load_json, save_json


THIS_DIR = Path(__file__).resolve().parent
MRG_TREE_DIR = THIS_DIR.parent
WORKSPACE_ROOT = MRG_TREE_DIR.parent
MODULE3_DIR = MRG_TREE_DIR / "03_frontier_tree_search"
MODULE3_MAIN = MODULE3_DIR / "main.py"
DATA_ROOT = WORKSPACE_ROOT / "xiaorong_data" / "Gemma31b"
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "ablations" / "test"
DEFAULT_EXPERIMENTS_JSON = THIS_DIR / "ablation_config.json"

DEFAULT_DATASETS = ["daily", "hug", "mul", "ultratool"]
SUMMARY_FILE_NAME = "ablation_metrics_main_table.json"
SUMMARY_LOCK_TIMEOUT_SEC = 600
SUMMARY_LOCK_STALE_SEC = 7200


# ==================== Manual Run Config ====================
# 直接运行:
# python MRG-Tree/Ablation/auto_ablation.py
#
# 常改项优先放在这里，命令行参数只是可选覆盖。
MODEL_NAME = "Gemma31b"
LLM_MODEL = "none"
DAG_LLM_MODEL = None
PATH_LLM_MODEL = None
LLM_BASE_URL = "http://127.0.0.1:11434/v1/"
LLM_API_KEY = "ollama"
LLM_TEMPERATURE = 0.0

DATA_FOLDER_NAME = "Gemma31b"
DEVICE = "cuda"
MRG_MODEL_FILENAME = "model_best.pth"

MRG_KG_FILENAME = "service_kg_multi.gpickle"
TRAIN_TASKS_FILENAME = "data.json"
SERVICE_FILENAME = "tool_desc.json"
GRAPH_FILENAME = "graph_desc.json"
INPUT_FILENAME = "stage2_dag.json"
OUTPUT_FILENAME = "stage3_tree_candidates.json"
UNFINISHED_FILENAME = "unfinished_stage3_tree_candidates.json"

EMBEDDING_URL = "http://127.0.0.1:11434/v1/embeddings"
EMBEDDING_MODEL = "mxbai-embed-large:latest"
EMBEDDING_TIMEOUT = "300"

DATA_ROOT_DIR = str(DATA_ROOT)
TRAIN_ROOT_DIR = None
TEST_ROOT_DIR = None
OUTPUT_ROOT_DIR = None
SUMMARY_JSON_PATH = None

DATASETS_TO_RUN = DEFAULT_DATASETS
EXPERIMENTS_JSON_PATH = str(DEFAULT_EXPERIMENTS_JSON)
ONLY_EXPERIMENTS = None

TASK_LIMIT = None
ENABLE_RESUME = True
SAVE_EVERY_N = 20
PRINT_EVERY_N = 10
VERBOSE_PROGRESS = True
STAGE3_WORKERS = 4
TASK_TIMEOUT = 1800
# ==========================================================


RESULT_SORT_KEYS = ["strict_ohr", "real_er", "soft_ohr", "mp", "mr"]
METRIC_KEYS = [
    "avg_precision",
    "avg_recall",
    "avg_dgd",
    "loose_granularity_acc",
    "strict_ohr",
    "real_er",
    "all_er",
    "er_soft",
    "soft_ohr",
]

PARAM_TO_ENV = {
    "USE_BEAM": "MRG_TREE_USE_BEAM",
    "BEAM_WIDTH": "MRG_TREE_BEAM_WIDTH",
    "USE_DYNAMIC_BEAM": "MRG_TREE_USE_DYNAMIC_BEAM",
    "BEAM_MIN": "MRG_TREE_BEAM_MIN",
    "BEAM_MAX": "MRG_TREE_BEAM_MAX",
    "BEAM_THRESHOLD_HIGH": "MRG_TREE_BEAM_THRESHOLD_HIGH",
    "BEAM_THRESHOLD_LOW": "MRG_TREE_BEAM_THRESHOLD_LOW",
    "STEP1_SEMANTIC_KEEP_RATIO": "MRG_TREE_STEP1_SEMANTIC_KEEP_RATIO",
    "HISTORY_TOPK": "MRG_TREE_HISTORY_TOPK",
    "HISTORY_BONUS_STEP1": "MRG_TREE_HISTORY_BONUS_STEP1",
    "HISTORY_BONUS_LATER": "MRG_TREE_HISTORY_BONUS_LATER",
    "STRICT_NEIGHBOR_ONLY_AFTER_STEP1": "MRG_TREE_STRICT_NEIGHBOR_ONLY_AFTER_STEP1",
    "REQUIREMENT_TOPK": "MRG_TREE_REQUIREMENT_TOPK",
    "REQUIREMENT_MIN_PLAN_SIZE": "MRG_TREE_REQUIREMENT_MIN_PLAN_SIZE",
    "REQUIREMENT_COVERAGE_THRESHOLD": "MRG_TREE_REQUIREMENT_COVERAGE_THRESHOLD",
    "TREE_BUDGET_MULTIPLIER": "MRG_TREE_BUDGET_MULTIPLIER",
    "TREE_QUEUE_SIZE": "MRG_TREE_QUEUE_SIZE",
    "TREE_CHILDREN_PER_REQUIREMENT": "MRG_TREE_CHILDREN_PER_REQUIREMENT",
    "TREE_RELATION_KEEP": "MRG_TREE_RELATION_KEEP",
    "TREE_MATCH_THRESHOLD": "MRG_TREE_MATCH_THRESHOLD",
    "TREE_MRG_PROB_THRESHOLD": "MRG_TREE_MRG_PROB_THRESHOLD",
    "TREE_ALPHA_MATCH": "MRG_TREE_ALPHA_MATCH",
    "TREE_BETA_MRG": "MRG_TREE_BETA_MRG",
    "TREE_ETA_HISTORY": "MRG_TREE_ETA_HISTORY",
    "TREE_GAMMA_DEP": "MRG_TREE_GAMMA_DEP",
    "TREE_RHO_REDUNDANCY": "MRG_TREE_RHO_REDUNDANCY",
    "TREE_FRONTIER_SCORE_THRESHOLD": "MRG_TREE_FRONTIER_SCORE_THRESHOLD",
    "TREE_ENABLE_STATE_DOMINANCE": "MRG_TREE_ENABLE_STATE_DOMINANCE",
    "TREE_USE_DEP_SIG": "MRG_TREE_USE_DEP_SIG",
    "IGNORE_DAG_ORDER": "MRG_TREE_IGNORE_DAG_ORDER",
    "DISABLE_DEPENDS_ON": "MRG_TREE_DISABLE_DEPENDS_ON",
    "MODULE4_TOP_K": "MRG_TREE_MODULE4_TOP_K",
    "MODULE4_INCLUDE_M_MINUS_1": "MRG_TREE_MODULE4_INCLUDE_M_MINUS_1",
}

DEFAULT_MODULE3_CONFIG = {
    "USE_BEAM": True,
    "BEAM_WIDTH": 5,
    "USE_DYNAMIC_BEAM": False,
    "BEAM_MIN": 1,
    "BEAM_MAX": 10,
    "BEAM_THRESHOLD_HIGH": 0.30,
    "BEAM_THRESHOLD_LOW": 0.05,
    "STEP1_SEMANTIC_KEEP_RATIO": 0.5,
    "HISTORY_TOPK": 10,
    "HISTORY_BONUS_STEP1": 0.15,
    "HISTORY_BONUS_LATER": 0.00,
    "STRICT_NEIGHBOR_ONLY_AFTER_STEP1": True,
    "REQUIREMENT_TOPK": 24,
    "REQUIREMENT_MIN_PLAN_SIZE": 2,
    "REQUIREMENT_COVERAGE_THRESHOLD": 0.55,
    "TREE_BUDGET_MULTIPLIER": 8,
    "TREE_QUEUE_SIZE": 24,
    "TREE_CHILDREN_PER_REQUIREMENT": 4,
    "TREE_RELATION_KEEP": 1,
    "TREE_MATCH_THRESHOLD": 0.05,
    "TREE_MRG_PROB_THRESHOLD": 0.55,
    "TREE_ALPHA_MATCH": 1.0,
    "TREE_BETA_MRG": 1.0,
    "TREE_ETA_HISTORY": 0.6,
    "TREE_GAMMA_DEP": 0.4,
    "TREE_RHO_REDUNDANCY": 0.25,
    "TREE_FRONTIER_SCORE_THRESHOLD": 0.0,
    "TREE_ENABLE_STATE_DOMINANCE": True,
    "TREE_USE_DEP_SIG": True,
    "IGNORE_DAG_ORDER": False,
    "DISABLE_DEPENDS_ON": False,
    "MODULE4_TOP_K": 5,
    "MODULE4_INCLUDE_M_MINUS_1": True,
}


def _experiment_engine(exp: Dict[str, Any]) -> str:
    return str(exp.get("engine", "mrg_tree")).strip().lower() or "mrg_tree"


def _clean_flags_for_output(flags: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in flags.items() if not str(key).startswith("__")}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _slugify(name: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(name).strip())
    return value.strip("_") or "item"


def _normalize_workers(value: Optional[int]) -> int:
    try:
        workers = int(value or 1)
    except Exception:
        workers = 1
    return max(1, workers)


def _normalize_timeout(value: Optional[int]) -> int:
    try:
        timeout = int(value or 1800)
    except Exception:
        timeout = 1800
    return max(60, timeout)


def _cleanup_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _experiment_flags(exp: Dict[str, Any]) -> Dict[str, Any]:
    flags = {}
    if isinstance(exp.get("flags"), dict):
        flags.update(exp["flags"])
    if isinstance(exp.get("infer"), dict):
        flags.update(exp["infer"])
    return flags


def _load_experiments(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    if not isinstance(data, list):
        raise ValueError(f"Experiment config must be a list: {path}")
    return data


def _filter_experiments(experiments: List[Dict[str, Any]], only_names: Optional[List[str]]) -> List[Dict[str, Any]]:
    if not only_names:
        return experiments
    allow = {name.strip() for name in only_names if str(name).strip()}
    return [exp for exp in experiments if exp.get("name") in allow]


def _sort_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _score_tuple(item: Dict[str, Any]):
        if item.get("status") not in {"ok", "partial_error"}:
            return (1,)

        metrics = item.get("macro_avg", {})
        values = [0]
        for key in RESULT_SORT_KEYS:
            mapped = {
                "mp": "avg_precision",
                "mr": "avg_recall",
                "strict_ohr": "strict_ohr",
                "real_er": "real_er",
                "soft_ohr": "soft_ohr",
            }[key]
            values.append(-float(metrics.get(mapped, 0.0)))
        values.append(item.get("name", ""))
        return tuple(values)

    return sorted(results, key=_score_tuple)


def _load_existing_results(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = load_json(str(path))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return data


def _save_results(path: Path, results: List[Dict[str, Any]]) -> None:
    _ensure_dir(path.parent)
    save_json(str(path), _sort_results(results))


def _macro_average(metrics_by_dataset: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    valid_metrics = [metrics for metrics in metrics_by_dataset.values() if isinstance(metrics, dict) and "error" not in metrics]
    if not valid_metrics:
        return {}

    averaged = {}
    for key in METRIC_KEYS:
        averaged[key] = sum(float(metrics.get(key, 0.0)) for metrics in valid_metrics) / len(valid_metrics)
    return averaged


def _summary_lock_path(path: Path) -> Path:
    return path.with_name(path.name + ".lock")


def _acquire_summary_lock(path: Path) -> Path:
    lock_path = _summary_lock_path(path)
    deadline = time.time() + SUMMARY_LOCK_TIMEOUT_SEC
    while True:
        try:
            lock_path.mkdir()
            owner = {"pid": os.getpid(), "created_at": time.time()}
            save_json(str(lock_path / "owner.json"), owner)
            return lock_path
        except FileExistsError:
            try:
                mtime = lock_path.stat().st_mtime
                if time.time() - mtime > SUMMARY_LOCK_STALE_SEC:
                    shutil.rmtree(lock_path, ignore_errors=True)
                    continue
            except FileNotFoundError:
                continue

            if time.time() >= deadline:
                raise TimeoutError(f"Timed out waiting for summary lock: {lock_path}")
            time.sleep(1.0)


def _release_summary_lock(lock_path: Path) -> None:
    shutil.rmtree(lock_path, ignore_errors=True)


def _merge_result_entries(existing: Optional[Dict[str, Any]], new_entry: Dict[str, Any]) -> Dict[str, Any]:
    if not existing:
        merged = deepcopy(new_entry)
        merged["macro_avg"] = _macro_average(merged.get("datasets", {}))
        merged["status"] = "ok" if all("error" not in metrics for metrics in merged.get("datasets", {}).values()) else "partial_error"
        return merged

    merged = deepcopy(existing)
    for key in [
        "name",
        "engine",
        "description",
        "model_name",
        "llm_model",
        "dag_llm_model",
        "path_llm_model",
        "llm_base_url",
        "llm_temperature",
        "flags",
    ]:
        merged[key] = new_entry.get(key, merged.get(key))

    merged_datasets = dict(existing.get("datasets", {}))
    merged_datasets.update(new_entry.get("datasets", {}))
    merged["datasets"] = merged_datasets

    merged_artifacts = dict(existing.get("artifacts", {}))
    merged_artifacts.update(new_entry.get("artifacts", {}))
    merged["artifacts"] = merged_artifacts

    merged["duration_sec"] = round(float(existing.get("duration_sec", 0.0)) + float(new_entry.get("duration_sec", 0.0)), 2)
    merged["macro_avg"] = _macro_average(merged_datasets)
    merged["status"] = "ok" if all("error" not in metrics for metrics in merged_datasets.values()) else "partial_error"
    return merged


def _upsert_result_with_lock(path: Path, result_entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    lock_path = _acquire_summary_lock(path)
    try:
        results = _load_existing_results(path)
        existing = next((item for item in results if item.get("name") == result_entry.get("name")), None)
        merged_entry = _merge_result_entries(existing, result_entry)
        results = [item for item in results if item.get("name") != result_entry.get("name")]
        results.append(merged_entry)
        _save_results(path, results)
        return results
    finally:
        _release_summary_lock(lock_path)


def _build_roots() -> Dict[str, Path]:
    data_root = Path(DATA_ROOT_DIR)
    train_root = Path(TRAIN_ROOT_DIR) if TRAIN_ROOT_DIR else data_root / "train"
    test_root = Path(TEST_ROOT_DIR) if TEST_ROOT_DIR else data_root / "test"
    output_root = Path(OUTPUT_ROOT_DIR) if OUTPUT_ROOT_DIR else DEFAULT_OUTPUT_ROOT
    summary_path = Path(SUMMARY_JSON_PATH) if SUMMARY_JSON_PATH else output_root.parent / SUMMARY_FILE_NAME
    return {
        "data_root": data_root,
        "train_root": train_root,
        "test_root": test_root,
        "output_root": output_root,
        "summary_path": summary_path,
    }


def _build_dataset_paths(output_root: Path, exp_slug: str, dataset: str, create_dir: bool = True) -> Dict[str, Path]:
    dataset_dir = output_root / dataset / exp_slug
    if create_dir:
        _ensure_dir(dataset_dir)
    return {
        "dataset_dir": dataset_dir,
        "prediction": dataset_dir / OUTPUT_FILENAME,
        "unfinished": dataset_dir / UNFINISHED_FILENAME,
        "metrics": dataset_dir / "metrics.json",
        "error": dataset_dir / "dataset_error.json",
        "config": dataset_dir / "config.json",
    }


def _build_data_paths(roots: Dict[str, Path], dataset: str) -> Dict[str, Path]:
    train_dir = roots["train_root"] / dataset
    test_dir = roots["test_root"] / dataset
    return {
        "train_dir": train_dir,
        "test_dir": test_dir,
        "input_file": test_dir / INPUT_FILENAME,
        "tool_desc_file": test_dir / SERVICE_FILENAME,
        "graph_file": test_dir / GRAPH_FILENAME,
        "kg_file": train_dir / MRG_KG_FILENAME,
        "train_tasks_file": train_dir / TRAIN_TASKS_FILENAME,
        "model_file": train_dir / MRG_MODEL_FILENAME,
    }


def _get_task_id(task: Dict[str, Any]) -> str:
    return str(task.get("annotation_id", ""))


def _is_valid_prediction_task(task: Dict[str, Any]) -> bool:
    if not isinstance(task, dict):
        return False
    if task.get("annotation_id") is None:
        return False
    if "service_path_list" not in task or "recom_result" not in task:
        return False
    if task.get("service_path_list") is None or task.get("recom_result") is None:
        return False
    if not isinstance(task.get("service_path_list"), list):
        return False
    if not isinstance(task.get("recom_result"), list):
        return False
    return True


def _load_total_tasks(test_root: Path, dataset: str) -> int:
    input_file = test_root / dataset / INPUT_FILENAME
    try:
        data = load_json(str(input_file))
    except Exception:
        return 0
    if TASK_LIMIT is not None:
        data = data[:TASK_LIMIT]
    return len(data)


def _diagnose_prediction_file(path: Path, total_tasks: int) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False, "valid_count": 0, "invalid_count": 0, "complete": False}
    try:
        data = load_json(str(path))
    except Exception:
        return {"exists": True, "valid_count": 0, "invalid_count": total_tasks or 1, "complete": False}
    if not isinstance(data, list):
        return {"exists": True, "valid_count": 0, "invalid_count": total_tasks or 1, "complete": False}

    valid_count = 0
    invalid_count = 0
    seen_ids = set()
    for item in data:
        task_id = _get_task_id(item)
        if task_id in seen_ids:
            invalid_count += 1
            continue
        seen_ids.add(task_id)
        if _is_valid_prediction_task(item):
            valid_count += 1
        else:
            invalid_count += 1

    complete = total_tasks > 0 and valid_count >= total_tasks and invalid_count == 0
    return {
        "exists": True,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "complete": complete,
    }


def _experiment_artifacts_complete(
    output_root: Path,
    exp_name: str,
    datasets: List[str],
    test_root: Path,
) -> bool:
    exp_slug = _slugify(exp_name)
    for dataset in datasets:
        total_tasks = _load_total_tasks(test_root, dataset)
        paths = _build_dataset_paths(output_root, exp_slug, dataset, create_dir=False)
        pred_info = _diagnose_prediction_file(paths["prediction"], total_tasks)
        if not paths["metrics"].exists():
            return False
        if not pred_info["complete"]:
            return False
    return True


def _load_full_config_if_compatible(full_config_path: Path, current_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not full_config_path.exists():
        return None
    full_config = load_json(str(full_config_path))
    keys_to_match = [
        "engine",
        "model_name",
        "llm_model",
        "dag_llm_model",
        "path_llm_model",
        "llm_base_url",
        "llm_temperature",
        "flags",
        "train_dir",
        "test_dir",
        "limit",
        "model_file",
    ]
    for key in keys_to_match:
        if full_config.get(key) != current_config.get(key):
            return None
    return full_config


def _seed_reusable_outputs_from_source(
    dataset: str,
    exp_name: str,
    current_paths: Dict[str, Path],
    source_paths: Dict[str, Path],
    current_config: Dict[str, Any],
    resume: bool,
    source_label: str,
) -> None:
    if not resume or source_label == exp_name:
        return

    full_config = _load_full_config_if_compatible(source_paths["config"], current_config)
    if full_config is None:
        if VERBOSE_PROGRESS:
            print(f"[{dataset}] reuse from {source_label} skipped: config missing or incompatible")
        return

    copied = []
    for key in ["prediction", "metrics"]:
        src = source_paths[key]
        dst = current_paths[key]
        if dst.exists() or not src.exists():
            continue
        shutil.copy2(src, dst)
        copied.append(dst.name)

    if copied and VERBOSE_PROGRESS:
        print(f"[{dataset}] reuse from {source_label} -> {', '.join(copied)}")


def _build_env(dataset: str, data_paths: Dict[str, Path], output_paths: Dict[str, Path], flags: Dict[str, Any]) -> Dict[str, str]:
    env = os.environ.copy()
    env["MRG_TREE_ABLATION_ENGINE"] = str(flags.get("__engine__", "mrg_tree"))
    env["MRG_SC_DATA_ROOT"] = str(data_paths["train_dir"].parents[1])
    env["MRG_SC_DATASET"] = dataset
    env["MRG_SC_DEVICE"] = DEVICE
    env["MRG_SC_EMBEDDING_URL"] = EMBEDDING_URL
    env["MRG_SC_EMBEDDING_MODEL"] = EMBEDDING_MODEL
    env["MRG_SC_EMBEDDING_TIMEOUT"] = str(EMBEDDING_TIMEOUT)

    env["MRG_TREE_KG_PATH"] = str(data_paths["kg_file"])
    env["MRG_TREE_MODEL_PATH"] = str(data_paths["model_file"])
    env["MRG_TREE_TRAIN_TASKS_FILE"] = str(data_paths["train_tasks_file"])
    env["MRG_TREE_SERVICE_FILE"] = str(data_paths["tool_desc_file"])

    env["MRG_TREE_MODULE3_INPUT_FILE"] = str(data_paths["input_file"])
    env["MRG_TREE_MODULE3_OUTPUT_FILE"] = str(output_paths["prediction"])
    env["MRG_TREE_MODULE3_UNFINISHED_FILE"] = str(output_paths["unfinished"])
    env["MRG_TREE_MODULE3_SAVE_INTERVAL"] = str(SAVE_EVERY_N)
    env["MRG_TREE_MODULE3_MAX_WORKERS"] = str(_normalize_workers(flags.get("stage3_workers", STAGE3_WORKERS)))
    env["MRG_TREE_MODULE3_TASK_TIMEOUT"] = str(TASK_TIMEOUT)

    config = deepcopy(DEFAULT_MODULE3_CONFIG)
    config.update({k: v for k, v in flags.items() if k in PARAM_TO_ENV})
    for key, value in config.items():
        env[PARAM_TO_ENV[key]] = str(value)

    for key, value in flags.items():
        if key.startswith("MRG_"):
            env[key] = str(value)

    if TASK_LIMIT is not None:
        env["MRG_TREE_ABLATION_TASK_LIMIT"] = str(TASK_LIMIT)

    return env


def _build_dataset_result_entry(
    dataset: str,
    data_paths: Dict[str, Path],
    output_paths: Dict[str, Path],
    exp: Dict[str, Any],
    flags: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "experiment": exp["name"],
        "engine": _experiment_engine(exp),
        "description": exp.get("description", ""),
        "model_name": MODEL_NAME,
        "llm_model": LLM_MODEL,
        "dag_llm_model": DAG_LLM_MODEL or LLM_MODEL,
        "path_llm_model": PATH_LLM_MODEL or LLM_MODEL,
        "llm_base_url": LLM_BASE_URL,
        "llm_temperature": LLM_TEMPERATURE,
        "dataset": dataset,
        "flags": _clean_flags_for_output(flags),
        "train_dir": str(data_paths["train_dir"]),
        "test_dir": str(data_paths["test_dir"]),
        "model_file": str(data_paths["model_file"]),
        "limit": TASK_LIMIT,
        "stage3_workers": int(flags.get("stage3_workers", STAGE3_WORKERS)),
        "output_dir": str(output_paths["dataset_dir"]),
    }


def _run_single_dataset(
    dataset: str,
    exp: Dict[str, Any],
    roots: Dict[str, Path],
    full_flags: Dict[str, Any],
    resume: bool,
) -> Dict[str, Any]:
    flags = _experiment_flags(exp)
    flags["__engine__"] = _experiment_engine(exp)
    data_paths = _build_data_paths(roots, dataset)
    output_paths = _build_dataset_paths(roots["output_root"], _slugify(exp["name"]), dataset)

    if not data_paths["train_dir"].exists():
        return {"error": f"missing train dir: {data_paths['train_dir']}"}
    if not data_paths["test_dir"].exists():
        return {"error": f"missing test dir: {data_paths['test_dir']}"}
    if not data_paths["input_file"].exists():
        return {"error": f"missing input file: {data_paths['input_file']}"}

    config_payload = _build_dataset_result_entry(dataset, data_paths, output_paths, exp, flags)
    save_json(str(output_paths["config"]), config_payload)

    source_name = exp.get("reuse_from")
    if not source_name and exp.get("reuse_from_full", False):
        source_name = "full_model"
    if source_name:
        source_paths = _build_dataset_paths(roots["output_root"], _slugify(source_name), dataset, create_dir=False)
        _seed_reusable_outputs_from_source(
            dataset=dataset,
            exp_name=exp["name"],
            current_paths=output_paths,
            source_paths=source_paths,
            current_config=config_payload,
            resume=resume,
            source_label=source_name,
        )

    total_tasks = _load_total_tasks(roots["test_root"], dataset)
    pred_info = _diagnose_prediction_file(output_paths["prediction"], total_tasks)
    if resume and output_paths["metrics"].exists() and pred_info["complete"]:
        return load_json(str(output_paths["metrics"]))

    env = _build_env(dataset, data_paths, output_paths, flags)
    subprocess.run(
        [sys.executable, str(MODULE3_MAIN)],
        cwd=str(MODULE3_DIR),
        env=env,
        check=True,
    )

    metrics = evaluate_predictions(
        load_json(str(output_paths["prediction"]))[:TASK_LIMIT] if TASK_LIMIT is not None else load_json(str(output_paths["prediction"])),
        str(data_paths["tool_desc_file"]),
        str(data_paths["graph_file"]),
        show_progress=True,
    )
    save_json(str(output_paths["metrics"]), metrics)
    if output_paths["error"].exists():
        output_paths["error"].unlink()
    return metrics


def _build_result_entry(
    exp: Dict[str, Any],
    dataset_metrics: Dict[str, Dict[str, Any]],
    dataset_dirs: Dict[str, str],
    duration_sec: float,
    flags: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "name": exp["name"],
        "engine": _experiment_engine(exp),
        "description": exp.get("description", ""),
        "status": "ok" if all("error" not in metrics for metrics in dataset_metrics.values()) else "partial_error",
        "model_name": MODEL_NAME,
        "llm_model": LLM_MODEL,
        "dag_llm_model": DAG_LLM_MODEL or LLM_MODEL,
        "path_llm_model": PATH_LLM_MODEL or LLM_MODEL,
        "llm_base_url": LLM_BASE_URL,
        "llm_temperature": LLM_TEMPERATURE,
        "flags": _clean_flags_for_output(flags),
        "duration_sec": round(duration_sec, 2),
        "datasets": dataset_metrics,
        "macro_avg": _macro_average(dataset_metrics),
        "artifacts": dataset_dirs,
    }


def _find_completed_result(results: List[Dict[str, Any]], name: str, required_datasets: List[str]) -> Optional[Dict[str, Any]]:
    for item in results:
        if item.get("name") != name or item.get("status") != "ok":
            continue
        dataset_metrics = item.get("datasets", {})
        if all(dataset in dataset_metrics and "error" not in dataset_metrics[dataset] for dataset in required_datasets):
            return item
    return None


def run_experiment(
    exp: Dict[str, Any],
    datasets: List[str],
    roots: Dict[str, Path],
    resume: bool,
    results_path: Path,
) -> Dict[str, Any]:
    exp_name = exp["name"]
    flags = _experiment_flags(exp)
    full_flags = _experiment_flags({"name": "full_model", "infer": {}})

    start_time = time.time()
    dataset_metrics = {}
    dataset_dirs = {}

    for dataset in datasets:
        output_paths = _build_dataset_paths(roots["output_root"], _slugify(exp_name), dataset)
        dataset_dirs[dataset] = str(output_paths["dataset_dir"])

        try:
            if VERBOSE_PROGRESS:
                print(
                    f"\n=== Dataset: {dataset} | Experiment: {exp_name} | Model: {MODEL_NAME} | "
                    f"resume={'on' if resume else 'off'} | "
                    f"stage3_workers={int(flags.get('stage3_workers', STAGE3_WORKERS))} ==="
                )
            metrics = _run_single_dataset(dataset, exp, roots, full_flags, resume)
            dataset_metrics[dataset] = metrics
        except Exception as exc:
            error_payload = {
                "dataset": dataset,
                "experiment": exp_name,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            print(f"\n[{dataset}] pipeline failed: {error_payload['error']}")
            dataset_metrics[dataset] = {"error": error_payload["error"]}
            save_json(str(output_paths["error"]), error_payload)

        _cleanup_memory()

    duration_sec = time.time() - start_time
    result_entry = _build_result_entry(exp, dataset_metrics, dataset_dirs, duration_sec, flags)
    _upsert_result_with_lock(results_path, result_entry)
    return result_entry


def parse_args():
    parser = argparse.ArgumentParser(description="Run MRG-Tree ablation experiments across multiple datasets.")
    parser.add_argument("--model-name", default=MODEL_NAME, help="Experiment model tag used for output grouping.")
    parser.add_argument("--data-folder-name", default=DATA_FOLDER_NAME, help="Data folder name under Data/.")
    parser.add_argument("--mrg-model-filename", default=MRG_MODEL_FILENAME, help="MRG-SC checkpoint filename under each train/<dataset>/ directory.")
    parser.add_argument("--mrg-kg-filename", default=MRG_KG_FILENAME, help="MRG-SC knowledge graph filename under each train/<dataset>/ directory.")
    parser.add_argument("--llm-model", default=LLM_MODEL, help="Reserved LLM model tag stored in configs and summary.")
    parser.add_argument("--dag-llm-model", default=DAG_LLM_MODEL, help="Reserved DAG LLM model tag.")
    parser.add_argument("--path-llm-model", default=PATH_LLM_MODEL, help="Reserved path LLM model tag.")
    parser.add_argument("--llm-base-url", default=LLM_BASE_URL, help="Reserved base URL stored in configs and summary.")
    parser.add_argument("--llm-api-key", default=LLM_API_KEY, help="Reserved API key field for record consistency.")
    parser.add_argument("--llm-temperature", type=float, default=LLM_TEMPERATURE, help="Reserved temperature field.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS_TO_RUN, help="Datasets to run.")
    parser.add_argument("--experiments-json", default=EXPERIMENTS_JSON_PATH, help="Path to ablation experiment config JSON.")
    parser.add_argument("--data-root", default=DATA_ROOT_DIR, help="Base data directory.")
    parser.add_argument("--train-root", default=TRAIN_ROOT_DIR, help="Optional explicit train root.")
    parser.add_argument("--test-root", default=TEST_ROOT_DIR, help="Optional explicit test root.")
    parser.add_argument("--output-root", default=OUTPUT_ROOT_DIR, help="Optional explicit output root.")
    parser.add_argument("--summary-path", default=SUMMARY_JSON_PATH, help="Optional explicit summary JSON path.")
    parser.add_argument("--only-experiments", nargs="*", default=ONLY_EXPERIMENTS, help="Optional experiment name whitelist.")
    parser.add_argument("--task-limit", type=int, default=TASK_LIMIT, help="Only run the first N tasks.")
    parser.add_argument("--disable-resume", action="store_true", help="Disable resume.")
    parser.add_argument("--stage3-workers", type=int, default=STAGE3_WORKERS, help="Workers used inside module3.")
    parser.add_argument("--task-timeout", type=int, default=TASK_TIMEOUT, help="Per-task timeout passed to module3.")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY_N, help="Module3 autosave interval.")
    return parser.parse_args()


def main():
    global MODEL_NAME
    global DATA_FOLDER_NAME
    global MRG_MODEL_FILENAME
    global MRG_KG_FILENAME
    global LLM_MODEL
    global DAG_LLM_MODEL
    global PATH_LLM_MODEL
    global LLM_BASE_URL
    global LLM_API_KEY
    global LLM_TEMPERATURE
    global DATASETS_TO_RUN
    global EXPERIMENTS_JSON_PATH
    global DATA_ROOT_DIR
    global TRAIN_ROOT_DIR
    global TEST_ROOT_DIR
    global OUTPUT_ROOT_DIR
    global SUMMARY_JSON_PATH
    global ONLY_EXPERIMENTS
    global TASK_LIMIT
    global ENABLE_RESUME
    global STAGE3_WORKERS
    global TASK_TIMEOUT
    global SAVE_EVERY_N

    args = parse_args()
    MODEL_NAME = args.model_name
    DATA_FOLDER_NAME = args.data_folder_name
    MRG_MODEL_FILENAME = args.mrg_model_filename
    MRG_KG_FILENAME = args.mrg_kg_filename
    LLM_MODEL = args.llm_model
    DAG_LLM_MODEL = args.dag_llm_model or args.llm_model
    PATH_LLM_MODEL = args.path_llm_model or args.llm_model
    LLM_BASE_URL = args.llm_base_url
    LLM_API_KEY = args.llm_api_key
    LLM_TEMPERATURE = float(args.llm_temperature)
    DATASETS_TO_RUN = args.datasets
    EXPERIMENTS_JSON_PATH = args.experiments_json
    DATA_ROOT_DIR = args.data_root
    TRAIN_ROOT_DIR = args.train_root
    TEST_ROOT_DIR = args.test_root
    OUTPUT_ROOT_DIR = args.output_root
    SUMMARY_JSON_PATH = args.summary_path
    ONLY_EXPERIMENTS = args.only_experiments
    TASK_LIMIT = args.task_limit
    ENABLE_RESUME = not args.disable_resume
    STAGE3_WORKERS = _normalize_workers(args.stage3_workers)
    TASK_TIMEOUT = _normalize_timeout(args.task_timeout)
    SAVE_EVERY_N = _normalize_workers(args.save_every)

    roots = _build_roots()
    _ensure_dir(roots["output_root"])

    experiments = _filter_experiments(_load_experiments(Path(EXPERIMENTS_JSON_PATH)), ONLY_EXPERIMENTS)
    existing_results = _load_existing_results(roots["summary_path"])

    if VERBOSE_PROGRESS:
        print(f"MODEL_NAME       = {MODEL_NAME}")
        print(f"LLM_MODEL        = {LLM_MODEL}")
        print(f"DAG_LLM_MODEL    = {DAG_LLM_MODEL}")
        print(f"PATH_LLM_MODEL   = {PATH_LLM_MODEL}")
        print(f"DATA_FOLDER_NAME = {DATA_FOLDER_NAME}")
        print(f"MRG_MODEL_FILE  = {MRG_MODEL_FILENAME}")
        print(f"MRG_KG_FILE     = {MRG_KG_FILENAME}")
        print(f"TRAIN_ROOT       = {roots['train_root']}")
        print(f"TEST_ROOT        = {roots['test_root']}")
        print(f"OUTPUT_ROOT      = {roots['output_root']}")
        print(f"SUMMARY_PATH     = {roots['summary_path']}")
        print(f"DATASETS         = {DATASETS_TO_RUN}")
        print(f"EXPERIMENTS_JSON = {EXPERIMENTS_JSON_PATH}")

    progress = tqdm(experiments, desc="Ablation Experiments", ncols=100)
    for exp in progress:
        exp_name = exp["name"]
        progress.set_postfix(current=exp_name)

        completed = _find_completed_result(existing_results, exp_name, DATASETS_TO_RUN)
        if ENABLE_RESUME and completed and _experiment_artifacts_complete(
            output_root=roots["output_root"],
            exp_name=exp_name,
            datasets=DATASETS_TO_RUN,
            test_root=roots["test_root"],
        ):
            if VERBOSE_PROGRESS:
                print(f"[summary] skip experiment '{exp_name}' because datasets {DATASETS_TO_RUN} are already complete")
            continue

        result_entry = run_experiment(
            exp=exp,
            datasets=DATASETS_TO_RUN,
            roots=roots,
            resume=ENABLE_RESUME,
            results_path=roots["summary_path"],
        )
        existing_results = _load_existing_results(roots["summary_path"])
        if VERBOSE_PROGRESS:
            macro = result_entry.get("macro_avg", {})
            print(
                f"[done] {exp_name} | "
                f"MP={macro.get('avg_precision', 0.0):.4f} | "
                f"MR={macro.get('avg_recall', 0.0):.4f} | "
                f"DGD={macro.get('avg_dgd', 0.0):.4f} | "
                f"OHR={macro.get('strict_ohr', 0.0):.4f} | "
                f"ER={macro.get('real_er', 0.0):.4f}"
            )

    print(f"\nSaved summary to: {roots['summary_path']}")


if __name__ == "__main__":
    main()
