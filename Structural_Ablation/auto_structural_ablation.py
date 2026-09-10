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

STAGE1_MAIN = MRG_TREE_DIR / "02_stage1_stage3_dag" / "stage1" / "main.py"
STAGE2_MAIN = MRG_TREE_DIR / "02_stage1_stage3_dag" / "stage2" / "main.py"
STAGE3_MAIN = MRG_TREE_DIR / "02_stage1_stage3_dag" / "stage3" / "main.py"
STAGE4_MAIN = MRG_TREE_DIR / "03_frontier_tree_search" / "main.py"

DATA_ROOT = WORKSPACE_ROOT / "xiaorong_data" / "Gemma31b"
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "structural_ablations"
DEFAULT_EXPERIMENTS_JSON = THIS_DIR / "structural_ablation_main_table.json"

DEFAULT_DATASETS = ["daily", "hug", "mul", "ultratool"]
SUMMARY_FILE_NAME = "structural_ablation_metrics_main_table.json"
STAGE_ORDER = ["stage1", "stage2", "stage3", "stage4"]
STAGE_LABELS = {
    "stage1": "Stage1",
    "stage2": "Stage2",
    "stage3": "Stage3",
    "stage4": "Stage4",
}
FLAG_DEFAULTS = {
    "stage3_variant": "full",
    "search_variant": "frontier_tree",
}
STAGE_FLAG_KEYS = {
    "stage1": [],
    "stage2": [],
    "stage3": ["stage3_variant"],
    "stage4": ["search_variant"],
}
SUMMARY_LOCK_TIMEOUT_SEC = 600
SUMMARY_LOCK_STALE_SEC = 6 * 60 * 60


# ==================== Manual Run Config ====================
MODEL_NAME = "Gemma31b"
LLM_MODEL = "llama3.1:70b"
LLM_BASE_URL = "http://127.0.0.1:11434/v1/"
LLM_API_KEY = "ollama"
LLM_TEMPERATURE = 0.0

DEVICE = "cuda"
MRG_MODEL_FILENAME = "model_best.pth"
MRG_KG_FILENAME = "service_kg_multi.gpickle"

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
# Default manual-run shortcut: execute only the full setting unless overridden by --only.
ONLY_EXPERIMENTS = ["full"]

TASK_LIMIT = None
ENABLE_RESUME = True
VERBOSE_PROGRESS = True

STAGE1_WORKERS = 4
STAGE2_WORKERS = 4
STAGE3_WORKERS = 4
STAGE4_WORKERS = 4
STAGE1_SAVE_EVERY = 50
STAGE2_SAVE_EVERY = 50
STAGE3_SAVE_EVERY = 50
STAGE4_SAVE_EVERY = 20
STAGE4_TASK_TIMEOUT = 1800
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
}

STAGE_FLAG_KEYS["stage4"] = ["search_variant", *PARAM_TO_ENV.keys()]

DEFAULT_STAGE4_CONFIG = {
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
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _slugify(name: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(name).strip())
    return value.strip("_") or "item"


def _normalize_workers(value: Optional[int], default: int = 1) -> int:
    try:
        workers = int(value or default)
    except Exception:
        workers = default
    return max(1, workers)


def _normalize_timeout(value: Optional[int], default: int = 1800) -> int:
    try:
        timeout = int(value or default)
    except Exception:
        timeout = default
    return max(60, timeout)


def _cleanup_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


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
        "description",
        "model_name",
        "llm_model",
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
    summary_path = Path(SUMMARY_JSON_PATH) if SUMMARY_JSON_PATH else output_root / _slugify(MODEL_NAME) / SUMMARY_FILE_NAME
    return {
        "data_root": data_root,
        "train_root": train_root,
        "test_root": test_root,
        "output_root": output_root,
        "summary_path": summary_path,
    }


def _build_dataset_paths(output_root: Path, model_slug: str, exp_slug: str, dataset: str, create_dir: bool = True) -> Dict[str, Path]:
    dataset_dir = output_root / model_slug / exp_slug / dataset
    if create_dir:
        _ensure_dir(dataset_dir)
    return {
        "dataset_dir": dataset_dir,
        "stage1": dataset_dir / "stage1_tasknum.json",
        "stage2": dataset_dir / "stage15_services.json",
        "stage3": dataset_dir / "stage2_dag.json",
        "stage4": dataset_dir / "stage3_tree_candidates.json",
        "stage1_unfinished": dataset_dir / "unfinished_stage1.json",
        "stage2_unfinished": dataset_dir / "unfinished_stage15.json",
        "stage3_unfinished": dataset_dir / "unfinished_stage22.json",
        "stage4_unfinished": dataset_dir / "unfinished_stage3_tree_candidates.json",
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
        "train_tasks_file": train_dir / "data.json",
        "test_tasks_file": test_dir / "data.json",
        "service_file": test_dir / "tool_desc.json",
        "graph_file": test_dir / "graph_desc.json",
        "kg_file": train_dir / MRG_KG_FILENAME,
        "model_file": train_dir / MRG_MODEL_FILENAME,
        "stage1_embeddings": train_dir / "stage1_history_embeddings.npy",
        "stage2_embeddings": train_dir / "stage2_train_task_embeddings.npy",
    }


def _load_total_tasks(test_root: Path, dataset: str) -> int:
    input_file = test_root / dataset / "data.json"
    try:
        data = load_json(str(input_file))
    except Exception:
        return 0
    if TASK_LIMIT is not None:
        data = data[:TASK_LIMIT]
    return len(data)


def _task_id(task: Dict[str, Any]) -> str:
    return str(task.get("annotation_id", ""))


def _is_valid_stage_output(stage_key: str, task: Dict[str, Any]) -> bool:
    if stage_key == "stage1":
        try:
            return int(task.get("pred_task_num", 0)) > 0
        except Exception:
            return False
    if stage_key == "stage2":
        return isinstance(task.get("st1.5_service"), list) and len(task.get("st1.5_service", [])) > 0
    if stage_key == "stage3":
        return isinstance(task.get("nodes"), list) and len(task.get("nodes", [])) > 0
    if stage_key == "stage4":
        return (
            isinstance(task.get("service_path_list"), list)
            and isinstance(task.get("recom_result"), list)
            and task.get("service_path_list") is not None
            and task.get("recom_result") is not None
        )
    return True


def _diagnose_stage_file(stage_key: str, path: Path, total_tasks: int) -> Dict[str, Any]:
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
        task_id = _task_id(item)
        if task_id in seen_ids:
            invalid_count += 1
            continue
        seen_ids.add(task_id)
        if _is_valid_stage_output(stage_key, item):
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


def _load_valid_stage_tasks(stage_key: str, path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = load_json(str(path))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if _is_valid_stage_output(stage_key, item)]


def _effective_flag_value(flags: Dict[str, Any], key: str) -> Any:
    return flags.get(key, FLAG_DEFAULTS.get(key))


def _reusable_stages_from_full(exp_name: str, full_flags: Dict[str, Any], current_flags: Dict[str, Any]) -> List[str]:
    if exp_name == "full":
        return []

    first_changed_stage = None
    for stage_name in STAGE_ORDER:
        for key in STAGE_FLAG_KEYS.get(stage_name, []):
            if _effective_flag_value(full_flags, key) != _effective_flag_value(current_flags, key):
                first_changed_stage = stage_name
                break
        if first_changed_stage is not None:
            break

    if first_changed_stage is None:
        return list(STAGE_ORDER)

    change_index = STAGE_ORDER.index(first_changed_stage)
    return STAGE_ORDER[:change_index]


def _load_full_config_if_compatible(full_config_path: Path, current_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not full_config_path.exists():
        return None
    try:
        full_config = load_json(str(full_config_path))
    except Exception:
        return None

    keys_to_match = [
        "model_name",
        "llm_model",
        "llm_base_url",
        "train_dir",
        "test_dir",
        "limit",
        "model_file",
        "kg_file",
    ]
    for key in keys_to_match:
        if full_config.get(key) != current_config.get(key):
            return None
    return full_config


def _seed_reusable_outputs_from_full(
    dataset: str,
    exp_name: str,
    current_paths: Dict[str, Path],
    full_paths: Dict[str, Path],
    full_flags: Dict[str, Any],
    current_flags: Dict[str, Any],
    current_config: Dict[str, Any],
    resume: bool,
) -> None:
    if not resume or exp_name == "full":
        return

    full_config = _load_full_config_if_compatible(full_paths["config"], current_config)
    if full_config is None:
        if VERBOSE_PROGRESS:
            print(f"[{dataset}] reuse from full skipped: config missing or incompatible")
        return

    reusable_stages = _reusable_stages_from_full(exp_name, full_flags, current_flags)
    copied = []
    for stage_key in reusable_stages:
        src = full_paths[stage_key]
        dst = current_paths[stage_key]
        if dst.exists() or not src.exists():
            continue
        shutil.copy2(src, dst)
        copied.append(dst.name)

    if copied and VERBOSE_PROGRESS:
        print(f"[{dataset}] reuse from full -> {', '.join(copied)}")


def _base_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["MRG_SC_DEVICE"] = DEVICE
    env["MRG_SC_EMBEDDING_URL"] = EMBEDDING_URL
    env["MRG_SC_EMBEDDING_MODEL"] = EMBEDDING_MODEL
    env["MRG_SC_EMBEDDING_TIMEOUT"] = str(EMBEDDING_TIMEOUT)
    env["MRG_SC_LLM_BASE_URL"] = LLM_BASE_URL
    env["MRG_SC_LLM_MODEL"] = LLM_MODEL
    env["MRG_SC_LLM_API_KEY"] = LLM_API_KEY
    if TASK_LIMIT is not None:
        env["MRG_TREE_ABLATION_TASK_LIMIT"] = str(TASK_LIMIT)
    return env


def _stage1_env(data_paths: Dict[str, Path], output_paths: Dict[str, Path]) -> Dict[str, str]:
    env = _base_env()
    env["MRG_TREE_STAGE1_TRAIN_DIR"] = str(data_paths["train_dir"])
    env["MRG_TREE_STAGE1_TEST_DIR"] = str(data_paths["test_dir"])
    env["MRG_TREE_STAGE1_TRAIN_TASKS_FILE"] = str(data_paths["train_tasks_file"])
    env["MRG_TREE_STAGE1_INPUT_FILE"] = str(data_paths["test_tasks_file"])
    env["MRG_TREE_STAGE1_HISTORY_EMBEDDINGS_PATH"] = str(data_paths["stage1_embeddings"])
    env["MRG_TREE_STAGE1_OUTPUT_FILE"] = str(output_paths["stage1"])
    env["MRG_TREE_STAGE1_UNFINISHED_FILE"] = str(output_paths["stage1_unfinished"])
    env["MRG_TREE_STAGE1_SAVE_INTERVAL"] = str(STAGE1_SAVE_EVERY)
    env["MRG_TREE_STAGE1_MAX_WORKERS"] = str(STAGE1_WORKERS)
    return env


def _stage2_env(data_paths: Dict[str, Path], output_paths: Dict[str, Path]) -> Dict[str, str]:
    env = _base_env()
    env["MRG_TREE_STAGE2_TRAIN_DIR"] = str(data_paths["train_dir"])
    env["MRG_TREE_STAGE2_TEST_DIR"] = str(data_paths["test_dir"])
    env["MRG_TREE_STAGE2_TRAIN_TASKS_FILE"] = str(data_paths["train_tasks_file"])
    env["MRG_TREE_STAGE2_TRAIN_EMBEDDINGS_PATH"] = str(data_paths["stage2_embeddings"])
    env["MRG_TREE_STAGE2_INPUT_FILE"] = str(output_paths["stage1"])
    env["MRG_TREE_STAGE2_SERVICE_FILE"] = str(data_paths["service_file"])
    env["MRG_TREE_STAGE2_OUTPUT_FILE"] = str(output_paths["stage2"])
    env["MRG_TREE_STAGE2_UNFINISHED_FILE"] = str(output_paths["stage2_unfinished"])
    env["MRG_TREE_STAGE2_SAVE_INTERVAL"] = str(STAGE2_SAVE_EVERY)
    env["MRG_TREE_STAGE2_MAX_WORKERS"] = str(STAGE2_WORKERS)
    return env


def _stage3_env(data_paths: Dict[str, Path], output_paths: Dict[str, Path], flags: Dict[str, Any]) -> Dict[str, str]:
    env = _base_env()
    env["MRG_TREE_STAGE3_TOOL_DESC_FILE"] = str(data_paths["service_file"])
    env["MRG_TREE_STAGE3_INPUT_FILE"] = str(output_paths["stage2"])
    env["MRG_TREE_STAGE3_OUTPUT_FILE"] = str(output_paths["stage3"])
    env["MRG_TREE_STAGE3_UNFINISHED_FILE"] = str(output_paths["stage3_unfinished"])
    env["MRG_TREE_STAGE3_SAVE_INTERVAL"] = str(STAGE3_SAVE_EVERY)
    env["MRG_TREE_STAGE3_MAX_WORKERS"] = str(STAGE3_WORKERS)
    env["MRG_TREE_STAGE3_VARIANT"] = str(flags.get("stage3_variant", FLAG_DEFAULTS["stage3_variant"]))
    return env


def _stage4_env(dataset: str, data_paths: Dict[str, Path], output_paths: Dict[str, Path], flags: Dict[str, Any]) -> Dict[str, str]:
    env = _base_env()
    env["MRG_TREE_ABLATION_ENGINE"] = "mrg_tree"
    env["MRG_SC_DATA_ROOT"] = str(data_paths["train_dir"].parents[1])
    env["MRG_SC_DATASET"] = dataset
    env["MRG_TREE_KG_PATH"] = str(data_paths["kg_file"])
    env["MRG_TREE_MODEL_PATH"] = str(data_paths["model_file"])
    env["MRG_TREE_TRAIN_TASKS_FILE"] = str(data_paths["train_tasks_file"])
    env["MRG_TREE_SERVICE_FILE"] = str(data_paths["service_file"])
    env["MRG_TREE_MODULE3_INPUT_FILE"] = str(output_paths["stage3"])
    env["MRG_TREE_MODULE3_OUTPUT_FILE"] = str(output_paths["stage4"])
    env["MRG_TREE_MODULE3_UNFINISHED_FILE"] = str(output_paths["stage4_unfinished"])
    env["MRG_TREE_MODULE3_SAVE_INTERVAL"] = str(STAGE4_SAVE_EVERY)
    env["MRG_TREE_MODULE3_MAX_WORKERS"] = str(STAGE4_WORKERS)
    env["MRG_TREE_MODULE3_TASK_TIMEOUT"] = str(STAGE4_TASK_TIMEOUT)
    env["MRG_TREE_SEARCH_VARIANT"] = str(flags.get("search_variant", FLAG_DEFAULTS["search_variant"]))

    config = deepcopy(DEFAULT_STAGE4_CONFIG)
    config.update({key: value for key, value in flags.items() if key in PARAM_TO_ENV})
    for key, value in config.items():
        env[PARAM_TO_ENV[key]] = str(value)
    return env


def _run_stage_script(script_path: Path, env: Dict[str, str]) -> None:
    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(script_path.parent),
        env=env,
        check=True,
    )


def _run_stage_if_needed(
    *,
    stage_key: str,
    stage_label: str,
    script_path: Path,
    env: Dict[str, str],
    output_path: Path,
    total_tasks: int,
    resume: bool,
    dataset: str,
    allow_partial: bool = False,
) -> List[Dict[str, Any]]:
    stage_info = _diagnose_stage_file(stage_key, output_path, total_tasks)
    if resume and stage_info["complete"]:
        if VERBOSE_PROGRESS:
            print(f"[{dataset}] {stage_label}: resume hit, skip with {stage_info['valid_count']}/{total_tasks}")
        return _load_valid_stage_tasks(stage_key, output_path)

    if VERBOSE_PROGRESS:
        print(f"[{dataset}] {stage_label}: run -> {script_path.name}")
    _run_stage_script(script_path, env)

    stage_info = _diagnose_stage_file(stage_key, output_path, total_tasks)
    valid_tasks = _load_valid_stage_tasks(stage_key, output_path)
    if stage_info["complete"]:
        return valid_tasks

    if allow_partial and valid_tasks:
        if VERBOSE_PROGRESS:
            print(
                f"[{dataset}] {stage_label}: continue with partial output "
                f"valid={stage_info['valid_count']}/{total_tasks}, invalid={stage_info['invalid_count']}"
            )
        return valid_tasks

    if not stage_info["complete"]:
        raise RuntimeError(
            f"{stage_label} output incomplete: valid={stage_info['valid_count']}/{total_tasks}, "
            f"invalid={stage_info['invalid_count']}, path={output_path}"
        )
    return valid_tasks


def _build_dataset_result_entry(
    dataset: str,
    data_paths: Dict[str, Path],
    output_paths: Dict[str, Path],
    exp: Dict[str, Any],
    flags: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "experiment": exp["name"],
        "description": exp.get("description", ""),
        "model_name": MODEL_NAME,
        "llm_model": LLM_MODEL,
        "llm_base_url": LLM_BASE_URL,
        "llm_temperature": LLM_TEMPERATURE,
        "dataset": dataset,
        "flags": flags,
        "train_dir": str(data_paths["train_dir"]),
        "test_dir": str(data_paths["test_dir"]),
        "model_file": str(data_paths["model_file"]),
        "kg_file": str(data_paths["kg_file"]),
        "limit": TASK_LIMIT,
        "output_dir": str(output_paths["dataset_dir"]),
    }


def _run_single_dataset(
    dataset: str,
    exp: Dict[str, Any],
    roots: Dict[str, Path],
    full_flags: Dict[str, Any],
    resume: bool,
    model_slug: str,
) -> Dict[str, Any]:
    flags = deepcopy(exp.get("flags", {}))
    data_paths = _build_data_paths(roots, dataset)
    output_paths = _build_dataset_paths(roots["output_root"], model_slug, _slugify(exp["name"]), dataset)

    if not data_paths["train_dir"].exists():
        return {"error": f"missing train dir: {data_paths['train_dir']}"}
    if not data_paths["test_dir"].exists():
        return {"error": f"missing test dir: {data_paths['test_dir']}"}

    total_tasks = _load_total_tasks(roots["test_root"], dataset)
    config_payload = _build_dataset_result_entry(dataset, data_paths, output_paths, exp, flags)
    save_json(str(output_paths["config"]), config_payload)

    full_paths = _build_dataset_paths(roots["output_root"], model_slug, _slugify("full"), dataset)
    _seed_reusable_outputs_from_full(
        dataset=dataset,
        exp_name=exp["name"],
        current_paths=output_paths,
        full_paths=full_paths,
        full_flags=full_flags,
        current_flags=flags,
        current_config=config_payload,
        resume=resume,
    )

    stage1_tasks = _run_stage_if_needed(
        stage_key="stage1",
        stage_label=STAGE_LABELS["stage1"],
        script_path=STAGE1_MAIN,
        env=_stage1_env(data_paths, output_paths),
        output_path=output_paths["stage1"],
        total_tasks=total_tasks,
        resume=resume,
        dataset=dataset,
        allow_partial=True,
    )
    if not stage1_tasks:
        raise RuntimeError(f"Stage1 produced no valid tasks for dataset={dataset}")

    stage2_tasks = _run_stage_if_needed(
        stage_key="stage2",
        stage_label=STAGE_LABELS["stage2"],
        script_path=STAGE2_MAIN,
        env=_stage2_env(data_paths, output_paths),
        output_path=output_paths["stage2"],
        total_tasks=len(stage1_tasks),
        resume=resume,
        dataset=dataset,
        allow_partial=True,
    )
    if not stage2_tasks:
        raise RuntimeError(f"Stage2 produced no valid tasks for dataset={dataset}")
    del stage1_tasks

    stage3_tasks = _run_stage_if_needed(
        stage_key="stage3",
        stage_label=STAGE_LABELS["stage3"],
        script_path=STAGE3_MAIN,
        env=_stage3_env(data_paths, output_paths, flags),
        output_path=output_paths["stage3"],
        total_tasks=len(stage2_tasks),
        resume=resume,
        dataset=dataset,
        allow_partial=True,
    )
    if not stage3_tasks:
        raise RuntimeError(f"Stage3 produced no valid tasks for dataset={dataset}")
    del stage2_tasks

    predictions = _run_stage_if_needed(
        stage_key="stage4",
        stage_label=STAGE_LABELS["stage4"],
        script_path=STAGE4_MAIN,
        env=_stage4_env(dataset, data_paths, output_paths, flags),
        output_path=output_paths["stage4"],
        total_tasks=len(stage3_tasks),
        resume=resume,
        dataset=dataset,
        allow_partial=True,
    )
    if not predictions:
        raise RuntimeError(f"Stage4 produced no valid predictions for dataset={dataset}")

    metrics = evaluate_predictions(
        predictions,
        str(data_paths["service_file"]),
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
) -> Dict[str, Any]:
    return {
        "name": exp["name"],
        "description": exp.get("description", ""),
        "status": "ok" if all("error" not in metrics for metrics in dataset_metrics.values()) else "partial_error",
        "model_name": MODEL_NAME,
        "llm_model": LLM_MODEL,
        "llm_base_url": LLM_BASE_URL,
        "llm_temperature": LLM_TEMPERATURE,
        "flags": exp.get("flags", {}),
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


def _experiment_artifacts_complete(
    output_root: Path,
    model_slug: str,
    exp_name: str,
    datasets: List[str],
    test_root: Path,
) -> bool:
    exp_slug = _slugify(exp_name)
    for dataset in datasets:
        total_tasks = _load_total_tasks(test_root, dataset)
        paths = _build_dataset_paths(output_root, model_slug, exp_slug, dataset, create_dir=False)
        pred_info = _diagnose_stage_file("stage4", paths["stage4"], total_tasks)
        if not paths["metrics"].exists():
            return False
        if not pred_info["complete"]:
            return False
    return True


def run_experiment(
    exp: Dict[str, Any],
    datasets: List[str],
    roots: Dict[str, Path],
    resume: bool,
    results_path: Path,
    full_flags: Dict[str, Any],
) -> Dict[str, Any]:
    exp_name = exp["name"]
    model_slug = _slugify(MODEL_NAME)

    start_time = time.time()
    dataset_metrics = {}
    dataset_dirs = {}

    for dataset in datasets:
        output_paths = _build_dataset_paths(roots["output_root"], model_slug, _slugify(exp_name), dataset)
        dataset_dirs[dataset] = str(output_paths["dataset_dir"])

        try:
            if VERBOSE_PROGRESS:
                print(
                    f"\n=== Dataset: {dataset} | Experiment: {exp_name} | Model: {MODEL_NAME} | "
                    f"resume={'on' if resume else 'off'} | "
                    f"stage3_variant={exp.get('flags', {}).get('stage3_variant', 'full')} | "
                    f"search_variant={exp.get('flags', {}).get('search_variant', 'frontier_tree')} ==="
                )
            metrics = _run_single_dataset(
                dataset=dataset,
                exp=exp,
                roots=roots,
                full_flags=full_flags,
                resume=resume,
                model_slug=model_slug,
            )
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
    result_entry = _build_result_entry(exp, dataset_metrics, dataset_dirs, duration_sec)
    _upsert_result_with_lock(results_path, result_entry)
    return result_entry


def parse_args():
    parser = argparse.ArgumentParser(description="Run structural ablation experiments for MRG-Tree.")
    parser.add_argument("--model-name", default=MODEL_NAME, help="Experiment model tag used for output grouping.")
    parser.add_argument("--llm-model", default=LLM_MODEL, help="LLM model name used by Stage3 DAG planning.")
    parser.add_argument("--llm-base-url", default=LLM_BASE_URL, help="Base URL of the LLM service.")
    parser.add_argument("--llm-api-key", default=LLM_API_KEY, help="API key passed to the LLM endpoint.")
    parser.add_argument("--llm-temperature", type=float, default=LLM_TEMPERATURE, help="Reserved temperature field.")
    parser.add_argument("--mrg-model-filename", default=MRG_MODEL_FILENAME, help="MRG-SC checkpoint filename in each train/<dataset>/ directory.")
    parser.add_argument("--mrg-kg-filename", default=MRG_KG_FILENAME, help="Knowledge graph filename in each train/<dataset>/ directory.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS_TO_RUN, help="Datasets to run.")
    parser.add_argument("--experiments-json", default=EXPERIMENTS_JSON_PATH, help="Path to structural ablation config JSON.")
    parser.add_argument("--data-root", default=DATA_ROOT_DIR, help="Base data directory.")
    parser.add_argument("--train-root", default=TRAIN_ROOT_DIR, help="Optional explicit train root.")
    parser.add_argument("--test-root", default=TEST_ROOT_DIR, help="Optional explicit test root.")
    parser.add_argument("--output-root", default=OUTPUT_ROOT_DIR, help="Optional explicit output root.")
    parser.add_argument("--summary-path", default=SUMMARY_JSON_PATH, help="Optional explicit summary JSON path.")
    parser.add_argument("--only", nargs="*", default=ONLY_EXPERIMENTS, help="Optional experiment name whitelist.")
    parser.add_argument("--limit", type=int, default=TASK_LIMIT, help="Only run the first N tasks.")
    parser.add_argument("--disable-resume", action="store_true", help="Disable stage-level resume.")
    parser.add_argument("--stage1-workers", type=int, default=STAGE1_WORKERS, help="Workers used in Stage1 task-number prediction.")
    parser.add_argument("--stage2-workers", type=int, default=STAGE2_WORKERS, help="Workers used in Stage2 service recall.")
    parser.add_argument("--stage3-workers", type=int, default=STAGE3_WORKERS, help="Workers used in Stage3 DAG generation.")
    parser.add_argument("--stage4-workers", type=int, default=STAGE4_WORKERS, help="Workers used in Stage4 tree reasoning.")
    parser.add_argument("--stage4-task-timeout", type=int, default=STAGE4_TASK_TIMEOUT, help="Per-task timeout passed to Stage4.")
    return parser.parse_args()


def main():
    global MODEL_NAME
    global LLM_MODEL
    global LLM_BASE_URL
    global LLM_API_KEY
    global LLM_TEMPERATURE
    global MRG_MODEL_FILENAME
    global MRG_KG_FILENAME
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
    global STAGE1_WORKERS
    global STAGE2_WORKERS
    global STAGE3_WORKERS
    global STAGE4_WORKERS
    global STAGE4_TASK_TIMEOUT

    args = parse_args()
    MODEL_NAME = args.model_name
    LLM_MODEL = args.llm_model
    LLM_BASE_URL = args.llm_base_url
    LLM_API_KEY = args.llm_api_key
    LLM_TEMPERATURE = float(args.llm_temperature)
    MRG_MODEL_FILENAME = args.mrg_model_filename
    MRG_KG_FILENAME = args.mrg_kg_filename
    DATASETS_TO_RUN = args.datasets
    EXPERIMENTS_JSON_PATH = args.experiments_json
    DATA_ROOT_DIR = args.data_root
    TRAIN_ROOT_DIR = args.train_root
    TEST_ROOT_DIR = args.test_root
    OUTPUT_ROOT_DIR = args.output_root
    SUMMARY_JSON_PATH = args.summary_path
    ONLY_EXPERIMENTS = args.only
    TASK_LIMIT = args.limit
    ENABLE_RESUME = not args.disable_resume
    STAGE1_WORKERS = _normalize_workers(args.stage1_workers, STAGE1_WORKERS)
    STAGE2_WORKERS = _normalize_workers(args.stage2_workers, STAGE2_WORKERS)
    STAGE3_WORKERS = _normalize_workers(args.stage3_workers, STAGE3_WORKERS)
    STAGE4_WORKERS = _normalize_workers(args.stage4_workers, STAGE4_WORKERS)
    STAGE4_TASK_TIMEOUT = _normalize_timeout(args.stage4_task_timeout, STAGE4_TASK_TIMEOUT)

    roots = _build_roots()
    _ensure_dir(roots["output_root"])

    all_experiments = _load_experiments(Path(EXPERIMENTS_JSON_PATH))
    full_experiment = next((exp for exp in all_experiments if exp.get("name") == "full"), None)
    full_flags = deepcopy(full_experiment.get("flags", {})) if full_experiment else {}
    experiments = _filter_experiments(all_experiments, ONLY_EXPERIMENTS)
    experiments.sort(key=lambda exp: (0 if exp.get("name") == "full" else 1, exp.get("name", "")))
    existing_results = _load_existing_results(roots["summary_path"])

    if VERBOSE_PROGRESS:
        print(f"MODEL_NAME       = {MODEL_NAME}")
        print(f"LLM_MODEL        = {LLM_MODEL}")
        print(f"LLM_BASE_URL     = {LLM_BASE_URL}")
        print(f"TRAIN_ROOT       = {roots['train_root']}")
        print(f"TEST_ROOT        = {roots['test_root']}")
        print(f"OUTPUT_ROOT      = {roots['output_root']}")
        print(f"SUMMARY_PATH     = {roots['summary_path']}")
        print(f"DATASETS         = {DATASETS_TO_RUN}")
        print(f"EXPERIMENTS_JSON = {EXPERIMENTS_JSON_PATH}")

    model_slug = _slugify(MODEL_NAME)
    progress = tqdm(experiments, desc="Structural Ablations", ncols=100)
    for exp in progress:
        exp_name = exp["name"]
        progress.set_postfix(current=exp_name)

        completed = _find_completed_result(existing_results, exp_name, DATASETS_TO_RUN)
        if ENABLE_RESUME and completed and _experiment_artifacts_complete(
            output_root=roots["output_root"],
            model_slug=model_slug,
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
            full_flags=full_flags,
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
                f"ER={macro.get('real_er', 0.0):.4f} | "
                f"SoftOHR={macro.get('soft_ohr', 0.0):.4f}"
            )

    print(f"\nSaved summary to: {roots['summary_path']}")


if __name__ == "__main__":
    main()
