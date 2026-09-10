import importlib
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from file_utils import read_json_file
from dag_generator import DAGGenerator as DefaultDAGGenerator


MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parents[1]
STRUCTURAL_VARIANT_DIR = PROJECT_ROOT / "Structural_Ablation" / "stage3_variants"

DEFAULT_BASE_DATA_DIR = "../../Data_no_self/Llama3.1:70b-bge/hug"
BASE_DATA_DIR = os.getenv("MRG_TREE_STAGE3_BASE_DATA_DIR", DEFAULT_BASE_DATA_DIR).strip()

TOOL_DESC_PATH = os.getenv(
    "MRG_TREE_STAGE3_TOOL_DESC_FILE",
    os.path.join(BASE_DATA_DIR, "tool_desc.json"),
).strip()
INPUT_FILE = os.getenv(
    "MRG_TREE_STAGE3_INPUT_FILE",
    os.path.join(BASE_DATA_DIR, "stage15_services.json"),
).strip()
OUTPUT_FILE = os.getenv(
    "MRG_TREE_STAGE3_OUTPUT_FILE",
    os.path.join(BASE_DATA_DIR, "stage2_dag.json"),
).strip()
UNFINISHED_FILE = os.getenv(
    "MRG_TREE_STAGE3_UNFINISHED_FILE",
    os.path.join(BASE_DATA_DIR, "unfinished_stage22.json"),
).strip()

SAVE_INTERVAL = max(1, int(os.getenv("MRG_TREE_STAGE3_SAVE_INTERVAL", "50").strip()))
MAX_WORKERS = max(1, int(os.getenv("MRG_TREE_STAGE3_MAX_WORKERS", str(min(6, os.cpu_count() or 4))).strip()))
MAX_RETRIES = max(1, int(os.getenv("MRG_TREE_STAGE3_MAX_RETRIES", "3").strip()))
TASK_LIMIT_RAW = os.getenv("MRG_TREE_ABLATION_TASK_LIMIT", "").strip()
TASK_LIMIT = int(TASK_LIMIT_RAW) if TASK_LIMIT_RAW else None
STAGE3_VARIANT = os.getenv("MRG_TREE_STAGE3_VARIANT", "full").strip().lower() or "full"


generator = None
processed_tasks = []
processed_ids = set()
save_lock = threading.Lock()
task_counter = 0


def _task_id(task):
    return str(task.get("annotation_id", ""))


def _is_valid_processed_task(task):
    return isinstance(task.get("nodes"), list) and len(task.get("nodes", [])) > 0


def save_progress():
    processed_tasks.sort(key=lambda item: _task_id(item))
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file_obj:
        json.dump(processed_tasks, file_obj, indent=4, ensure_ascii=False)
    print(f"Auto-saved {len(processed_tasks)} tasks")


def _prepare_resume_cache(input_tasks):
    if not os.path.exists(OUTPUT_FILE):
        return [], set(), list(input_tasks), 0

    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as file_obj:
            existing = json.load(file_obj)
    except Exception as exc:
        print(f"[resume] failed to read cache, rerun all: {exc}")
        return [], set(), list(input_tasks), 0

    if not isinstance(existing, list):
        print("[resume] cached output is not a list, rerun all")
        return [], set(), list(input_tasks), 0

    input_ids = {_task_id(task) for task in input_tasks}
    valid_tasks = []
    invalid_count = 0
    for task in existing:
        if _task_id(task) not in input_ids:
            continue
        if _is_valid_processed_task(task):
            valid_tasks.append(task)
        else:
            invalid_count += 1

    valid_tasks.sort(key=lambda item: _task_id(item))
    valid_ids = {_task_id(task) for task in valid_tasks}
    remaining = [task for task in input_tasks if _task_id(task) not in valid_ids]
    return valid_tasks, valid_ids, remaining, invalid_count


def _prime_output_file(valid_tasks, invalid_count):
    if invalid_count > 0 or not os.path.exists(OUTPUT_FILE):
        os.makedirs(os.path.dirname(OUTPUT_FILE) or ".", exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as file_obj:
            json.dump(valid_tasks, file_obj, indent=4, ensure_ascii=False)
        print(f"[resume] primed output with {len(valid_tasks)} valid cached tasks")


def _variant_class():
    if STAGE3_VARIANT == "full":
        return DefaultDAGGenerator

    module_map = {
        "dag_no_service": "dag_no_service",
        "dag_no_length": "dag_no_length",
        "dag_hard_length": "dag_hard_length",
        "linear_planner": "linear_planner",
    }
    module_name = module_map.get(STAGE3_VARIANT)
    if not module_name:
        raise ValueError(f"Unsupported stage3 variant: {STAGE3_VARIANT}")

    variant_dir_str = str(STRUCTURAL_VARIANT_DIR)
    if variant_dir_str not in sys.path:
        sys.path.insert(0, variant_dir_str)
    module = importlib.import_module(module_name)
    module = importlib.reload(module)
    return module.DAGGenerator


def _log_task_result(stage_name, annotation_id, attempts, ok, detail):
    status = "success" if ok else "failed"
    print(f"[{stage_name}] annotation_id={annotation_id} attempts={attempts} result={status} detail={detail}")


def process_single_task(task):
    global task_counter

    query = task.get("confirmed_task", "")
    task_num = task.get("pred_task_num", 3)
    service_ids = task.get("st1.5_service", [])
    annotation_id = _task_id(task)
    dag = None
    attempts_used = 0
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        attempts_used = attempt
        try:
            dag = generator.generate_dag(query, task_num, service_ids)
            if dag is None:
                raise ValueError("Invalid workflow sequence")
            task["nodes"] = dag.get("nodes", [])
            task["edges"] = dag.get("edges", [])
            break
        except Exception as exc:
            last_error = exc
            if attempt >= MAX_RETRIES:
                _log_task_result("Stage3", annotation_id, attempts_used, False, f"{type(exc).__name__}: {exc}")
                raise
    else:
        _log_task_result("Stage3", annotation_id, attempts_used, False, f"{type(last_error).__name__}: {last_error}")
        raise last_error

    need_save = False
    with save_lock:
        processed_tasks.append(task)
        processed_ids.add(_task_id(task))
        task_counter += 1

        if task_counter % 5 == 0:
            print(f"Processed {task_counter} workflows")

        if len(processed_tasks) % SAVE_INTERVAL == 0:
            need_save = True

    if need_save:
        save_progress()

    _log_task_result("Stage3", annotation_id, attempts_used, True, f"node_count={len(task.get('nodes', []))}")
    return task


def process_tasks(tasks):
    unfinished_tasks = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_task, task): task for task in tasks}

        for future in as_completed(futures):
            task = futures[future]
            try:
                future.result(timeout=600)
            except Exception as exc:
                print(f"fail: {exc}")
                print(f"task: {task.get('confirmed_task', '')}")
                unfinished_tasks.append(task)

    return unfinished_tasks


def main():
    global generator
    global processed_tasks
    global processed_ids
    global task_counter

    print("=== ENTER STAGE3 ===")
    print(f"Variant      : {STAGE3_VARIANT}")
    print(f"Input file   : {INPUT_FILE}")
    print(f"Output file  : {OUTPUT_FILE}")
    print(f"Tool desc    : {TOOL_DESC_PATH}")

    tasks = read_json_file(INPUT_FILE)
    if TASK_LIMIT is not None:
        tasks = tasks[:TASK_LIMIT]

    generator_cls = _variant_class()
    generator = generator_cls(TOOL_DESC_PATH)

    processed_tasks, processed_ids, remaining_tasks, invalid_count = _prepare_resume_cache(tasks)
    task_counter = len(processed_tasks)
    _prime_output_file(processed_tasks, invalid_count)

    if task_counter > 0:
        print(f"Resume from {task_counter} processed tasks")
    if invalid_count > 0:
        print(f"Discard invalid cached tasks: {invalid_count}")

    print(f"Total tasks : {len(tasks)}")
    print(f"Done        : {len(processed_tasks)}")
    print(f"Remaining   : {len(remaining_tasks)}")
    print(f"Workers     : {MAX_WORKERS}")

    if not remaining_tasks:
        print("All tasks already finished")
        return

    unfinished_tasks = process_tasks(remaining_tasks)
    save_progress()

    if unfinished_tasks:
        with open(UNFINISHED_FILE, "w", encoding="utf-8") as file_obj:
            json.dump(unfinished_tasks, file_obj, indent=4, ensure_ascii=False)
        print(f"Unfinished tasks saved: {len(unfinished_tasks)}")

    print("Stage3 finished")
    print(f"Total workflows: {len(processed_tasks)}")


if __name__ == "__main__":
    main()
