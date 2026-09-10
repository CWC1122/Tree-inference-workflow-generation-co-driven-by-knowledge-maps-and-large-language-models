import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import tempfile

from tqdm import tqdm


MODULE_ROOT = Path(__file__).resolve().parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from searcher_factory import build_searcher


INPUT_FILE = os.getenv("MRG_TREE_MODULE3_INPUT_FILE", "").strip()
OUTPUT_FILE = os.getenv("MRG_TREE_MODULE3_OUTPUT_FILE", "").strip()
UNFINISHED_FILE = os.getenv("MRG_TREE_MODULE3_UNFINISHED_FILE", "").strip()
SAVE_INTERVAL = max(1, int(os.getenv("MRG_TREE_MODULE3_SAVE_INTERVAL", "50").strip()))
MAX_WORKERS = max(1, int(os.getenv("MRG_TREE_MODULE3_MAX_WORKERS", str(min(4, os.cpu_count() or 4))).strip()))
TASK_TIMEOUT = max(60, int(os.getenv("MRG_TREE_MODULE3_TASK_TIMEOUT", "1800").strip()))
TASK_LIMIT_RAW = os.getenv("MRG_TREE_ABLATION_TASK_LIMIT", "").strip()
TASK_LIMIT = int(TASK_LIMIT_RAW) if TASK_LIMIT_RAW else None


searcher = None
processed_tasks = []
processed_ids = set()
save_lock = threading.Lock()
task_counter = 0


def _task_id(task: dict) -> str:
    return str(task.get("annotation_id", ""))


def _is_valid_cached_task(task: dict) -> bool:
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


def build_service_path_list(module3_result: dict) -> list[dict]:
    if "candidate_paths" not in module3_result:
        path_ids = list(module3_result.get("pred_path", []))
        return [
            {
                "path_id": "path_1",
                "path_ids": path_ids,
                "tree_score": float(module3_result.get("score", 0.0)),
                "covered_requirements": [],
                "selected_requirements": [],
            }
        ]

    service_path_list = []
    for candidate in module3_result.get("candidate_paths", []):
        service_path_list.append(
            {
                "path_id": candidate.get("path_id", ""),
                "path_ids": list(candidate.get("path_ids", [])),
                "tree_score": float(candidate.get("tree_score", 0.0)),
                "covered_requirements": list(candidate.get("covered_requirements", [])),
                "selected_requirements": list(candidate.get("selected_requirements", [])),
            }
        )
    return service_path_list


def build_recom_result(module3_result: dict) -> list[dict]:
    service_ids = module3_result.get("best_path_ids")
    if service_ids is None:
        service_ids = module3_result.get("pred_path", [])
    return [
        searcher._tool_payload(service_id)
        for service_id in service_ids
    ]


def append_module3_fields(item: dict, module3_result: dict) -> dict:
    updated = dict(item)
    updated["service_path_list"] = build_service_path_list(module3_result)
    updated["recom_result"] = build_recom_result(module3_result)
    updated["module3_search_mode"] = module3_result.get("search_mode", "")
    updated["module3_fallback_reason"] = module3_result.get("fallback_reason", "")
    updated["module3_search_stats"] = dict(module3_result.get("search_stats", {}))
    return updated


def dump_json(path_str: str, payload) -> None:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".tmp_module3_", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, ensure_ascii=False, indent=4)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def save_progress() -> None:
    processed_tasks.sort(key=lambda item: str(item.get("annotation_id", "")))
    dump_json(OUTPUT_FILE, processed_tasks)
    print(f"Auto-saved {len(processed_tasks)} tasks")


def _prepare_resume_cache(input_tasks: list[dict]):
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
        if _is_valid_cached_task(task):
            valid_tasks.append(task)
        else:
            invalid_count += 1

    valid_tasks.sort(key=lambda item: _task_id(item))
    valid_ids = {_task_id(task) for task in valid_tasks}
    remaining = [task for task in input_tasks if _task_id(task) not in valid_ids]
    return valid_tasks, valid_ids, remaining, invalid_count


def _prime_output_file(valid_tasks: list[dict], invalid_count: int) -> None:
    if invalid_count > 0 or not os.path.exists(OUTPUT_FILE):
        dump_json(OUTPUT_FILE, valid_tasks)
        print(f"[resume] primed output with {len(valid_tasks)} valid cached tasks")


def process_single_task(task: dict) -> dict:
    global task_counter

    current_task_id = task.get("annotation_id")
    if current_task_id is not None:
        current_task_id = str(current_task_id)
    if hasattr(searcher, "search_candidate_paths"):
        module3_result = searcher.search_candidate_paths(
            task_text=task.get("confirmed_task", ""),
            pred_task_num=task.get("pred_task_num", 3),
            current_task_id=current_task_id,
            task_record=task,
        )
    else:
        module3_result = searcher.search_service_path(
            task_text=task.get("confirmed_task", ""),
            pred_task_num=task.get("pred_task_num", 3),
            return_debug=True,
            current_task_id=current_task_id,
            task_record=task,
        )
    result = append_module3_fields(task, module3_result)

    need_save = False
    with save_lock:
        processed_tasks.append(result)
        annotation_id = task.get("annotation_id")
        if annotation_id is not None:
            processed_ids.add(str(annotation_id))
        task_counter += 1

        if task_counter % 10 == 0:
            print(f"Processed {task_counter}")

        if len(processed_tasks) % SAVE_INTERVAL == 0:
            need_save = True

    if need_save:
        save_progress()

    return result


def process_tasks(tasks: list[dict]) -> list[dict]:
    unfinished = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_task, task): task for task in tasks}

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Module3 tree search",
            ncols=100,
            unit="task",
        ):
            task = futures[future]
            try:
                future.result(timeout=TASK_TIMEOUT)
            except Exception as exc:
                failed_item = dict(task)
                failed_item["_error"] = str(exc)
                unfinished.append(failed_item)

    return unfinished


def main() -> None:
    global searcher
    global processed_tasks
    global processed_ids
    global task_counter

    if not INPUT_FILE or not OUTPUT_FILE or not UNFINISHED_FILE:
        raise ValueError("Please set MRG_TREE_MODULE3_INPUT_FILE / OUTPUT_FILE / UNFINISHED_FILE.")

    with open(INPUT_FILE, "r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    if TASK_LIMIT is not None:
        data = data[:TASK_LIMIT]
    total_tasks = len(data)
    print(f"Total tasks: {total_tasks}")

    searcher = build_searcher()
    searcher.precompute_task_embeddings(
        [item.get("confirmed_task", "") for item in data],
        desc="Embedding module3 tasks",
    )

    processed_tasks, processed_ids, remaining, invalid_count = _prepare_resume_cache(data)
    task_counter = len(processed_tasks)
    _prime_output_file(processed_tasks, invalid_count)
    if task_counter > 0:
        print(f"Resume from {task_counter} processed tasks")
    if invalid_count > 0:
        print(f"Discard invalid cached tasks: {invalid_count}")

    print(f"Done      : {len(processed_tasks)}")
    print(f"Remaining : {len(remaining)}")
    print(f"Workers   : {MAX_WORKERS}")

    if not remaining:
        print("All tasks already finished")
        return

    unfinished = process_tasks(remaining)

    save_progress()
    if unfinished:
        dump_json(UNFINISHED_FILE, unfinished)
        print(f"failed: {len(unfinished)} -> {UNFINISHED_FILE}")

    print(f"done: {len(processed_tasks)} -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
