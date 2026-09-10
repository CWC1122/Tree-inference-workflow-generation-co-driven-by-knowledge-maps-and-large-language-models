import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from embedding_cache import build_or_load_embeddings
from embedding_client import EmbeddingClient
from file_utils import read_json_file
from train_service_retriever import TrainTaskServiceRetriever


DEFAULT_TRAIN_DATA_DIR = "../../new_Data/train/mul"
DEFAULT_TEST_DATA_DIR = "../../new_Data/test/mul"

TRAIN_DATA_DIR = os.getenv("MRG_TREE_STAGE2_TRAIN_DIR", DEFAULT_TRAIN_DATA_DIR).strip()
TEST_DATA_DIR = os.getenv("MRG_TREE_STAGE2_TEST_DIR", DEFAULT_TEST_DATA_DIR).strip()

TRAIN_TASKS_FILE = os.getenv(
    "MRG_TREE_STAGE2_TRAIN_TASKS_FILE",
    os.path.join(TRAIN_DATA_DIR, "data.json"),
).strip()
TRAIN_EMBEDDINGS_PATH = os.getenv(
    "MRG_TREE_STAGE2_TRAIN_EMBEDDINGS_PATH",
    os.path.join(TRAIN_DATA_DIR, "stage2_train_task_embeddings.npy"),
).strip()
INPUT_FILE = os.getenv(
    "MRG_TREE_STAGE2_INPUT_FILE",
    os.path.join(TEST_DATA_DIR, "stage1_tasknum.json"),
).strip()
SERVICE_FILE = os.getenv(
    "MRG_TREE_STAGE2_SERVICE_FILE",
    os.path.join(TEST_DATA_DIR, "tool_desc.json"),
).strip()
OUTPUT_FILE = os.getenv(
    "MRG_TREE_STAGE2_OUTPUT_FILE",
    os.path.join(TEST_DATA_DIR, "stage15_services.json"),
).strip()
UNFINISHED_FILE = os.getenv(
    "MRG_TREE_STAGE2_UNFINISHED_FILE",
    os.path.join(TEST_DATA_DIR, "unfinished_stage15.json"),
).strip()

SAVE_INTERVAL = max(1, int(os.getenv("MRG_TREE_STAGE2_SAVE_INTERVAL", "50").strip()))
MAX_WORKERS = max(1, int(os.getenv("MRG_TREE_STAGE2_MAX_WORKERS", str(min(6, os.cpu_count() or 4))).strip()))
MAX_RETRIES = max(1, int(os.getenv("MRG_TREE_STAGE2_MAX_RETRIES", "3").strip()))
TASK_LIMIT_RAW = os.getenv("MRG_TREE_ABLATION_TASK_LIMIT", "").strip()
TASK_LIMIT = int(TASK_LIMIT_RAW) if TASK_LIMIT_RAW else None


recall_engine = None
embedder = None
processed_tasks = []
processed_ids = set()
save_lock = threading.Lock()
task_counter = 0


def _task_id(task):
    return str(task.get("annotation_id", ""))


def _is_valid_processed_task(task):
    values = task.get("st1.5_service")
    return isinstance(values, list) and len(values) > 0


def compute_k(task_num):
    return min(max(12, int(task_num) * 4), 24)


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


def _log_task_result(stage_name, annotation_id, attempts, ok, detail):
    status = "success" if ok else "failed"
    print(f"[{stage_name}] annotation_id={annotation_id} attempts={attempts} result={status} detail={detail}")


def process_single_task(task):
    global task_counter

    query = task.get("confirmed_task", "")
    annotation_id = _task_id(task)
    attempts_used = 0
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        attempts_used = attempt
        try:
            task_num = task.get("pred_task_num", 3)
            candidate_size = compute_k(task_num)

            try:
                query_embedding = embedder.get_embedding(query)
            except Exception as exc:
                print(f"[Query Embedding Error] {exc}")
                query_embedding = None

            top_services = recall_engine.recall(query_embedding, candidate_size)
            task["st1.5_service"] = [service["action_uid"] for service in top_services]
            break
        except Exception as exc:
            last_error = RuntimeError(f"Stage2 error: {exc}")
            if attempt >= MAX_RETRIES:
                _log_task_result("Stage2", annotation_id, attempts_used, False, f"{type(last_error).__name__}: {last_error}")
                raise last_error
    else:
        _log_task_result("Stage2", annotation_id, attempts_used, False, f"{type(last_error).__name__}: {last_error}")
        raise last_error

    need_save = False
    with save_lock:
        processed_tasks.append(task)
        processed_ids.add(_task_id(task))
        task_counter += 1

        if task_counter % 10 == 0:
            print(f"Processed {task_counter}")

        if len(processed_tasks) % SAVE_INTERVAL == 0:
            need_save = True

    if need_save:
        save_progress()
    _log_task_result("Stage2", annotation_id, attempts_used, True, f"service_count={len(task.get('st1.5_service', []))}")
    return task


def process_tasks(tasks):
    unfinished = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_task, task): task for task in tasks}

        for future in as_completed(futures):
            task = futures[future]
            try:
                future.result(timeout=300)
            except Exception as exc:
                print(f"fail: {exc}")
                print(f"task: {task.get('confirmed_task', '')}")
                unfinished.append(task)

    return unfinished


def main():
    global recall_engine
    global embedder
    global processed_tasks
    global processed_ids
    global task_counter

    print("=== ENTER STAGE2 ===")
    print(f"Train split  : {TRAIN_DATA_DIR}")
    print(f"Test split   : {TEST_DATA_DIR}")
    print(f"Input file   : {INPUT_FILE}")
    print(f"Output file  : {OUTPUT_FILE}")

    tasks = read_json_file(INPUT_FILE)
    if TASK_LIMIT is not None:
        tasks = tasks[:TASK_LIMIT]
    train_tasks = read_json_file(TRAIN_TASKS_FILE)
    services = read_json_file(SERVICE_FILE)

    print(f"Loaded test tasks  : {len(tasks)}")
    print(f"Loaded train tasks : {len(train_tasks)}")
    print(f"Loaded services    : {len(services)}")

    embedder = EmbeddingClient()
    embeddings, valid_train_tasks = build_or_load_embeddings(
        train_tasks,
        TRAIN_EMBEDDINGS_PATH,
    )

    valid_service_ids = [service["action_uid"] for service in services]
    recall_engine = TrainTaskServiceRetriever(
        valid_train_tasks,
        embeddings,
        valid_service_ids,
    )

    processed_tasks, processed_ids, remaining, invalid_count = _prepare_resume_cache(tasks)
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
        recall_engine.print_stats()
        return

    unfinished = process_tasks(remaining)
    save_progress()

    if unfinished:
        with open(UNFINISHED_FILE, "w", encoding="utf-8") as file_obj:
            json.dump(unfinished, file_obj, indent=4, ensure_ascii=False)
        print(f"Unfinished tasks saved: {len(unfinished)}")

    recall_engine.print_stats()
    print("Stage2 finished")
    print(f"Processed tasks: {len(processed_tasks)}")


if __name__ == "__main__":
    main()
