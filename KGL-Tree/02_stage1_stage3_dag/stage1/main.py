import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from file_utils import read_json_file
from embedding_client import EmbeddingClient
from embedding_cache import build_or_load_embeddings
from task_retriever import TaskRetriever


DEFAULT_TRAIN_DATA_DIR = "../../new_Data/train/mul"
DEFAULT_TEST_DATA_DIR = "../../new_Data/test/mul"

TRAIN_DATA_DIR = os.getenv("MRG_TREE_STAGE1_TRAIN_DIR", DEFAULT_TRAIN_DATA_DIR).strip()
TEST_DATA_DIR = os.getenv("MRG_TREE_STAGE1_TEST_DIR", DEFAULT_TEST_DATA_DIR).strip()

TRAIN_TASKS_FILE_PATH = os.getenv(
    "MRG_TREE_STAGE1_TRAIN_TASKS_FILE",
    os.path.join(TRAIN_DATA_DIR, "data.json"),
).strip()
TASKS_FILE_PATH = os.getenv(
    "MRG_TREE_STAGE1_INPUT_FILE",
    os.path.join(TEST_DATA_DIR, "data.json"),
).strip()
HISTORY_EMBEDDINGS_PATH = os.getenv(
    "MRG_TREE_STAGE1_HISTORY_EMBEDDINGS_PATH",
    os.path.join(TRAIN_DATA_DIR, "stage1_history_embeddings.npy"),
).strip()
UNFINISHED_TASKS_PATH = os.getenv(
    "MRG_TREE_STAGE1_UNFINISHED_FILE",
    os.path.join(TEST_DATA_DIR, "unfinished_stage1.json"),
).strip()
PROCESSED_TASKS_PATH = os.getenv(
    "MRG_TREE_STAGE1_OUTPUT_FILE",
    os.path.join(TEST_DATA_DIR, "stage1_tasknum.json"),
).strip()

SAVE_INTERVAL = max(1, int(os.getenv("MRG_TREE_STAGE1_SAVE_INTERVAL", "50").strip()))
MAX_WORKERS = max(1, int(os.getenv("MRG_TREE_STAGE1_MAX_WORKERS", str(os.cpu_count() or 4)).strip()))
MAX_RETRIES = max(1, int(os.getenv("MRG_TREE_STAGE1_MAX_RETRIES", "3").strip()))
TASK_LIMIT_RAW = os.getenv("MRG_TREE_ABLATION_TASK_LIMIT", "").strip()
TASK_LIMIT = int(TASK_LIMIT_RAW) if TASK_LIMIT_RAW else None


retriever = None
embedder = None
processed_tasks = []
processed_ids = set()
task_counter = 0
save_lock = threading.Lock()


def _task_id(task):
    return str(task.get("annotation_id", ""))


def _is_valid_processed_task(task):
    if not isinstance(task, dict):
        return False
    if task.get("annotation_id") is None:
        return False
    try:
        return int(task.get("pred_task_num", 0)) > 0
    except Exception:
        return False


def save_progress():
    processed_tasks.sort(key=lambda item: _task_id(item))
    with open(PROCESSED_TASKS_PATH, "w", encoding="utf-8") as file_obj:
        json.dump(processed_tasks, file_obj, indent=4, ensure_ascii=False)
    print(f"Auto-saved {len(processed_tasks)} tasks")


def _prepare_resume_cache(input_tasks):
    if not os.path.exists(PROCESSED_TASKS_PATH):
        return [], set(), list(input_tasks), 0

    try:
        with open(PROCESSED_TASKS_PATH, "r", encoding="utf-8") as file_obj:
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
    if invalid_count > 0 or not os.path.exists(PROCESSED_TASKS_PATH):
        os.makedirs(os.path.dirname(PROCESSED_TASKS_PATH) or ".", exist_ok=True)
        with open(PROCESSED_TASKS_PATH, "w", encoding="utf-8") as file_obj:
            json.dump(valid_tasks, file_obj, indent=4, ensure_ascii=False)
        print(f"[resume] primed output with {len(valid_tasks)} valid cached tasks")


def _log_task_result(stage_name, annotation_id, attempts, ok, detail):
    status = "success" if ok else "failed"
    print(f"[{stage_name}] annotation_id={annotation_id} attempts={attempts} result={status} detail={detail}")


def process_single_task(task):
    global task_counter

    query = task.get("confirmed_task", "")
    annotation_id = _task_id(task)
    last_error = None
    attempts_used = 0
    for attempt in range(1, MAX_RETRIES + 1):
        attempts_used = attempt
        try:
            try:
                query_embedding = embedder.get_embedding(query)
            except Exception as exc:
                print(f"[Query Embedding Error] {exc}")
                query_embedding = None

            task_num = retriever.predict_task_num(
                query_embedding,
                recom_num=task.get("recom_num", 1),
            )
            task["pred_task_num"] = task_num
            break
        except Exception as exc:
            last_error = exc
            if attempt >= MAX_RETRIES:
                _log_task_result("Stage1", annotation_id, attempts_used, False, f"{type(exc).__name__}: {exc}")
                raise
    else:
        _log_task_result("Stage1", annotation_id, attempts_used, False, f"{type(last_error).__name__}: {last_error}")
        raise last_error

    need_save = False
    with save_lock:
        processed_tasks.append(task)
        processed_ids.add(_task_id(task))
        task_counter += 1

        if task_counter % 50 == 0:
            print(f"Processed {task_counter} tasks")

        if len(processed_tasks) % SAVE_INTERVAL == 0:
            need_save = True

    if need_save:
        save_progress()

    _log_task_result("Stage1", annotation_id, attempts_used, True, f"pred_task_num={task.get('pred_task_num')}")
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
                print(f"fail: {exc}, task: {task.get('confirmed_task', '')}")
                unfinished_tasks.append(task)
    return unfinished_tasks


def main():
    global retriever
    global embedder
    global processed_tasks
    global processed_ids
    global task_counter

    print("=== ENTER STAGE1 ===")
    print(f"Train split  : {TRAIN_DATA_DIR}")
    print(f"Test split   : {TEST_DATA_DIR}")
    print(f"Input file   : {TASKS_FILE_PATH}")
    print(f"Output file  : {PROCESSED_TASKS_PATH}")

    tasks = read_json_file(TASKS_FILE_PATH)
    if TASK_LIMIT is not None:
        tasks = tasks[:TASK_LIMIT]
    train_tasks = read_json_file(TRAIN_TASKS_FILE_PATH)

    print(f"Loaded test tasks  : {len(tasks)}")
    print(f"Loaded train tasks : {len(train_tasks)}")

    embedder = EmbeddingClient()
    embeddings, valid_train_tasks = build_or_load_embeddings(
        train_tasks,
        HISTORY_EMBEDDINGS_PATH,
    )
    retriever = TaskRetriever(valid_train_tasks, embeddings, have_self=True)

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
        retriever.print_stats()
        return

    unfinished_tasks = process_tasks(remaining)
    save_progress()

    if unfinished_tasks:
        with open(UNFINISHED_TASKS_PATH, "w", encoding="utf-8") as file_obj:
            json.dump(unfinished_tasks, file_obj, indent=4, ensure_ascii=False)
        print(f"unfinished saved: {len(unfinished_tasks)}")

    retriever.print_stats()
    print("Stage1 finished")
    print(f"Processed tasks: {len(processed_tasks)}")


if __name__ == "__main__":
    main()
