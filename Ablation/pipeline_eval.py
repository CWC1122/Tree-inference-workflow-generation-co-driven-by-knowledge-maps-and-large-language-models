import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def save_json(path: str, data: Any) -> None:
    parent_dir = os.path.dirname(path) or "."
    os.makedirs(parent_dir, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".tmp_json_", suffix=".json", dir=parent_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file_obj:
            json.dump(data, file_obj, ensure_ascii=False, indent=2)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def build_adjacency(graph_data: List[Dict[str, Any]]) -> Dict[str, set]:
    adjacency = {}
    for link in graph_data:
        src = link["source"]
        dst = link["target"]
        adjacency.setdefault(src, set()).add(dst)
    return adjacency


def build_api_repr_dict(tool_desc: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {api["target_action_reprs"]: api for api in tool_desc}


def calculate_task_metrics(recom_result, actual_apis_reprs):
    recom_true_positives = 0
    actual_true_positives = 0
    total_true = 0

    recom_result_reprs = set()
    for api in recom_result:
        if api is not None and "target_action_reprs" in api:
            recom_result_reprs.add(api["target_action_reprs"])

    for api_repr in recom_result_reprs:
        if api_repr in actual_apis_reprs:
            recom_true_positives += 1

    for api_repr in actual_apis_reprs:
        if api_repr in recom_result_reprs:
            actual_true_positives += 1

    precision = recom_true_positives / len(recom_result) if len(recom_result) > 0 else 0.0
    recall = actual_true_positives / len(actual_apis_reprs) if len(actual_apis_reprs) > 0 else 0.0

    if precision == 1.0 and recall == 1.0:
        total_true = 1.0

    return precision, recall, total_true


def calculate_task_granularity_deviation(recom_result, actual_apis_reprs):
    actual_steps = len(actual_apis_reprs)
    if actual_steps == 0:
        return 0.0
    return abs(len(recom_result) - actual_steps) / actual_steps


def _calculate_er_core(action_reprs, api_by_repr, adjacency_list):
    num_pairs = len(action_reprs) - 1
    if num_pairs <= 0:
        return 1.0, False

    executable_pairs = 0
    for idx in range(num_pairs):
        cur = api_by_repr.get(action_reprs[idx])
        nxt = api_by_repr.get(action_reprs[idx + 1])
        if not cur or not nxt:
            continue

        cur_id = cur.get("action_uid")
        nxt_id = nxt.get("action_uid")
        if cur_id in adjacency_list and nxt_id in adjacency_list[cur_id]:
            executable_pairs += 1

    return executable_pairs / num_pairs, True


def calculate_executability_soft(task, api_by_repr, adjacency_list):
    action_reprs = [step["target_action_reprs"] for step in task.get("recom_result", [])]
    total_steps = len(action_reprs)
    if total_steps <= 1:
        return 1.0

    satisfied = 0
    total = total_steps - 1
    for idx in range(1, total_steps):
        cur = api_by_repr.get(action_reprs[idx])
        if not cur:
            continue

        cur_id = cur.get("action_uid")
        ok = False
        for jdx in range(idx):
            prev = api_by_repr.get(action_reprs[jdx])
            if not prev:
                continue
            prev_id = prev.get("action_uid")
            if prev_id in adjacency_list and cur_id in adjacency_list[prev_id]:
                ok = True
                break

        if ok:
            satisfied += 1

    return satisfied / total if total > 0 else 0.0


def calculate_soft_ohr(task, api_by_repr, adjacency_list):
    recom_result = task.get("recom_result", [])
    pred_reprs = [step["target_action_reprs"] for step in recom_result]
    gt_reprs = set(task.get("action_reprs", []))

    if not set(pred_reprs).issubset(gt_reprs):
        return 0.0

    er_soft = calculate_executability_soft(task, api_by_repr, adjacency_list)
    return 1.0 if er_soft == 1.0 else 0.0


def evaluate_predictions(
    predicted_tasks: List[Dict[str, Any]],
    tool_desc_path: str,
    graph_desc_path: str,
    show_progress: bool = True,
) -> Dict[str, Any]:
    tool_desc = load_json(tool_desc_path)
    graph_data = load_json(graph_desc_path)
    api_by_repr = build_api_repr_dict(tool_desc)
    adjacency = build_adjacency(graph_data)

    total_precision = 0.0
    total_recall = 0.0
    total_whole_precision = 0.0
    total_dgd = 0.0
    total_loose_granularity_acc = 0.0
    total_executability_soft = 0.0
    total_soft_ohr = 0.0
    total_all_er = 0.0
    total_real_er = 0.0
    num_real_er_tasks = 0

    iterator = predicted_tasks
    if show_progress:
        iterator = tqdm(predicted_tasks, desc="Evaluating predictions", ncols=100)

    total_tasks = len(predicted_tasks)
    for task in iterator:
        recom_result = task.get("recom_result", [])
        actual_apis_reprs = task.get("action_reprs", [])
        action_reprs = [step["target_action_reprs"] for step in recom_result]

        precision, recall, whole_precision = calculate_task_metrics(recom_result, actual_apis_reprs)
        total_precision += precision
        total_recall += recall
        total_whole_precision += whole_precision
        total_dgd += calculate_task_granularity_deviation(recom_result, actual_apis_reprs)

        pred_steps = len(recom_result)
        gt_steps = len(actual_apis_reprs)
        if abs(pred_steps - gt_steps) <= 1:
            total_loose_granularity_acc += 1.0

        er_score, is_real_valid = _calculate_er_core(action_reprs, api_by_repr, adjacency)
        total_all_er += er_score
        if is_real_valid:
            total_real_er += er_score
            num_real_er_tasks += 1

        er_soft = calculate_executability_soft(task, api_by_repr, adjacency)
        soft_ohr = calculate_soft_ohr(task, api_by_repr, adjacency)
        total_executability_soft += er_soft
        total_soft_ohr += soft_ohr

    denom = max(1, total_tasks)
    return {
        "avg_precision": total_precision / denom,
        "avg_recall": total_recall / denom,
        "avg_dgd": total_dgd / denom,
        "loose_granularity_acc": total_loose_granularity_acc / denom,
        "strict_ohr": total_whole_precision / denom,
        "er_soft": total_executability_soft / denom,
        "soft_ohr": total_soft_ohr / denom,
        "all_er": total_all_er / denom,
        "real_er": total_real_er / num_real_er_tasks if num_real_er_tasks > 0 else 0.0,
        "total_tasks": total_tasks,
        "real_er_task_count": num_real_er_tasks,
    }
