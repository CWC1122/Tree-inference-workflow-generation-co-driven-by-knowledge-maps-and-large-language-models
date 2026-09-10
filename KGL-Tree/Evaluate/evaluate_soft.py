#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026-03-30
# @Author : wenchao
# @Desc : 任务评估指标计算脚本（CR、ER-soft、softOHR、Real_ER、All_ER）
import json
import os
import numpy as np

# ==================== 路径配置 ====================
BASE_DATA_DIR = os.getenv("MRG_TREE_EVAL_BASE_DIR", "../../Data/Gemma31b_bge/test/hug")
INPUT_TEST_TASK = os.path.join(
    BASE_DATA_DIR,
    os.getenv("MRG_TREE_EVAL_INPUT_FILE", "stage3_tree_candidates.json"),
)
INPUT_TEST_API = os.path.join(
    BASE_DATA_DIR,
    os.getenv("MRG_TREE_EVAL_API_FILE", "tool_desc.json"),
)
INPUT_GRAPH = os.path.join(
    BASE_DATA_DIR,
    os.getenv("MRG_TREE_EVAL_GRAPH_FILE", "graph_desc.json"),
)
OUTPUT_RESULT = os.path.join(
    BASE_DATA_DIR,
    os.getenv("MRG_TREE_EVAL_OUTPUT_FILE", "evaluate_result_weight_sorted.json"),
)

print(f"数据基础目录: {os.path.abspath(BASE_DATA_DIR)}")
TEST_NUM = 9999  # 可在此修改测试样本数，例如 10、100、1000


def read_json_file(file_path):
    with open(file_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


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

    precision = recom_true_positives / len(recom_result) if len(recom_result) > 0 else 0
    recall = actual_true_positives / len(actual_apis_reprs) if len(actual_apis_reprs) > 0 else 0

    if precision == 1.0 and recall == 1.0:
        total_true = 1.0

    return precision, recall, total_true


def calculate_highest_metrics(all_recom_results, actual_apis_reprs):
    highest_precision = 0.0
    highest_recall = 0.0

    for api_list in all_recom_results:
        current_precision, current_recall, _ = calculate_task_metrics(api_list, actual_apis_reprs)
        highest_precision = max(highest_precision, current_precision)
        highest_recall = max(highest_recall, current_recall)

    return highest_precision, highest_recall


def calculate_task_granularity_deviation(recom_result, actual_steps):
    if len(actual_steps) == 0:
        return 0
    return abs(len(recom_result) - len(actual_steps)) / len(actual_steps)


def graph_f1_score(pred, gt):
    if len(pred) == 0 or len(gt) == 0:
        return 0.0

    intersect = set(pred) & set(gt)
    precision = len(intersect) / len(pred)
    recall = len(intersect) / len(gt)
    return 2 * precision * recall / (precision + recall + 1e-9)


def batch_graph_f1_score(pred_list, gt_list):
    score_list = [graph_f1_score(pred, gt) for pred, gt in zip(pred_list, gt_list)]
    return round(float(np.mean(np.array(score_list))), 4) if score_list else 0.0


def batch_graph_accuracy(pred_list, gt_list):
    score_list = [float(graph_f1_score(pred, gt) >= 0.99) for pred, gt in zip(pred_list, gt_list)]
    return round(sum(score_list) / len(score_list), 4) if score_list else 0.0


def batch_sequence_accuracy(pred_list, gt_list):
    score_list = [float(pred == gt) for pred, gt in zip(pred_list, gt_list)]
    return round(sum(score_list) / len(score_list), 4) if score_list else 0.0


def calculate_sequence_ohr(recom_result, actual_apis_reprs):
    pred_reprs_in_order = [
        api["target_action_reprs"]
        for api in recom_result
        if api is not None and "target_action_reprs" in api
    ]
    return 1.0 if pred_reprs_in_order == actual_apis_reprs else 0.0


def build_sequence_links(node_ids):
    return [f"{node_ids[i]}, {node_ids[i + 1]}" for i in range(len(node_ids) - 1)]


def calculate_graph_main_metrics(test_task):
    pred_nodes = []
    gt_nodes = []
    pred_links = []
    gt_links = []
    valid_link_pred = []
    valid_link_gt = []

    for task in test_task:
        pred_node_ids = [
            service.get("action_uid")
            for service in task.get("recom_result", [])
            if isinstance(service, dict) and service.get("action_uid")
        ]
        gt_node_ids = [node_id for node_id in task.get("action_id_list", []) if node_id]

        pred_nodes.append(pred_node_ids)
        gt_nodes.append(gt_node_ids)
        current_pred_links = build_sequence_links(pred_node_ids)
        current_gt_links = build_sequence_links(gt_node_ids)

        pred_links.append(current_pred_links)
        gt_links.append(current_gt_links)

        # 单步任务没有可评估的边，因此不纳入 l-F1。
        if len(gt_node_ids) > 1:
            valid_link_pred.append(current_pred_links)
            valid_link_gt.append(current_gt_links)

    return {
        "node_f1": batch_graph_f1_score(pred_nodes, gt_nodes),
        "link_f1": batch_graph_f1_score(valid_link_pred, valid_link_gt),
        "accuracy": batch_graph_accuracy(pred_nodes, gt_nodes),
        "sequence_accuracy": batch_sequence_accuracy(pred_nodes, gt_nodes),
    }


def build_adjacency(graph_data):
    adjacency_list = {}
    for link in graph_data:
        source = link["source"]
        target = link["target"]
        if source not in adjacency_list:
            adjacency_list[source] = set()
        adjacency_list[source].add(target)
    return adjacency_list


def _calculate_er_core(action_reprs, test_api_dict, adjacency_list):
    num_pairs = len(action_reprs) - 1

    if num_pairs <= 0:
        return 1.0, False

    executable_pairs = 0
    for i in range(num_pairs):
        cur = test_api_dict.get(action_reprs[i])
        nxt = test_api_dict.get(action_reprs[i + 1])

        if not cur or not nxt:
            continue

        cur_id = cur.get("action_uid")
        nxt_id = nxt.get("action_uid")

        if cur_id in adjacency_list and nxt_id in adjacency_list[cur_id]:
            executable_pairs += 1

    score = executable_pairs / num_pairs
    return score, True


def calculate_executability_soft(task, test_api_dict, adjacency_list):
    action_reprs = [service["target_action_reprs"] for service in task.get("recom_result", [])]

    n = len(action_reprs)
    if n <= 1:
        return 1.0

    satisfied = 0
    total = n - 1

    for i in range(1, n):
        cur = test_api_dict.get(action_reprs[i])
        if not cur:
            continue

        cur_id = cur.get("action_uid")
        ok = False

        for j in range(i):
            prev = test_api_dict.get(action_reprs[j])
            if not prev:
                continue

            prev_id = prev.get("action_uid")
            if prev_id in adjacency_list and cur_id in adjacency_list[prev_id]:
                ok = True
                break

        if ok:
            satisfied += 1

    return satisfied / total if total > 0 else 0.0


def calculate_soft_ohr(task, test_api_dict, adjacency_list):
    recom_result = task.get("recom_result", [])
    pred_reprs = [service["target_action_reprs"] for service in recom_result]
    gt_reprs = set(task.get("action_reprs", []))

    if not set(pred_reprs).issubset(gt_reprs):
        return 0.0

    er_soft = calculate_executability_soft(task, test_api_dict, adjacency_list)
    return 1.0 if er_soft == 1.0 else 0.0


def calculate_average_metrics(test_task, test_api_dict, graph_data):
    adjacency_list = build_adjacency(graph_data)

    total_precision = 0.0
    total_recall = 0.0
    total_highest_precision = 0.0
    total_highest_recall = 0.0
    total_granularity_deviation = 0.0
    total_loose_granularity_acc = 0.0
    total_executability_soft = 0.0
    total_whole_precision = 0.0
    total_soft_ohr = 0.0
    total_sequence_ohr = 0.0

    total_all_er = 0.0
    total_real_er = 0.0
    num_real_er_tasks = 0
    num_tasks = len(test_task)

    for task in test_task:
        recom_result = task.get("recom_result", [])
        all_recom_results = task.get("all_recom_results", [])
        actual_apis_reprs = task.get("action_reprs", [])
        action_reprs = [service["target_action_reprs"] for service in recom_result]

        precision, recall, whole_precision = calculate_task_metrics(recom_result, actual_apis_reprs)
        total_precision += precision
        total_recall += recall
        total_whole_precision += whole_precision

        highest_precision, highest_recall = calculate_highest_metrics(all_recom_results, actual_apis_reprs)
        total_highest_precision += highest_precision
        total_highest_recall += highest_recall

        granularity_deviation = calculate_task_granularity_deviation(recom_result, actual_apis_reprs)
        total_granularity_deviation += granularity_deviation

        pred_steps = len(recom_result)
        gt_steps = len(actual_apis_reprs)
        if abs(pred_steps - gt_steps) <= 1:
            total_loose_granularity_acc += 1

        er_score, is_real_valid = _calculate_er_core(action_reprs, test_api_dict, adjacency_list)
        total_all_er += er_score

        if is_real_valid:
            total_real_er += er_score
            num_real_er_tasks += 1

        er_soft = calculate_executability_soft(task, test_api_dict, adjacency_list)
        soft_ohr = calculate_soft_ohr(task, test_api_dict, adjacency_list)
        sequence_ohr = calculate_sequence_ohr(recom_result, actual_apis_reprs)
        total_executability_soft += er_soft
        total_soft_ohr += soft_ohr
        total_sequence_ohr += sequence_ohr

    avg_all_er = total_all_er / num_tasks if num_tasks > 0 else 0.0
    avg_real_er = total_real_er / num_real_er_tasks if num_real_er_tasks > 0 else 0.0

    return (
        total_precision / num_tasks if num_tasks > 0 else 0.0,
        total_recall / num_tasks if num_tasks > 0 else 0.0,
        total_highest_precision / num_tasks if num_tasks > 0 else 0.0,
        total_highest_recall / num_tasks if num_tasks > 0 else 0.0,
        total_granularity_deviation / num_tasks if num_tasks > 0 else 0.0,
        total_loose_granularity_acc / num_tasks if num_tasks > 0 else 0.0,
        avg_real_er,
        avg_all_er,
        total_executability_soft / num_tasks if num_tasks > 0 else 0.0,
        total_whole_precision / num_tasks if num_tasks > 0 else 0.0,
        total_soft_ohr / num_tasks if num_tasks > 0 else 0.0,
        total_sequence_ohr / num_tasks if num_tasks > 0 else 0.0,
        num_tasks,
        num_real_er_tasks,
    )


def clean_data(data):
    if isinstance(data, dict):
        return {k: clean_data(v) for k, v in data.items() if k != "vector"}
    if isinstance(data, list):
        return [clean_data(item) for item in data]
    if isinstance(data, np.ndarray):
        return data.tolist()
    return data


def main():
    test_task = read_json_file(INPUT_TEST_TASK)
    test_api = read_json_file(INPUT_TEST_API)
    graph_data = read_json_file(INPUT_GRAPH)

    test_api_dict = {api["target_action_reprs"]: api for api in test_api}
    graph_main_metrics = calculate_graph_main_metrics(test_task)

    (
        avg_p,
        avg_r,
        avg_hp,
        avg_hr,
        avg_gd,
        avg_lga,
        avg_real_er,
        avg_all_er,
        avg_er_soft,
        avg_wp,
        avg_soft_ohr,
        avg_sequence_ohr,
        total_count,
        real_er_count,
    ) = calculate_average_metrics(test_task, test_api_dict, graph_data)

    print(f"Total tasks: {total_count}")
    print(f"Average precision: {avg_p:.4f}")
    print(f"Average recall: {avg_r:.4f}")
    print(f"Average granularity deviation: {avg_gd:.4f}")
    print(f"Loose granularity accuracy (±1 step): {avg_lga:.4f}")
    print("-" * 40)
    print(f"All_ER: {avg_all_er:.4f}")
    print(f"Real_ER (valid multi-step tasks: {real_er_count}): {avg_real_er:.4f}")
    print("-" * 40)
    print(f"ER-soft: {avg_er_soft:.4f}")
    print(f"Strict OHR: {avg_wp:.4f}")
    print(f"shunxu_OHR: {avg_sequence_ohr:.4f}")
    print(f"softOHR: {avg_soft_ohr:.4f}")
    print("-" * 40)
    print(f"n-F1: {graph_main_metrics['node_f1']:.4f}")
    print(f"l-F1: {graph_main_metrics['link_f1']:.4f}")
    print(f"Acc: {graph_main_metrics['accuracy']:.4f}")
    print(f"shunxu_acc: {graph_main_metrics['sequence_accuracy']:.4f}")

    with open(OUTPUT_RESULT, "w", encoding="utf-8") as f:
        json.dump(clean_data(test_task), f, indent=4, ensure_ascii=False)

    print(f"Result saved to {OUTPUT_RESULT}")


if __name__ == "__main__":
    main()
