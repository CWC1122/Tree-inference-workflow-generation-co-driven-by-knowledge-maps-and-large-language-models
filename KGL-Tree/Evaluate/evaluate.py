#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026-03-30
# @Author : wenchao
# @Desc : 任务评估指标计算脚本（路径统一配置版）

# =============================
# 🔥 全局路径配置（所有文件路径在这里修改）
# =============================
INPUT_TEST_TASK = r"../../Data/mul303/topo/stage5.5_top1_no_filter.json"       # 测试任务文件
INPUT_TEST_API = r"../../Experience/Gemma27b/Taskbench/mul/tool_desc.json"  # API服务文件
INPUT_GRAPH = r"../../Experience/Gemma27b/Taskbench/mul/graph_desc.json"     # 依赖图文件
OUTPUT_RESULT = "../../Data/mul303/topo/evaluate_result_no_filter.json"                      # 评估结果输出文件

# =============================
# 依赖库导入
# =============================
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os
import gensim.downloader as api
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from transformers import BertTokenizer


# 读取 JSON 文件
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# 计算单个任务的精确率和召回率
def calculate_task_metrics(recom_result, actual_apis_reprs):
    recom_true_positives = 0
    actual_true_positives = 0
    total_true = 0
    recom_result_reprs = set()

    #去重一下
    # for api in recom_result:
    #     if api is not None and 'target_action_reprs' in api:
    #         recom_result_reprs.add(api['target_action_reprs'])
    #         if api['target_action_reprs'] in actual_apis_reprs:
    #             recom_true_positives += 1
    for api in recom_result:
        if api is not None and 'target_action_reprs' in api:
            recom_result_reprs.add(api['target_action_reprs'])
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
    if total_true == 1.0:
        print("recom_result_reprs",recom_result_reprs)
        print("actual_apis_reprs",actual_apis_reprs)
        print("recom_true_positives: ", recom_true_positives, "actual_true_positives: ", actual_true_positives,
              "len(recom_result)", len(recom_result), "len(actual_apis_reprs)", len(actual_apis_reprs),
              "total_true:",total_true)
    print("recom_true_positives: ", recom_true_positives, "actual_true_positives: ", actual_true_positives,
          "len(recom_result)", len(recom_result), "len(actual_apis_reprs)", len(actual_apis_reprs),
          "total_true:",total_true)

    return precision, recall, total_true


# 计算单个任务的最高精确率和最高召回率
def calculate_highest_metrics(all_recom_results, actual_apis_reprs):
    highest_precision = 0.0
    highest_recall = 0.0

    for api_list in all_recom_results:
        current_precision, current_recall ,total= calculate_task_metrics(api_list, actual_apis_reprs)
        if current_precision > highest_precision:
            highest_precision = current_precision
        if current_recall > highest_recall:
            highest_recall = current_recall

    return highest_precision, highest_recall


# 计算单个任务的分解粒度偏差
def calculate_task_granularity_deviation(recom_result, actual_steps):
    recom_result_count = len(recom_result)
    actual_steps_count = len(actual_steps)
    deviation = abs(recom_result_count - actual_steps_count) / actual_steps_count if actual_steps_count > 0 else 0
    return deviation


# 提取 target_action_reprs 的中心部分
def extract_center_part(target_action_reprs):
    center = re.sub(r'\[.*?\]\s*', '', target_action_reprs)  # 去掉标签
    center = re.sub(r'\s*->.*$', '', center)  # 去掉操作
    return center.strip()


# 计算单个任务的可执行性
def calculate_executability(task, test_api_dict, graph_data):
    """
    计算任务的可执行性，基于 graph_desc.json 中的边信息判断两个服务是否可执行。

    :param task: 任务数据，包含推荐的API序列
    :param test_api_dict: API字典，用于查找API的详细信息
    :param graph_data: 从 graph_desc.json 文件中读取的图数据
    :return: 可执行性得分
    """
    action_reprs = [s['target_action_reprs'] for s in task.get('recom_result', [])]
    num_pairs = len(action_reprs) - 1
    if num_pairs <= 0:
        return 0.0

    # 构建图的邻接表
    adjacency_list = {}
    for link in graph_data:
        source = link['source']
        target = link['target']
        if source not in adjacency_list:
            adjacency_list[source] = set()
        adjacency_list[source].add(target)

    executable_pairs = 0
    for i in range(num_pairs):
        current_repr = action_reprs[i]
        next_repr = action_reprs[i + 1]

        # 获取当前API和下一个API的ID
        current_api = test_api_dict.get(current_repr)
        next_api = test_api_dict.get(next_repr)

        if not current_api or not next_api:
            continue  # 如果API不存在，视为不可执行
        # # 检查 current_repr 和 next_repr 是否在 task 的 action_reprs 中连续存在
        # task_action_reprs = task.get('action_reprs', [])
        # for j in range(len(task_action_reprs) - 1):
        #     if task_action_reprs[j] == current_repr and task_action_reprs[j + 1] == next_repr:
        #         executable_pairs += 1
        #         break
        # else:
        # 如果不在 action_reprs 中连续存在，则检查 next_api 的 cleaned_html 是否包含 current_repr 的中心部分
        # 从API描述中提取ID（假设API的ID存储在某个字段中，例如 'id'）
        current_api_id = current_api.get('action_uid')
        next_api_id = next_api.get('action_uid')

        if not current_api_id or not next_api_id:
            continue  # 如果API ID不存在，视为不可执行

        # 检查图中是否存在从 current_api_id 到 next_api_id 的边
        if current_api_id in adjacency_list and next_api_id in adjacency_list[current_api_id]:
            executable_pairs += 1

    return executable_pairs / num_pairs if num_pairs > 0 else 0.0

# 计算所有任务的平均精确率、平均召回率、平均最高精确率、平均最高召回率、平均分解粒度偏差和平均可执行性
def calculate_average_metrics(test_task, test_api_dict, graph_data):
    total_precision = 0
    total_recall = 0
    total_highest_precision = 0
    total_highest_recall = 0
    total_granularity_deviation = 0
    total_executability = 0  # 新增：总可执行性
    total_whole_precision = 0
    num_tasks = len(test_task)

    for task in test_task:
        recom_result = task.get('recom_result', [])
        all_recom_results = task.get('all_recom_results', [])
        actual_apis_reprs = task.get('action_reprs', [])

        # 普通精确率和召回率
        precision, recall, whole_precision = calculate_task_metrics(recom_result, actual_apis_reprs)
        total_precision += precision
        total_recall += recall
        total_whole_precision += whole_precision
        if whole_precision == 1.0:
            print("咋这么好呢", task['confirmed_task'])

        # 最高精确率和最高召回率
        highest_precision, highest_recall = calculate_highest_metrics(all_recom_results, actual_apis_reprs)
        total_highest_precision += highest_precision
        total_highest_recall += highest_recall

        # 计算分解粒度偏差
        granularity_deviation = calculate_task_granularity_deviation(recom_result, task.get('action_reprs', []))
        total_granularity_deviation += granularity_deviation

        # 计算可执行性
        executability = calculate_executability(task, test_api_dict, graph_data)
        total_executability += executability

    average_precision = total_precision / num_tasks if num_tasks > 0 else 0
    average_recall = total_recall / num_tasks if num_tasks > 0 else 0
    average_highest_precision = total_highest_precision / num_tasks if num_tasks > 0 else 0
    average_highest_recall = total_highest_recall / num_tasks if num_tasks > 0 else 0
    average_granularity_deviation = total_granularity_deviation / num_tasks if num_tasks > 0 else 0
    average_executability = total_executability / num_tasks if num_tasks > 0 else 0
    average_total_whole_precision = total_whole_precision / num_tasks if num_tasks > 0 else 0

    print("一共完成了多少个任务", num_tasks)
    return (average_precision, average_recall, average_highest_precision,
            average_highest_recall, average_granularity_deviation, average_executability, average_total_whole_precision)

# # 处理单个任务
# def process_single_task(task, test_api, graph_data):
#     print("正在处理 test_task:", task.get('confirmed_task', '无确认任务'))
#     start_time = time.time()
#     recom_result, all_recom_results = process_task(task, test_api,graph_data)
#     elapsed_time = time.time() - start_time
#     print(f"Task {task.get('confirmed_task', '无确认任务')} processed in {elapsed_time:.2f} seconds")
#     task['recom_result'] = recom_result
#     # task['all_recom_results'] = all_recom_results
#     task['all_recom_results'] = []
#     return task

#清理数据
def clean_data(data):
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            if key == "vector":
                continue  # 跳过这个键
            if isinstance(value, np.ndarray):
                cleaned[key] = value.tolist()
            elif isinstance(value, (dict, list)):
                cleaned[key] = clean_data(value)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                cleaned[key] = value
            # 其他类型的值将被忽略
        return cleaned
    elif isinstance(data, list):
        cleaned = []
        for item in data:
            if isinstance(item, np.ndarray):
                cleaned.append(item.tolist())
            elif isinstance(item, (dict, list)):
                cleaned.append(clean_data(item))
            elif isinstance(item, (str, int, float, bool)) or item is None:
                cleaned.append(item)
            # 其他类型的值将被忽略
        return cleaned
    else:
        return data


# 主函数
def main():
    # 读取数据（使用顶部配置的路径）
    test_task = read_json_file(INPUT_TEST_TASK)
    test_api = read_json_file(INPUT_TEST_API)
    graph_data = read_json_file(INPUT_GRAPH)

    # 构建API字典
    test_api_dict = {api['target_action_reprs']: api for api in test_api}

    completed_tasks = test_task

    # 计算平均精确率、平均召回率、平均最高精确率、平均最高召回率、平均分解粒度偏差和平均可执行性
    (average_precision, average_recall, average_highest_precision,
     average_highest_recall, average_granularity_deviation,
     average_executability, average_total_whole_precision) = calculate_average_metrics(completed_tasks, test_api_dict, graph_data)
    print(f"平均精确率: {average_precision:.4f}")
    print(f"平均召回率: {average_recall:.4f}")
    print(f"平均分解粒度偏差: {average_granularity_deviation:.4f}")
    print(f"平均可执行性: {average_executability:.4f}")
    print(f"平均整体成功率: {average_total_whole_precision:.4f}")

    # 将结果保存到当前文件夹下的文件中
    completed_tasks_cleaned = clean_data(completed_tasks)
    with open(OUTPUT_RESULT, 'w', encoding='utf-8') as f:
        json.dump(completed_tasks_cleaned, f, indent=4, ensure_ascii=False)
    print(f"结果已保存至 {OUTPUT_RESULT}")


if __name__ == "__main__":
    main()