import os
import json
import math
from pathlib import Path
import sys
import networkx as nx
import requests
from collections import defaultdict
import numpy as np
import pickle
from concurrent.futures import ThreadPoolExecutor


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mrg_sc_runtime import (
    build_data_path,
    get_dataset_name,
    get_embedding_model,
    get_embedding_timeout,
    get_embedding_url,
    get_split_dir,
)

# ==================== 1. 文件路径配置 ====================
DATASET_NAME = get_dataset_name()
BASE_DATA_DIR = str(get_split_dir("train", DATASET_NAME))
TOOL_DESC_PATH = build_data_path("train", "tool_desc.json", DATASET_NAME)
GRAPH_DESC_PATH = build_data_path("train", "graph_desc.json", DATASET_NAME)
TASKS_FILE_PATH = build_data_path("train", "data.json", DATASET_NAME)
KG_OUTPUT_PATH = build_data_path("train", "service_kg_multi.gpickle", DATASET_NAME)

# ==================== 2. Embedding 客户端 ====================
class EmbeddingClient:
    def __init__(self):
        self.url = get_embedding_url()
        self.model = get_embedding_model()
        self.timeout = float(get_embedding_timeout())

    def get_embedding(self, text):
        payload = {"model": self.model, "input": text or ""}
        response = requests.post(self.url, json=payload, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()
        if "data" not in data or not data["data"]:
            raise RuntimeError(f"Unexpected embedding response: {data}")
        return data["data"][0]["embedding"]


# 1) 图类型改成 MultiDiGraph
# KG = nx.DiGraph(name="ServiceKG")
KG = nx.MultiDiGraph(name="ServiceKG")
embed_client = EmbeddingClient()

# 关系优先级：WORKFLOW 最高
REL_PRIORITY = {
    "DEPENDS_ON": 1,
    "ALTERNATIVE_TO": 1,
    "COMPLEMENTS": 2,
    "WORKFLOW": 3,
    "USES": 0,
}

# 2) upsert_edge 改成“同(u,v,type)聚合，不同type并存”
def upsert_edge(u, v, rel_type, weight=1.0, count=1, **extra):
    """
    MultiDiGraph版本：
    - (u,v,rel_type) 已存在：累加 count，weight 取 max，更新 extra
    - 不存在：新增一条 key=rel_type 的边
    这样同一对节点可有多条不同 type 的边，不会覆盖。
    """
    key = rel_type  # 每种关系类型一个独立边槽位

    if KG.has_edge(u, v, key=key):
        old = KG[u][v][key]
        old["count"] = int(old.get("count", 1)) + int(count)
        old["weight"] = float(max(old.get("weight", 1.0), float(weight)))
        for k, v_ in extra.items():
            old[k] = v_
    else:
        KG.add_edge(
            u, v,
            key=key,
            type=rel_type,
            weight=float(weight),
            count=int(count),
            **extra
        )



# ==================== 3. 多线程 embedding ====================
def get_batch_embeddings(texts, max_workers=8):
    def worker(t):
        return embed_client.get_embedding(t)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        embeddings = list(executor.map(worker, texts))

    mat = np.array(embeddings, dtype=np.float32)
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / (norm + 1e-8)
    return mat


# ==================== 4. 加载 Service ====================
def load_service_entities():
    print("📦 加载 Service 实体...")
    with open(TOOL_DESC_PATH, "r", encoding="utf-8") as f:
        tool_desc = json.load(f)

    for tool in tool_desc:
        sid = tool["action_uid"]
        KG.add_node(
            f"Service:{sid}",
            type="Service",
            id=sid,
            name=sid,
            description=tool.get("target_action_reprs", ""),
            input_types=json.dumps(tool.get("input-type", [])),
            output_types=json.dumps(tool.get("output-type", [])),
        )

    print(f"✅ 服务数：{len([n for n, d in KG.nodes(data=True) if d.get('type') == 'Service'])}")


# ==================== 5. 依赖关系 ====================
def load_core_dependencies():
    print("🔗 加载 DEPENDS_ON ...")
    with open(GRAPH_DESC_PATH, "r", encoding="utf-8") as f:
        graph_desc = json.load(f)

    cnt = 0
    for d in graph_desc:
        s = f"Service:{d['source']}"
        t = f"Service:{d['target']}"
        if KG.has_node(s) and KG.has_node(t):
            upsert_edge(
                s, t,
                rel_type="DEPENDS_ON",
                weight=float(d.get("weight", 1.0)),
                count=int(d.get("count", 1)),
                source="graph_desc"
            )
            cnt += 1

    print(f"✅ DEPENDS_ON 写入：{cnt}")


# ==================== 6. Task + USES ====================
def load_task_nodes_and_uses():
    print("📝 加载 Task 节点与 USES ...")
    with open(TASKS_FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    task_cnt, uses_cnt = 0, 0
    for item in data:
        tid = item["annotation_id"]
        t_node = f"Task:{tid}"

        KG.add_node(t_node, type="Task", id=tid)
        task_cnt += 1

        actions = item.get("action_id_list", [])
        for aid in actions:
            s_node = f"Service:{aid}"
            if KG.has_node(s_node):
                # Task->Service，不参与服务间关系优先级竞争
                KG.add_edge(t_node, s_node, type="USES", weight=1.0, count=1)
                uses_cnt += 1

    print(f"✅ Task：{task_cnt} | USES：{uses_cnt}")


# ==================== 7. WORKFLOW（按出现次数聚合） ====================
def generate_workflow_relations():
    print("🔥 生成 WORKFLOW（按频次聚合）...")
    with open(TASKS_FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    wf_count = defaultdict(int)
    for item in data:
        actions = item.get("action_id_list", [])
        for i in range(len(actions) - 1):
            s1 = f"Service:{actions[i]}"
            s2 = f"Service:{actions[i+1]}"
            if KG.has_node(s1) and KG.has_node(s2):
                wf_count[(s1, s2)] += 1

    if not wf_count:
        print("✅ WORKFLOW：0")
        return

    max_cnt = max(wf_count.values())
    edge_num = 0
    for (u, v), c in wf_count.items():
        # 推荐：log 归一化，避免高频边权重过大
        norm_w = math.log1p(c) / math.log1p(max_cnt) if max_cnt > 1 else 1.0
        upsert_edge(
            u, v,
            rel_type="WORKFLOW",
            weight=float(round(norm_w, 6)),
            count=int(c),
            wf_raw_count=int(c),
            wf_max_count=int(max_cnt),
        )
        edge_num += 1

    print(f"✅ WORKFLOW 边数：{edge_num} | max_count={max_cnt}")


# ==================== 8. ALTERNATIVE ====================
def generate_alternative_relations(sim_thresh=0.85):
    print("🔄 生成 ALTERNATIVE_TO ...")

    services = [n for n, d in KG.nodes(data=True) if d["type"] == "Service"]
    descs = [KG.nodes[s]["description"] for s in services]
    embs = get_batch_embeddings(descs)
    sim_mat = embs @ embs.T

    def overlap(a, b):
        return len(set(a) & set(b)) > 0

    def has_io(i, o):
        return len(i) > 0 or len(o) > 0

    cnt = 0
    for i in range(len(services)):
        for j in range(i + 1, len(services)):
            sim = float(sim_mat[i, j])
            if sim < sim_thresh:
                continue

            s_i, s_j = services[i], services[j]
            i1 = json.loads(KG.nodes[s_i]["input_types"])
            i2 = json.loads(KG.nodes[s_j]["input_types"])
            o1 = json.loads(KG.nodes[s_i]["output_types"])
            o2 = json.loads(KG.nodes[s_j]["output_types"])

            has_io_1 = has_io(i1, o1)
            has_io_2 = has_io(i2, o2)

            if has_io_1 and has_io_2:
                if not (overlap(i1, i2) and overlap(o1, o2)):
                    continue
            elif not has_io_1 and not has_io_2:
                pass
            else:
                continue

            upsert_edge(s_i, s_j, "ALTERNATIVE_TO", weight=sim, count=1)
            upsert_edge(s_j, s_i, "ALTERNATIVE_TO", weight=sim, count=1)
            cnt += 1

    print(f"✅ ALTERNATIVE：{cnt}")


# ==================== 9. COMPLEMENTS ====================
def generate_complements_relations(min_co=3):
    print("🔄 生成 COMPLEMENTS ...")

    co = defaultdict(int)
    tasks = [n for n, d in KG.nodes(data=True) if d["type"] == "Task"]

    for t in tasks:
        used = [n for n in KG.neighbors(t) if KG.nodes[n]["type"] == "Service"]
        for i in range(len(used)):
            for j in range(i + 1, len(used)):
                u, v = used[i], used[j]
                key = tuple(sorted([u, v]))
                co[key] += 1

    services = [n for n, d in KG.nodes(data=True) if d["type"] == "Service"]
    descs = [KG.nodes[s]["description"] for s in services]
    embs = get_batch_embeddings(descs)
    sim_mat = embs @ embs.T
    idx = {s: i for i, s in enumerate(services)}

    cnt = 0
    for (u, v), c in co.items():
        if c < min_co:
            continue
        sim = float(sim_mat[idx[u], idx[v]])
        if not (0.4 < sim < 0.8):
            continue

        w = min(c / 100.0, 1.0)
        upsert_edge(u, v, "COMPLEMENTS", weight=w, count=int(c), co_count=int(c))
        upsert_edge(v, u, "COMPLEMENTS", weight=w, count=int(c), co_count=int(c))
        cnt += 1

    print(f"✅ COMPLEMENTS：{cnt}")


# ==================== 10. 保存 ====================
def save_kg():
    print("💾 保存 KG ...")
    with open(KG_OUTPUT_PATH, "wb") as f:
        pickle.dump(KG, f)

    rel_stat = defaultdict(int)
    for _, _, d in KG.edges(data=True):
        rel_stat[d.get("type", "UNKNOWN")] += 1

    print("\n📊 KG统计：")
    print(f"Nodes: {KG.number_of_nodes()}")
    print(f"Edges: {KG.number_of_edges()}")
    print(f"Relation dist: {dict(rel_stat)}")


# ==================== 主流程 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 构建 Service KG 可重复边")
    print("=" * 60)

    load_service_entities()
    load_core_dependencies()
    load_task_nodes_and_uses()          # 先有 Task/USES，供 COMPLEMENTS 统计
    generate_complements_relations()
    generate_alternative_relations()
    generate_workflow_relations()       # 最后写，且按 count 聚合 + 最高优先级

    save_kg()

    print("=" * 60)
    print("🎉 完成")
    print("=" * 60)
