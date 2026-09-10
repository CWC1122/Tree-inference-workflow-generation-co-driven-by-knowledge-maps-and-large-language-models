import os
import json
import math
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm


@dataclass
class Sample:
    task_id: str
    task_text: str
    path_service_ids: List[str]
    prev_service_node: Optional[str]
    target_service_node: str
    seq_len: int
    step_idx: int


@dataclass
class TaskRecord:
    task_id: str
    task_text: str
    action_id_list: List[str]
    recom_num: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_json_load(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return []
    return []


def compose_service_text(service_node: str, attrs: Dict) -> str:
    sid = attrs.get("id", service_node.replace("Service:", ""))
    desc = attrs.get("description", "") or ""
    ins = safe_json_load(attrs.get("input_types", attrs.get("input-type", [])))
    outs = safe_json_load(attrs.get("output_types", attrs.get("output-type", [])))

    parts = [f"Service: {sid}", f"Description: {desc}"]
    if ins:
        parts.append(f"Input: {', '.join(ins)}")
    if outs:
        parts.append(f"Output: {', '.join(outs)}")
    return "\n".join(parts)


def wf_repeat_from_count(count: int, cap: int) -> int:
    count = max(1, int(count))
    rep = 1 + int(math.log2(1 + count))
    return min(cap, rep)


def load_graph(
    kg_path: str,
    rel_types: Sequence[str],
    dep_repeat: int,
    wf_dep_bonus: int,
    workflow_repeat_cap: int,
    workflow_repeat_cap_no_dep: int,
    device: torch.device,
):
    with open(kg_path, "rb") as f:
        kg = pickle.load(f)

    service_nodes = [n for n, d in kg.nodes(data=True) if d.get("type") == "Service"]
    idx_map = {n: i for i, n in enumerate(service_nodes)}

    rel2id: Dict[str, int] = {}
    for rel in rel_types:
        rel2id[rel] = len(rel2id)
        if rel not in {"WORKFLOW", "DEPENDS_ON"}:
            rel2id[f"{rel}_REV"] = len(rel2id)
    relation_vocab = [key for key, _ in sorted(rel2id.items(), key=lambda item: item[1])]

    pair: Dict[Tuple[str, str], Dict[str, object]] = {}
    uses_by_task: Dict[str, List[str]] = defaultdict(list)

    def consume_service_edge(u: str, v: str, rel: str, edge_data: Dict) -> None:
        if u not in idx_map or v not in idx_map or rel not in rel_types:
            return

        rec = pair.setdefault((u, v), {"dep": False, "wf_count": 0, "comp": False, "alt": False})
        if rel == "DEPENDS_ON":
            rec["dep"] = True
        elif rel == "WORKFLOW":
            rec["wf_count"] = int(rec["wf_count"]) + int(edge_data.get("count", edge_data.get("wf_raw_count", 1)))
        elif rel == "COMPLEMENTS":
            rec["comp"] = True
        elif rel == "ALTERNATIVE_TO":
            rec["alt"] = True

    is_multi = getattr(kg, "is_multigraph", lambda: False)()
    if is_multi:
        iterator = kg.edges(keys=True, data=True)
        for u, v, _, edge_data in iterator:
            rel = edge_data.get("type")
            if rel == "USES" and u.startswith("Task:") and v.startswith("Service:"):
                uses_by_task[u.replace("Task:", "")].append(v.replace("Service:", ""))
            consume_service_edge(u, v, rel, edge_data)
    else:
        iterator = kg.edges(data=True)
        for u, v, edge_data in iterator:
            rel = edge_data.get("type")
            if rel == "USES" and u.startswith("Task:") and v.startswith("Service:"):
                uses_by_task[u.replace("Task:", "")].append(v.replace("Service:", ""))
            consume_service_edge(u, v, rel, edge_data)

    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_type: List[int] = []
    dep_neighbors = {s: set() for s in service_nodes}
    fallback_neighbors = {s: set() for s in service_nodes}

    def add_directed(u: str, v: str, rel: str) -> None:
        edge_src.append(idx_map[u])
        edge_dst.append(idx_map[v])
        edge_type.append(rel2id[rel])

    for (u, v), rel_info in pair.items():
        has_dep = bool(rel_info["dep"])
        wf_count = int(rel_info["wf_count"])
        has_comp = bool(rel_info["comp"])
        has_alt = bool(rel_info["alt"])

        if has_dep:
            for _ in range(dep_repeat):
                add_directed(u, v, "DEPENDS_ON")
            dep_neighbors[u].add(v)

        if wf_count > 0:
            if has_dep:
                rep = wf_repeat_from_count(wf_count, workflow_repeat_cap) + wf_dep_bonus
            else:
                rep = wf_repeat_from_count(wf_count, workflow_repeat_cap_no_dep)
            for _ in range(max(1, int(rep))):
                add_directed(u, v, "WORKFLOW")
            fallback_neighbors[u].add(v)

        if has_comp:
            add_directed(u, v, "COMPLEMENTS")
            add_directed(v, u, "COMPLEMENTS_REV")
            fallback_neighbors[u].add(v)
            fallback_neighbors[v].add(u)

        if has_alt:
            add_directed(u, v, "ALTERNATIVE_TO")
            add_directed(v, u, "ALTERNATIVE_TO_REV")

    if not edge_type:
        raise RuntimeError("No valid graph edges built. Check KG relations and node names.")

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long, device=device)
    edge_type_t = torch.tensor(edge_type, dtype=torch.long, device=device)
    start_nodes = service_nodes[:]

    return (
        kg,
        service_nodes,
        idx_map,
        edge_index,
        edge_type_t,
        len(rel2id),
        relation_vocab,
        dep_neighbors,
        fallback_neighbors,
        start_nodes,
        uses_by_task,
    )


def load_task_records(tasks_file: str) -> Dict[str, TaskRecord]:
    with open(tasks_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    records: Dict[str, TaskRecord] = {}
    for item in data:
        task_id = str(item.get("annotation_id", "unknown"))
        records[task_id] = TaskRecord(
            task_id=task_id,
            task_text=item.get("confirmed_task", ""),
            action_id_list=list(item.get("action_id_list", [])),
            recom_num=int(item.get("recom_num", 0) or 0),
        )
    return records


def build_samples(idx_map: Dict[str, int], tasks_file: str) -> List[Sample]:
    with open(tasks_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples: List[Sample] = []
    for item in tqdm(data, desc=f"Building samples ({tasks_file})"):
        task_id = str(item.get("annotation_id", "unknown"))
        task_text = item.get("confirmed_task", "")
        actions = list(item.get("action_id_list", []))
        seq_len = len(actions)

        for idx, action_id in enumerate(actions):
            target_node = f"Service:{action_id}"
            if target_node not in idx_map:
                continue

            samples.append(
                Sample(
                    task_id=task_id,
                    task_text=task_text,
                    path_service_ids=actions[:idx],
                    prev_service_node=f"Service:{actions[idx - 1]}" if idx > 0 else None,
                    target_service_node=target_node,
                    seq_len=seq_len,
                    step_idx=idx + 1,
                )
            )
    return samples


def split_by_task(samples: List[Sample], val_ratio: float, seed: int):
    task_ids = sorted({sample.task_id for sample in samples})
    train_ids, val_ids = train_test_split(task_ids, test_size=val_ratio, random_state=seed)
    train_set, val_set = set(train_ids), set(val_ids)
    train_samples = [sample for sample in samples if sample.task_id in train_set]
    val_samples = [sample for sample in samples if sample.task_id in val_set]
    return train_samples, val_samples, train_set, val_set


def _load_embedding_cache(cache_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    cache: Dict[str, torch.Tensor] = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            raw = pickle.load(f)
        cache = {key: torch.tensor(value, dtype=torch.float32, device=device) for key, value in raw.items()}
    return cache


def _save_embedding_cache(cache_path: str, cache: Dict[str, torch.Tensor]) -> None:
    if not cache_path:
        return
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    cpu_cache = {key: value.detach().cpu().numpy() for key, value in cache.items()}
    with open(cache_path, "wb") as f:
        pickle.dump(cpu_cache, f)


def build_or_load_service_embeddings(
    service_nodes,
    kg,
    cache_path: str,
    embed_client,
    device: torch.device,
) -> torch.Tensor:
    texts = [compose_service_text(node, kg.nodes[node]) for node in service_nodes]
    cache = _load_embedding_cache(cache_path, device)

    missing = [text for text in texts if text not in cache]
    for text in tqdm(missing, desc="Service embeddings (missing)"):
        cache[text] = torch.tensor(embed_client.get_embedding(text), dtype=torch.float32, device=device)

    _save_embedding_cache(cache_path, cache)
    tensor = torch.stack([cache[text] for text in texts], dim=0)
    return F.normalize(tensor, dim=1)


def build_or_load_task_emb_cache(task_texts, cache_path: str, embed_client, device: torch.device):
    cache = _load_embedding_cache(cache_path, device)

    missing = [text for text in task_texts if text not in cache]
    for text in tqdm(missing, desc="Task embeddings (missing)"):
        cache[text] = torch.tensor(embed_client.get_embedding(text), dtype=torch.float32, device=device)

    _save_embedding_cache(cache_path, cache)
    return cache
