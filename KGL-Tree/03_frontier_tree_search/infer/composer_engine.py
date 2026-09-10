import json
import math
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from embedding_client import EmbeddingClient
from model_defs import GraphEncoder, PathEncoder, PosProjector, QueryFusion, TaskProjector


class StepAwareQueryFusion(nn.Module):
    """
    gate input: [task, path, prev, pos, step_feat(3)]
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 4 + 3, dim),
            nn.GELU(),
            nn.Linear(dim, 4),
        )
        self.ln = nn.LayerNorm(dim)

    def forward(
        self,
        task_emb: torch.Tensor,
        path_emb: torch.Tensor,
        prev_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        step_idx: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> torch.Tensor:
        step_idx_f = step_idx.float()
        seq_len_f = seq_len.float().clamp(min=1.0)

        step_norm = step_idx_f / seq_len_f
        is_first = (step_idx == 1).float()
        is_single = (seq_len == 1).float()
        step_feat = torch.stack([step_norm, is_first, is_single], dim=-1)

        cat = torch.cat([task_emb, path_emb, prev_emb, pos_emb, step_feat], dim=-1)
        weight = torch.softmax(self.gate(cat), dim=-1)

        query = (
            weight[:, 0:1] * task_emb
            + weight[:, 1:2] * path_emb
            + weight[:, 2:3] * prev_emb
            + weight[:, 3:4] * pos_emb
        )
        return F.normalize(self.ln(query), dim=-1)


@dataclass
class BeamState:
    path_ids: List[str]
    last_node: str
    score: float
    step_candidates: List[List[str]]


@dataclass
class HistoryTask:
    task_id: str
    task_text: str
    service_counts: Dict[str, int]


class RGCNPathSearcher:
    def __init__(
        self,
        kg_path: str,
        model_path: str,
        tool_desc_path: str,
        history_tasks_file: Optional[str] = None,
        device: str = "cuda",
        use_beam: bool = True,
        beam_width: int = 5,
        use_dynamic_beam: bool = True,
        beam_min: int = 1,
        beam_max: int = 10,
        beam_threshold_high: float = 0.3,
        beam_threshold_low: float = 0.05,
        step1_semantic_prefilter: Optional[bool] = None,
        step1_semantic_keep_ratio: Optional[float] = None,
        history_topk: Optional[int] = None,
        history_bonus_step1: Optional[float] = None,
        history_bonus_later: Optional[float] = None,
        strict_neighbor_only_after_step1: Optional[bool] = None,
        dependency_first_decoding: Optional[bool] = None,
        service_emb_cache_path: Optional[str] = None,
        text_emb_cache_path: Optional[str] = None,
    ):
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.embedder = EmbeddingClient()
        self.service_emb_cache_path = service_emb_cache_path
        self.text_emb_cache_path = text_emb_cache_path
        self.service_text_cache: Dict[str, torch.Tensor] = self._load_embedding_cache(self.service_emb_cache_path)
        self.text_cache: Dict[str, torch.Tensor] = self._load_embedding_cache(self.text_emb_cache_path)
        self.history_cache: Dict[Tuple[Optional[str], str], Tuple[Dict[str, float], List[Tuple[str, float]]]] = {}

        self.use_beam = bool(use_beam)
        self.beam_width = max(1, int(beam_width))
        self.use_dynamic_beam = bool(use_dynamic_beam)
        self.beam_min = max(1, int(beam_min))
        self.beam_max = max(self.beam_min, int(beam_max))
        self.beam_threshold_high = float(beam_threshold_high)
        self.beam_threshold_low = float(beam_threshold_low)

        with open(tool_desc_path, "r", encoding="utf-8") as f:
            tools = json.load(f)
        self.tool_map = {tool["action_uid"]: tool for tool in tools}

        with open(kg_path, "rb") as f:
            kg = pickle.load(f)

        ckpt = torch.load(model_path, map_location=self.device)
        hidden_dim = int(ckpt["hidden_dim"])
        num_relations = int(ckpt["num_relations"])
        dep_priority = ckpt.get("dep_priority", {})
        self.dep_repeat = int(dep_priority.get("DEP_REPEAT", 3))
        self.wf_dep_bonus = int(dep_priority.get("WF_DEP_BONUS", 2))
        self.workflow_repeat_cap = int(dep_priority.get("WORKFLOW_REPEAT_CAP", 5))
        self.workflow_repeat_cap_no_dep = int(dep_priority.get("WORKFLOW_REPEAT_CAP_NO_DEP", 2))

        raw_service_nodes = ckpt["service_nodes"]
        seen = set()
        self.service_nodes: List[str] = []
        for node in raw_service_nodes:
            if isinstance(node, str) and node not in seen:
                self.service_nodes.append(node)
                seen.add(node)
        self.idx_map = {node: idx for idx, node in enumerate(self.service_nodes)}
        self.enabled_rel_types = set(
            ckpt.get("enabled_rel_types", ["DEPENDS_ON", "COMPLEMENTS", "ALTERNATIVE_TO", "WORKFLOW"])
        )

        if "use_step_aware_fusion" in ckpt:
            self.use_step_aware = bool(ckpt["use_step_aware_fusion"])
            self.query_fusion_type = "step_aware" if self.use_step_aware else "query_fusion"
            self.n_query_inputs = int(ckpt.get("query_fusion_n_inputs", 4))
        else:
            self.query_fusion_type = str(ckpt.get("query_fusion_type", "legacy")).lower()
            if self.query_fusion_type == "step_aware":
                self.use_step_aware = True
                self.n_query_inputs = 4
            else:
                q_w = ckpt["query_fusion"]["gate.0.weight"]
                gate_in = int(q_w.shape[1])
                self.n_query_inputs = gate_in // hidden_dim
                if self.n_query_inputs not in (3, 4):
                    raise RuntimeError(f"Unsupported query fusion inputs: {self.n_query_inputs}")
                self.use_step_aware = False

        self.use_positional = bool(ckpt.get("use_positional_signal", self.n_query_inputs == 4))
        self.use_path_encoder = bool(ckpt.get("use_path_encoder", True))
        self.use_dependency_first_decoding = bool(
            ckpt.get("use_dependency_first_decoding", True)
            if dependency_first_decoding is None
            else dependency_first_decoding
        )

        ckpt_pref = bool(ckpt.get("step1_semantic_prefilter", False))
        ckpt_ratio = float(ckpt.get("step1_semantic_topk_dynamic_ratio", 0.3))
        self.step1_semantic_prefilter = ckpt_pref if step1_semantic_prefilter is None else bool(step1_semantic_prefilter)
        self.step1_semantic_keep_ratio = ckpt_ratio if step1_semantic_keep_ratio is None else float(step1_semantic_keep_ratio)
        self.step1_semantic_keep_ratio = min(1.0, max(0.01, self.step1_semantic_keep_ratio))

        self.history_tasks_file = history_tasks_file or ckpt.get("history_tasks_file")
        if not self.history_tasks_file:
            raise RuntimeError(
                "Missing history_tasks_file. Pass the training data json path so inference can retrieve train-task history."
            )
        if not os.path.exists(self.history_tasks_file):
            raise RuntimeError(f"history_tasks_file not found: {self.history_tasks_file}")

        self.history_topk = int(ckpt.get("history_topk", 4) if history_topk is None else history_topk)
        self.history_bonus_step1 = float(
            ckpt.get("history_bonus_step1", 0.20) if history_bonus_step1 is None else history_bonus_step1
        )
        self.history_bonus_later = float(
            ckpt.get("history_bonus_later", 0.08) if history_bonus_later is None else history_bonus_later
        )
        self.strict_neighbor_only_after_step1 = bool(
            ckpt.get("strict_neighbor_only_after_step1", True)
            if strict_neighbor_only_after_step1 is None
            else strict_neighbor_only_after_step1
        )
        self.history_train_task_ids = {
            str(task_id) for task_id in ckpt.get("history_train_task_ids", [])
        }

        relation_vocab = ckpt.get("relation_vocab")
        if relation_vocab:
            rel2id = {str(name): idx for idx, name in enumerate(relation_vocab)}
        else:
            rel2id = self._build_rel2id_by_num_relations(num_relations)
        pair_relations: Dict[Tuple[str, str], Dict[str, object]] = {}
        self.dep_neighbors = {node: set() for node in self.service_nodes}
        self.workflow_neighbors = {node: set() for node in self.service_nodes}
        self.complement_neighbors = {node: set() for node in self.service_nodes}
        self.alternative_neighbors = {node: set() for node in self.service_nodes}
        self.fallback_neighbors = {node: set() for node in self.service_nodes}
        self.uses_by_task: Dict[str, List[str]] = defaultdict(list)

        is_multi = getattr(kg, "is_multigraph", lambda: False)()
        if is_multi:
            for u, v, _, edge_data in kg.edges(keys=True, data=True):
                rel = edge_data.get("type")
                if rel == "USES" and u.startswith("Task:") and v.startswith("Service:"):
                    self.uses_by_task[u.replace("Task:", "")].append(v.replace("Service:", ""))
                self._consume_service_edge(pair_relations, u, v, edge_data)
        else:
            for u, v, edge_data in kg.edges(data=True):
                rel = edge_data.get("type")
                if rel == "USES" and u.startswith("Task:") and v.startswith("Service:"):
                    self.uses_by_task[u.replace("Task:", "")].append(v.replace("Service:", ""))
                self._consume_service_edge(pair_relations, u, v, edge_data)

        edge_src, edge_dst, edge_type = self._build_rgcn_edges(pair_relations, rel2id)

        if not edge_src:
            raise RuntimeError("No valid graph edges after filtering. Check KG/model relation settings.")

        n_nodes = len(self.service_nodes)
        if min(min(edge_src), min(edge_dst)) < 0 or max(max(edge_src), max(edge_dst)) >= n_nodes:
            raise RuntimeError("edge_index out of bounds before RGCN.")
        if min(edge_type) < 0 or max(edge_type) >= num_relations:
            raise RuntimeError("edge_type out of bounds.")

        self.start_nodes = self.service_nodes[:]

        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long, device=self.device)
        edge_type_t = torch.tensor(edge_type, dtype=torch.long, device=self.device)

        service_texts = []
        for node in self.service_nodes:
            if node in kg.nodes:
                service_texts.append(self._compose_service_text(node, kg.nodes[node]))
            else:
                service_texts.append("")
        service_text_embs = torch.stack(
            [self._embed_text(text, use_service_cache=True) for text in tqdm(service_texts, desc="Embedding services (missing)")],
            dim=0,
        )
        self._save_embedding_cache(self.service_emb_cache_path, self.service_text_cache)
        service_text_embs = F.normalize(service_text_embs, dim=1)

        in_dim = service_text_embs.size(1)
        self.encoder = GraphEncoder(in_dim, hidden_dim, num_relations).to(self.device)
        self.path_encoder = PathEncoder(hidden_dim).to(self.device) if self.use_path_encoder else None
        self.task_proj = TaskProjector(in_dim, hidden_dim).to(self.device)

        if self.use_step_aware:
            self.query_fusion = StepAwareQueryFusion(hidden_dim).to(self.device)
        else:
            self.query_fusion = QueryFusion(hidden_dim, self.n_query_inputs).to(self.device)

        self.pos_proj = None
        if self.use_positional:
            self.pos_proj = PosProjector(hidden_dim).to(self.device)

        self.encoder.load_state_dict(ckpt["encoder"])
        if self.path_encoder is not None and ckpt.get("path_encoder") is not None:
            self.path_encoder.load_state_dict(ckpt["path_encoder"])
        self.task_proj.load_state_dict(ckpt["task_proj"])
        self.query_fusion.load_state_dict(ckpt["query_fusion"])

        if self.use_positional:
            if "pos_proj" not in ckpt:
                raise RuntimeError("Checkpoint expects positional signal but missing 'pos_proj'.")
            self.pos_proj.load_state_dict(ckpt["pos_proj"])
            self.pos_proj.eval()

        self.encoder.eval()
        if self.path_encoder is not None:
            self.path_encoder.eval()
        self.task_proj.eval()
        self.query_fusion.eval()

        with torch.no_grad():
            self.service_embs = self.encoder(service_text_embs, edge_index, edge_type_t)
        self.emb_dim = self.service_embs.size(1)

        self.history_tasks = self._build_history_tasks()
        if self.history_tasks:
            history_embs = [
                self._embed_text(task.task_text)
                for task in tqdm(self.history_tasks, desc="Embedding train history tasks (missing)")
            ]
            self._save_embedding_cache(self.text_emb_cache_path, self.text_cache)
            self.history_task_embs = F.normalize(torch.stack(history_embs, dim=0), dim=1)
            self.history_task_ids = [task.task_id for task in self.history_tasks]
            self.history_service_counts = [task.service_counts for task in self.history_tasks]
            self.history_task_index = {task_id: idx for idx, task_id in enumerate(self.history_task_ids)}
        else:
            self.history_task_embs = None
            self.history_task_ids = []
            self.history_service_counts = []
            self.history_task_index = {}

        print(
            f"[QueryMode] fusion_type={self.query_fusion_type}, use_step_aware={self.use_step_aware}, "
            f"use_path_encoder={self.use_path_encoder}, use_positional={self.use_positional}, "
            f"dependency_first={self.use_dependency_first_decoding}, "
            f"step1_prefilter={self.step1_semantic_prefilter}, step1_keep_ratio={self.step1_semantic_keep_ratio}, "
            f"history_topk={self.history_topk}, history_bonus_step1={self.history_bonus_step1}, "
            f"history_bonus_later={self.history_bonus_later}, strict_neighbor_only_after_step1={self.strict_neighbor_only_after_step1}, "
            f"dep_repeat={self.dep_repeat}, wf_dep_bonus={self.wf_dep_bonus}, "
            f"history_tasks={len(self.history_tasks)}"
        )

    def _safe_json_load(self, value):
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return []
        return []

    def _compose_service_text(self, service_node: str, attrs: Dict) -> str:
        sid = attrs.get("id", service_node.replace("Service:", ""))
        desc = attrs.get("description", "") or ""
        ins = self._safe_json_load(attrs.get("input_types", attrs.get("input-type", [])))
        outs = self._safe_json_load(attrs.get("output_types", attrs.get("output-type", [])))

        parts = [f"Service: {sid}", f"Description: {desc}"]
        if ins:
            parts.append(f"Input: {', '.join(ins)}")
        if outs:
            parts.append(f"Output: {', '.join(outs)}")
        return "\n".join(parts)

    def _load_embedding_cache(self, cache_path: Optional[str]) -> Dict[str, torch.Tensor]:
        if not cache_path or not os.path.exists(cache_path):
            return {}

        with open(cache_path, "rb") as f:
            raw = pickle.load(f)
        return {key: torch.tensor(value, dtype=torch.float32, device=self.device) for key, value in raw.items()}

    def _save_embedding_cache(self, cache_path: Optional[str], cache: Dict[str, torch.Tensor]) -> None:
        if not cache_path:
            return

        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        cpu_cache = {key: value.detach().cpu().numpy() for key, value in cache.items()}
        with open(cache_path, "wb") as f:
            pickle.dump(cpu_cache, f)

    def _load_task_records(self, tasks_file: str) -> Dict[str, Dict[str, object]]:
        with open(tasks_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[history_debug] raw_data_json_count = {len(data)}")

        records: Dict[str, Dict[str, object]] = {}
        for item in data:
            task_id = str(item.get("annotation_id", "unknown"))
            records[task_id] = {
                "task_text": item.get("confirmed_task", ""),
                "action_id_list": list(item.get("action_id_list", [])),
            }
        print(f"[history_debug] unique_annotation_id_count = {len(records)}")
        return records

    def _build_history_tasks(self) -> List[HistoryTask]:
        task_records = self._load_task_records(self.history_tasks_file)
        allowed_ids = self.history_train_task_ids or set(task_records.keys())
        print(f"[history_debug] allowed_history_task_id_count = {len(allowed_ids)}")

        history_tasks: List[HistoryTask] = []
        empty_task_text_count = 0
        for task_id in sorted(allowed_ids):
            record = task_records.get(task_id)
            if record is None:
                continue

            service_counts: Dict[str, int] = defaultdict(int)
            used_services = list(self.uses_by_task.get(task_id, [])) or list(record.get("action_id_list", []))
            for service_id in used_services:
                node = f"Service:{service_id}"
                if node in self.idx_map:
                    service_counts[node] += 1

            task_text = str(record.get("task_text", "") or "")
            if not task_text:
                empty_task_text_count += 1
                continue

            history_tasks.append(
                HistoryTask(
                    task_id=str(task_id),
                    task_text=task_text,
                    service_counts=dict(service_counts),
                )
            )
        print(f"[history_debug] empty_confirmed_task_filtered_count = {empty_task_text_count}")
        print(f"[history_debug] final_history_task_count = {len(history_tasks)}")
        return history_tasks

    def _dynamic_step1_keep_k(self, n: int) -> int:
        return max(1, int(math.ceil(n * self.step1_semantic_keep_ratio)))

    def _wf_repeat_from_count(self, count: int, cap: int) -> int:
        count = max(1, int(count))
        rep = 1 + int(math.log2(1 + count))
        return min(cap, rep)

    def _build_rel2id_by_num_relations(self, num_relations: int) -> Dict[str, int]:
        if num_relations == 4:
            keys = ["DEPENDS_ON", "COMPLEMENTS", "ALTERNATIVE_TO", "WORKFLOW"]
        elif num_relations == 6:
            keys = ["DEPENDS_ON", "COMPLEMENTS", "COMPLEMENTS_REV", "ALTERNATIVE_TO", "ALTERNATIVE_TO_REV", "WORKFLOW"]
        elif num_relations == 7:
            keys = ["DEPENDS_ON", "DEPENDS_ON_REV", "COMPLEMENTS", "COMPLEMENTS_REV", "ALTERNATIVE_TO", "ALTERNATIVE_TO_REV", "WORKFLOW"]
        elif num_relations == 8:
            keys = ["DEPENDS_ON", "DEPENDS_ON_REV", "COMPLEMENTS", "COMPLEMENTS_REV", "ALTERNATIVE_TO", "ALTERNATIVE_TO_REV", "WORKFLOW", "WORKFLOW_REV"]
        else:
            raise RuntimeError(f"Unsupported num_relations={num_relations}")
        return {key: idx for idx, key in enumerate(keys)}

    def _consume_service_edge(
        self,
        pair_relations: Dict[Tuple[str, str], Dict[str, object]],
        u: str,
        v: str,
        edge_data: Dict,
    ) -> None:
        if u not in self.idx_map or v not in self.idx_map:
            return

        rel = edge_data.get("type")
        if rel not in self.enabled_rel_types:
            return

        rec = pair_relations.setdefault((u, v), {"dep": False, "wf_count": 0, "comp": False, "alt": False})
        if rel == "DEPENDS_ON":
            rec["dep"] = True
        elif rel == "WORKFLOW":
            rec["wf_count"] = int(rec["wf_count"]) + int(edge_data.get("count", edge_data.get("wf_raw_count", 1)))
        elif rel == "COMPLEMENTS":
            rec["comp"] = True
        elif rel == "ALTERNATIVE_TO":
            rec["alt"] = True

    def _build_rgcn_edges(
        self,
        pair_relations: Dict[Tuple[str, str], Dict[str, object]],
        rel2id: Dict[str, int],
    ) -> Tuple[List[int], List[int], List[int]]:
        edge_src: List[int] = []
        edge_dst: List[int] = []
        edge_type: List[int] = []

        def add_directed(u: str, v: str, rel: str) -> None:
            edge_src.append(self.idx_map[u])
            edge_dst.append(self.idx_map[v])
            edge_type.append(rel2id[rel])

        for (u, v), rel_info in pair_relations.items():
            has_dep = bool(rel_info["dep"])
            wf_count = int(rel_info["wf_count"])
            has_comp = bool(rel_info["comp"])
            has_alt = bool(rel_info["alt"])

            if has_dep:
                for _ in range(self.dep_repeat):
                    add_directed(u, v, "DEPENDS_ON")
                self.dep_neighbors[u].add(v)

            if wf_count > 0:
                if has_dep:
                    rep = self._wf_repeat_from_count(wf_count, self.workflow_repeat_cap) + self.wf_dep_bonus
                else:
                    rep = self._wf_repeat_from_count(wf_count, self.workflow_repeat_cap_no_dep)
                for _ in range(max(1, int(rep))):
                    add_directed(u, v, "WORKFLOW")
                self.workflow_neighbors[u].add(v)
                self.fallback_neighbors[u].add(v)

            if has_comp:
                add_directed(u, v, "COMPLEMENTS")
                if "COMPLEMENTS_REV" in rel2id:
                    add_directed(v, u, "COMPLEMENTS_REV")
                self.complement_neighbors[u].add(v)
                self.complement_neighbors[v].add(u)
                self.fallback_neighbors[u].add(v)
                self.fallback_neighbors[v].add(u)

            if has_alt:
                add_directed(u, v, "ALTERNATIVE_TO")
                if "ALTERNATIVE_TO_REV" in rel2id:
                    add_directed(v, u, "ALTERNATIVE_TO_REV")
                self.alternative_neighbors[u].add(v)
                self.alternative_neighbors[v].add(u)

        return edge_src, edge_dst, edge_type

    def _embed_text(self, text: str, use_service_cache: bool = False) -> torch.Tensor:
        key = (text or "").strip()
        cache = self.service_text_cache if use_service_cache else self.text_cache
        if key in cache:
            return cache[key]

        vec = np.asarray(self.embedder.get_embedding(key), dtype=np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        ten = torch.tensor(vec, dtype=torch.float32, device=self.device)
        cache[key] = ten
        return ten

    def _encode_path(self, path_service_ids: List[str]) -> torch.Tensor:
        if not self.use_path_encoder or self.path_encoder is None:
            return torch.zeros((1, self.emb_dim), dtype=self.service_embs.dtype, device=self.device)

        if not path_service_ids:
            seq = torch.zeros((1, 1, self.emb_dim), device=self.device)
            lengths = torch.tensor([1], dtype=torch.long, device=self.device)
            return self.path_encoder(seq, lengths)

        vecs = []
        for service_id in path_service_ids:
            node = f"Service:{service_id}"
            if node in self.idx_map:
                vecs.append(self.service_embs[self.idx_map[node]])

        if not vecs:
            seq = torch.zeros((1, 1, self.emb_dim), device=self.device)
            lengths = torch.tensor([1], dtype=torch.long, device=self.device)
            return self.path_encoder(seq, lengths)

        seq = pad_sequence([torch.stack(vecs)], batch_first=True)
        lengths = torch.tensor([len(vecs)], dtype=torch.long, device=self.device)
        return self.path_encoder(seq, lengths)

    def _make_query(self, task_emb: torch.Tensor, path_ids: List[str], step_idx_1based: int, total_steps: int) -> torch.Tensor:
        path_emb = self._encode_path(path_ids)

        if path_ids:
            prev_node = f"Service:{path_ids[-1]}"
            if prev_node in self.idx_map:
                prev_emb = self.service_embs[self.idx_map[prev_node]].unsqueeze(0)
            else:
                prev_emb = torch.zeros((1, self.emb_dim), device=self.device)
        else:
            prev_emb = torch.zeros((1, self.emb_dim), device=self.device)

        if self.use_positional:
            step = step_idx_1based
            total = max(1, total_steps)
            pos_feat = torch.tensor([[float(step) / total, float(total - step) / total]], dtype=torch.float32, device=self.device)
            pos_emb = self.pos_proj(pos_feat)
        else:
            pos_emb = torch.zeros_like(task_emb)

        if self.use_step_aware:
            step_idx_t = torch.tensor([step_idx_1based], dtype=torch.long, device=self.device)
            seq_len_t = torch.tensor([max(1, total_steps)], dtype=torch.long, device=self.device)
            return self.query_fusion(task_emb, path_emb, prev_emb, pos_emb, step_idx_t, seq_len_t)

        if self.use_positional:
            return self.query_fusion(task_emb, path_emb, prev_emb, pos_emb)
        return self.query_fusion(task_emb, path_emb, prev_emb)

    def _history_bonus_scale(self, step_idx_1based: int) -> float:
        return self.history_bonus_step1 if int(step_idx_1based) == 1 else self.history_bonus_later

    def _get_history_prior(
        self,
        task_text: str,
        current_task_id: Optional[str] = None,
    ) -> Tuple[Dict[str, float], List[Tuple[str, float]]]:
        current_task_id = str(current_task_id) if current_task_id is not None else None
        cache_key = (current_task_id, task_text)
        if cache_key in self.history_cache:
            return self.history_cache[cache_key]

        if self.history_task_embs is None or not task_text:
            result = ({}, [])
            self.history_cache[cache_key] = result
            return result

        query_emb = F.normalize(self._embed_text(task_text).unsqueeze(0), dim=1)
        sims = torch.matmul(query_emb, self.history_task_embs.t()).squeeze(0)

        if current_task_id is not None and current_task_id in self.history_task_index:
            sims[self.history_task_index[current_task_id]] = -1e9

        ranked = torch.argsort(sims, descending=True).tolist()
        prior = defaultdict(float)
        matched: List[Tuple[str, float]] = []
        used = 0

        for pos in ranked:
            if used >= self.history_topk:
                break

            sim = float(sims[pos].item())
            if sim <= 0.0:
                continue

            matched.append((self.history_task_ids[pos], sim))
            for node, count in self.history_service_counts[pos].items():
                prior[node] += sim * float(count)
            used += 1

        if prior:
            max_score = max(prior.values())
            norm_prior = {node: score / max_score for node, score in prior.items()}
        else:
            norm_prior = {}

        result = (norm_prior, matched)
        self.history_cache[cache_key] = result
        return result

    def _candidate_nodes_from_prefix(self, prefix_ids: List[str], task_emb: Optional[torch.Tensor] = None) -> List[str]:
        if not prefix_ids:
            cand_nodes = list(self.start_nodes)
        else:
            prev_node = f"Service:{prefix_ids[-1]}"
            dep_nodes = sorted(self.dep_neighbors.get(prev_node, set()))
            fallback_nodes = sorted(self.fallback_neighbors.get(prev_node, set()))
            if self.use_dependency_first_decoding:
                if dep_nodes:
                    cand_nodes = dep_nodes
                else:
                    cand_nodes = fallback_nodes
            else:
                cand_nodes = list(dict.fromkeys(dep_nodes + fallback_nodes))

            if not cand_nodes and self.strict_neighbor_only_after_step1:
                return []
            if not cand_nodes:
                cand_nodes = list(self.service_nodes)

        cand_nodes = [node for node in cand_nodes if node in self.idx_map]

        if (not prefix_ids) and self.step1_semantic_prefilter and task_emb is not None and len(cand_nodes) > 1:
            cand_idx = torch.tensor([self.idx_map[node] for node in cand_nodes], dtype=torch.long, device=self.device)
            sims = torch.matmul(task_emb, self.service_embs[cand_idx].t()).squeeze(0)
            keep_k = min(self._dynamic_step1_keep_k(len(cand_nodes)), sims.numel())
            keep_local = torch.topk(sims, k=keep_k).indices.tolist()
            cand_nodes = [cand_nodes[idx] for idx in keep_local]

        return cand_nodes

    def _candidate_history_bonus(
        self,
        cand_nodes: List[str],
        history_prior: Dict[str, float],
        step_idx_1based: int,
    ) -> torch.Tensor:
        if not cand_nodes or not history_prior:
            return torch.zeros(len(cand_nodes), dtype=torch.float32, device=self.device)

        scale = self._history_bonus_scale(step_idx_1based)
        values = [float(history_prior.get(node, 0.0)) * scale for node in cand_nodes]
        return torch.tensor(values, dtype=torch.float32, device=self.device)

    def precompute_task_embeddings(self, task_texts: List[str], desc: str = "Embedding input tasks") -> None:
        unique_texts = []
        seen = set()
        for text in task_texts:
            key = (text or "").strip()
            if not key or key in seen or key in self.text_cache:
                continue
            seen.add(key)
            unique_texts.append(key)

        for text in tqdm(unique_texts, desc=desc):
            self._embed_text(text)
        self._save_embedding_cache(self.text_emb_cache_path, self.text_cache)

    def get_candidate_ids_for_prefix(
        self,
        prefix_ids: List[str],
        task_text: str = "",
        current_task_id: Optional[str] = None,
    ) -> List[str]:
        task_emb = self.task_proj(self._embed_text(task_text).unsqueeze(0)) if task_text else None
        cand_nodes = self._candidate_nodes_from_prefix(prefix_ids, task_emb=task_emb)
        return [node.replace("Service:", "") for node in cand_nodes]

    def _choose_dynamic_beam(self, sorted_scores: List[float]) -> int:
        if not self.use_beam:
            return 1
        if not self.use_dynamic_beam:
            return self.beam_width
        if len(sorted_scores) <= 1:
            return self.beam_min

        gap12 = sorted_scores[0] - sorted_scores[1]
        if gap12 >= self.beam_threshold_high:
            return self.beam_min
        if gap12 <= self.beam_threshold_low:
            return self.beam_max

        ratio = (self.beam_threshold_high - gap12) / (self.beam_threshold_high - self.beam_threshold_low + 1e-8)
        keep = self.beam_min + ratio * (self.beam_max - self.beam_min)
        return int(max(self.beam_min, min(self.beam_max, round(keep))))

    def _tool_payload(self, sid: str) -> Dict[str, object]:
        tool = self.tool_map.get(sid)
        if not tool:
            return {
                "action_uid": sid,
                "target_action_reprs": "",
                "input-type": [],
                "output-type": [],
                "id": sid,
            }

        return {
            "action_uid": tool.get("action_uid", sid),
            "target_action_reprs": tool.get("target_action_reprs", ""),
            "input-type": tool.get("input-type", []),
            "output-type": tool.get("output-type", []),
            "id": tool.get("action_uid", sid),
        }

    @torch.no_grad()
    def search_service_path(
        self,
        task_text: str,
        pred_task_num: int,
        return_debug: bool = False,
        current_task_id: Optional[str] = None,
        task_record: Optional[Dict[str, object]] = None,
    ):
        total_steps = max(1, int(pred_task_num))
        task_emb = self.task_proj(self._embed_text(task_text).unsqueeze(0))
        history_prior, matched_history = self._get_history_prior(task_text, current_task_id=current_task_id)

        beams = [BeamState(path_ids=[], last_node="", score=0.0, step_candidates=[])]
        max_expand = max(self.beam_width, self.beam_max) if self.use_dynamic_beam else self.beam_width
        max_expand = max(1, max_expand)

        for step in range(total_steps):
            step_idx = step + 1
            all_new: List[BeamState] = []

            for beam in beams:
                cand_nodes = self._candidate_nodes_from_prefix(beam.path_ids, task_emb=task_emb)
                if not cand_nodes:
                    continue

                cand_ids = [node.replace("Service:", "") for node in cand_nodes]
                query = self._make_query(task_emb, beam.path_ids, step_idx, total_steps)
                cand_idx = torch.tensor([self.idx_map[node] for node in cand_nodes], dtype=torch.long, device=self.device)
                logits = torch.matmul(query, self.service_embs[cand_idx].t()).squeeze(0)
                logits = logits + self._candidate_history_bonus(cand_nodes, history_prior, step_idx)

                k_expand = min(max_expand, logits.numel())
                top = torch.topk(logits, k=k_expand)
                for score, local_idx in zip(top.values.tolist(), top.indices.tolist()):
                    node = cand_nodes[local_idx]
                    sid = node.replace("Service:", "")
                    all_new.append(
                        BeamState(
                            path_ids=beam.path_ids + [sid],
                            last_node=node,
                            score=beam.score + float(score),
                            step_candidates=beam.step_candidates + [cand_ids],
                        )
                    )

            if not all_new:
                break

            all_new.sort(key=lambda item: item.score, reverse=True)
            if not self.use_beam:
                keep = 1
            else:
                step_scores = [item.score for item in all_new[:max(3, self.beam_max)]]
                keep = self._choose_dynamic_beam(step_scores)
            beams = all_new[:max(1, keep)]

        best = beams[0] if beams else BeamState(path_ids=[], last_node="", score=0.0, step_candidates=[])

        if return_debug:
            return {
                "pred_path": best.path_ids,
                "candidates": best.step_candidates,
                "score": best.score,
                "matched_history": [
                    {"task_id": task_id, "similarity": float(sim)}
                    for task_id, sim in matched_history
                ],
                "history_prior_top_services": [
                    {
                        "service_id": node.replace("Service:", ""),
                        "score": float(score),
                    }
                    for node, score in sorted(history_prior.items(), key=lambda x: x[1], reverse=True)[:10]
                ],
            }

        return [self._tool_payload(sid) for sid in best.path_ids]
