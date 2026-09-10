import math
from typing import Dict, List, Sequence, Tuple

import torch

from .data_utils import Sample


def half_keep_k(size: int, keep_ratio: float) -> int:
    return max(1, int(math.ceil(size * keep_ratio)))


def history_bonus_scale(step_idx: int, step1_bonus: float, later_bonus: float) -> float:
    return float(step1_bonus if int(step_idx) == 1 else later_bonus)


def step1_semantic_prefilter(
    cand_idx: List[int],
    task_emb: torch.Tensor,
    service_embs: torch.Tensor,
    keep_ratio: float,
) -> List[int]:
    if len(cand_idx) <= 1:
        return cand_idx

    cand_t = torch.tensor(cand_idx, dtype=torch.long, device=service_embs.device)
    sims = torch.matmul(task_emb, service_embs[cand_t].t()).squeeze(0)
    keep_k = min(half_keep_k(len(cand_idx), keep_ratio), sims.numel())
    keep_local = torch.topk(sims, k=keep_k).indices.tolist()
    return [cand_idx[idx] for idx in keep_local]


def build_candidate_bonus(
    cand_ids: Sequence[int],
    service_nodes: Sequence[str],
    history_prior: Dict[str, float],
    bonus_scale: float,
    device: torch.device,
) -> torch.Tensor:
    if not history_prior or bonus_scale <= 0.0:
        return torch.zeros(len(cand_ids), dtype=torch.float32, device=device)

    values = [float(history_prior.get(service_nodes[idx], 0.0)) * bonus_scale for idx in cand_ids]
    return torch.tensor(values, dtype=torch.float32, device=device)


def build_dense_history_bonus(
    service_nodes: Sequence[str],
    history_prior: Dict[str, float],
    bonus_scale: float,
    device: torch.device,
) -> torch.Tensor:
    if not history_prior or bonus_scale <= 0.0:
        return torch.zeros(len(service_nodes), dtype=torch.float32, device=device)

    values = [float(history_prior.get(node, 0.0)) * bonus_scale for node in service_nodes]
    return torch.tensor(values, dtype=torch.float32, device=device)


def _dedupe_keep_order(indices: List[int]) -> List[int]:
    return list(dict.fromkeys(indices))


def _limit_candidates(
    cand_ids: List[int],
    target_idx: int,
    max_candidates: int,
    query_vec: torch.Tensor,
    service_embs: torch.Tensor,
    service_nodes: Sequence[str],
    history_prior: Dict[str, float],
    bonus_scale: float,
) -> List[int]:
    if len(cand_ids) <= max_candidates:
        return cand_ids

    cand_t = torch.tensor(cand_ids, dtype=torch.long, device=service_embs.device)
    scores = torch.matmul(query_vec, service_embs[cand_t].t()).squeeze(0)
    scores = scores + build_candidate_bonus(cand_ids, service_nodes, history_prior, bonus_scale, service_embs.device)

    keep_k = min(max_candidates, scores.numel())
    keep_local = torch.topk(scores, k=keep_k).indices.tolist()
    limited = [cand_ids[idx] for idx in keep_local]

    if target_idx in cand_ids and target_idx not in limited and limited:
        limited[-1] = target_idx
    return _dedupe_keep_order(limited)


def build_candidate_set(
    sample: Sample,
    idx_map: Dict[str, int],
    dep_neighbors: Dict[str, set],
    fallback_neighbors: Dict[str, set],
    start_nodes: Sequence[str],
    task_emb: torch.Tensor,
    query_vec: torch.Tensor,
    service_embs: torch.Tensor,
    service_nodes: Sequence[str],
    history_prior: Dict[str, float],
    max_candidates: int,
    enable_step1_prefilter: bool,
    step1_keep_ratio: float,
    history_bonus_step1: float,
    history_bonus_later: float,
    use_dependency_first: bool,
) -> Tuple[List[int], int, bool]:
    target_idx = idx_map[sample.target_service_node]
    is_step1 = sample.prev_service_node is None

    if is_step1:
        base_nodes = list(start_nodes)
    else:
        dep_base = list(dep_neighbors.get(sample.prev_service_node, set()))
        fallback_base = list(fallback_neighbors.get(sample.prev_service_node, set()))
        if use_dependency_first:
            if dep_base:
                base_nodes = dep_base
            else:
                base_nodes = fallback_base
        else:
            base_nodes = dep_base + fallback_base

    cand_ids = _dedupe_keep_order([idx_map[node] for node in base_nodes if node in idx_map])

    if is_step1 and enable_step1_prefilter and len(cand_ids) > 1:
        cand_ids = step1_semantic_prefilter(cand_ids, task_emb, service_embs, step1_keep_ratio)

    if is_step1 and target_idx not in cand_ids:
        cand_ids.append(target_idx)

    if not is_step1 and target_idx not in cand_ids:
        return cand_ids[:max_candidates], -1, False

    bonus_scale = history_bonus_scale(sample.step_idx, history_bonus_step1, history_bonus_later)
    cand_ids = _limit_candidates(
        cand_ids,
        target_idx,
        max_candidates,
        query_vec.detach(),
        service_embs,
        service_nodes,
        history_prior,
        bonus_scale,
    )

    if target_idx not in cand_ids:
        if is_step1 and cand_ids:
            cand_ids[-1] = target_idx
            cand_ids = _dedupe_keep_order(cand_ids)
        else:
            return cand_ids, -1, False

    label = cand_ids.index(target_idx)
    return cand_ids, label, True
