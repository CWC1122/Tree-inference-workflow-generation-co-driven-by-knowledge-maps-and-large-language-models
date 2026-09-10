from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .planner import RequirementPlan


@dataclass(frozen=True)
class RelationNeighborhoods:
    dep: Dict[str, set]
    workflow: Dict[str, set]
    complement: Dict[str, set]
    alternative: Dict[str, set]
    fallback: Dict[str, set]


class RequirementServiceGrounder:
    def __init__(
        self,
        *,
        service_nodes: Sequence[str],
        idx_map: Dict[str, int],
        service_embs: torch.Tensor,
        task_proj: Callable[[torch.Tensor], torch.Tensor],
        embed_text: Callable[[str], torch.Tensor],
        tool_map: Dict[str, Dict[str, object]],
        safe_json_load: Callable[[Any], Any],
        neighborhoods: RelationNeighborhoods,
        device: torch.device,
        requirement_topk: int,
        coverage_threshold: float,
        children_per_requirement: int,
        relation_keep: int,
        disable_depends_on: bool = False,
    ):
        self.service_nodes = list(service_nodes)
        self.idx_map = idx_map
        self.service_embs = service_embs
        self.task_proj = task_proj
        self.embed_text = embed_text
        self.tool_map = tool_map
        self.safe_json_load = safe_json_load
        self.neighborhoods = neighborhoods
        self.device = device
        self.requirement_topk = max(1, int(requirement_topk))
        self.coverage_threshold = float(coverage_threshold)
        self.children_per_requirement = max(1, int(children_per_requirement))
        self.relation_keep = max(1, int(relation_keep))
        self.disable_depends_on = bool(disable_depends_on)

    def extract_candidate_pool(self, task_record: Optional[Dict[str, Any]]) -> List[str]:
        if not task_record:
            return []

        for key in ("st1.5_service", "stage15_service", "candidate_services", "service_candidates"):
            values = task_record.get(key)
            if isinstance(values, list) and values:
                return [str(value) for value in values if str(value).strip()]
        return []

    def resolve_candidate_service_nodes(
        self,
        candidate_service_nodes: Optional[Sequence[str]],
    ) -> List[str]:
        if not candidate_service_nodes:
            return list(self.service_nodes)

        seen = set()
        resolved: List[str] = []
        for node in candidate_service_nodes:
            normalized = str(node)
            if not normalized.startswith("Service:"):
                normalized = f"Service:{normalized}"
            if normalized in self.idx_map and normalized not in seen:
                resolved.append(normalized)
                seen.add(normalized)

        return resolved or list(self.service_nodes)

    def ground_plan(
        self,
        plan: RequirementPlan,
        history_prior: Dict[str, float],
        candidate_service_nodes: Optional[Sequence[str]] = None,
    ) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, Dict[str, float]]]:
        del history_prior  # history is handled later in the unified tree score.

        candidate_lists: Dict[str, List[Tuple[str, float]]] = {}
        score_lookup: Dict[str, Dict[str, float]] = {}
        service_nodes = self.resolve_candidate_service_nodes(candidate_service_nodes)
        cand_idx = torch.tensor(
            [self.idx_map[service_node] for service_node in service_nodes],
            dtype=torch.long,
            device=self.device,
        )

        for req_node in plan.nodes:
            req_emb = self.task_proj(self.embed_text(req_node.text).unsqueeze(0))
            sims = torch.matmul(req_emb, self.service_embs[cand_idx].t()).squeeze(0)
            scored = [
                (service_node, float(sims[local_idx].item()))
                for local_idx, service_node in enumerate(service_nodes)
            ]
            scored.sort(key=lambda item: item[1], reverse=True)
            candidate_lists[req_node.req_id] = scored
            score_lookup[req_node.req_id] = {service_node: score for service_node, score in scored}

        return candidate_lists, score_lookup

    def dependency_constrained_candidates(
        self,
        prefix_ids: Sequence[str],
        scored_candidates: Sequence[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        if not scored_candidates:
            return []

        if self.disable_depends_on:
            return list(scored_candidates)

        if not prefix_ids:
            return list(scored_candidates)

        support_nodes = set()
        for service_id in prefix_ids:
            support_nodes.update(self.neighborhoods.dep.get(f"Service:{service_id}", set()))

        return [
            (service_node, score)
            for service_node, score in scored_candidates
            if service_node in support_nodes
        ]

    def has_selection_evidence(
        self,
        match_score: float,
        mrg_prob: float,
        history_score: float,
        match_threshold: float,
        mrg_prob_threshold: float,
    ) -> bool:
        return (
            match_score > match_threshold
            or mrg_prob > mrg_prob_threshold
            or history_score > 0.0
        )

    def dependency_relation_score(
        self,
        prefix_ids: Sequence[str],
        service_node: str,
        root_admissible: bool,
    ) -> float:
        if not prefix_ids:
            return 0.6 if root_admissible else 0.0

        last_node = f"Service:{prefix_ids[-1]}"
        if service_node in self.neighborhoods.dep.get(last_node, set()):
            return 1.0

        for service_id in prefix_ids:
            historical = f"Service:{service_id}"
            if service_node in self.neighborhoods.dep.get(historical, set()):
                return 0.8
        return 0.0

    def mark_requirement_covered(
        self,
        covered_req_ids: Sequence[str],
        req_id: str,
    ) -> Tuple[str, ...]:
        ordered = list(covered_req_ids)
        if req_id not in ordered:
            ordered.append(req_id)
        return tuple(ordered)
