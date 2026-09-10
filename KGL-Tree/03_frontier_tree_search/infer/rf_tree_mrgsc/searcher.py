import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from composer_engine import RGCNPathSearcher

from .grounding import RelationNeighborhoods, RequirementServiceGrounder
from .planner import RequirementPlan, RequirementPlanner
from .tree_pruning import FrontierTreePruner


@dataclass
class FrontierState:
    path_ids: List[str]
    covered_req_ids: Tuple[str, ...]
    frontier_req_ids: Tuple[str, ...]
    tree_score: float
    step_candidates: List[List[str]]
    selected_requirements: List[str]


class FrontierTreeComposer(RGCNPathSearcher):
    def __init__(
        self,
        *args,
        requirement_topk: int = 24,
        requirement_min_plan_size: int = 2,
        requirement_coverage_threshold: float = 0.55,
        tree_budget_multiplier: int = 8,
        tree_queue_size: int = 24,
        tree_children_per_requirement: int = 4,
        tree_relation_keep: int = 1,
        tree_match_threshold: float = 0.05,
        tree_mrg_prob_threshold: float = 0.55,
        tree_alpha_match: float = 1.0,
        tree_beta_mrg: float = 1.0,
        tree_eta_history: float = 0.6,
        tree_gamma_dep: float = 0.4,
        tree_rho_redundancy: float = 0.25,
        tree_frontier_score_threshold: float = 0.0,
        tree_enable_state_dominance: bool = True,
        tree_use_dep_sig: bool = True,
        ignore_dag_order: bool = False,
        disable_depends_on: bool = False,
        candidate_top_k: int = 5,
        candidate_include_m_minus_1: bool = True,
        search_variant: str = "frontier_tree",
        **kwargs,
    ):
        legacy_top_k = kwargs.pop("module4_top_k", None)
        legacy_include_m1 = kwargs.pop("module4_include_m_minus_1", None)
        for legacy_key in (
            "critic_frontier_score_threshold",
            "critic_enable_state_dominance",
            "critic_use_dep_sig",
            "critic_verifier_top_k",
            "critic_verifier_include_m_minus_1",
            "critic_verifier_temperature",
            "critic_llm_score_weight",
            "module4_llm_temperature",
            "final_lambda_coverage",
            "final_lambda_exec",
            "final_lambda_redundancy",
            "final_lambda_length",
            "final_coverage_threshold",
            "final_exec_threshold",
            "module4_delete_mrg_drop_tolerance",
        ):
            kwargs.pop(legacy_key, None)
        if legacy_top_k is not None:
            candidate_top_k = legacy_top_k
        if legacy_include_m1 is not None:
            candidate_include_m_minus_1 = legacy_include_m1

        super().__init__(*args, **kwargs)
        self.requirement_min_plan_size = max(1, int(requirement_min_plan_size))
        self.tree_budget_multiplier = max(1, int(tree_budget_multiplier))
        self.tree_queue_size = max(1, int(tree_queue_size))
        self.tree_top_b = max(1, int(tree_children_per_requirement))
        self.tree_match_threshold = float(tree_match_threshold)
        self.tree_mrg_prob_threshold = float(tree_mrg_prob_threshold)
        self.tree_alpha_match = float(tree_alpha_match)
        self.tree_beta_mrg = float(tree_beta_mrg)
        self.tree_eta_history = float(tree_eta_history)
        self.tree_gamma_dep = float(tree_gamma_dep)
        self.tree_rho_redundancy = float(tree_rho_redundancy)
        self.candidate_top_k = max(1, int(candidate_top_k))
        self.candidate_include_m_minus_1 = bool(candidate_include_m_minus_1)
        normalized_variant = str(search_variant or "frontier_tree").strip().lower()
        if normalized_variant in {"full", "tree", "frontier"}:
            normalized_variant = "frontier_tree"
        if normalized_variant in {"topological_sequential", "topo_sequential"}:
            normalized_variant = "topological_sequential"
        if normalized_variant not in {"frontier_tree", "greedy_frontier", "topological_sequential"}:
            normalized_variant = "frontier_tree"
        self.search_variant = normalized_variant

        self.requirement_planner = RequirementPlanner(
            min_plan_size=self.requirement_min_plan_size,
            ignore_edges=ignore_dag_order,
        )
        self.neighborhoods = RelationNeighborhoods(
            dep=self.dep_neighbors,
            workflow=self.workflow_neighbors,
            complement=self.complement_neighbors,
            alternative=self.alternative_neighbors,
            fallback=self.fallback_neighbors,
        )
        self.requirement_grounder = RequirementServiceGrounder(
            service_nodes=self.service_nodes,
            idx_map=self.idx_map,
            service_embs=self.service_embs,
            task_proj=self.task_proj,
            embed_text=self._embed_text,
            tool_map=self.tool_map,
            safe_json_load=self._safe_json_load,
            neighborhoods=self.neighborhoods,
            device=self.device,
            requirement_topk=requirement_topk,
            coverage_threshold=requirement_coverage_threshold,
            children_per_requirement=tree_children_per_requirement,
            relation_keep=tree_relation_keep,
            disable_depends_on=disable_depends_on,
        )
        self.tree_pruner = FrontierTreePruner(
            dep_neighbors=self.dep_neighbors,
            frontier_score_threshold=tree_frontier_score_threshold,
            enable_state_dominance=tree_enable_state_dominance,
            use_dep_sig=tree_use_dep_sig,
        )

    def _compute_tree_score(
        self,
        *,
        match_score: float,
        mrg_prob: float,
        history_score: float,
        dep_rel: float,
        gain: float,
        is_duplicate: bool,
    ) -> float:
        redundancy_penalty = 1.0 if is_duplicate else 0.0
        log_p_mrg = math.log(max(mrg_prob, 1e-8))
        return (
            gain
            * (
                self.tree_alpha_match * match_score
                + self.tree_beta_mrg * log_p_mrg
                + self.tree_eta_history * history_score
                + self.tree_gamma_dep * dep_rel
            )
            - self.tree_rho_redundancy * redundancy_penalty
        )

    def _build_dag_summary(self, plan: RequirementPlan) -> List[Dict[str, object]]:
        return [
            {
                "id": node.req_id,
                "text": node.text,
                "predecessors": list(node.predecessors),
            }
            for node in plan.nodes
        ]

    def _build_dependency_evidence(self, path_ids: Sequence[str]) -> List[Dict[str, object]]:
        evidence: List[Dict[str, object]] = []
        for idx, service_id in enumerate(path_ids):
            if idx == 0:
                evidence.append({"service_id": service_id, "supported_by": [], "mode": "root"})
                continue

            current_node = f"Service:{service_id}"
            supporters = []
            for prev_id in path_ids[:idx]:
                prev_node = f"Service:{prev_id}"
                if current_node in self.dep_neighbors.get(prev_node, set()):
                    supporters.append(prev_id)
            evidence.append({"service_id": service_id, "supported_by": supporters, "mode": "depends_on"})
        return evidence

    def _build_service_details(self, path_ids: Sequence[str]) -> List[Dict[str, object]]:
        details = []
        for service_id in path_ids:
            payload = self._tool_payload(service_id)
            details.append(
                {
                    "service_id": payload.get("id", service_id),
                    "service_name": payload.get("action_uid", service_id),
                    "description": payload.get("target_action_reprs", ""),
                    "input_types": payload.get("input-type", []),
                    "output_types": payload.get("output-type", []),
                }
            )
        return details

    def _state_to_candidate(
        self,
        state: FrontierState,
        path_id: str,
        plan: RequirementPlan,
    ) -> Dict[str, object]:
        covered_requirements = [
            {"id": req_id, "text": plan.node_map[req_id].text}
            for req_id in state.covered_req_ids
            if req_id in plan.node_map
        ]
        return {
            "path_id": path_id,
            "path_ids": list(state.path_ids),
            "tree_score": float(state.tree_score),
            "step_candidates": [list(step) for step in state.step_candidates],
            "covered_req_ids": list(state.covered_req_ids),
            "frontier_req_ids": list(state.frontier_req_ids),
            "selected_requirements": list(state.selected_requirements),
            "covered_requirements": covered_requirements,
            "service_details": self._build_service_details(state.path_ids),
            "dependency_evidence": self._build_dependency_evidence(state.path_ids),
            "total_requirement_count": plan.size,
            "source": "module3_tree",
        }

    def _register_state(
        self,
        registry: Dict[Tuple[str, ...], FrontierState],
        state: FrontierState,
    ) -> None:
        if not state.path_ids:
            return

        key = tuple(state.path_ids)
        existed = registry.get(key)
        if existed is None or state.tree_score > existed.tree_score:
            registry[key] = state

    def _topological_requirement_order(self, plan: RequirementPlan) -> List[str]:
        covered: Tuple[str, ...] = ()
        ordered: List[str] = []
        seen = set()
        while True:
            frontier = plan.frontier(covered)
            frontier = [node for node in frontier if node.req_id not in seen]
            if not frontier:
                break
            next_node = frontier[0]
            ordered.append(next_node.req_id)
            seen.add(next_node.req_id)
            covered = self.requirement_grounder.mark_requirement_covered(covered, next_node.req_id)
        for node in plan.nodes:
            if node.req_id not in seen:
                ordered.append(node.req_id)
        return ordered

    def _score_frontier_expansions(
        self,
        *,
        state: FrontierState,
        frontier_req_ids: Sequence[str],
        query: torch.Tensor,
        candidate_lists: Dict[str, List[Tuple[str, float]]],
        score_lookup: Dict[str, Dict[str, float]],
        history_prior: Dict[str, float],
        predicted_steps: int,
        plan: RequirementPlan,
    ) -> Tuple[List[Tuple[float, FrontierState]], float]:
        scored_expansions: List[Tuple[float, FrontierState]] = []
        best_frontier_score = float("-inf")

        for req_id in frontier_req_ids:
            req_candidates = self.requirement_grounder.dependency_constrained_candidates(
                state.path_ids,
                candidate_lists.get(req_id, []),
            )
            if not req_candidates:
                continue

            candidate_ids_for_req: List[str] = []
            for service_node, match_score in req_candidates:
                service_idx = self.idx_map[service_node]
                mrg_logit = float(torch.matmul(query, self.service_embs[service_idx]).item())
                mrg_prob = float(torch.sigmoid(torch.tensor(mrg_logit)).item())
                hist_score = float(history_prior.get(service_node, 0.0))

                if not self.requirement_grounder.has_selection_evidence(
                    match_score=match_score,
                    mrg_prob=mrg_prob,
                    history_score=hist_score,
                    match_threshold=self.tree_match_threshold,
                    mrg_prob_threshold=self.tree_mrg_prob_threshold,
                ):
                    continue

                dep_rel = self.requirement_grounder.dependency_relation_score(
                    state.path_ids,
                    service_node,
                    root_admissible=True,
                )
                if dep_rel <= 0.0:
                    continue

                service_id = service_node.replace("Service:", "")
                candidate_ids_for_req.append(service_id)
                gain = max(0.0, float(score_lookup[req_id].get(service_node, match_score)))
                psi = self._compute_tree_score(
                    match_score=match_score,
                    mrg_prob=mrg_prob,
                    history_score=hist_score,
                    dep_rel=dep_rel,
                    gain=gain,
                    is_duplicate=service_id in state.path_ids,
                )
                best_frontier_score = max(best_frontier_score, psi)

                covered_req_ids = self.requirement_grounder.mark_requirement_covered(
                    state.covered_req_ids,
                    req_id,
                )
                frontier_req_ids_next = tuple(node.req_id for node in plan.frontier(covered_req_ids))
                child = FrontierState(
                    path_ids=state.path_ids + [service_id],
                    covered_req_ids=covered_req_ids,
                    frontier_req_ids=frontier_req_ids_next,
                    tree_score=state.tree_score + psi,
                    step_candidates=state.step_candidates + [candidate_ids_for_req[:]],
                    selected_requirements=state.selected_requirements + [req_id],
                )
                scored_expansions.append((psi, child))

        return scored_expansions, best_frontier_score

    def _collect_candidate_paths(
        self,
        states: Sequence[FrontierState],
        plan: RequirementPlan,
    ) -> List[Dict[str, object]]:
        ordered_states = list(states)
        ordered_states.sort(key=lambda item: item.tree_score, reverse=True)
        candidates = [
            self._state_to_candidate(state, f"path_{idx + 1}", plan)
            for idx, state in enumerate(ordered_states)
        ]
        selected = self._select_output_candidates(candidates, plan.size)
        if selected:
            return selected
        return candidates[: self.candidate_top_k]

    def _select_output_candidates(
        self,
        candidates: Sequence[Dict[str, object]],
        target_length: int,
    ) -> List[Dict[str, object]]:
        allowed_lengths = {target_length}
        if self.candidate_include_m_minus_1:
            allowed_lengths.add(max(1, target_length - 1))

        filtered = [
            candidate
            for candidate in candidates
            if len(candidate.get("path_ids", [])) in allowed_lengths
        ]
        filtered.sort(key=lambda item: float(item.get("tree_score", 0.0)), reverse=True)
        return filtered[: self.candidate_top_k]

    def _build_history_summary(
        self,
        matched_history,
        history_prior: Dict[str, float],
    ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
        matched_history_payload = [
            {"task_id": task_id, "similarity": float(sim)}
            for task_id, sim in matched_history
        ]
        history_prior_payload = [
            {
                "service_id": node.replace("Service:", ""),
                "score": float(score),
            }
            for node, score in sorted(history_prior.items(), key=lambda item: item[1], reverse=True)[:10]
        ]
        return matched_history_payload, history_prior_payload

    def _build_module3_result(
        self,
        *,
        search_mode: str,
        fallback_reason: str,
        plan: RequirementPlan,
        matched_history,
        history_prior: Dict[str, float],
        candidate_paths: Sequence[Dict[str, object]],
        predicted_steps: int,
        search_stats: Optional[Dict[str, object]] = None,
    ) -> Dict[str, Any]:
        best_candidate = candidate_paths[0] if candidate_paths else {}
        matched_history_payload, history_prior_payload = self._build_history_summary(
            matched_history,
            history_prior,
        )
        return {
            "search_mode": search_mode,
            "fallback_reason": fallback_reason,
            "predicted_steps": predicted_steps,
            "planner_source": plan.source,
            "planner_confidence": float(plan.confidence),
            "requirement_plan": self._build_dag_summary(plan),
            "matched_history": matched_history_payload,
            "history_prior_top_services": history_prior_payload,
            "candidate_paths": list(candidate_paths),
            "best_path_ids": list(best_candidate.get("path_ids", [])),
            "best_tree_score": float(best_candidate.get("tree_score", 0.0)),
            "pred_path": list(best_candidate.get("path_ids", [])),
            "candidates": list(best_candidate.get("step_candidates", [])),
            "search_stats": dict(search_stats or {}),
        }

    def _fallback_module3_result(
        self,
        *,
        task_text: str,
        predicted_steps: int,
        current_task_id: Optional[str],
        task_record: Optional[Dict[str, Any]],
        fallback_reason: str,
        plan: RequirementPlan,
        search_stats: Optional[Dict[str, object]] = None,
    ) -> Dict[str, Any]:
        beam_debug = super().search_service_path(
            task_text=task_text,
            pred_task_num=predicted_steps,
            return_debug=True,
            current_task_id=current_task_id,
            task_record=task_record,
        )
        path_ids = list(beam_debug.get("pred_path", []))
        candidate_paths = [
            {
                "path_id": "path_1",
                "path_ids": path_ids,
                "tree_score": float(beam_debug.get("score", 0.0)),
                "step_candidates": list(beam_debug.get("candidates", [])),
                "covered_req_ids": [],
                "frontier_req_ids": [],
                "selected_requirements": [],
                "covered_requirements": [],
                "service_details": self._build_service_details(path_ids),
                "dependency_evidence": self._build_dependency_evidence(path_ids),
                "total_requirement_count": plan.size,
                "source": "beam_fallback",
            }
        ]
        return {
            "search_mode": "beam_fallback",
            "fallback_reason": fallback_reason,
            "predicted_steps": predicted_steps,
            "planner_source": plan.source,
            "planner_confidence": float(plan.confidence),
            "requirement_plan": self._build_dag_summary(plan),
            "matched_history": list(beam_debug.get("matched_history", [])),
            "history_prior_top_services": list(beam_debug.get("history_prior_top_services", [])),
            "candidate_paths": candidate_paths,
            "best_path_ids": path_ids,
            "best_tree_score": float(beam_debug.get("score", 0.0)),
            "pred_path": path_ids,
            "candidates": list(beam_debug.get("candidates", [])),
            "search_stats": dict(search_stats or {}),
        }

    def _run_greedy_frontier_search(
        self,
        *,
        task_emb: torch.Tensor,
        task_text: str,
        current_task_id: Optional[str],
        task_record: Optional[Dict[str, Any]],
        plan: RequirementPlan,
        predicted_steps: int,
        history_prior: Dict[str, float],
        matched_history,
        candidate_lists: Dict[str, List[Tuple[str, float]]],
        score_lookup: Dict[str, Dict[str, float]],
        base_search_stats: Dict[str, object],
    ) -> Dict[str, Any]:
        state = FrontierState(
            path_ids=[],
            covered_req_ids=(),
            frontier_req_ids=tuple(node.req_id for node in plan.frontier(())),
            tree_score=0.0,
            step_candidates=[],
            selected_requirements=[],
        )
        expanded = 0
        generated_children = 0
        kept_children = 0
        peak_frontier_size = len(state.frontier_req_ids)
        loop_start = time.perf_counter()

        while state.frontier_req_ids and expanded < max(plan.size, predicted_steps):
            peak_frontier_size = max(peak_frontier_size, len(state.frontier_req_ids))
            step_idx = len(state.path_ids) + 1
            query = self._make_query(task_emb, state.path_ids, step_idx, predicted_steps)
            scored_expansions, _ = self._score_frontier_expansions(
                state=state,
                frontier_req_ids=state.frontier_req_ids,
                query=query,
                candidate_lists=candidate_lists,
                score_lookup=score_lookup,
                history_prior=history_prior,
                predicted_steps=predicted_steps,
                plan=plan,
            )
            if not scored_expansions:
                break

            generated_children += len(scored_expansions)
            scored_expansions.sort(key=lambda item: item[0], reverse=True)
            state = scored_expansions[0][1]
            kept_children += 1
            expanded += 1

        loop_end = time.perf_counter()
        candidate_paths = self._collect_candidate_paths([state], plan)
        search_stats = {
            **base_search_stats,
            "search_mode": "greedy_frontier",
            "tree_budget": int(max(plan.size, predicted_steps)),
            "budget_exhausted": bool(expanded >= max(plan.size, predicted_steps)),
            "search_loop_time_ms": float((loop_end - loop_start) * 1000.0),
            "expanded_states": int(expanded),
            "completed_states": int(1 if len(state.covered_req_ids) >= plan.size else 0),
            "generated_children": int(generated_children),
            "kept_children": int(kept_children),
            "peak_current_layer": 1,
            "peak_next_layer": 1 if kept_children > 0 else 0,
            "peak_candidate_registry": int(1 if candidate_paths else 0),
            "peak_frontier_size": int(peak_frontier_size),
            "candidate_pool_size": 1,
            "candidate_path_count": int(len(candidate_paths)),
            "frontier_pruned": 0,
            "dominated_pruned": 0,
            "dominance_cache_size": 0,
        }
        if not candidate_paths:
            return self._fallback_module3_result(
                task_text=task_text,
                current_task_id=current_task_id,
                task_record=task_record,
                fallback_reason="greedy_search_empty",
                predicted_steps=predicted_steps,
                plan=plan,
                search_stats={
                    **search_stats,
                    "search_mode": "beam_fallback",
                    "fallback_used": True,
                    "fallback_reason": "greedy_search_empty",
                },
            )

        return self._build_module3_result(
            search_mode="greedy_frontier",
            fallback_reason="",
            plan=plan,
            matched_history=matched_history,
            history_prior=history_prior,
            candidate_paths=candidate_paths,
            predicted_steps=predicted_steps,
            search_stats=search_stats,
        )

    def _run_topological_sequential_search(
        self,
        *,
        task_emb: torch.Tensor,
        task_text: str,
        current_task_id: Optional[str],
        task_record: Optional[Dict[str, Any]],
        plan: RequirementPlan,
        predicted_steps: int,
        history_prior: Dict[str, float],
        matched_history,
        candidate_lists: Dict[str, List[Tuple[str, float]]],
        score_lookup: Dict[str, Dict[str, float]],
        base_search_stats: Dict[str, object],
    ) -> Dict[str, Any]:
        order = self._topological_requirement_order(plan)
        state = FrontierState(
            path_ids=[],
            covered_req_ids=(),
            frontier_req_ids=tuple(order),
            tree_score=0.0,
            step_candidates=[],
            selected_requirements=[],
        )
        expanded = 0
        generated_children = 0
        kept_children = 0
        peak_frontier_size = len(order)
        loop_start = time.perf_counter()

        for req_id in order:
            step_idx = len(state.path_ids) + 1
            query = self._make_query(task_emb, state.path_ids, step_idx, predicted_steps)
            single_frontier = [req_id]
            scored_expansions, _ = self._score_frontier_expansions(
                state=state,
                frontier_req_ids=single_frontier,
                query=query,
                candidate_lists=candidate_lists,
                score_lookup=score_lookup,
                history_prior=history_prior,
                predicted_steps=predicted_steps,
                plan=plan,
            )
            expanded += 1
            if not scored_expansions:
                continue

            generated_children += len(scored_expansions)
            scored_expansions.sort(key=lambda item: item[0], reverse=True)
            state = scored_expansions[0][1]
            kept_children += 1

        loop_end = time.perf_counter()
        candidate_paths = self._collect_candidate_paths([state], plan)
        search_stats = {
            **base_search_stats,
            "search_mode": "topological_sequential",
            "tree_budget": int(max(len(order), predicted_steps)),
            "budget_exhausted": False,
            "search_loop_time_ms": float((loop_end - loop_start) * 1000.0),
            "expanded_states": int(expanded),
            "completed_states": int(1 if len(state.covered_req_ids) >= plan.size else 0),
            "generated_children": int(generated_children),
            "kept_children": int(kept_children),
            "peak_current_layer": 1,
            "peak_next_layer": 1 if kept_children > 0 else 0,
            "peak_candidate_registry": int(1 if candidate_paths else 0),
            "peak_frontier_size": int(peak_frontier_size),
            "candidate_pool_size": 1,
            "candidate_path_count": int(len(candidate_paths)),
            "frontier_pruned": 0,
            "dominated_pruned": 0,
            "dominance_cache_size": 0,
            "topological_order": list(order),
        }
        if not candidate_paths:
            return self._fallback_module3_result(
                task_text=task_text,
                current_task_id=current_task_id,
                task_record=task_record,
                fallback_reason="topological_search_empty",
                predicted_steps=predicted_steps,
                plan=plan,
                search_stats={
                    **search_stats,
                    "search_mode": "beam_fallback",
                    "fallback_used": True,
                    "fallback_reason": "topological_search_empty",
                },
            )

        return self._build_module3_result(
            search_mode="topological_sequential",
            fallback_reason="",
            plan=plan,
            matched_history=matched_history,
            history_prior=history_prior,
            candidate_paths=candidate_paths,
            predicted_steps=predicted_steps,
            search_stats=search_stats,
        )

    def _resolve_predicted_steps(
        self,
        pred_task_num: int,
        task_record: Optional[Dict[str, Any]],
    ) -> int:
        if task_record:
            nodes = task_record.get("nodes")
            if isinstance(nodes, list) and nodes:
                return max(1, len(nodes))
        return max(1, int(pred_task_num))

    @torch.no_grad()
    def search_candidate_paths(
        self,
        task_text: str,
        pred_task_num: int,
        current_task_id: Optional[str] = None,
        task_record: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        predicted_steps = self._resolve_predicted_steps(pred_task_num, task_record)
        plan = self.requirement_planner.build_plan(task_text, predicted_steps, task_record=task_record)

        base_search_stats: Dict[str, object] = {
            "search_mode": self.search_variant,
            "predicted_steps": int(predicted_steps),
            "plan_size": int(plan.size),
            "planner_confidence": float(plan.confidence),
            "tree_budget": 0,
            "budget_exhausted": False,
            "search_loop_time_ms": 0.0,
            "expanded_states": 0,
            "completed_states": 0,
            "generated_children": 0,
            "kept_children": 0,
            "peak_current_layer": 0,
            "peak_next_layer": 0,
            "peak_candidate_registry": 0,
            "peak_frontier_size": 0,
            "candidate_pool_size": 0,
            "candidate_path_count": 0,
        }

        if plan.size < self.requirement_min_plan_size or plan.confidence <= 0.0:
            return self._fallback_module3_result(
                task_text=task_text,
                current_task_id=current_task_id,
                task_record=task_record,
                fallback_reason=f"planner_confidence={plan.confidence:.2f}, plan_size={plan.size}",
                predicted_steps=predicted_steps,
                plan=plan,
                search_stats={
                    **base_search_stats,
                    "search_mode": "beam_fallback",
                    "fallback_used": True,
                    "fallback_reason": f"planner_confidence={plan.confidence:.2f}, plan_size={plan.size}",
                },
            )

        task_emb = self.task_proj(self._embed_text(task_text).unsqueeze(0))
        history_prior, matched_history = self._get_history_prior(task_text, current_task_id=current_task_id)
        candidate_pool = self.requirement_grounder.extract_candidate_pool(task_record)
        candidate_service_nodes = self.requirement_grounder.resolve_candidate_service_nodes(candidate_pool)
        candidate_lists, score_lookup = self.requirement_grounder.ground_plan(
            plan,
            history_prior,
            candidate_service_nodes=candidate_service_nodes,
        )
        if not any(candidate_lists.values()):
            return self._fallback_module3_result(
                task_text=task_text,
                current_task_id=current_task_id,
                task_record=task_record,
                fallback_reason="no_requirement_candidates",
                predicted_steps=predicted_steps,
                plan=plan,
                search_stats={
                    **base_search_stats,
                    "search_mode": "beam_fallback",
                    "fallback_used": True,
                    "fallback_reason": "no_requirement_candidates",
                },
            )

        if self.search_variant == "greedy_frontier":
            return self._run_greedy_frontier_search(
                task_emb=task_emb,
                task_text=task_text,
                current_task_id=current_task_id,
                task_record=task_record,
                plan=plan,
                predicted_steps=predicted_steps,
                history_prior=history_prior,
                matched_history=matched_history,
                candidate_lists=candidate_lists,
                score_lookup=score_lookup,
                base_search_stats=base_search_stats,
            )

        if self.search_variant == "topological_sequential":
            return self._run_topological_sequential_search(
                task_emb=task_emb,
                task_text=task_text,
                current_task_id=current_task_id,
                task_record=task_record,
                plan=plan,
                predicted_steps=predicted_steps,
                history_prior=history_prior,
                matched_history=matched_history,
                candidate_lists=candidate_lists,
                score_lookup=score_lookup,
                base_search_stats=base_search_stats,
            )

        self.tree_pruner.reset()
        tree_budget = max(plan.size * self.tree_budget_multiplier, predicted_steps * self.tree_budget_multiplier)
        search_stats = dict(base_search_stats)
        search_stats["tree_budget"] = int(tree_budget)
        root = FrontierState(
            path_ids=[],
            covered_req_ids=(),
            frontier_req_ids=tuple(node.req_id for node in plan.frontier(())),
            tree_score=0.0,
            step_candidates=[],
            selected_requirements=[],
        )
        current_layer: List[FrontierState] = [root]
        completed_states: List[FrontierState] = []
        candidate_registry: Dict[Tuple[str, ...], FrontierState] = {}
        expanded = 0
        generated_children = 0
        kept_children = 0
        peak_current_layer = len(current_layer)
        peak_next_layer = 0
        peak_candidate_registry = 0
        peak_frontier_size = len(root.frontier_req_ids)
        loop_start = time.perf_counter()

        while current_layer and expanded < tree_budget:
            current_layer.sort(key=lambda item: item.tree_score, reverse=True)
            current_layer = current_layer[: self.tree_queue_size]
            peak_current_layer = max(peak_current_layer, len(current_layer))

            next_layer: List[FrontierState] = []
            for state in current_layer:
                if expanded >= tree_budget:
                    break

                peak_frontier_size = max(peak_frontier_size, len(state.frontier_req_ids))

                if self.tree_pruner.is_dominated(
                    covered_req_ids=state.covered_req_ids,
                    frontier_req_ids=state.frontier_req_ids,
                    path_ids=state.path_ids,
                    tree_score=state.tree_score,
                ):
                    expanded += 1
                    continue

                if not state.frontier_req_ids and len(state.covered_req_ids) >= plan.size:
                    completed_states.append(state)
                    self._register_state(candidate_registry, state)
                    peak_candidate_registry = max(peak_candidate_registry, len(candidate_registry))
                    expanded += 1
                    continue

                scored_expansions: List[Tuple[float, FrontierState]] = []
                best_frontier_score = float("-inf")
                step_idx = len(state.path_ids) + 1
                query = self._make_query(task_emb, state.path_ids, step_idx, predicted_steps)

                for req_id in state.frontier_req_ids:
                    req_candidates = self.requirement_grounder.dependency_constrained_candidates(
                        state.path_ids,
                        candidate_lists.get(req_id, []),
                    )
                    if not req_candidates:
                        continue

                    candidate_ids_for_req: List[str] = []
                    for service_node, match_score in req_candidates:
                        service_idx = self.idx_map[service_node]
                        mrg_logit = float(torch.matmul(query, self.service_embs[service_idx]).item())
                        mrg_prob = float(torch.sigmoid(torch.tensor(mrg_logit)).item())
                        hist_score = float(history_prior.get(service_node, 0.0))

                        if not self.requirement_grounder.has_selection_evidence(
                            match_score=match_score,
                            mrg_prob=mrg_prob,
                            history_score=hist_score,
                            match_threshold=self.tree_match_threshold,
                            mrg_prob_threshold=self.tree_mrg_prob_threshold,
                        ):
                            continue

                        dep_rel = self.requirement_grounder.dependency_relation_score(
                            state.path_ids,
                            service_node,
                            root_admissible=True,
                        )
                        if dep_rel <= 0.0:
                            continue

                        candidate_ids_for_req.append(service_node.replace("Service:", ""))
                        gain = max(0.0, float(score_lookup[req_id].get(service_node, match_score)))
                        psi = self._compute_tree_score(
                            match_score=match_score,
                            mrg_prob=mrg_prob,
                            history_score=hist_score,
                            dep_rel=dep_rel,
                            gain=gain,
                            is_duplicate=service_node.replace("Service:", "") in state.path_ids,
                        )
                        best_frontier_score = max(best_frontier_score, psi)

                        covered_req_ids = self.requirement_grounder.mark_requirement_covered(
                            state.covered_req_ids,
                            req_id,
                        )
                        frontier_req_ids = tuple(node.req_id for node in plan.frontier(covered_req_ids))
                        child = FrontierState(
                            path_ids=state.path_ids + [service_node.replace("Service:", "")],
                            covered_req_ids=covered_req_ids,
                            frontier_req_ids=frontier_req_ids,
                            tree_score=state.tree_score + psi,
                            step_candidates=state.step_candidates + [candidate_ids_for_req[:]],
                            selected_requirements=state.selected_requirements + [req_id],
                        )
                        scored_expansions.append((psi, child))

                if self.tree_pruner.should_prune_frontier(
                    frontier_req_ids=state.frontier_req_ids,
                    covered_req_count=len(state.covered_req_ids),
                    total_req_count=plan.size,
                    best_frontier_score=best_frontier_score,
                ):
                    expanded += 1
                    continue

                if not scored_expansions:
                    expanded += 1
                    continue

                generated_children += len(scored_expansions)
                scored_expansions.sort(key=lambda item: item[0], reverse=True)
                top_children = [child for _, child in scored_expansions[: self.tree_top_b]]
                kept_children += len(top_children)
                next_layer.extend(top_children)
                peak_next_layer = max(peak_next_layer, len(next_layer))
                expanded += 1

            next_layer.sort(key=lambda item: item.tree_score, reverse=True)
            next_layer = next_layer[: self.tree_queue_size]
            for state in next_layer:
                self._register_state(candidate_registry, state)
            peak_candidate_registry = max(peak_candidate_registry, len(candidate_registry))
            current_layer = next_layer

        loop_end = time.perf_counter()
        search_stats.update(
            {
                "budget_exhausted": bool(expanded >= tree_budget),
                "search_loop_time_ms": float((loop_end - loop_start) * 1000.0),
                "expanded_states": int(expanded),
                "completed_states": int(len(completed_states)),
                "generated_children": int(generated_children),
                "kept_children": int(kept_children),
                "peak_current_layer": int(peak_current_layer),
                "peak_next_layer": int(peak_next_layer),
                "peak_candidate_registry": int(peak_candidate_registry),
                "peak_frontier_size": int(peak_frontier_size),
            }
        )
        search_stats.update(self.tree_pruner.export_stats())

        candidate_pool = completed_states + list(candidate_registry.values()) + current_layer
        search_stats["candidate_pool_size"] = int(len(candidate_pool))
        candidate_states = self._collect_candidate_paths(candidate_pool, plan)
        search_stats["candidate_path_count"] = int(len(candidate_states))
        if not candidate_states:
            return self._fallback_module3_result(
                task_text=task_text,
                current_task_id=current_task_id,
                task_record=task_record,
                fallback_reason="tree_search_empty",
                predicted_steps=predicted_steps,
                plan=plan,
                search_stats={
                    **search_stats,
                    "search_mode": "beam_fallback",
                    "fallback_used": True,
                    "fallback_reason": "tree_search_empty",
                },
            )

        return self._build_module3_result(
            search_mode=self.search_variant,
            fallback_reason="",
            plan=plan,
            matched_history=matched_history,
            history_prior=history_prior,
            candidate_paths=candidate_states,
            predicted_steps=predicted_steps,
            search_stats=search_stats,
        )

    @torch.no_grad()
    def search_service_path(
        self,
        task_text: str,
        pred_task_num: int,
        return_debug: bool = False,
        current_task_id: Optional[str] = None,
        task_record: Optional[Dict[str, Any]] = None,
    ):
        result = self.search_candidate_paths(
            task_text=task_text,
            pred_task_num=pred_task_num,
            current_task_id=current_task_id,
            task_record=task_record,
        )
        if return_debug:
            return result
        return [self._tool_payload(sid) for sid in result.get("best_path_ids", [])]

    def _path_executability_soft(self, path_ids: Sequence[str]) -> float:
        if len(path_ids) <= 1:
            return 1.0

        satisfied = 0
        total = len(path_ids) - 1
        for idx in range(1, len(path_ids)):
            current_node = f"Service:{path_ids[idx]}"
            ok = False
            for prev_idx in range(idx):
                prev_node = f"Service:{path_ids[prev_idx]}"
                if current_node in self.dep_neighbors.get(prev_node, set()):
                    ok = True
                    break
            if ok:
                satisfied += 1
        return satisfied / float(total)
