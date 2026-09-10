from typing import Dict, List, Sequence, Tuple


class FrontierTreePruner:
    def __init__(
        self,
        *,
        dep_neighbors,
        frontier_score_threshold: float = 0.0,
        enable_state_dominance: bool = True,
        use_dep_sig: bool = True,
    ):
        self.dep_neighbors = dep_neighbors
        self.frontier_score_threshold = float(frontier_score_threshold)
        self.enable_state_dominance = bool(enable_state_dominance)
        self.use_dep_sig = bool(use_dep_sig)
        self.reset()

    def reset(self) -> None:
        self._dominance_best: Dict[Tuple[object, ...], float] = {}
        self.frontier_pruned = 0
        self.dominated_pruned = 0

    def should_prune_frontier(
        self,
        frontier_req_ids: Sequence[str],
        covered_req_count: int,
        total_req_count: int,
        best_frontier_score: float,
    ) -> bool:
        if not frontier_req_ids and covered_req_count < total_req_count:
            self.frontier_pruned += 1
            return True
        if frontier_req_ids and best_frontier_score < self.frontier_score_threshold:
            self.frontier_pruned += 1
            return True
        return False

    def _dep_signature(self, path_ids: Sequence[str]) -> Tuple[str, ...]:
        support = set()
        for service_id in path_ids:
            support.update(self.dep_neighbors.get(f"Service:{service_id}", set()))
        return tuple(sorted(node.replace("Service:", "") for node in support))

    def build_state_key(
        self,
        covered_req_ids: Sequence[str],
        frontier_req_ids: Sequence[str],
        path_ids: Sequence[str],
    ) -> Tuple[object, ...]:
        key: List[object] = [
            tuple(sorted(covered_req_ids)),
            tuple(sorted(frontier_req_ids)),
            path_ids[-1] if path_ids else "",
            len(path_ids),
        ]
        if self.use_dep_sig:
            key.append(self._dep_signature(path_ids))
        return tuple(key)

    def is_dominated(
        self,
        covered_req_ids: Sequence[str],
        frontier_req_ids: Sequence[str],
        path_ids: Sequence[str],
        tree_score: float,
    ) -> bool:
        if not self.enable_state_dominance:
            return False

        key = self.build_state_key(covered_req_ids, frontier_req_ids, path_ids)
        best = self._dominance_best.get(key)
        if best is not None and best >= tree_score:
            self.dominated_pruned += 1
            return True
        self._dominance_best[key] = tree_score
        return False

    def export_stats(self) -> Dict[str, int]:
        return {
            "frontier_pruned": int(self.frontier_pruned),
            "dominated_pruned": int(self.dominated_pruned),
            "dominance_cache_size": int(len(self._dominance_best)),
        }
