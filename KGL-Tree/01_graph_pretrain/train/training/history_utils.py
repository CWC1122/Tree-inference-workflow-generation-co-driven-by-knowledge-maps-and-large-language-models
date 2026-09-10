from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from .data_utils import TaskRecord


@dataclass
class HistoryTask:
    task_id: str
    task_text: str
    service_counts: Dict[str, int]


def build_history_tasks(
    task_records: Dict[str, TaskRecord],
    allowed_task_ids: Set[str],
    uses_by_task: Dict[str, List[str]],
    idx_map: Dict[str, int],
) -> List[HistoryTask]:
    history_tasks: List[HistoryTask] = []

    for task_id in sorted(allowed_task_ids):
        record = task_records.get(task_id)
        if record is None:
            continue

        service_counts = defaultdict(int)
        use_list = list(uses_by_task.get(task_id, [])) or list(record.action_id_list)
        for service_id in use_list:
            node = f"Service:{service_id}"
            if node in idx_map:
                service_counts[node] += 1

        history_tasks.append(
            HistoryTask(
                task_id=task_id,
                task_text=record.task_text,
                service_counts=dict(service_counts),
            )
        )

    return history_tasks


class SimilarTaskRetriever:
    def __init__(
        self,
        history_tasks: Iterable[HistoryTask],
        task_cache: Dict[str, torch.Tensor],
        topk: int,
        device: torch.device,
    ):
        self.topk = max(1, int(topk))
        self.device = device
        self.history_tasks = list(history_tasks)
        self.task_ids = [task.task_id for task in self.history_tasks]
        self.service_counts = [task.service_counts for task in self.history_tasks]
        self.task_index = {task_id: idx for idx, task_id in enumerate(self.task_ids)}
        self.prior_cache: Dict[Tuple[Optional[str], str], Dict[str, float]] = {}

        if self.history_tasks:
            embeddings = [task_cache[task.task_text] for task in self.history_tasks]
            self.task_embs = F.normalize(torch.stack(embeddings, dim=0), dim=1)
        else:
            self.task_embs = None

    def get_service_prior(self, task_id: Optional[str], task_text: str, task_cache: Dict[str, torch.Tensor]) -> Dict[str, float]:
        cache_key = (task_id, task_text)
        if cache_key in self.prior_cache:
            return self.prior_cache[cache_key]

        if self.task_embs is None or self.task_embs.numel() == 0 or task_text not in task_cache:
            self.prior_cache[cache_key] = {}
            return {}

        query_emb = F.normalize(task_cache[task_text].unsqueeze(0), dim=1)
        sims = torch.matmul(query_emb, self.task_embs.t()).squeeze(0)

        if task_id is not None and task_id in self.task_index:
            sims[self.task_index[task_id]] = -1e9

        ranked = torch.argsort(sims, descending=True).tolist()
        prior = defaultdict(float)
        used = 0

        for pos in ranked:
            if used >= self.topk:
                break

            sim = float(sims[pos].item())
            if sim <= 0.0:
                continue

            for node, count in self.service_counts[pos].items():
                prior[node] += sim * float(count)
            used += 1

        if prior:
            max_score = max(prior.values())
            norm_prior = {node: score / max_score for node, score in prior.items()}
        else:
            norm_prior = {}

        self.prior_cache[cache_key] = norm_prior
        return norm_prior
