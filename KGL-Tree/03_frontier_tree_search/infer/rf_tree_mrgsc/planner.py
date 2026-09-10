import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_TEXT_KEYS = (
    "text",
    "requirement",
    "description",
    "name",
    "title",
    "subtask",
)
_NODE_KEYS = (
    "requirement_dag",
    "requirement_plan",
    "requirement_nodes",
    "nodes",
    "subtasks",
    "subtask_list",
)
_EDGE_KEYS = (
    "edges",
    "requirement_edges",
    "dependencies",
)


@dataclass(frozen=True)
class RequirementNode:
    req_id: str
    text: str
    predecessors: Tuple[str, ...] = ()
    expected_inputs: Tuple[str, ...] = ()
    expected_outputs: Tuple[str, ...] = ()


@dataclass
class RequirementPlan:
    nodes: List[RequirementNode]
    source: str
    confidence: float
    task_text: str = ""
    node_map: Dict[str, RequirementNode] = field(init=False)
    successors: Dict[str, List[str]] = field(init=False)

    def __post_init__(self) -> None:
        self.node_map = {node.req_id: node for node in self.nodes}
        self.successors = {node.req_id: [] for node in self.nodes}
        for node in self.nodes:
            for pred in node.predecessors:
                if pred in self.successors:
                    self.successors[pred].append(node.req_id)

    @property
    def size(self) -> int:
        return len(self.nodes)

    @property
    def req_ids(self) -> List[str]:
        return [node.req_id for node in self.nodes]

    def frontier(self, covered_ids: Iterable[str]) -> List[RequirementNode]:
        covered = set(covered_ids)
        ready: List[RequirementNode] = []
        for node in self.nodes:
            if node.req_id in covered:
                continue
            if all(pred in covered for pred in node.predecessors):
                ready.append(node)
        return ready

    def covered_ratio(self, covered_ids: Iterable[str]) -> float:
        total = max(1, self.size)
        return len(set(covered_ids)) / float(total)


class RequirementPlanner:
    def __init__(self, min_plan_size: int = 2, ignore_edges: bool = False):
        self.min_plan_size = max(1, int(min_plan_size))
        self.ignore_edges = bool(ignore_edges)

    def build_plan(
        self,
        task_text: str,
        predicted_steps: int,
        task_record: Optional[Dict[str, Any]] = None,
    ) -> RequirementPlan:
        explicit = self._build_explicit_plan(task_text, predicted_steps, task_record or {})
        if explicit is not None:
            return explicit
        return self._build_heuristic_plan(task_text, predicted_steps)

    def _build_explicit_plan(
        self,
        task_text: str,
        predicted_steps: int,
        task_record: Dict[str, Any],
    ) -> Optional[RequirementPlan]:
        raw_nodes = None
        raw_edges = None
        source = ""

        for key in _NODE_KEYS:
            value = task_record.get(key)
            if not value:
                continue
            source = key
            if key == "requirement_dag":
                dag = self._coerce_json(value)
                if isinstance(dag, dict):
                    raw_nodes = dag.get("nodes") or dag.get("requirements")
                    raw_edges = dag.get("edges") or dag.get("links")
                elif isinstance(dag, list):
                    raw_nodes = dag
            else:
                raw_nodes = self._coerce_json(value)
            break

        if raw_nodes is None:
            return None

        nodes = self._normalize_nodes(raw_nodes)
        if not nodes:
            return None

        if raw_edges is None:
            for edge_key in _EDGE_KEYS:
                if task_record.get(edge_key):
                    raw_edges = self._coerce_json(task_record[edge_key])
                    break

        edges = self._normalize_edges(raw_edges, nodes)
        plan_nodes = self._build_plan_nodes(nodes, edges)
        if not plan_nodes:
            return None

        confidence = 1.0 if len(plan_nodes) >= self.min_plan_size else 0.8
        return RequirementPlan(
            nodes=plan_nodes,
            source=f"explicit:{source}",
            confidence=confidence,
            task_text=task_text,
        )

    def _build_heuristic_plan(self, task_text: str, predicted_steps: int) -> RequirementPlan:
        target_steps = max(1, int(predicted_steps))
        segments = self._split_task_text(task_text)
        if len(segments) > target_steps:
            segments = self._merge_segments(segments, target_steps)

        if len(segments) <= 1:
            confidence = 0.0
        else:
            confidence = min(0.65, 0.35 + 0.1 * len(segments))

        nodes: List[RequirementNode] = []
        for idx, segment in enumerate(segments or [task_text.strip() or "complete the task"]):
            req_id = f"R{idx + 1}"
            predecessors = () if idx == 0 else (f"R{idx}",)
            nodes.append(
                RequirementNode(
                    req_id=req_id,
                    text=segment,
                    predecessors=predecessors,
                )
            )

        return RequirementPlan(
            nodes=nodes,
            source="heuristic",
            confidence=confidence,
            task_text=task_text,
        )

    def _coerce_json(self, value: Any) -> Any:
        if isinstance(value, (dict, list)):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                return json.loads(text)
            except Exception:
                return value
        return value

    def _normalize_nodes(self, raw_nodes: Any) -> List[Dict[str, Any]]:
        if isinstance(raw_nodes, dict):
            raw_nodes = raw_nodes.get("nodes") or raw_nodes.get("requirements") or []

        nodes: List[Dict[str, Any]] = []
        if not isinstance(raw_nodes, list):
            return nodes

        for idx, item in enumerate(raw_nodes):
            if isinstance(item, str):
                text = item.strip()
                if not text:
                    continue
                nodes.append({"id": f"R{idx + 1}", "text": text})
                continue

            if not isinstance(item, dict):
                continue

            node_id = str(item.get("id") or item.get("req_id") or f"R{idx + 1}")
            text = self._pick_text(item)
            if not text:
                continue
            nodes.append(
                {
                    "id": node_id,
                    "text": text,
                    "predecessors": self._to_string_list(
                        item.get("predecessors") or item.get("parents") or item.get("deps") or []
                    ),
                    "expected_inputs": self._to_string_list(
                        item.get("expected_inputs") or item.get("inputs") or []
                    ),
                    "expected_outputs": self._to_string_list(
                        item.get("expected_outputs") or item.get("outputs") or []
                    ),
                }
            )
        return nodes

    def _normalize_edges(
        self,
        raw_edges: Any,
        nodes: Sequence[Dict[str, Any]],
    ) -> List[Tuple[str, str]]:
        if self.ignore_edges:
            return []

        node_ids = {str(node["id"]) for node in nodes}
        edges: List[Tuple[str, str]] = []

        if isinstance(raw_edges, list):
            for item in raw_edges:
                if isinstance(item, dict):
                    src = str(item.get("source") or item.get("from") or item.get("src") or "")
                    dst = str(item.get("target") or item.get("to") or item.get("dst") or "")
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    src = str(item[0])
                    dst = str(item[1])
                else:
                    continue

                if src in node_ids and dst in node_ids:
                    edges.append((src, dst))

        if edges:
            return edges

        if any(node.get("predecessors") for node in nodes):
            for node in nodes:
                for pred in node.get("predecessors", []):
                    if pred in node_ids:
                        edges.append((pred, str(node["id"])))
            return edges

        for idx in range(1, len(nodes)):
            edges.append((str(nodes[idx - 1]["id"]), str(nodes[idx]["id"])))
        return edges

    def _build_plan_nodes(
        self,
        nodes: Sequence[Dict[str, Any]],
        edges: Sequence[Tuple[str, str]],
    ) -> List[RequirementNode]:
        predecessors: Dict[str, List[str]] = {str(node["id"]): [] for node in nodes}
        for src, dst in edges:
            if dst in predecessors and src not in predecessors[dst]:
                predecessors[dst].append(src)

        plan_nodes: List[RequirementNode] = []
        for node in nodes:
            req_id = str(node["id"])
            plan_nodes.append(
                RequirementNode(
                    req_id=req_id,
                    text=str(node["text"]),
                    predecessors=tuple(predecessors.get(req_id, [])),
                    expected_inputs=tuple(node.get("expected_inputs", [])),
                    expected_outputs=tuple(node.get("expected_outputs", [])),
                )
            )
        return plan_nodes

    def _pick_text(self, item: Dict[str, Any]) -> str:
        for key in _TEXT_KEYS:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _to_string_list(self, value: Any) -> List[str]:
        if isinstance(value, str):
            parsed = self._coerce_json(value)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
            if value.strip():
                return [part.strip() for part in re.split(r"[,;/|]+", value) if part.strip()]
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    def _split_task_text(self, task_text: str) -> List[str]:
        text = (task_text or "").strip()
        if not text:
            return []

        normalized = re.sub(r"\s+", " ", text)
        separators = [
            r"\bthen\b",
            r"\band then\b",
            r"\bafter that\b",
            r"\bbefore\b",
            r"\bafter\b",
            r"\band\b",
            r"\bwith\b",
            r"\busing\b",
            r";",
            r",",
        ]
        pattern = "|".join(separators)
        pieces = [part.strip(" .") for part in re.split(pattern, normalized, flags=re.IGNORECASE) if part.strip(" .")]

        cleaned: List[str] = []
        for part in pieces:
            if len(part.split()) < 2 and cleaned:
                cleaned[-1] = f"{cleaned[-1]} {part}".strip()
                continue
            cleaned.append(part)
        return cleaned

    def _merge_segments(self, segments: Sequence[str], target_steps: int) -> List[str]:
        if target_steps <= 0 or len(segments) <= target_steps:
            return list(segments)

        chunk_size = int(math.ceil(len(segments) / float(target_steps)))
        merged: List[str] = []
        for idx in range(0, len(segments), chunk_size):
            merged.append(" ; ".join(segments[idx : idx + chunk_size]))
        return merged[:target_steps]
