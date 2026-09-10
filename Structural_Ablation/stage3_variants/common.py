import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


VARIANT_DIR = Path(__file__).resolve().parent
STRUCTURAL_ROOT = VARIANT_DIR.parent
PROJECT_ROOT = STRUCTURAL_ROOT.parent
STAGE3_ROOT = PROJECT_ROOT / "02_stage1_stage3_dag" / "stage3"
MODULE_ROOT = PROJECT_ROOT / "02_stage1_stage3_dag"

for path in (str(STAGE3_ROOT), str(MODULE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from localLLM.localLLM import localLLM
from utils import extract_json, load_dependency_types, fix_json_string


class BaseStructuralDAGGenerator:
    planning_mode = "linear"
    length_mode = "soft"
    use_service_awareness = True

    def __init__(self, tool_desc_path: str):
        self.dep_types = load_dependency_types(tool_desc_path)
        self.service_dict = self._load_service_dict(tool_desc_path)

    def _load_service_dict(self, path: str) -> Dict[str, Dict]:
        with open(path, "r", encoding="utf-8") as file_obj:
            services = json.load(file_obj)
        return {service["action_uid"]: service for service in services}

    def build_node_template(self, task_num: int) -> List[Dict]:
        return [{"id": index + 1, "requirement": ""} for index in range(max(1, int(task_num)))]

    def _service_context(self, service_ids: Optional[List[str]]) -> Tuple[str, str]:
        blocks = []
        selected_services = []
        for service_id in service_ids or []:
            service = self.service_dict.get(service_id)
            if not service:
                continue
            selected_services.append(service)
            blocks.append(
                "\n".join(
                    [
                        f"Service ID: {service['action_uid']}",
                        f"Description: {service.get('target_action_reprs', '')}",
                        f"Input: {service.get('input-type', [])}",
                        f"Output: {service.get('output-type', [])}",
                    ]
                )
            )

        has_io = any(("input-type" in service and "output-type" in service) for service in selected_services)
        dep_str = ", ".join(self.dep_types) if has_io else ""
        return ("\n\n".join(blocks) if blocks else "None"), dep_str

    def _build_length_guidance(self, task_num: int) -> str:
        if self.length_mode == "none":
            return """
You are NOT given any estimated number of subtasks in this variant.

Rules:
1. Decide the number of subtasks yourself.
2. Use as few subtasks as necessary to solve the task well.
3. Number nodes starting from 1.
4. Node IDs must be consecutive with no gaps.
5. Every subtask must contribute directly to the user task.
"""

        node_template = json.dumps(self.build_node_template(task_num), ensure_ascii=False, indent=2)
        if self.length_mode == "hard":
            return f"""
You are given an exact target number of subtasks: {task_num}.

IMPORTANT:
- You MUST generate exactly {task_num} meaningful subtasks.
- You MUST use the node template exactly as provided.
- You MUST NOT leave nodes unused.
- You MUST NOT create fewer or more subtasks than {task_num}.

Node template:
{node_template}
"""

        return f"""
You are given an estimated number of subtasks: {task_num}.

IMPORTANT:
- This number is ONLY a reference derived from similar historical tasks.
- It is NOT a strict requirement.

Rules:
1. If the task can be completed with fewer subtasks, use fewer nodes.
2. If the task requires more subtasks, you may exceed this number.
3. Do NOT create unnecessary or unrelated subtasks just to match the number.
4. Every subtask must contribute directly to solving the user task.

Node template:
{node_template}
"""

    def _build_node_rules(self) -> str:
        if self.length_mode == "none":
            return """
Rules:
1. Create only necessary nodes with meaningful requirements.
2. Do NOT invent unrelated capabilities.
3. Node IDs must start at 1 and increase consecutively.
4. Each node requirement must describe a needed capability, not the final answer.
"""

        return """
Rules:
1. You MAY leave some nodes unused if not needed (they will be removed later).
2. Fill only necessary nodes with meaningful requirements.
3. Do NOT invent unrelated capabilities.
4. You MUST follow the node template exactly for IDs.
"""

    def _build_structure_constraints(self, dep_str: str) -> str:
        edge_types_block = dep_str if dep_str else "(empty)"

        if self.planning_mode == "linear":
            return f"""
Workflow Constraints (STRICT)

- The workflow MUST be a SINGLE ordered sequence.
- Do NOT create any branching.
- Each internal node should have exactly one predecessor and one successor.
- The first node has no predecessor.
- The last node has no successor.
- Do NOT create self-loops.
- Edges must connect existing nodes only.

Edge types must be chosen only from:
{edge_types_block}

If the edge type list above is empty, you MUST set all edge types to "none".
"""

        return f"""
DAG Constraints (STRICT)

- The graph MUST be a DAG (no cycles).
- The graph MUST be fully connected (single component).
- Do NOT create independent subgraphs.
- Do NOT generate self-loops.
- Edges must connect existing nodes only.

If the task contains multiple independent parts, you MUST connect them into one executable workflow.

Edge types must be chosen only from:
{edge_types_block}

If the edge type list above is empty, you MUST set all edge types to "none".
"""

    def _build_service_guidance(self, service_text: str) -> str:
        if not self.use_service_awareness:
            return """
Service Awareness

No candidate service list is provided in this variant.
Decompose the task purely based on the user request and general task semantics.
"""

        return f"""
Service Awareness

You are given candidate services:

{service_text}

Rules:
1. Each subtask SHOULD map to a service.
2. Prefer service-oriented wording.
3. Make the decomposition easy for later service matching.
"""

    def build_prompt(self, task: str, task_num: int, service_ids: Optional[List[str]]) -> str:
        service_text, dep_str = self._service_context(service_ids if self.use_service_awareness else None)
        structure_constraints = self._build_structure_constraints(dep_str)
        service_guidance = self._build_service_guidance(service_text)

        return f"""
You are a task planner for a service composition system.

Your job is NOT to solve the task.
Your job is to decompose the task into capability requirements that can later be matched to services.

Decomposition Guidance
{self._build_length_guidance(task_num)}

Subtask Definition

A subtask must describe WHAT capability is needed, not the final answer.

Good examples:
- "Analyze the sentiment of the text"
- "Translate the text into English"
- "Detect objects in the image"
- "Transcribe speech from the audio"

Bad examples:
- "The sentiment is positive"
- "The translated result is ..."
- "The detected object is a cat"

Graph Construction Rules
{self._build_node_rules()}

{structure_constraints}

{service_guidance}

Special Case

If there is only 1 node, the graph has no edges.

Output Format (STRICT JSON)
{{
  "nodes": [
    {{"id": 1, "requirement": "..."}}
  ],
  "edges": [
    {{"source": 1, "target": 2, "type": "none"}}
  ]
}}

User Task:
{task}
"""

    def remove_self_loops(self, edges: List[Dict]) -> List[Dict]:
        return [edge for edge in edges if edge.get("source") != edge.get("target")]

    def is_dag(self, nodes: List[Dict], edges: List[Dict]) -> bool:
        graph = {node["id"]: [] for node in nodes}
        for edge in edges:
            src = edge.get("source")
            dst = edge.get("target")
            if src in graph and dst in graph:
                graph[src].append(dst)

        visited = set()
        stack = set()

        def dfs(node_id):
            if node_id in stack:
                return False
            if node_id in visited:
                return True

            stack.add(node_id)
            for nxt in graph[node_id]:
                if not dfs(nxt):
                    return False
            stack.remove(node_id)
            visited.add(node_id)
            return True

        return all(dfs(node_id) for node_id in graph)

    def force_connect_dag(self, nodes: List[Dict], edges: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        if not nodes:
            return nodes, edges

        graph = {node["id"]: set() for node in nodes}
        for edge in edges:
            src = edge.get("source")
            dst = edge.get("target")
            if src in graph and dst in graph:
                graph[src].add(dst)
                graph[dst].add(src)

        visited = set()
        components = []

        def dfs(node_id, component):
            component.append(node_id)
            visited.add(node_id)
            for nxt in graph[node_id]:
                if nxt not in visited:
                    dfs(nxt, component)

        for node in nodes:
            node_id = node["id"]
            if node_id not in visited:
                component = []
                dfs(node_id, component)
                components.append(component)

        components.sort(key=lambda component: min(component))
        if len(components) <= 1:
            return nodes, edges

        new_edges = list(edges)
        for index in range(len(components) - 1):
            src = components[index][-1]
            dst = components[index + 1][0]
            new_edges.append({"source": src, "target": dst, "type": "none"})
        return nodes, new_edges

    def remove_cycles(self, nodes: List[Dict], edges: List[Dict]) -> List[Dict]:
        clean_edges = []
        for edge in edges:
            clean_edges.append(edge)
            if not self.is_dag(nodes, clean_edges):
                clean_edges.pop()
        return clean_edges

    def is_connected(self, nodes: List[Dict], edges: List[Dict]) -> bool:
        if not nodes:
            return False

        graph = {node["id"]: set() for node in nodes}
        for edge in edges:
            src = edge.get("source")
            dst = edge.get("target")
            if src in graph and dst in graph:
                graph[src].add(dst)
                graph[dst].add(src)

        visited = set()

        def dfs(node_id):
            visited.add(node_id)
            for nxt in graph[node_id]:
                if nxt not in visited:
                    dfs(nxt)

        dfs(nodes[0]["id"])
        return len(visited) == len(nodes)

    def remove_empty_nodes(self, nodes: List[Dict], edges: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        valid_nodes = [node for node in nodes if str(node.get("requirement", "")).strip()]
        if not valid_nodes:
            return [], []

        valid_ids = {node["id"] for node in valid_nodes}
        clean_edges = [
            edge
            for edge in edges
            if edge.get("source") in valid_ids and edge.get("target") in valid_ids
        ]

        id_map = {}
        new_nodes = []
        for new_id, node in enumerate(valid_nodes, start=1):
            id_map[node["id"]] = new_id
            new_nodes.append({"id": new_id, "requirement": node["requirement"]})

        new_edges = []
        for edge in clean_edges:
            new_edges.append(
                {
                    "source": id_map[edge["source"]],
                    "target": id_map[edge["target"]],
                    "type": edge.get("type", "none"),
                }
            )

        return new_nodes, new_edges

    def _force_linear_edges(self, nodes: List[Dict], raw_edges: List[Dict]) -> List[Dict]:
        if len(nodes) <= 1:
            return []

        edge_type_map = {}
        for edge in raw_edges:
            src = edge.get("source")
            dst = edge.get("target")
            if isinstance(src, int) and isinstance(dst, int):
                edge_type_map[(src, dst)] = edge.get("type", "none")

        linear_edges = []
        for index in range(len(nodes) - 1):
            src = nodes[index]["id"]
            dst = nodes[index + 1]["id"]
            linear_edges.append(
                {
                    "source": src,
                    "target": dst,
                    "type": edge_type_map.get((src, dst), "none"),
                }
            )
        return linear_edges

    def _finalize_graph(self, nodes: List[Dict], edges: List[Dict], task_num: int):
        nodes, edges = self.remove_empty_nodes(nodes, edges)
        if self.length_mode == "hard" and len(nodes) != max(1, int(task_num)):
            return None

        if self.planning_mode == "linear":
            edges = self._force_linear_edges(nodes, edges)
        else:
            nodes, edges = self.force_connect_dag(nodes, edges)
            edges = self.remove_self_loops(edges)
            edges = self.remove_cycles(nodes, edges)

        if not self.is_dag(nodes, edges):
            return None
        if not self.is_connected(nodes, edges):
            return None

        if self.planning_mode == "linear":
            edges = self._force_linear_edges(nodes, edges)

        return {"nodes": nodes, "edges": edges}

    def generate_dag(self, task: str, task_num: int, service_ids: Optional[List[str]] = None):
        prompt = self.build_prompt(task, task_num, service_ids)
        messages = [
            {
                "role": "system",
                "content": "You are a strict JSON generator. Output valid JSON only. No explanation.",
            },
            {"role": "user", "content": prompt},
        ]

        llm_model = os.getenv("MRG_SC_LLM_MODEL", "llama3.1:70b").strip() or "llama3.1:70b"
        llm_base_url = os.getenv("MRG_SC_LLM_BASE_URL", "http://localhost:11434/v1/").strip()
        llm_api_key = os.getenv("MRG_SC_LLM_API_KEY", "ollama").strip() or "ollama"

        response = localLLM(
            messages,
            model=llm_model,
            temperature=0.0,
            base_url=llm_base_url,
            api_key=llm_api_key,
        )
        response = fix_json_string(response)
        dag = extract_json(response)
        if not dag:
            return None

        nodes = dag.get("nodes", [])
        edges = dag.get("edges", [])
        if not isinstance(nodes, list) or not isinstance(edges, list):
            return None
        return self._finalize_graph(nodes, edges, task_num)
