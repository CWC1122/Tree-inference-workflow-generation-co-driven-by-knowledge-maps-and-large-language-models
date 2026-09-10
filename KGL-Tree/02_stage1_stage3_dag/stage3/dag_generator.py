import json
import os
import sys
from typing import Dict, List, Optional, Tuple


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from localLLM.localLLM import localLLM
from utils import extract_json, fix_json_string, load_dependency_types


class DAGGenerator:
    def __init__(self, tool_desc_path: str):
        self.dep_types = load_dependency_types(tool_desc_path)
        self.service_dict = self.load_service_dict(tool_desc_path)

    def load_service_dict(self, path: str) -> Dict[str, Dict]:
        with open(path, "r", encoding="utf-8") as file_obj:
            services = json.load(file_obj)
        return {service["action_uid"]: service for service in services}

    def build_node_template(self, task_num: int) -> List[Dict]:
        task_num = max(1, int(task_num))
        return [{"id": index + 1, "requirement": ""} for index in range(task_num)]

    def build_prompt(self, task: str, nodes: List[Dict], service_ids: Optional[List[str]]):
        node_count = len(nodes)

        service_blocks = []
        selected_services = []
        for service_id in service_ids or []:
            service = self.service_dict.get(service_id)
            if not service:
                continue

            selected_services.append(service)
            block = "\n".join(
                [
                    f"Service ID: {service['action_uid']}",
                    f"Description: {service.get('target_action_reprs', '')}",
                    f"Input: {service.get('input-type', [])}",
                    f"Output: {service.get('output-type', [])}",
                ]
            )
            service_blocks.append(block)

        has_io = any(("input-type" in service and "output-type" in service) for service in selected_services)
        dep_str = ", ".join(self.dep_types) if has_io else ""
        service_text = "\n\n".join(service_blocks) if service_blocks else "None"

        return f"""
You are a task planner for a service composition system.

Your job is NOT to solve the task.
Your job is to decompose the task into capability requirements that can be matched to services and arranged into one executable workflow.

Decomposition Guidance

You are given an estimated number of subtasks: {node_count}.

IMPORTANT:
- This number is ONLY a reference derived from similar historical tasks.
- It is NOT a strict requirement.

Rules:
1. If the task can be completed with fewer subtasks, use fewer nodes.
2. If the task requires more subtasks, you may exceed this number.
3. Do NOT create unnecessary or unrelated subtasks just to match the number.
4. Every subtask MUST contribute directly to solving the user task.

Subtask Definition

A subtask must describe WHAT capability is needed, NOT the final answer.

Good examples:
- "Analyze the sentiment of the text"
- "Translate the text into English"
- "Detect objects in the image"
- "Transcribe speech from the audio"

Bad examples:
- "The sentiment is positive"
- "The translated result is ..."
- "The detected object is a cat"

Workflow Construction Rules

You MUST follow the node template EXACTLY for IDs.

Node template:
{json.dumps(nodes, indent=2)}

Rules:
1. You MAY leave some nodes unused if not needed; they will be removed later.
2. Fill only necessary nodes with meaningful requirements.
3. Do NOT invent unrelated capabilities.
4. The final workflow MUST be a single ordered execution sequence.
5. Do NOT create branching, parallel subflows, or independent components.
6. Every internal node should have exactly one predecessor and one successor.
7. The first node has no predecessor and the last node has no successor.
8. Do NOT generate self-loops.
9. Edges must only connect existing nodes.

Edge Constraints

Edge types must be chosen ONLY from:
{dep_str}

IMPORTANT:
- If the edge type list above is EMPTY, you MUST set ALL edge types to "none".
- If multiple independent parts exist, arrange them into one reasonable end-to-end execution order.

Example:
{{"source":1,"target":2,"type":"none"}}

Service Awareness

You are given candidate services:
{service_text}

Rules:
1. Each subtask SHOULD map to a service.
2. Prefer service-oriented wording.
3. Make decomposition easy for service matching.

Special Case

If there is only 1 node:
- The workflow has no edges.

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

        def dfs(node_id: int) -> bool:
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

        def dfs(node_id: int) -> None:
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

    def force_linear_edges(self, nodes: List[Dict], edges: List[Dict]) -> List[Dict]:
        if len(nodes) <= 1:
            return []

        edge_type_map = {}
        for edge in edges:
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

    def generate_dag(self, task: str, task_num: int, service_ids: Optional[List[str]] = None):
        node_template = self.build_node_template(task_num)
        prompt = self.build_prompt(task, node_template, service_ids)

        messages = [
            {"role": "system", "content": "You are a strict JSON generator. Output valid JSON only. No explanation."},
            {"role": "user", "content": prompt},
        ]

        response = localLLM(
            messages,
            model=os.getenv("MRG_SC_LLM_MODEL", "llama3.1:70b").strip() or "llama3.1:70b",
            temperature=0.0,
            base_url=os.getenv("MRG_SC_LLM_BASE_URL", "http://localhost:11434/v1/").strip(),
            api_key=os.getenv("MRG_SC_LLM_API_KEY", "ollama").strip() or "ollama",
        )

        response = fix_json_string(response)
        dag = extract_json(response)
        if not dag:
            print("fail: JSON parse error")
            return None

        nodes = dag.get("nodes", [])
        edges = dag.get("edges", [])
        if not isinstance(nodes, list) or not isinstance(edges, list):
            return None

        nodes, edges = self.remove_empty_nodes(nodes, edges)
        edges = self.remove_self_loops(edges)
        edges = self.force_linear_edges(nodes, edges)

        if not self.is_dag(nodes, edges):
            return None
        if not self.is_connected(nodes, edges):
            return None

        return {"nodes": nodes, "edges": edges}
