import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


THIS_DIR = Path(__file__).resolve().parent
MRG_TREE_DIR = THIS_DIR.parent
WORKSPACE_ROOT = MRG_TREE_DIR.parent
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "xiaorong_data" / "Gemma31b" / "structural_ablations"
DEFAULT_DATASETS = ["daily", "hug", "mul", "ultratool"]
DEFAULT_PREDICTION_FILE = "stage3_tree_candidates.json"

METRIC_KEYS = [
    "search_loop_time_ms",
    "expanded_states",
    "completed_states",
    "generated_children",
    "kept_children",
    "peak_current_layer",
    "peak_next_layer",
    "peak_candidate_registry",
    "peak_frontier_size",
    "frontier_pruned",
    "dominated_pruned",
    "candidate_pool_size",
    "candidate_path_count",
]


def _slugify(name: str) -> str:
    import re

    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(name).strip())
    return value.strip("_") or "item"


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _collect_task_stats(predictions: List[Dict[str, Any]]) -> Dict[str, float]:
    per_key_values: Dict[str, List[float]] = {key: [] for key in METRIC_KEYS}
    valid_tasks = 0

    for task in predictions:
        stats = task.get("module3_search_stats", {})
        if not isinstance(stats, dict) or not stats:
            continue
        valid_tasks += 1
        for key in METRIC_KEYS:
            value = stats.get(key)
            if value is None:
                continue
            try:
                per_key_values[key].append(float(value))
            except Exception:
                continue

    summary = {key: _mean(values) for key, values in per_key_values.items()}
    summary["valid_task_count"] = valid_tasks
    return summary


def _collect_experiment_stats(output_root: Path, model_name: str, exp_name: str, datasets: List[str], prediction_filename: str) -> Dict[str, Any]:
    model_slug = _slugify(model_name)
    exp_slug = _slugify(exp_name)
    dataset_stats = {}
    macro_inputs: Dict[str, List[float]] = {key: [] for key in METRIC_KEYS}

    for dataset in datasets:
        prediction_path = output_root / model_slug / exp_slug / dataset / prediction_filename
        if not prediction_path.exists():
            dataset_stats[dataset] = {"error": f"missing prediction file: {prediction_path}"}
            continue

        predictions = _load_json(prediction_path)
        if not isinstance(predictions, list):
            dataset_stats[dataset] = {"error": f"prediction is not a list: {prediction_path}"}
            continue

        stats = _collect_task_stats(predictions)
        dataset_stats[dataset] = stats
        if stats.get("valid_task_count", 0) > 0:
            for key in METRIC_KEYS:
                macro_inputs[key].append(float(stats.get(key, 0.0)))

    macro_avg = {key: _mean(values) for key, values in macro_inputs.items()}
    return {
        "name": exp_name,
        "datasets": dataset_stats,
        "macro_avg": macro_avg,
    }


def _attach_relative_deltas(results: List[Dict[str, Any]], baseline_name: str) -> None:
    baseline = next((item for item in results if item.get("name") == baseline_name), None)
    if not baseline:
        return

    base_metrics = baseline.get("macro_avg", {})
    for item in results:
        deltas = {}
        for key in METRIC_KEYS:
            base_value = float(base_metrics.get(key, 0.0))
            cur_value = float(item.get("macro_avg", {}).get(key, 0.0))
            if base_value == 0.0:
                continue
            deltas[key] = (cur_value - base_value) / base_value
        item["relative_to_baseline"] = deltas


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate module3 search-efficiency statistics from structural-ablation outputs.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Root directory of structural ablation outputs.")
    parser.add_argument("--model-name", default="Gemma31b", help="Model tag used in the output directory layout.")
    parser.add_argument("--experiments", nargs="+", required=True, help="Experiment names to summarize.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, help="Datasets to include.")
    parser.add_argument("--prediction-file", default=DEFAULT_PREDICTION_FILE, help="Prediction filename inside each dataset artifact directory.")
    parser.add_argument("--baseline", default=None, help="Optional baseline experiment name for relative deltas.")
    parser.add_argument("--output-json", default=None, help="Optional explicit output JSON path.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    results = [
        _collect_experiment_stats(output_root, args.model_name, exp_name, args.datasets, args.prediction_file)
        for exp_name in args.experiments
    ]

    if args.baseline:
        _attach_relative_deltas(results, args.baseline)

    output_json = (
        Path(args.output_json)
        if args.output_json
        else output_root / _slugify(args.model_name) / "structural_ablation_efficiency_summary.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as file_obj:
        json.dump(results, file_obj, ensure_ascii=False, indent=2)

    print(f"Saved efficiency summary to: {output_json}")


if __name__ == "__main__":
    main()
