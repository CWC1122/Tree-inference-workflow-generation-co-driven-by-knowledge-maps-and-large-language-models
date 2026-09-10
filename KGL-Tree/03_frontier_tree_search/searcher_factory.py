import os
import sys
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parent
INFER_ROOT = MODULE_ROOT / "infer"

if str(INFER_ROOT) not in sys.path:
    sys.path.insert(0, str(INFER_ROOT))

from composer_engine import RGCNPathSearcher
from rf_tree_mrgsc import FrontierTreeComposer


def _get_env_str(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    return int(raw)


def _get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    return float(raw)


def _get_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _build_paths() -> dict:
    data_root = Path(_get_env_str("MRG_SC_DATA_ROOT", str(Path(__file__).resolve().parents[1] / "Data")))
    dataset_name = _get_env_str("MRG_SC_DATASET", "mul")
    train_dir = data_root / "train" / dataset_name
    test_dir = data_root / "test" / dataset_name
    return {
        "train_dir": train_dir,
        "test_dir": test_dir,
        "kg_path": Path(_get_env_str("MRG_TREE_KG_PATH", str(train_dir / "service_kg_multi.gpickle"))),
        "model_path": Path(_get_env_str("MRG_TREE_MODEL_PATH", str(train_dir / "mrg_sc_model_best.pth"))),
        "train_tasks_file": Path(_get_env_str("MRG_TREE_TRAIN_TASKS_FILE", str(train_dir / "data.json"))),
        "service_file": Path(_get_env_str("MRG_TREE_SERVICE_FILE", str(test_dir / "tool_desc.json"))),
    }


def _get_engine() -> str:
    raw = os.getenv("MRG_TREE_ABLATION_ENGINE", "mrg_tree").strip().lower()
    return raw or "mrg_tree"


def _get_search_variant() -> str:
    raw = os.getenv("MRG_TREE_SEARCH_VARIANT", "frontier_tree").strip().lower()
    return raw or "frontier_tree"


def build_searcher():
    paths = _build_paths()
    common_kwargs = dict(
        kg_path=str(paths["kg_path"]),
        model_path=str(paths["model_path"]),
        tool_desc_path=str(paths["service_file"]),
        history_tasks_file=str(paths["train_tasks_file"]),
        device=_get_env_str("MRG_SC_DEVICE", "cuda"),
        use_beam=_get_env_bool("MRG_TREE_USE_BEAM", True),
        beam_width=_get_env_int("MRG_TREE_BEAM_WIDTH", 5),
        use_dynamic_beam=_get_env_bool("MRG_TREE_USE_DYNAMIC_BEAM", False),
        beam_min=_get_env_int("MRG_TREE_BEAM_MIN", 1),
        beam_max=_get_env_int("MRG_TREE_BEAM_MAX", 10),
        beam_threshold_high=_get_env_float("MRG_TREE_BEAM_THRESHOLD_HIGH", 0.30),
        beam_threshold_low=_get_env_float("MRG_TREE_BEAM_THRESHOLD_LOW", 0.05),
        step1_semantic_keep_ratio=_get_env_float("MRG_TREE_STEP1_SEMANTIC_KEEP_RATIO", 0.5),
        history_topk=_get_env_int("MRG_TREE_HISTORY_TOPK", 10),
        history_bonus_step1=_get_env_float("MRG_TREE_HISTORY_BONUS_STEP1", 0.15),
        history_bonus_later=_get_env_float("MRG_TREE_HISTORY_BONUS_LATER", 0.00),
        strict_neighbor_only_after_step1=_get_env_bool("MRG_TREE_STRICT_NEIGHBOR_ONLY_AFTER_STEP1", True),
    )

    if _get_engine() == "mrg_sc":
        return RGCNPathSearcher(**common_kwargs)

    return FrontierTreeComposer(
        **common_kwargs,
        search_variant=_get_search_variant(),
        requirement_topk=_get_env_int("MRG_TREE_REQUIREMENT_TOPK", 24),
        requirement_min_plan_size=_get_env_int("MRG_TREE_REQUIREMENT_MIN_PLAN_SIZE", 2),
        requirement_coverage_threshold=_get_env_float("MRG_TREE_REQUIREMENT_COVERAGE_THRESHOLD", 0.55),
        tree_budget_multiplier=_get_env_int("MRG_TREE_BUDGET_MULTIPLIER", 8),
        tree_queue_size=_get_env_int("MRG_TREE_QUEUE_SIZE", 24),
        tree_children_per_requirement=_get_env_int("MRG_TREE_CHILDREN_PER_REQUIREMENT", 4),
        tree_relation_keep=_get_env_int("MRG_TREE_RELATION_KEEP", 1),
        tree_match_threshold=_get_env_float("MRG_TREE_MATCH_THRESHOLD", 0.05),
        tree_mrg_prob_threshold=_get_env_float("MRG_TREE_MRG_PROB_THRESHOLD", 0.55),
        tree_alpha_match=_get_env_float("MRG_TREE_ALPHA_MATCH", 1.0),
        tree_beta_mrg=_get_env_float("MRG_TREE_BETA_MRG", 1.0),
        tree_eta_history=_get_env_float("MRG_TREE_ETA_HISTORY", 0.6),
        tree_gamma_dep=_get_env_float("MRG_TREE_GAMMA_DEP", 0.4),
        tree_rho_redundancy=_get_env_float("MRG_TREE_RHO_REDUNDANCY", 0.25),
        tree_frontier_score_threshold=_get_env_float("MRG_TREE_FRONTIER_SCORE_THRESHOLD", 0.0),
        tree_enable_state_dominance=_get_env_bool("MRG_TREE_ENABLE_STATE_DOMINANCE", True),
        tree_use_dep_sig=_get_env_bool("MRG_TREE_USE_DEP_SIG", True),
        ignore_dag_order=_get_env_bool("MRG_TREE_IGNORE_DAG_ORDER", False),
        disable_depends_on=_get_env_bool("MRG_TREE_DISABLE_DEPENDS_ON", False),
        candidate_top_k=_get_env_int("MRG_TREE_MODULE4_TOP_K", 5),
        candidate_include_m_minus_1=_get_env_bool("MRG_TREE_MODULE4_INCLUDE_M_MINUS_1", True),
    )
