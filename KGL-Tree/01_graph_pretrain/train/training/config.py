from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mrg_sc_runtime import build_data_path, get_dataset_name, get_device, get_split_dir


DATASET_NAME = get_dataset_name()
BASE_DATA_DIR = str(get_split_dir("train", DATASET_NAME))
TEST_DATA_DIR = str(get_split_dir("test", DATASET_NAME))

KG_PATH = build_data_path("train", "service_kg_multi.gpickle", DATASET_NAME)
TASKS_FILE_PATH = build_data_path("train", "data.json", DATASET_NAME)
TEST_TASKS_FILE_PATH = build_data_path("test", "data.json", DATASET_NAME)

MODEL_PATH = build_data_path("train", "mrg_sc_model_best.pth", DATASET_NAME)
LAST_MODEL_PATH = build_data_path("train", "mrg_sc_model_last.pth", DATASET_NAME)
TASK_EMB_CACHE_PATH = build_data_path("train", "task_emb_cache.pkl", DATASET_NAME)
SERVICE_EMB_CACHE_PATH = build_data_path("train", "service_emb_cache.pkl", DATASET_NAME)

SEED = 42
VAL_RATIO = 0.1
BATCH_SIZE = 64
EPOCHS = 40
LR = 2e-4
WEIGHT_DECAY = 1e-5
HIDDEN_DIM = 512
DROPOUT = 0.2
PATIENCE = 7
TEMPERATURE = 0.07
LABEL_SMOOTHING = 0.05
MAX_CANDIDATES = 64

REL_TYPES = ["DEPENDS_ON", "COMPLEMENTS", "ALTERNATIVE_TO", "WORKFLOW"]
INFER_ALLOWED = {"DEPENDS_ON", "WORKFLOW", "COMPLEMENTS"}

WORKFLOW_REPEAT_CAP = 3
WORKFLOW_REPEAT_CAP_NO_DEP = 2
DEP_REPEAT = 3
WF_DEP_BONUS = 2

SINGLE_STEP_LOSS_WEIGHT = 1.8

ENABLE_STEP1_SEMANTIC_PREFILTER = True
DYNAMIC_KEEP_RATIO = 0.5

USE_STEP_AWARE_FUSION = True
USE_PATH_ENCODER = True
USE_POSITIONAL_SIGNAL = True
USE_DEPENDENCY_FIRST_DECODING = True

HISTORY_TOPK = 10
HISTORY_BONUS_STEP1 = 0.15
HISTORY_BONUS_LATER = 0.00

STRICT_NEIGHBOR_ONLY_AFTER_STEP1 = True

DEVICE = torch.device(get_device())
