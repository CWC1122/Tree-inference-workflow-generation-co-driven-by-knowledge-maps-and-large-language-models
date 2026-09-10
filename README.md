# KGL-Tree

Official code release for **KGL-Tree: Knowledge-Graph- and Large-Language-Model-Collaborative Tree-Inference Service Composition**.

KGL-Tree is a knowledge-graph-guided service composition framework for open-ended natural-language requests. It combines multi-relation service graph representation learning, task-length and service retrieval priors, structured requirement planning, and prior-enhanced frontier tree search with state-dominance pruning.

The repository provides the main KGL-Tree pipeline, baseline implementations, evaluation scripts, and ablation experiment runners. The full processed datasets are not redistributed in this repository. Please prepare the data according to [Data/README.md](Data/README.md).

## Method Pipeline

The implementation is organized into the following stages:

1. `01_graph_pretrain/`: build a multi-relation service graph and pretrain the graph representation model.
2. `02_stage1_stage3_dag/`: perform early-stage task-length prediction, similar-task or service retrieval, and structured requirement planning.
3. `03_frontier_tree_search/`: ground planned requirements to candidate services and perform prior-enhanced frontier tree search with structural constraints and state-dominance pruning.
4. `Evaluate/`: calculate the service-composition evaluation metrics.

The directory name `02_stage1_stage3_dag` and intermediate file name `stage2_dag.json` are retained for compatibility with the current implementation. In the KGL-Tree pipeline, this directory should be understood as the structured planning stage.

## Repository Layout

```text
KGL-Tree-master/
├── Baseline/
│   ├── BIKER.py
│   ├── Greedy.py
│   ├── KCAR.py
│   ├── LLMDirect.py
│   └── DHGL/
├── Data/
│   └── README.md
├── MRG-Tree/
│   ├── 01_graph_pretrain/
│   │   ├── graph_build/
│   │   └── train/
│   ├── 02_stage1_stage3_dag/
│   │   ├── stage1/
│   │   ├── stage2/
│   │   └── stage3/
│   ├── 03_frontier_tree_search/
│   │   └── infer/
│   ├── Ablation/
│   ├── Structural_Ablation/
│   ├── Evaluate/
│   └── run_all.py
├── LICENSE
├── README.md
└── requirements.txt
```

## Environment Setup

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

The versions used by the release are specified in `requirements.txt`. The file is intentionally kept unchanged so that the released code and the reported experiments use the same dependency specification.

The pipeline also requires:

- a CUDA-capable PyTorch environment if GPU inference or graph pretraining is used;
- an OpenAI-compatible embedding service, such as an Ollama-compatible local endpoint;
- an OpenAI-compatible chat endpoint for the LLM-based planning stages and the LLM baselines.

## Data Preparation

The expected processed-data layout is:

```text
Data/
└── <data-config>/
    ├── train/
    │   └── <dataset>/
    │       ├── data.json
    │       ├── tool_desc.json
    │       └── graph_desc.json
    └── test/
        └── <dataset>/
            ├── data.json
            ├── tool_desc.json
            └── graph_desc.json
```

The repository supports the code-level dataset names `ultratool`, `mul`, `hug`, and `daily`. The raw resources are based on UltraTool and the JARVIS/TaskBench data. See [Data/README.md](Data/README.md) for the required fields and preparation notes.

The following files are produced during execution and are normally kept outside version control:

```text
service_kg_multi.gpickle
gnn1_rgcn_best.pth
stage1_history_embeddings.npy
stage2_train_task_embeddings.npy
stage1_tasknum.json
stage15_services.json
stage2_dag.json
stage3_tree_candidates.json
evaluate_result_weight_sorted.json
```

## Configuration

The launcher files contain the experiment-level settings near the top of each file. Before running the pipeline, check at least:

- `DATA_FOLDER_NAME`
- `DATABASE_NAME`
- `DEVICE`
- embedding endpoint and embedding model
- LLM endpoint and LLM model
- input and output data paths

The current code also accepts environment variables. The `MRG_SC_*` prefix is retained as a compatibility prefix from an earlier internal project name; it does not change the method name or the identity of this repository.

Common variables include:

```text
MRG_SC_DATA_ROOT
MRG_SC_DATASET
MRG_SC_DATASETS
MRG_SC_DEVICE
MRG_SC_EMBEDDING_URL
MRG_SC_EMBEDDING_MODEL
MRG_SC_EMBEDDING_TIMEOUT
MRG_SC_LLM_MODEL
MRG_SC_LLM_BASE_URL
MRG_SC_LLM_API_KEY
```

The tree-search launcher additionally exposes configuration variables with the `MRG_TREE_*` prefix, including beam-search, history-prior, dependency, scoring, pruning, worker, timeout, and input/output settings.

For example, in PowerShell:

```powershell
$env:MRG_SC_DATA_ROOT = "$PWD\Data\Gemma31b_bge"
$env:MRG_SC_DATASET = "hug"
$env:MRG_SC_DEVICE = "cuda"
$env:MRG_SC_EMBEDDING_URL = "http://127.0.0.1:11434/v1/embeddings"
$env:MRG_SC_EMBEDDING_MODEL = "bge-m3:latest"
$env:MRG_SC_LLM_BASE_URL = "http://127.0.0.1:11434/v1/"
$env:MRG_SC_LLM_MODEL = "your-local-chat-model"
```

Because the current launchers preserve experiment-specific defaults, always verify the paths printed at startup before processing a full dataset.

## Running the Full Pipeline

From the repository root:

```bash
python MRG-Tree/run_all.py
```

The launcher executes the four main modules in order:

```text
01_graph_pretrain/run.py
02_stage1_stage3_dag/run.py
03_frontier_tree_search/run.py
Evaluate/evaluate_soft.py
```

The stages are resumable. Intermediate JSON files are written periodically, so an interrupted run can normally be continued after the input paths and configuration have been kept unchanged.

## Running Individual Modules

### Graph Pretraining

```bash
python MRG-Tree/01_graph_pretrain/run.py
```

This stage builds the multi-relation service graph and trains the graph representation model.

### Requirement Planning

```bash
python MRG-Tree/02_stage1_stage3_dag/run.py
```

This stage runs task-length prediction, service retrieval, and structured requirement planning. Its internal sub-stages can also be run directly:

```bash
python MRG-Tree/02_stage1_stage3_dag/stage1/main.py
python MRG-Tree/02_stage1_stage3_dag/stage2/main.py
python MRG-Tree/02_stage1_stage3_dag/stage3/main.py
```

### Frontier Tree Search

```bash
python MRG-Tree/03_frontier_tree_search/run.py
```

This stage performs service grounding, prior-enhanced tree expansion, structural filtering, and state-dominance pruning.

### Evaluation

```bash
python MRG-Tree/Evaluate/evaluate_soft.py
```

The evaluation scripts read the generated prediction file together with `tool_desc.json` and `graph_desc.json`. Depending on the selected evaluator, the output includes service-set matching, granularity, executability, and whole-sequence matching metrics. The exact input and output paths can be changed through the evaluator's environment variables.

## Baselines

The repository contains implementations or execution scripts for the following comparison methods:

- BIKER
- Greedy
- KCAR
- LLMDirect
- DHGL

Examples:

```bash
python Baseline/BIKER.py
python Baseline/Greedy.py
python Baseline/KCAR.py
python Baseline/LLMDirect.py
python Baseline/DHGL/stage1_decompose.py
python Baseline/DHGL/stage2_recall.py
python Baseline/DHGL/stage25_recall_test.py
python Baseline/DHGL/stage3_main.py
```

The baseline scripts share the data and endpoint configuration described above. For multi-dataset baseline runs, use `MRG_SC_DATASETS` as a comma-separated list of supported dataset names.

## Ablation and Efficiency Analysis

Run the module ablation experiments with:

```bash
python MRG-Tree/Ablation/auto_ablation.py
```

Run the structural ablation and search-efficiency experiments with:

```bash
python MRG-Tree/Structural_Ablation/auto_structural_ablation.py
python MRG-Tree/Structural_Ablation/efficiency_eval.py
```

The experiment configurations are stored in:

```text
MRG-Tree/Ablation/ablation_config.json
MRG-Tree/Structural_Ablation/structural_ablation_main_table.json
```

Generated ablation outputs are written under the configured data root and are excluded from the public release when they contain large intermediate artifacts.

## Reproducibility Notes

- Do not commit the processed train/test data, model checkpoints, embedding caches, or generated prediction files unless redistribution is permitted.
- Use the same data split, embedding service, LLM model, and launcher configuration when reproducing the reported results.
- The default values in individual launcher files reflect the original experimental environment and may need to be changed for a new machine.
- Some internal Python modules and generated filenames retain historical `MRG-SC`-related identifiers for compatibility. These identifiers are implementation details and do not indicate that the released method is MRG-SC.

## Citation

If you use this repository, please cite the KGL-Tree paper:

```text
KGL-Tree: Knowledge-Graph- and Large-Language-Model-Collaborative
Tree-Inference Service Composition.
```

Formal publication metadata will be added after the paper is publicly available.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

## Acknowledgements

This project uses or adapts publicly available resources from UltraTool, JARVIS/TaskBench, and related open-source research projects. Please consult the original projects for their respective licenses and citation requirements.
