# Data Preparation

This directory documents the data format expected by KGL-Tree. The full processed datasets are not redistributed with the repository. Please obtain the raw resources from their original sources and prepare the local files according to the layout below.

## Raw Data Sources

- UltraTool: https://github.com/JoeYing1019/UltraTool
- JARVIS / TaskBench: https://github.com/microsoft/JARVIS

Please follow the licenses and citation requirements of the original datasets.

## Processed-Data Layout

The code expects a configurable data root with separate training and test splits:

```text
<data-root>/
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

The supported code-level dataset names are:

```text
ultratool
mul
hug
daily
```

The exact data-root name, such as a model or embedding configuration, is controlled by the launcher files and environment variables. Keep the same directory convention for both `train` and `test`.

## Required Files

### `data.json`

Each task record should contain at least:

- `annotation_id`: unique task identifier;
- `confirmed_task`: natural-language task request;
- `action_id_list`: ground-truth service identifiers in execution order;
- `action_reprs`: ground-truth service descriptions in execution order.

The following metadata fields are supported when available:

- `domain`
- `subdomain`
- `website`

### `tool_desc.json`

Each service record should contain at least:

- `action_uid`: unique service identifier;
- `target_action_reprs`: textual service representation;
- `input-type`: service input type;
- `output-type`: service output type.

The service representation used in `data.json` and `tool_desc.json` should be normalized consistently.

### `graph_desc.json`

Each relation record should contain:

- `source`: source service identifier;
- `target`: target service identifier;
- `type`: relation type.

The graph can contain dependency, transition, substitution, complementarity, or other relation labels supported by the graph-construction code. Legacy preprocessing files may include additional fields such as `count` or `weight`; these fields are optional for the public data format unless a downstream script explicitly requires them.

## Generated Runtime Files

After the pipeline starts, the following files may be generated in the corresponding train or test directory:

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
unfinished_*.json
```

These files are runtime artifacts rather than raw dataset files. They can be regenerated from the prepared input data and are excluded from the public release when they are large or machine-specific.

## Preparation Workflow

1. Download or otherwise obtain the permitted raw UltraTool and JARVIS/TaskBench resources.
2. Split the data into the training and test partitions used by the experiment.
3. Convert task annotations to `data.json`.
4. Convert service metadata to `tool_desc.json`.
5. Convert service relations to `graph_desc.json`.
6. Place the files under `<data-root>/train/<dataset>/` and `<data-root>/test/<dataset>/`.
7. Update the launcher configuration or set `MRG_SC_DATA_ROOT` and `MRG_SC_DATASET`.
8. Confirm the paths printed by the launcher before starting a large run.

No raw-data download or preprocessing script is included in the current release. The conversion from the original resources to the three processed JSON files should preserve service identifiers and execution order.

## Configuration Variables

The runtime currently uses the following compatibility variables:

```text
MRG_SC_DATA_ROOT
MRG_SC_DATASET
MRG_SC_DATASETS
```

The `MRG_SC_*` prefix is retained by the implementation for backward compatibility with earlier internal code. It does not denote the method used in this repository.

## Data Release and Privacy

Do not commit private, licensed, or restricted data to the repository. Before publishing a processed dataset, verify that redistribution is permitted by the original dataset license and by any task or service provider terms.
