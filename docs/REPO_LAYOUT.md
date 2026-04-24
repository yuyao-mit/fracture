# Repository Layout

This document defines the intended repository structure for fracture prediction experiments based on:

- four neural operators already implemented under `src/models`
- future hybrid neural-network + FEM models
- reproducible ICML-style benchmarking

## Design Principles

1. Keep source code, configs, manifests, and paper-facing assets in git.
2. Keep large datasets, checkpoints, logs, predictions, and figures outside git.
3. Separate baseline operator experiments from hybrid NN+FEM experiments.
4. Make experiment identity reproducible through config files and tracked manifests.

## Recommended Top-Level Structure

```text
fracture/
├── README.md
├── EXPERIMENTS.md
├── docs/
│   └── REPO_LAYOUT.md
├── configs/
│   ├── paths/
│   ├── models/
│   ├── solver/
│   └── experiments/
├── scripts/
│   └── slurm/
├── src/
│   ├── models/
│   ├── data/
│   ├── fem/
│   ├── hybrid/
│   ├── training/
│   ├── evaluation/
│   └── utils/
├── metadata/
│   ├── splits/
│   └── runs/
└── tests/
```

## What Lives in Git

### `src/`

- `src/models`: the four existing neural operators
- `src/data`: dataset indexing, loaders, transforms, mesh helpers
- `src/fem`: FEM wrappers, PDE interfaces, solver adapters
- `src/hybrid`: NN-to-FEM coupling logic
- `src/training`: training loops, callbacks, checkpoint policy helpers
- `src/evaluation`: metrics, evaluation harness, result aggregation
- `src/utils`: shared utilities

### `configs/`

- `configs/paths`: machine- or cluster-specific path templates
- `configs/models`: per-model config templates
- `configs/solver`: FEM and solver settings
- `configs/experiments`: benchmark definitions

### `metadata/`

- `metadata/splits`: tracked train/val/test split manifests
- `metadata/runs`: tracked experiment registry, result summaries, and paper tables that are small enough to version

### `docs/`

- architecture notes
- experiment protocol notes
- paper asset conventions

## What Lives Outside Git

These should remain on shared storage:

- dataset: `/ocean/projects/mch250029p/shared/fracture`
- checkpoints: `/ocean/projects/mch250029p/shared/ckpt`

Recommended external run-artifact root:

```text
/ocean/projects/mch250029p/shared/experiments/fracture
```

Recommended subdirectories under the external run root:

```text
/ocean/projects/mch250029p/shared/experiments/fracture
├── logs
├── metrics
├── predictions
├── figures
├── tables
└── exports
```

This keeps the repo clean while making experiment outputs discoverable on the cluster.

## Experiment Storage Policy

Use this split between repo and shared storage:

- Repo stores experiment definitions and manifests.
- Shared storage stores heavy outputs.

### In Git

- experiment config YAML
- split manifest JSON/CSV
- small aggregated metrics tables
- plotting scripts
- paper-ready selected figures if they are lightweight and stable

### Outside Git

- full checkpoints
- per-sample predictions
- tensorboard or wandb logs
- raw evaluation dumps
- large generated figures

## Experiment Naming Convention

Each experiment should have a stable run name:

```text
{family}_{task}_{split}_{model}_{seed}
```

Examples:

- `baseline_fracture_id_operatorA_seed0`
- `baseline_fracture_ood_geometry_operatorB_seed1`
- `hybrid_fracture_ood_load_paramfem_seed0`

Use the run name consistently in:

- config filenames
- checkpoint subdirectories
- metric exports
- plot exports

## Suggested Ownership by Directory

### Baseline Neural Operators

- code: `src/models`
- model configs: `configs/models`
- experiment configs: `configs/experiments/id`, `configs/experiments/ood`, `configs/experiments/low_data`

### Hybrid NN+FEM

- code: `src/hybrid`, `src/fem`
- solver configs: `configs/solver`
- hybrid configs: `configs/models`
- experiment configs: `configs/experiments/ablations`, `configs/experiments/ood`

### Evaluation

- code: `src/evaluation`
- manifests: `metadata/runs`
- plots and tables: external shared storage, with selected summaries optionally copied into git later

## Minimal Workflow

1. Define data path config in `configs/paths`.
2. Define one model config in `configs/models`.
3. Define one experiment config in `configs/experiments`.
4. Launch training from `scripts/`.
5. Save checkpoints to `/ocean/projects/mch250029p/shared/ckpt`.
6. Save logs and metrics to `/ocean/projects/mch250029p/shared/experiments/fracture`.
7. Commit only configs, manifests, and summarized outputs.
