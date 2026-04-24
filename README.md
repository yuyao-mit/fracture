# Fracture

This repository contains code for learning-based fracture prediction. The codebase is intended to run with data and training artifacts stored on a shared filesystem rather than checked into the repository.

## Shared Data and Checkpoint Locations

The actual dataset and saved checkpoints used by this project live under:

```text
/ocean/projects/mch250029p/shared
```

Directory layout:

```text
/ocean/projects/mch250029p/shared
├── ckpt
└── fracture
    ├── test
    ├── train
    └── val
```

Absolute paths:

- Dataset root: `/ocean/projects/mch250029p/shared/fracture`
- Training split: `/ocean/projects/mch250029p/shared/fracture/train`
- Validation split: `/ocean/projects/mch250029p/shared/fracture/val`
- Test split: `/ocean/projects/mch250029p/shared/fracture/test`
- Checkpoint root: `/ocean/projects/mch250029p/shared/ckpt`

The repository itself does not store the full dataset or the training checkpoints. Any training, evaluation, or inference job should point to the shared paths above.

## Recommended Path Convention

Use the following environment variables in scripts, job files, or local shells:

```bash
export FRACTURE_DATA_ROOT=/ocean/projects/mch250029p/shared/fracture
export FRACTURE_CKPT_ROOT=/ocean/projects/mch250029p/shared/ckpt
```

Then map splits as:

```bash
$FRACTURE_DATA_ROOT/train
$FRACTURE_DATA_ROOT/val
$FRACTURE_DATA_ROOT/test
```

## Typical Usage

Every experiment is defined by one YAML in `configs/experiments/`. The training and evaluation entrypoints resolve the config (including its `defaults:` chain), derive the canonical run name, and write checkpoints / metrics / wandb logs under shared storage.

```bash
# ID benchmark for FNO
python scripts/train.py --config configs/experiments/id/fno.yaml
python scripts/evaluate.py --config configs/experiments/id/fno.yaml

# Ad-hoc overrides
python scripts/train.py --config configs/experiments/id/fno.yaml \
    training.epochs=50 wandb.mode=offline

# Hybrid NN+FEM (parameter-to-solver) on the ID split
python scripts/train.py --config configs/experiments/id/paramfem.yaml
```

Cluster-specific path defaults live in `configs/paths/shared_paths.yaml` (gitignored); a tracked template is in `configs/paths/shared_paths.example.yaml`.

Run names follow `{family}_{task}_{split}_{model_id}_s{seed}`; see `EXPERIMENTS.md`. The run registry is appended to `metadata/runs/<benchmark_group>.csv` after each run.

When running training or evaluation from this repository, make sure the code reads data from the shared dataset directory and writes or loads checkpoints from the shared checkpoint directory.

Examples:

```bash
DATA_ROOT=/ocean/projects/mch250029p/shared/fracture
CKPT_ROOT=/ocean/projects/mch250029p/shared/ckpt
```

If your training or evaluation entrypoint accepts explicit arguments, pass these absolute paths rather than repo-local placeholder folders.

## Model Implementations

The primary model code lives under:

```text
/jet/home/yyao6/research/fracture/src/models
```

This directory contains the four neural operator implementations currently used in this project. When selecting a learning-based model for training, validation, testing, or comparison experiments, use the implementations in `src/models`.

In other words:

- shared data comes from `/ocean/projects/mch250029p/shared/fracture`
- checkpoints are stored in `/ocean/projects/mch250029p/shared/ckpt`
- model definitions are maintained in `src/models`

For new experiments, `src/models` should be treated as the default location for neural operator backbones and related model variants.

## Repository Layout

The repository is now organized around four tracked layers:

- `src/`: source code for models, data, FEM, hybrid coupling, training, and evaluation
- `configs/`: tracked configs for paths, models, solvers, and experiment definitions
- `metadata/`: tracked split manifests and lightweight run registries
- `scripts/`: thin entrypoints and cluster launch helpers

Additional structure:

- `docs/REPO_LAYOUT.md`: repository architecture and storage policy
- `tests/`: unit and smoke tests

This split is intentional:

- keep code and reproducibility assets in git
- keep large datasets, checkpoints, logs, predictions, and figures in shared storage

Recommended external artifact root:

```text
/ocean/projects/mch250029p/shared/experiments/fracture
```

Recommended subdirectories under that root:

```text
logs/
metrics/
predictions/
figures/
tables/
exports/
```

## Notes for Reproducibility

- Keep dataset access read-only when possible.
- Save new model checkpoints under `/ocean/projects/mch250029p/shared/ckpt`.
- Do not commit generated checkpoints or copied datasets into this repository.
- If a script currently assumes local relative paths, update the config or command-line arguments to use the shared paths above.

## Project Scope

This repository is being used for fracture prediction experiments, including hybrid neural-network and finite-element-method workflows. The code in the repo should be treated as the source of truth for models, solvers, and experiment setup, while the large binary artifacts remain in shared storage.
