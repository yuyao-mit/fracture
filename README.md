# Neural Operator for Phase-Field Fracture

A machine-learning framework that couples neural operators with finite element method (FEM) to accelerate phase-field fracture simulations. The core idea is to replace the expensive phase-field solve in each staggered iteration with a fast neural operator prediction, while retaining the FEM solve for the mechanical equilibrium.

## Overview

Phase-field fracture (AT-2 model) is solved via a staggered scheme:
1. **Mechanics sub-problem** — FEM (DOLFINx/FEniCSx) solves for displacement and stress/strain fields given the current damage field.
2. **Phase-field sub-problem** — a neural operator predicts the updated damage field `d` from the mechanical state, replacing the conventional variational solve.

Two training objectives are supported:

| Mode | Flag | Target |
|------|------|--------|
| State rollout | *(default)* | Next phase-field state(s) `d` |
| Latent physics inference | `--infer_latent_variable` | Latent physics variable (e.g., length scale `ℓ`) |

## Repository Structure

```
fracture/
├── requirements.txt
└── src/
    ├── train.py              # Training entry point
    ├── inference.py          # Evaluation / prediction entry point
    ├── models/
    │   ├── fno.py            # Fourier Neural Operator (FNO)
    │   ├── uno.py            # U-shaped Neural Operator (UNO)
    │   ├── codano.py         # CODANO
    │   ├── rno.py            # Recurrent Neural Operator (RNO)
    │   └── neuraloperator/   # neuralop library (submodule)
    ├── fem/
    │   ├── PFx_hybrid_v3.py          # Hybrid FEM + NN solver
    │   ├── PFx_hybrid_v3_FEM.py      # FEM mechanics solver
    │   └── PFx_hybrid_v3_utils.py    # Mesh building, I/O utilities
    └── utils/
        ├── dataset.py        # ChunkedScalarDatasetEfficient
        └── loss.py           # Masked MSE loss
```

## Data Format

Each simulation case is stored as a `.npz` file (named `case*.npz`) with the following arrays:

| Key | Shape | Description |
|-----|-------|-------------|
| `inputs` | `[T, C, H, W]` | 16-channel input fields per time step |
| `targets` | `[T, 1, H, W]` | Next phase-field state `d` |
| `latent_variables` | `[T, ...]` | Latent physics variables (e.g., `ℓ`) |
| `lc` | `[T]` | Length-scale scalar per step |

The 16 input channels (`C=16`) are:

```
[gc, ell, H, sxx, syy, sxy, exx, eyy, exy, pre_crack,
 Δexx, Δeyy, Δexy, Δsxx, Δsyy, Δsxy]
```

where `pre_crack` is a binary mask derived from NaN positions (existing crack), and `Δ` denotes incremental fields relative to the previous step.

## Installation

```bash
pip install -r requirements.txt
# Install DOLFINx separately (for the FEM solver)
# https://github.com/FEniCS/dolfinx
```

## Training

```bash
cd src
python train.py \
    --train-dir /path/to/train \
    --val-dir   /path/to/val \
    --ckpt_dir  /path/to/checkpoints \
    --model     fno \            # fno | uno | codano | rno
    --epochs    100 \
    --batch_size 8 \
    --input_steps 4 \
    --rollout_steps 1 \
    --lr 1e-4
```

To train in latent-physics-inference mode:

```bash
python train.py ... --infer_latent_variable
```

Metrics are logged to [Weights & Biases](https://wandb.ai). Pass `--wandb_mode disabled` to run without logging.

### Resuming from a checkpoint

```bash
python train.py ... --ckpt_path /path/to/checkpoints/epoch_0049.pt
```

## Inference

```bash
cd src
python inference.py \
    --data-dir  /path/to/test \
    --ckpt_path /path/to/checkpoints/epoch_0099.pt \
    --model     fno \
    --output_dir /path/to/outputs
```

Predictions and ground-truth targets are saved as `predictions.npy` and `targets.npy` under `--output_dir`. The masked MSE loss is printed to stdout.

## Available Models

| Name | Paper | Notes |
|------|-------|-------|
| `fno` | [Li et al., 2020](https://arxiv.org/abs/2010.08895) | 4-layer FNO, 12 Fourier modes |
| `uno` | [Ashiqur Rahman et al., 2022](https://arxiv.org/abs/2204.11127) | U-shaped with multi-scale skip connections |
| `codano` | [CODANO, 2024](https://arxiv.org/abs/2403.12553) | Attention-based neural operator |
| `rno` | [Fanaskov & Oseledets, 2023](https://arxiv.org/abs/2308.08794) | Recurrent operator for multi-step rollout |

All models share the same interface: they accept `[B, T_in, C_in, H, W]` tensors and return `[B, T_out, C_out, H, W]`.

## Hybrid FEM + NN Solver

The hybrid solver (`src/fem/PFx_hybrid_v3.py`) replaces the phase-field variational solve with a neural network corrector at every staggered iteration:

```bash
cd src/fem
python PFx_hybrid_v3.py \
    --xdmf  /path/to/mesh.xdmf \
    --nsteps 100 \
    --Gc 2.7e-3 \
    --ell 0.01 \
    --E 210.0 --nu 0.3 \
    --grid_nx 256 --grid_ny 256 \
    --out_dir /path/to/output
```

Outputs per run:
- `*_fields.npz` — nodal/cell fields and regular-grid snapshots
- `*.png` — damage field visualisations (every `--snapshot_every` steps)

## Loss Function

Training uses a **masked MSE** that excludes pixels inside existing cracks (identified by the `pre_crack` channel):

```
L = MSE(pred[mask==0], target[mask==0])
```

## License

See [LICENSE](LICENSE).
