# ICML Experiment Plan: Hybrid Neural Network + FEM for Fracture PDE Prediction

## Repo Constraints and Asset Locations

This experiment plan assumes the current repository layout and shared-storage setup:

- Neural-operator implementations live in `/jet/home/yyao6/research/fracture/src/models`
- The project currently uses the four neural operators implemented in `src/models`
- Dataset root is `/ocean/projects/mch250029p/shared/fracture`
- Checkpoints are stored in `/ocean/projects/mch250029p/shared/ckpt`

Any experiment script, config, or job submission should use the shared dataset and checkpoint paths above rather than repo-local copies.

## Concrete Model-ID Convention

This repo already contains four neural operators under `src/models`. From this point on, the experiment system should identify them by **the exact Python filename stem** used in `src/models`.

Examples of the rule:

- `src/models/fno.py` -> `model_id: fno`
- `src/models/deeponet.py` -> `model_id: deeponet`

Do not invent a second naming system for the same model. The filename stem should be reused consistently in:

- `configs/models/operator_<model_id>.yaml`
- run names
- checkpoint subdirectories
- metric exports
- table row names

For the rest of this document, let:

- `op1`
- `op2`
- `op3`
- `op4`

denote the four exact `model_id`s from `src/models`.

## Concrete Run Naming Convention

Every run should use the following canonical name:

```text
{family}_{task}_{split}_{model_id}_s{seed}
```

Required fields:

- `family`: `baseline` or `hybrid`
- `task`: `fracture`
- `split`: one of `id`, `ood_geometry`, `ood_load`, `ood_material`, `ood_resolution`, `lowdataXX`, `ablation_<name>`
- `model_id`: exact filename stem from `src/models` for operators, or a fixed hybrid id such as `paramfem` or `warmstart`
- `seed`: fixed to `0` for the current study

Examples:

- `baseline_fracture_id_op1_s0`
- `baseline_fracture_ood_geometry_op2_s0`
- `hybrid_fracture_id_paramfem_s0`
- `hybrid_fracture_ood_load_paramfem_s0`
- `hybrid_fracture_ablation_target_paramfem_s0`

## Concrete Artifact Storage Convention

Each run name maps to the same relative path structure under shared storage.

Checkpoint path:

```text
/ocean/projects/mch250029p/shared/ckpt/{run_name}/
```

Heavy experiment artifacts:

```text
/ocean/projects/mch250029p/shared/experiments/fracture/
├── logs/{run_name}/
├── metrics/{run_name}/
├── predictions/{run_name}/
├── figures/{run_name}/
└── tables/{benchmark_group}/
```

Tracked repo-side metadata:

```text
metadata/runs/{benchmark_group}.csv
metadata/runs/{benchmark_group}_summary.json
```

Expected files per run:

- `logs/{run_name}/stdout.log`
- `logs/{run_name}/stderr.log`
- `metrics/{run_name}/metrics.json`
- `metrics/{run_name}/per_case.csv`
- `predictions/{run_name}/...`
- `figures/{run_name}/...`

## Concrete Experiment Matrix

The experiment program should be run in **three layers**:

1. Full four-operator screening
2. Main paper benchmark
3. Hybrid ablations

### Layer 1. Four-Operator Screening

Purpose:

- train and rank all four neural operators fairly
- determine which two operator baselines deserve full OOD and low-data comparison

Run all four operators:

- `op1`
- `op2`
- `op3`
- `op4`

#### Matrix L1-A: In-Distribution Screening

| benchmark_group | split | models | seeds | total runs |
| --- | --- | --- | --- | --- |
| `screen_id` | `id` | `op1, op2, op3, op4` | `0` | `4` |

Run name template:

```text
baseline_fracture_id_{model_id}_s{seed}
```

Selection rule:

- rank by mean validation score first
- break ties using test `damage` / crack metric and runtime
- promote the top 2 models to Layer 2 as `best_op_a` and `best_op_b`

#### Matrix L1-B: Cheap Low-Data Screening

| benchmark_group | split | models | fractions | seeds | total runs |
| --- | --- | --- | --- | --- | --- |
| `screen_lowdata` | `lowdata10`, `lowdata25`, `lowdata100` | `op1, op2, op3, op4` | `10%, 25%, 100%` | `0` | `12` |

Run name template:

```text
baseline_fracture_lowdata{fraction}_{model_id}_s{seed}
```

Purpose:

- identify which operators degrade gracefully with less data
- avoid spending full OOD budget on weak baselines

### Layer 2. Main Paper Benchmark

Purpose:

- compare the best direct operators against the proposed hybrid model on the paper-critical splits

Models in this layer:

- `best_op_a`
- `best_op_b`
- `paramfem`
- optional `warmstart`

Where:

- `best_op_a`, `best_op_b` are selected from Layer 1
- `paramfem` is the main proposed hybrid model
- `warmstart` is included only if it is stable and cheap enough

#### Matrix L2-A: Main Accuracy Table

| benchmark_group | split | models | seeds | total runs without `warmstart` |
| --- | --- | --- | --- | --- |
| `main_id` | `id` | `best_op_a, best_op_b, paramfem` | `0` | `3` |

Run name templates:

```text
baseline_fracture_id_{best_op}_s{seed}
hybrid_fracture_id_paramfem_s{seed}
```

#### Matrix L2-B: OOD Benchmark Table

| benchmark_group | split | models | seeds | total runs without `warmstart` |
| --- | --- | --- | --- | --- |
| `main_ood_geometry` | `ood_geometry` | `best_op_a, best_op_b, paramfem` | `0` | `3` |
| `main_ood_load` | `ood_load` | `best_op_a, best_op_b, paramfem` | `0` | `3` |
| `main_ood_material` | `ood_material` | `best_op_a, best_op_b, paramfem` | `0` | `3` |
| `main_ood_resolution` | `ood_resolution` | `best_op_a, best_op_b, paramfem` | `0` | `3` |

Run name templates:

```text
baseline_fracture_ood_geometry_{best_op}_s{seed}
baseline_fracture_ood_load_{best_op}_s{seed}
baseline_fracture_ood_material_{best_op}_s{seed}
baseline_fracture_ood_resolution_{best_op}_s{seed}
hybrid_fracture_ood_geometry_paramfem_s{seed}
hybrid_fracture_ood_load_paramfem_s{seed}
hybrid_fracture_ood_material_paramfem_s{seed}
hybrid_fracture_ood_resolution_paramfem_s{seed}
```

#### Matrix L2-C: Full Low-Data Benchmark

| benchmark_group | split | models | fractions | seeds | total runs without `warmstart` |
| --- | --- | --- | --- | --- | --- |
| `main_lowdata` | `lowdata05`, `lowdata10`, `lowdata25`, `lowdata50`, `lowdata100` | `best_op_a, best_op_b, paramfem` | `5%, 10%, 25%, 50%, 100%` | `0` | `15` |

Run name templates:

```text
baseline_fracture_lowdata{fraction}_{best_op}_s{seed}
hybrid_fracture_lowdata{fraction}_paramfem_s{seed}
```

### Layer 3. Hybrid Ablations

Purpose:

- isolate why the hybrid method works
- keep ablation cost bounded by using only the main hybrid model

Hybrid variants:

- `paramfem_targetGc`
- `paramfem_targetdamage`
- `paramfem_latentfield`
- `warmstart`

#### Matrix L3-A: Prediction Target Ablation

| benchmark_group | split | hybrid variants | seeds | total runs |
| --- | --- | --- | --- | --- |
| `ablation_target` | `id` | `paramfem_targetGc, paramfem_targetdamage, paramfem_latentfield, warmstart` | `0` | `4` |

Run name template:

```text
hybrid_fracture_ablation_target_{variant}_s{seed}
```

#### Matrix L3-B: Coupling Ablation

| benchmark_group | split | hybrid variants | seeds | total runs |
| --- | --- | --- | --- | --- |
| `ablation_coupling` | `id` | `paramfem_fullsolve, paramfem_fewstep, warmstart` | `0` | `3` |

Run name template:

```text
hybrid_fracture_ablation_coupling_{variant}_s{seed}
```

#### Matrix L3-C: Loss Ablation

| benchmark_group | split | hybrid variants | seeds | total runs |
| --- | --- | --- | --- | --- |
| `ablation_loss` | `id` | `paramfem_solonly, paramfem_solphys, paramfem_solphysintermediate` | `0` | `3` |

Run name template:

```text
hybrid_fracture_ablation_loss_{variant}_s{seed}
```

#### Matrix L3-D: Intermediate-Field Representation Ablation

Question:

- is it better to predict a dense field, a low-rank latent code, or an elementwise representation before FEM?

| benchmark_group | split | hybrid variants | seeds | total runs |
| --- | --- | --- | --- | --- |
| `ablation_representation` | `id` | `paramfem_densefield, paramfem_lowrank, paramfem_elementwise` | `0` | `3` |

Run name template:

```text
hybrid_fracture_ablation_representation_{variant}_s{seed}
```

#### Matrix L3-E: Training-Regime Ablation

Question:

- does the method actually require end-to-end FEM gradients, or can a cheaper staged regime recover most of the gain?

| benchmark_group | split | hybrid variants | seeds | total runs |
| --- | --- | --- | --- | --- |
| `ablation_training_regime` | `id` | `paramfem_e2e, paramfem_twostage, paramfem_stopgrad` | `0` | `3` |

Run name template:

```text
hybrid_fracture_ablation_training_regime_{variant}_s{seed}
```

#### Matrix L3-F: Physics-Constraint Ablation

Question:

- which explicit physical constraints are actually responsible for the gain?

| benchmark_group | split | hybrid variants | seeds | total runs |
| --- | --- | --- | --- | --- |
| `ablation_constraints` | `id` | `paramfem_unconstrained, paramfem_positiveonly, paramfem_positive_irreversible, paramfem_fullconstraints` | `0` | `4` |

Run name template:

```text
hybrid_fracture_ablation_constraints_{variant}_s{seed}
```

#### Matrix L3-G: Solver-Budget Ablation

Question:

- how much of the hybrid gain comes from the full FEM refinement budget?

| benchmark_group | split | hybrid variants | seeds | total runs |
| --- | --- | --- | --- | --- |
| `ablation_solver_budget` | `id` | `paramfem_1step, paramfem_3step, paramfem_10step, paramfem_fullsolve` | `0` | `4` |

Run name template:

```text
hybrid_fracture_ablation_solver_budget_{variant}_s{seed}
```

#### Matrix L3-H: Input-Feature Ablation

Question:

- which conditioning signals are necessary for robust fracture prediction?

| benchmark_group | split | hybrid variants | seeds | total runs |
| --- | --- | --- | --- | --- |
| `ablation_inputs` | `id` | `paramfem_geomload, paramfem_geomloadbc, paramfem_allinputs` | `0` | `3` |

Run name template:

```text
hybrid_fracture_ablation_inputs_{variant}_s{seed}
```

#### Matrix L3-I: OOD Stress-Test Ablation

Question:

- do the same ablation conclusions still hold under the hardest unseen split?

Recommendation:

- only run the best 2 hybrid variants from L3-A to L3-H, plus the default `paramfem`, on one hard OOD split such as `ood_geometry` or `ood_material`

| benchmark_group | split | hybrid variants | seeds | total runs |
| --- | --- | --- | --- | --- |
| `ablation_oodstress` | `ood_geometry` | `top_hybrid_variant_a, top_hybrid_variant_b, paramfem` | `0` | `3` |

Run name template:

```text
hybrid_fracture_ablation_oodstress_{variant}_s{seed}
```

Recommended ablation priority if compute is limited:

1. `ablation_target`
2. `ablation_constraints`
3. `ablation_solver_budget`
4. `ablation_training_regime`
5. `ablation_representation`
6. `ablation_inputs`
7. `ablation_oodstress`

## Concrete Config Naming Convention

Use one file per model and one file per benchmark.

Model configs:

- `configs/models/operator_{op1}.yaml`
- `configs/models/operator_{op2}.yaml`
- `configs/models/operator_{op3}.yaml`
- `configs/models/operator_{op4}.yaml`
- `configs/models/hybrid_paramfem.yaml`
- `configs/models/hybrid_warmstart.yaml`

Benchmark configs:

- `configs/experiments/id/{model_id}.yaml`
- `configs/experiments/ood/geometry/{model_id}.yaml`
- `configs/experiments/ood/load/{model_id}.yaml`
- `configs/experiments/ood/material/{model_id}.yaml`
- `configs/experiments/ood/resolution/{model_id}.yaml`
- `configs/experiments/low_data/{model_id}_05.yaml`
- `configs/experiments/low_data/{model_id}_10.yaml`
- `configs/experiments/low_data/{model_id}_25.yaml`
- `configs/experiments/low_data/{model_id}_50.yaml`
- `configs/experiments/low_data/{model_id}_100.yaml`

Hybrid ablation configs:

- `configs/experiments/ablations/target_{variant}.yaml`
- `configs/experiments/ablations/coupling_{variant}.yaml`
- `configs/experiments/ablations/loss_{variant}.yaml`

## Concrete Result Registry Convention

Each benchmark group should have one tracked registry file:

- `metadata/runs/screen_id.csv`
- `metadata/runs/screen_lowdata.csv`
- `metadata/runs/main_id.csv`
- `metadata/runs/main_ood_geometry.csv`
- `metadata/runs/main_ood_load.csv`
- `metadata/runs/main_ood_material.csv`
- `metadata/runs/main_ood_resolution.csv`
- `metadata/runs/main_lowdata.csv`
- `metadata/runs/ablation_target.csv`
- `metadata/runs/ablation_coupling.csv`
- `metadata/runs/ablation_loss.csv`
- `metadata/runs/ablation_representation.csv`
- `metadata/runs/ablation_training_regime.csv`
- `metadata/runs/ablation_constraints.csv`
- `metadata/runs/ablation_solver_budget.csv`
- `metadata/runs/ablation_inputs.csv`
- `metadata/runs/ablation_oodstress.csv`

Each row should contain at least:

- `run_name`
- `model_id`
- `seed`
- `split`
- `config_path`
- `ckpt_path`
- `metrics_path`
- `status`
- `primary_score`

## Minimum Paper Run Budget

Assuming `warmstart` is optional:

- Layer 1: `4 + 12 = 16` runs
- Layer 2: `3 + 3 + 3 + 3 + 3 + 15 = 30` runs
- Layer 3 core: `4 + 3 + 3 = 10` runs
- Layer 3 expanded: `3 + 3 + 4 + 4 + 3 + 3 = 20` runs

Total planned budget:

- `56` runs for the minimal single-seed package
- `76` runs for the fuller single-seed package with expanded ablations

If this is too expensive, reduce in the following order:

1. drop `ood_resolution`
2. reduce Layer 1 low-data screening to `25%` and `100%`
3. keep only `ablation_target`, `ablation_constraints`, and `ablation_solver_budget`
4. run `ablation_oodstress` only for the final 2 best hybrid variants

## 1. Goal

Replace direct end-to-end neural prediction of PDE solution fields with a hybrid pipeline:

`input conditions -> neural network predicts FEM-relevant latent variables / parameter fields -> FEM solves PDE -> output displacement / stress / damage / crack path`

The paper should argue that this hybrid design improves:

- out-of-distribution generalization
- physical consistency
- sample efficiency
- interpretability

without losing too much inference efficiency relative to direct neural surrogates.

## 2. Core Paper Claim

The main ICML claim should be:

> Learning physically meaningful intermediate quantities for a downstream FEM solver is a better inductive bias than directly regressing PDE solution fields.

More concrete claims to validate:

1. Hybrid NN+FEM is more accurate than end-to-end neural surrogates on challenging fracture prediction tasks.
2. Hybrid NN+FEM generalizes better to unseen geometries, loads, and material distributions.
3. Hybrid NN+FEM produces more physically valid predictions, especially near crack initiation and propagation.
4. Hybrid NN+FEM is more data-efficient than direct field prediction.

## 3. Recommended Problem Setup

Start from a controlled but nontrivial benchmark:

- 2D quasi-static brittle fracture
- phase-field fracture or closely related damage-based formulation
- linear elasticity in the intact regime
- fixed mesh family for the first round of experiments

Inputs:

- geometry representation
- boundary conditions
- loading conditions
- material distribution
- optional initial crack / notch specification

Outputs to evaluate:

- displacement field `u(x)`
- stress field `sigma(x)`
- damage / phase field `d(x)`
- crack path
- force-displacement curve
- failure load / time-to-failure proxy

## 4. Primary Method Variants

We should not test only one hybrid model. The paper is stronger if it compares several coupling choices and then picks one as the main method.

### A. Direct End-to-End Baseline

Network directly predicts:

- `u(x)` only, or
- `u(x), d(x)` jointly

Candidate architectures:

- the four neural operators implemented in `src/models`
- optional raster or graph baseline only if already supported by the repo

This is the baseline we want to beat.

### B. Hybrid Parameter-to-Solver Model (Main Proposed Method)

Network predicts FEM-side latent quantities such as:

- fracture toughness field `Gc(x)`
- elastic modulus field `E(x)` if heterogeneity is relevant
- damage threshold / history field parameters
- low-dimensional latent field decoded onto the mesh

Then FEM solves the PDE using these predicted quantities.

This should be the main method because it is the cleanest "NN proposes, FEM enforces physics" story.

### C. Hybrid Warm-Start Model

Network predicts:

- initial guess for `u`
- initial guess for `d`
- reduced basis coefficients

Then FEM refines to the final solution.

This variant may not be the main method, but it is useful as an efficiency baseline.

### D. Constitutive Correction Model

Network predicts a correction term:

- `sigma = sigma_FEM + Delta sigma_NN`

or a correction to energy density / degradation law.

This is a higher-risk variant. It can be a stretch goal if the parameter-field approach works first.

## 5. Experimental Questions

Each experiment should map to a paper question.

### Q1. Accuracy

Does hybrid NN+FEM beat direct neural surrogates on forward prediction?

### Q2. Generalization

Does the hybrid method extrapolate better to unseen settings?

### Q3. Physics Validity

Does the hybrid method satisfy physical constraints better?

### Q4. Data Efficiency

How much data is needed before the hybrid method overtakes end-to-end surrogates?

### Q5. Compute Tradeoff

How much solver cost do we pay, and is the improvement worth it?

## 6. Dataset and Split Design

The train/test split design matters as much as the model. A weak split will not support an ICML claim.

### 6.1 In-Distribution Split

Random split over simulations generated from the same distribution.

Purpose:

- establish basic competitiveness

### 6.2 Geometry OOD Split

Train on:

- simple notched plates
- one family of holes / inclusions

Test on:

- unseen hole sizes
- unseen notch orientations
- multiple defects
- more complex boundary shapes

Purpose:

- test shape generalization

### 6.3 Load OOD Split

Train on:

- limited range of loading magnitudes and directions

Test on:

- larger magnitude
- rotated load direction
- shifted loading points
- mixed-mode loading

Purpose:

- test whether the hybrid model respects mechanics beyond interpolation

### 6.4 Material OOD Split

Train on:

- narrow material contrast / heterogeneity range

Test on:

- higher contrast
- sharper spatial variation
- unseen correlation length in heterogeneous materials

Purpose:

- test whether learning latent FEM inputs is more robust than direct solution regression

### 6.5 Resolution Transfer Split

Train on:

- coarse mesh or one resolution

Test on:

- finer mesh / denser evaluation grid

Purpose:

- test whether the method is tied to one discretization

### 6.6 Low-Data Split

Train with:

- 5%
- 10%
- 25%
- 50%
- 100%

of the full training set.

Purpose:

- measure sample efficiency

## 7. Baselines

At minimum, compare against:

1. The four neural operators already implemented in `src/models`
2. Proposed hybrid NN+FEM method
3. Pure FEM with known inputs if available as oracle
4. Reduced-order or warm-start FEM baseline if available

Recommended baseline list:

- all four repo-native neural operators in `src/models`
- optionally one non-operator baseline only if it is already implemented and easy to reproduce

The paper should not claim superiority over "neural operators" unless the comparison includes all four models currently maintained in `src/models`.

### 7.1 Baseline Protocol for the Four Neural Operators

All four neural operators in `src/models` should be trained and evaluated under the same protocol:

- same train/val/test split
- same shared dataset root: `/ocean/projects/mch250029p/shared/fracture`
- same checkpoint root: `/ocean/projects/mch250029p/shared/ckpt`
- same optimization budget as much as practical
- same evaluation metrics

This creates a credible baseline family before introducing the hybrid FEM-coupled model.

Do not add weak baselines just to increase the count. ICML reviewers will care more about whether the strongest relevant baselines are included.

## 8. Metrics

Use both ML metrics and physics metrics.

### 8.1 Field Error Metrics

- relative L2 error for displacement
- relative L2 error for stress
- relative L2 error for damage / phase field
- H1-like error for displacement if gradients are available

### 8.2 Fracture-Specific Metrics

- crack path IoU or centerline distance
- crack initiation location error
- crack propagation trajectory error
- failure load error
- area under force-displacement curve error

### 8.3 Physics Metrics

- equilibrium residual norm
- boundary condition violation
- energy consistency error
- irreversibility violation for damage / phase field
- elementwise negative damage or other nonphysical state count

### 8.4 Efficiency Metrics

- wall-clock inference time
- number of FEM Newton iterations
- memory usage
- amortized cost per sample

## 9. Main Result Tables and Figures

The paper should be designed around a small number of strong figures and tables.

### Table 1. In-Distribution Accuracy

Compare all methods on:

- displacement error
- stress error
- damage error
- failure load error
- runtime

### Table 2. OOD Generalization

Rows:

- geometry OOD
- load OOD
- material OOD
- resolution transfer

Columns:

- four neural operators from `src/models`
- hybrid warm-start
- proposed hybrid parameter-to-FEM

### Table 3. Data Efficiency

Rows:

- training fraction

Columns:

- per-model displacement error
- crack-path metric
- failure-load error

### Table 4. Physics Compliance

Columns:

- equilibrium residual
- BC violation
- irreversibility violation
- nonphysical prediction rate

### Figure 1. Pipeline Diagram

Show:

- direct NN pipeline
- proposed NN -> FEM pipeline

### Figure 2. Qualitative Fracture Predictions

For several test cases, show:

- ground truth
- direct NN prediction
- proposed method

Include:

- displacement field
- damage / crack field
- error map

### Figure 3. Force-Displacement Curves

Overlay:

- ground truth FEM
- direct NN
- proposed method

### Figure 4. OOD Stress Test

Show one or two hard unseen cases where direct surrogates fail but hybrid NN+FEM remains stable.

## 10. Critical Ablations

ICML reviewers will expect ablations that isolate why the method works.

### A1. What does the network predict?

Compare:

- direct solution fields using each neural operator in `src/models`
- latent field for `Gc(x)` or other fracture parameter
- low-rank coefficients decoded to parameter field
- FEM warm-start only

### A2. Differentiable vs Non-Differentiable Coupling

Compare:

- backprop through FEM if supported
- stop-gradient / two-stage training
- surrogate loss on intermediate fields

This tells us whether end-to-end differentiability is necessary.

### A3. Loss Function Design

Compare:

- solution loss only
- solution + physics regularization
- solution + intermediate-field supervision if available

### A4. Solver Coupling Strength

Compare:

- one-shot correction
- full FEM solve
- few-step FEM refinement

### A5. Mesh Dependence

Compare:

- training and testing on same mesh
- training coarse, testing fine

### A6. Intermediate Representation

Compare:

- dense field prediction
- low-rank latent representation
- elementwise parameterization

### A7. Training Regime

Compare:

- end-to-end differentiable training
- two-stage training
- stop-gradient coupling

### A8. Constraint Strength

Compare:

- unconstrained output
- positivity only
- positivity + irreversibility
- full physically constrained parameterization

### A9. Solver Budget

Compare:

- 1 FEM refinement step
- 3 FEM refinement steps
- 10 FEM refinement steps
- full solve

### A10. Input Conditioning

Compare:

- geometry + load only
- geometry + load + boundary conditions
- full conditioning including material / crack priors if available

## 11. Recommended Training Protocol

To make the results credible:

- use one fixed seed for the current study: `seed = 0`
- log the exact config, code version, and dataset split manifest for every run
- use the same train/val/test splits across methods
- match parameter count where possible
- tune all major baselines fairly
- keep the same data root and artifact root across all runs:
  `/ocean/projects/mch250029p/shared/fracture` and `/ocean/projects/mch250029p/shared/ckpt`

If time is limited, prioritize deeper ablations and stronger OOD analysis over multiple random seeds.

If the paper matures later, rerun only the final headline comparisons with additional seeds.

## 12. Minimal Viable ICML Experimental Package

If we need to de-risk quickly, the first complete story should be:

1. One fracture benchmark
2. All four neural operators from `src/models`
3. One proposed hybrid parameter-to-FEM method
4. In-distribution results
5. Geometry/load/material OOD results
6. Data-efficiency curve
7. Physics-consistency metrics
8. Qualitative crack-path figures

For the first pass, it is acceptable to narrow the hybrid comparison to the two strongest neural operators after an initial screening run across all four models.

If this package is strong, the paper is viable. Everything else is optional.

## 13. Recommended Execution Order

### Phase 1. Benchmark Construction

- finalize PDE formulation
- finalize mesh and geometry family
- generate training/validation/test simulations
- define ID and OOD splits early

### Phase 2. Baseline Reproduction

- train and evaluate all four neural operators in `src/models`
- identify the strongest one or two operator baselines for deeper comparison

### Phase 3. Proposed Method

- implement parameter-field prediction
- couple to FEM solver
- verify numerical stability

### Phase 4. Core Results

- run ID benchmark
- run OOD benchmark
- run low-data benchmark
- collect runtime metrics

### Phase 5. Ablations

- predicted quantity ablation
- loss ablation
- solver-coupling ablation

### Phase 6. Paper-Ready Analysis

- pick representative qualitative cases
- generate force-displacement plots
- generate error vs data plots
- generate runtime vs accuracy plot

## 14. Failure Modes to Watch

These are likely problems and should be explicitly monitored:

- network predicts parameter fields that are numerically unstable for FEM
- hybrid model overfits to one mesh family
- gradients through fracture evolution are unstable or too expensive
- runtime becomes too high to justify the method
- the hybrid model is only better on physics metrics but not on task metrics

Mitigations:

- constrain predicted parameters to physically valid ranges
- start with fixed mesh and quasi-static setup
- use curriculum on load severity
- keep a warm-start variant as fallback

## 15. Concrete Success Criteria

Before writing the paper, we should aim for:

1. At least one OOD setting where the proposed method clearly outperforms all direct surrogates.
2. A clear reduction in physics-violation metrics.
3. Competitive or better error in force-displacement and crack-path prediction.
4. Acceptable runtime overhead relative to the gain in robustness.

If we cannot get all four, the paper story is weaker and may need to shift toward a systems or scientific ML angle instead of a general ICML claim.

## 16. Immediate Next Steps in the Repo

Add or verify the following pieces:

- a simulation generator script for train/val/test and OOD splits
- a common evaluation harness for all four operators in `src/models`
- metrics for crack path, force-displacement, and physics residuals
- a hybrid model interface: `predict_latent_or_field -> fem_solve -> evaluate`
- experiment configs matching the naming convention in this document
- a model registry or config layer that cleanly selects among the four neural operators in `src/models`
- a run registry writer that updates `metadata/runs/*.csv`

## 17. Suggested File/Folder Outputs

Recommended outputs for reproducibility:

- `configs/models/operator_{op1}.yaml`
- `configs/models/operator_{op2}.yaml`
- `configs/models/operator_{op3}.yaml`
- `configs/models/operator_{op4}.yaml`
- `configs/models/hybrid_paramfem.yaml`
- `configs/models/hybrid_warmstart.yaml`
- `configs/experiments/id/{model_id}.yaml`
- `configs/experiments/ood/geometry/{model_id}.yaml`
- `configs/experiments/ood/load/{model_id}.yaml`
- `configs/experiments/ood/material/{model_id}.yaml`
- `configs/experiments/ood/resolution/{model_id}.yaml`
- `configs/experiments/low_data/{model_id}_05.yaml`
- `configs/experiments/low_data/{model_id}_10.yaml`
- `configs/experiments/low_data/{model_id}_25.yaml`
- `configs/experiments/low_data/{model_id}_50.yaml`
- `configs/experiments/low_data/{model_id}_100.yaml`
- `configs/experiments/ablations/target_{variant}.yaml`
- `configs/experiments/ablations/coupling_{variant}.yaml`
- `configs/experiments/ablations/loss_{variant}.yaml`
- `configs/experiments/ablations/representation_{variant}.yaml`
- `configs/experiments/ablations/training_regime_{variant}.yaml`
- `configs/experiments/ablations/constraints_{variant}.yaml`
- `configs/experiments/ablations/solver_budget_{variant}.yaml`
- `configs/experiments/ablations/inputs_{variant}.yaml`
- `configs/experiments/ablations/oodstress_{variant}.yaml`
- `metadata/runs/screen_id.csv`
- `metadata/runs/main_id.csv`
- `metadata/runs/main_ood_geometry.csv`
- `metadata/runs/main_ood_load.csv`
- `metadata/runs/main_ood_material.csv`
- `metadata/runs/main_ood_resolution.csv`
- `metadata/runs/main_lowdata.csv`
- `metadata/runs/ablation_target.csv`
- `metadata/runs/ablation_coupling.csv`
- `metadata/runs/ablation_loss.csv`
- `metadata/runs/ablation_representation.csv`
- `metadata/runs/ablation_training_regime.csv`
- `metadata/runs/ablation_constraints.csv`
- `metadata/runs/ablation_solver_budget.csv`
- `metadata/runs/ablation_inputs.csv`
- `metadata/runs/ablation_oodstress.csv`
- `scripts/make_splits.py`
- `scripts/evaluate.py`
- `scripts/collect_metrics.py`
- `scripts/plot_force_displacement.py`

## 18. Final Recommendation

For the first paper version, prioritize this exact message:

> Instead of directly regressing fracture PDE solutions, learn solver-compatible intermediate representations and let FEM enforce mechanics.

That is the cleanest and most defensible ICML narrative.
