#!/bin/bash
#SBATCH --job-name=fracture_train
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --account=mch250029p
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ── Configurable parameters ───────────────────────────────────────────────────
# Override any of these at submission time with --export, e.g.:
#   sbatch --export=ALL,MODEL=uno,EPOCHS=300 train_on_psc.sh
MODEL=${MODEL:-fno}               # fno | uno | codano | rno
TRAIN_DIR=${TRAIN_DIR:-$SCRATCH/fracture/data/train}
VAL_DIR=${VAL_DIR:-$SCRATCH/fracture/data/val}
CKPT_DIR=${CKPT_DIR:-$SCRATCH/fracture/checkpoints/$MODEL}
EPOCHS=${EPOCHS:-200}
BATCH_SIZE=${BATCH_SIZE:-8}
INPUT_STEPS=${INPUT_STEPS:-4}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-1}
LR=${LR:-1e-4}
NUM_WORKERS=${NUM_WORKERS:-4}
CKPT_EVERY=${CKPT_EVERY:-10}
WANDB_PROJECT=${WANDB_PROJECT:-fracture_ilp}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-${MODEL}_$(date +%m%d_%H%M)}

# Resume from checkpoint (optional):
#   sbatch --export=ALL,CKPT_PATH=/path/to/epoch_0049.pt train_on_psc.sh
CKPT_PATH=${CKPT_PATH:-}

# Latent-physics-inference mode (optional):
#   sbatch --export=ALL,INFER_LATENT=1 train_on_psc.sh
INFER_LATENT=${INFER_LATENT:-0}
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
source /jet/home/yyao6/miniconda3/etc/profile.d/conda.sh
conda activate ai4phasefield

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$REPO_ROOT/src"
LOG_DIR="$REPO_ROOT/src/scripts/logs"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

# ── Info ──────────────────────────────────────────────────────────────────────
echo "=============================="
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Python      : $(python --version)"
echo "Torch       : $(python -c 'import torch; print(torch.__version__)')"
echo "Model       : $MODEL"
echo "Train dir   : $TRAIN_DIR"
echo "Val dir     : $VAL_DIR"
echo "Ckpt dir    : $CKPT_DIR"
echo "Epochs      : $EPOCHS  |  Batch: $BATCH_SIZE  |  LR: $LR"
echo "Input steps : $INPUT_STEPS  |  Rollout: $ROLLOUT_STEPS"
echo "=============================="

# ── Build argument list ───────────────────────────────────────────────────────
ARGS=(
    --train-dir      "$TRAIN_DIR"
    --val-dir        "$VAL_DIR"
    --ckpt_dir       "$CKPT_DIR"
    --model          "$MODEL"
    --epochs         "$EPOCHS"
    --batch_size     "$BATCH_SIZE"
    --input_steps    "$INPUT_STEPS"
    --rollout_steps  "$ROLLOUT_STEPS"
    --lr             "$LR"
    --num_workers    "$NUM_WORKERS"
    --ckpt_every     "$CKPT_EVERY"
    --wandb_project  "$WANDB_PROJECT"
    --wandb_run_name "$WANDB_RUN_NAME"
    --device         cuda
)

[[ -n "$CKPT_PATH"     ]] && ARGS+=(--ckpt_path "$CKPT_PATH")
[[ "$INFER_LATENT" == "1" ]] && ARGS+=(--infer_latent_variable)

# ── Launch ────────────────────────────────────────────────────────────────────
echo "Command: python $SRC/train.py ${ARGS[*]}"
python "$SRC/train.py" "${ARGS[@]}"
