#!/bin/bash
#SBATCH --job-name=fracture_infer
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --account=mch250029p
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ── Configurable parameters ───────────────────────────────────────────────────
# Override at submission time with --export, e.g.:
#   sbatch --export=ALL,MODEL=uno,CKPT_PATH=/path/to/epoch_0099.pt inference_on_psc.sh
MODEL=${MODEL:-fno}               # fno | uno | codano | rno
DATA_DIR=${DATA_DIR:-$SCRATCH/fracture/data/test}
CKPT_PATH=${CKPT_PATH:?'ERROR: CKPT_PATH is required. Set via --export=ALL,CKPT_PATH=...'}
OUTPUT_DIR=${OUTPUT_DIR:-$SCRATCH/fracture/results/$(basename "$CKPT_PATH" .pt)_$(date +%m%d_%H%M)}
BATCH_SIZE=${BATCH_SIZE:-16}
INPUT_STEPS=${INPUT_STEPS:-4}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-1}
NUM_WORKERS=${NUM_WORKERS:-4}

# Latent-physics-inference mode (optional):
#   sbatch --export=ALL,CKPT_PATH=...,INFER_LATENT=1 inference_on_psc.sh
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
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# ── Info ──────────────────────────────────────────────────────────────────────
echo "=============================="
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Python      : $(python --version)"
echo "Torch       : $(python -c 'import torch; print(torch.__version__)')"
echo "Model       : $MODEL"
echo "Data dir    : $DATA_DIR"
echo "Checkpoint  : $CKPT_PATH"
echo "Output dir  : $OUTPUT_DIR"
echo "Batch size  : $BATCH_SIZE"
echo "=============================="

# ── Build argument list ───────────────────────────────────────────────────────
ARGS=(
    --data-dir      "$DATA_DIR"
    --ckpt_path     "$CKPT_PATH"
    --output_dir    "$OUTPUT_DIR"
    --model         "$MODEL"
    --batch_size    "$BATCH_SIZE"
    --input_steps   "$INPUT_STEPS"
    --rollout_steps "$ROLLOUT_STEPS"
    --num_workers   "$NUM_WORKERS"
    --device        cuda
)

[[ "$INFER_LATENT" == "1" ]] && ARGS+=(--infer_latent_variable)

# ── Launch ────────────────────────────────────────────────────────────────────
echo "Command: python $SRC/inference.py ${ARGS[*]}"
python "$SRC/inference.py" "${ARGS[@]}"

echo "=============================="
echo "Done. Results saved to: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
