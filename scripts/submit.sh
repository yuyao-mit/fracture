#!/bin/bash
#SBATCH --job-name=fracture
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --account=m4891_g

# Submit one config-driven run on NERSC Perlmutter.
#
# Required env: CONFIG (path to experiment yaml), RUN_NAME (for logs).
# Optional env:
#   RUN_LOG_DIR             where to write logs (default: pscratch experiments/logs/$RUN_NAME)
#   ENV_MODE                "conda" (default) | "module"
#   CONDA_ENV               conda env name when ENV_MODE=conda (default phasefield-ml)
#   OVERRIDES               space-separated dotted.key=value tokens (no spaces/metachars)
#   PREFLIGHT_CUDA_STRICT   1 (default) | 0
#
# Usage (preferred): invoked by scripts/submit_batch.py which fills in
#                    sbatch --account/--job-name/--output/--error (these override the
#                    #SBATCH lines above) and exports CONFIG/RUN_NAME/RUN_LOG_DIR/CONDA_ENV.
#
# QOS notes (Perlmutter GPU): `shared` (1 GPU / 1-4 of a node, MaxWall 2 days) is the
# default here so many screening runs pack onto few nodes. Override per submission with
# sbatch --qos=regular / --time=... if a single run needs a full node or longer wall.

set -euo pipefail

: "${CONFIG:?CONFIG=<path> required}"
: "${RUN_NAME:?RUN_NAME=<name> required}"
RUN_LOG_DIR="${RUN_LOG_DIR:-/pscratch/sd/y/yu_yao/MyQuota/fracture_experiments/logs/$RUN_NAME}"
ENV_MODE="${ENV_MODE:-conda}"
CONDA_ENV="${CONDA_ENV:-phasefield-ml}"
OVERRIDES="${OVERRIDES:-}"
PREFLIGHT_CUDA_STRICT="${PREFLIGHT_CUDA_STRICT:-1}"

mkdir -p "$RUN_LOG_DIR"

# Under SLURM, $0 is the spooled script copy, so trust SLURM_SUBMIT_DIR (the cwd at
# sbatch time) for the repo root and fall back to dirname-of-$0 for local runs.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

case "$ENV_MODE" in
  conda)
    # NERSC central Miniforge; the phasefield-ml env provides torch 2.10 + neuralop deps.
    source /global/common/software/nersc/pe/conda/24.10.0/Miniforge3-24.7.1-0/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV"
    PY="$(which python)"
    ;;
  module)
    module load conda
    conda activate "$CONDA_ENV"
    PY="$(which python)"
    ;;
  *)
    echo "unknown ENV_MODE=$ENV_MODE" >&2
    exit 2
    ;;
esac

cd "$REPO_ROOT"

# wandb-core's IPC port-file handshake times out on Lustre (an inherited $TMPDIR may
# point at pscratch); use fast node-local /tmp for transient service files instead.
export TMPDIR="/tmp/frac_${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-120}"

echo "============================================================"
echo "  run      : $RUN_NAME"
echo "  config   : $CONFIG"
echo "  overrides: $OVERRIDES"
echo "  log dir  : $RUN_LOG_DIR"
echo "  slurm job: ${SLURM_JOB_ID:-<local>}  on ${SLURMD_NODENAME:-$(hostname)}"
echo "  env mode : $ENV_MODE ($CONDA_ENV)"
echo "  python   : $PY  $($PY -V 2>&1)"
echo "  gpu      : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo N/A)"
echo "============================================================"

STRICT_FLAG=""
[[ "$PREFLIGHT_CUDA_STRICT" = "1" ]] && STRICT_FLAG="--strict-cuda"
$PY scripts/preflight.py --config "$CONFIG" $STRICT_FLAG $OVERRIDES

$PY scripts/train.py --config "$CONFIG" $OVERRIDES
