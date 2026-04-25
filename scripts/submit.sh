#!/bin/bash
#SBATCH --job-name=fracture
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --account=mch250029p

# Submit one config-driven run.
# Required env: CONFIG (path to experiment yaml), RUN_NAME (for logs), RUN_LOG_DIR.
# Optional env:
#   OVERRIDES               space-separated dotted.key=value tokens (no spaces/metachars)
#   ENV_MODE                "module" (default, uses PSC AI pytorch module) | "conda"
#   CONDA_ENV               env name when ENV_MODE=conda (default ai4phasefield)
#   PREFLIGHT_CUDA_STRICT   1 (default) | 0
#
# Usage (preferred): invoked by scripts/submit_batch.py which fills in
#                    sbatch --output/--error and exports CONFIG/RUN_NAME/etc.

set -euo pipefail

: "${CONFIG:?CONFIG=<path> required}"
: "${RUN_NAME:?RUN_NAME=<name> required}"
RUN_LOG_DIR="${RUN_LOG_DIR:-/ocean/projects/mch250029p/shared/experiments/fracture/logs/$RUN_NAME}"
ENV_MODE="${ENV_MODE:-module}"
CONDA_ENV="${CONDA_ENV:-ai4phasefield}"
OVERRIDES="${OVERRIDES:-}"
PREFLIGHT_CUDA_STRICT="${PREFLIGHT_CUDA_STRICT:-1}"

mkdir -p "$RUN_LOG_DIR"

case "$ENV_MODE" in
  module)
    # The 25.02 module's python is a broken symlink into another user's home on
    # current PSC nodes. 23.02 works directly. Deps live on /ocean (HOME quota
    # is full). The neuraloperator vendored package is on sys.path by script.
    MODULE_PY="/opt/packages/AI/pytorch_23.02-1.13.1-py3/bin/python"
    OCEAN_LIBS="/ocean/projects/mch250029p/yyao6/pylibs_pt1131"
    REPO_ROOT_EARLY="$(cd "$(dirname "$0")/.." && pwd)"
    NEURALOP_PATH="$REPO_ROOT_EARLY/src/models/neuraloperator"
    export PYTHONPATH="${OCEAN_LIBS}:${NEURALOP_PATH}:${PYTHONPATH:-}"
    PY="$MODULE_PY"
    ;;
  conda)
    source /jet/home/yyao6/miniconda3/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV"
    PY="$(which python)"
    ;;
  *)
    echo "unknown ENV_MODE=$ENV_MODE" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "============================================================"
echo "  run      : $RUN_NAME"
echo "  config   : $CONFIG"
echo "  overrides: $OVERRIDES"
echo "  log dir  : $RUN_LOG_DIR"
echo "  slurm job: ${SLURM_JOB_ID:-<local>}  on ${SLURMD_NODENAME:-$(hostname)}"
echo "  env mode : $ENV_MODE"
echo "  python   : $PY  $($PY -V 2>&1)"
echo "  gpu      : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "============================================================"

STRICT_FLAG=""
[[ "$PREFLIGHT_CUDA_STRICT" = "1" ]] && STRICT_FLAG="--strict-cuda"
$PY scripts/preflight.py --config "$CONFIG" $STRICT_FLAG $OVERRIDES

$PY scripts/train.py --config "$CONFIG" $OVERRIDES
