#!/usr/bin/env bash

set -euo pipefail
IFS=$' \n\t'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "ERROR: Python interpreter not found. Set PYTHON_BIN or create .venv." >&2
    exit 1
  fi
fi

if ! "$PYTHON_BIN" -c "import numpy" >/dev/null 2>&1; then
  echo "ERROR: numpy is not available for $PYTHON_BIN" >&2
  echo "       Activate/install your training environment first." >&2
  exit 1
fi

PRETRAIN_MODE="${PRETRAIN_MODE:-ssl}" # ssl | supervised
TEST_PROTOCOL="${TEST_PROTOCOL:-loso}" # loso | within_subject

if [[ "$PRETRAIN_MODE" != "ssl" && "$PRETRAIN_MODE" != "supervised" ]]; then
  echo "ERROR: PRETRAIN_MODE must be 'ssl' or 'supervised'" >&2
  exit 1
fi

if [[ "$TEST_PROTOCOL" != "loso" && "$TEST_PROTOCOL" != "within_subject" ]]; then
  echo "ERROR: TEST_PROTOCOL must be 'loso' or 'within_subject'" >&2
  exit 1
fi

# Source-only datasets for pretraining (BNCI intentionally excluded).
SOURCE_DATASETS_STR="${SOURCE_DATASETS:-physionetmi cho2017 lee2019_mi}"
SOURCE_DATASETS_STR="${SOURCE_DATASETS_STR//,/ }"
IFS=' ' read -r -a SOURCE_DATASETS <<<"$SOURCE_DATASETS_STR"
for ds in "${SOURCE_DATASETS[@]}"; do
  if [[ "$ds" == "bnci2014_001" ]]; then
    echo "ERROR: bnci2014_001 must not be in SOURCE_DATASETS (test-only requirement)." >&2
    exit 1
  fi
done

TEST_DATASET="${TEST_DATASET:-bnci2014_001}"
if [[ "$TEST_DATASET" != "bnci2014_001" ]]; then
  echo "WARNING: TEST_DATASET is '$TEST_DATASET' (expected bnci2014_001)." >&2
fi

DATA_PATH="${DATA_PATH:-/data/istiaqfuad/mne_data}"
SEED="${SEED:-42}"

PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-50}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-32}"
PRETRAIN_LR="${PRETRAIN_LR:-1e-3}"
PRETRAIN_WEIGHT_DECAY="${PRETRAIN_WEIGHT_DECAY:-0.0}"
PRETRAIN_VAL_SPLIT="${PRETRAIN_VAL_SPLIT:-0.2}"
DA_LAMBDA_GAMMA="${DA_LAMBDA_GAMMA:-10}"
PRETRAIN_DOMAIN_MODE="${PRETRAIN_DOMAIN_MODE:-dataset}"
MAX_SUBJECTS_PER_DATASET="${MAX_SUBJECTS_PER_DATASET:-}"
SKIP_FAILED_SUBJECTS="${SKIP_FAILED_SUBJECTS:-1}"
SUBJECT_LOAD_RETRIES="${SUBJECT_LOAD_RETRIES:-1}"
REDOWNLOAD_ON_FAILURE="${REDOWNLOAD_ON_FAILURE:-1}"
REDOWNLOAD_ONCE_PER_SUBJECT="${REDOWNLOAD_ONCE_PER_SUBJECT:-1}"
SKIP_KNOWN_FAILED_SUBJECTS="${SKIP_KNOWN_FAILED_SUBJECTS:-1}"

# DANN/SSL knobs.
DOMAIN_LOSS_WEIGHT="${DOMAIN_LOSS_WEIGHT:-1.0}"
SSL_DOMAIN_LOSS_WEIGHT="${SSL_DOMAIN_LOSS_WEIGHT:-0.2}"
SSL_WEIGHT="${SSL_WEIGHT:-1.0}"
SSL_TEMPERATURE="${SSL_TEMPERATURE:-0.2}"
SSL_PROJ_DIM="${SSL_PROJ_DIM:-128}"
SSL_HIDDEN_DIM="${SSL_HIDDEN_DIM:-256}"
SSL_NOISE_STD="${SSL_NOISE_STD:-0.02}"
SSL_TIME_MASK_RATIO="${SSL_TIME_MASK_RATIO:-0.1}"

EVAL_EPOCHS="${EVAL_EPOCHS:-30}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
EVAL_LR="${EVAL_LR:-1e-3}"
TEST_SUBJECTS_STR="${TEST_SUBJECTS:-}"

PRETRAIN_OUTPUT_DIR="${PRETRAIN_OUTPUT_DIR:-$ROOT_DIR/results/pretrain_cross_dataset}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$ROOT_DIR/results/$TEST_PROTOCOL}"

RUN_TAG="${RUN_TAG:-physio_cho_lee_${PRETRAIN_MODE}}"

mkdir -p "$PRETRAIN_OUTPUT_DIR" "$EVAL_OUTPUT_DIR"

echo "[0/3] Known failed-subject markers"
"$PYTHON_BIN" - <<PY
from pathlib import Path
import re

data_path = r"$DATA_PATH"
roots = [
    Path(data_path) / ".loader_markers" / "failed_subjects",
    Path.home() / ".cache" / "transfer-learning-bci-loader" / "failed_subjects",
]
seen = set()
groups = {}

for root in roots:
    if not root.exists():
        continue
    for marker in root.glob("*.marker"):
        key = str(marker.resolve())
        if key in seen:
            continue
        seen.add(key)
        m = re.match(r"(.+)_subject_(\d+)\.marker$", marker.name)
        if not m:
            continue
        ds = m.group(1)
        sid = int(m.group(2))
        groups.setdefault(ds, []).append(sid)

if not groups:
    print("  none")
else:
    for ds in sorted(groups):
        sids = sorted(set(groups[ds]))
        preview = sids[:20]
        suffix = " ..." if len(sids) > 20 else ""
        print(f"  {ds}: {len(sids)} failed subjects {preview}{suffix}")
PY

echo "[1/3] Starting pretraining"
echo "  mode: $PRETRAIN_MODE"
echo "  sources: ${SOURCE_DATASETS[*]}"
echo "  test-only dataset: $TEST_DATASET"

PRETRAIN_CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/training/pretrain_cross_dataset.py"
  --pretrain_mode "$PRETRAIN_MODE"
  --source_datasets "${SOURCE_DATASETS[@]}"
  --domain_mode "$PRETRAIN_DOMAIN_MODE"
  --data_path "$DATA_PATH"
  --epochs "$PRETRAIN_EPOCHS"
  --batch_size "$PRETRAIN_BATCH_SIZE"
  --lr "$PRETRAIN_LR"
  --weight_decay "$PRETRAIN_WEIGHT_DECAY"
  --val_split "$PRETRAIN_VAL_SPLIT"
  --domain_loss_weight "$DOMAIN_LOSS_WEIGHT"
  --ssl_domain_loss_weight "$SSL_DOMAIN_LOSS_WEIGHT"
  --da_lambda_gamma "$DA_LAMBDA_GAMMA"
  --ssl_weight "$SSL_WEIGHT"
  --ssl_temperature "$SSL_TEMPERATURE"
  --ssl_proj_dim "$SSL_PROJ_DIM"
  --ssl_hidden_dim "$SSL_HIDDEN_DIM"
  --ssl_noise_std "$SSL_NOISE_STD"
  --ssl_time_mask_ratio "$SSL_TIME_MASK_RATIO"
  --loader_euclidean_align
  --model_euclidean_alignment
  --model_riemannian_reweight
  --seed "$SEED"
  --deterministic
  --output_dir "$PRETRAIN_OUTPUT_DIR"
  --tag "$RUN_TAG"
  --subject_load_retries "$SUBJECT_LOAD_RETRIES"
)

if [[ -n "$MAX_SUBJECTS_PER_DATASET" ]]; then
  PRETRAIN_CMD+=(--max_subjects_per_dataset "$MAX_SUBJECTS_PER_DATASET")
fi
if [[ "$SKIP_FAILED_SUBJECTS" == "1" ]]; then
  PRETRAIN_CMD+=(--skip_failed_subjects)
else
  PRETRAIN_CMD+=(--no-skip-failed-subjects)
fi
if [[ "$REDOWNLOAD_ON_FAILURE" == "1" ]]; then
  PRETRAIN_CMD+=(--redownload_on_failure)
else
  PRETRAIN_CMD+=(--no-redownload-on-failure)
fi
if [[ "$REDOWNLOAD_ONCE_PER_SUBJECT" == "1" ]]; then
  PRETRAIN_CMD+=(--redownload_once_per_subject)
else
  PRETRAIN_CMD+=(--no-redownload-once-per-subject)
fi
if [[ "$SKIP_KNOWN_FAILED_SUBJECTS" == "1" ]]; then
  PRETRAIN_CMD+=(--skip_known_failed_subjects)
else
  PRETRAIN_CMD+=(--no-skip-known-failed-subjects)
fi

"${PRETRAIN_CMD[@]}"

echo "[2/3] Resolving latest pretrain checkpoint"
PRETRAIN_RUN_DIR="$($PYTHON_BIN - <<PY
from pathlib import Path
root = Path(r"$PRETRAIN_OUTPUT_DIR")
tag = r"$RUN_TAG"
candidates = [p for p in root.glob(f"pretrain_{tag}_*") if p.is_dir()]
if not candidates:
    raise SystemExit(1)
candidates.sort(key=lambda p: p.stat().st_mtime)
print(candidates[-1])
PY
)"

if [[ -z "$PRETRAIN_RUN_DIR" || ! -d "$PRETRAIN_RUN_DIR" ]]; then
  echo "ERROR: Could not find pretrain run directory under $PRETRAIN_OUTPUT_DIR" >&2
  exit 1
fi

CKPT_BEST="$PRETRAIN_RUN_DIR/checkpoints/pretrain_best.pt"
CKPT_LAST="$PRETRAIN_RUN_DIR/checkpoints/pretrain_last.pt"
if [[ -f "$CKPT_BEST" ]]; then
  INIT_CKPT="$CKPT_BEST"
elif [[ -f "$CKPT_LAST" ]]; then
  INIT_CKPT="$CKPT_LAST"
else
  echo "ERROR: No checkpoint found in $PRETRAIN_RUN_DIR/checkpoints" >&2
  exit 1
fi

echo "  pretrain run: $PRETRAIN_RUN_DIR"
echo "  init checkpoint: $INIT_CKPT"

echo "[3/3] Running BNCI evaluation ($TEST_PROTOCOL)"
if [[ "$TEST_PROTOCOL" == "loso" ]]; then
  EVAL_CMD=(
    "$PYTHON_BIN" "$ROOT_DIR/training/loso.py"
    --dataset "$TEST_DATASET"
    --init_checkpoint "$INIT_CKPT"
    --data_path "$DATA_PATH"
    --epochs "$EVAL_EPOCHS"
    --batch_size "$EVAL_BATCH_SIZE"
    --lr "$EVAL_LR"
    --loader_euclidean_align
    --model_euclidean_alignment
    --model_riemannian_reweight
    --seed "$SEED"
    --deterministic
    --output_dir "$EVAL_OUTPUT_DIR"
  )
  if [[ -n "$TEST_SUBJECTS_STR" ]]; then
    TEST_SUBJECTS_STR="${TEST_SUBJECTS_STR//,/ }"
    IFS=' ' read -r -a TEST_SUBJECTS <<<"$TEST_SUBJECTS_STR"
    EVAL_CMD+=(--subjects "${TEST_SUBJECTS[@]}")
  fi
  "${EVAL_CMD[@]}"
else
  EVAL_CMD=(
    "$PYTHON_BIN" "$ROOT_DIR/training/within_subject.py"
    --dataset "$TEST_DATASET"
    --init_checkpoint "$INIT_CKPT"
    --data_path "$DATA_PATH"
    --epochs "$EVAL_EPOCHS"
    --batch_size "$EVAL_BATCH_SIZE"
    --lr "$EVAL_LR"
    --loader_euclidean_align
    --model_euclidean_alignment
    --model_riemannian_reweight
    --seed "$SEED"
    --deterministic
    --output_dir "$EVAL_OUTPUT_DIR"
  )
  if [[ -n "$TEST_SUBJECTS_STR" ]]; then
    TEST_SUBJECTS_STR="${TEST_SUBJECTS_STR//,/ }"
    IFS=' ' read -r -a TEST_SUBJECTS <<<"$TEST_SUBJECTS_STR"
    EVAL_CMD+=(--subjects "${TEST_SUBJECTS[@]}")
  fi
  "${EVAL_CMD[@]}"
fi

echo "Done."
echo "- Pretrain directory: $PRETRAIN_RUN_DIR"
echo "- Checkpoint used: $INIT_CKPT"
echo "- Eval output root: $EVAL_OUTPUT_DIR"
