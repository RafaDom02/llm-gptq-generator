#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-9B}"
QUANT_MODEL="${QUANT_MODEL:-./Qwen3.5-9B-GPTQ-Int4}"
EVAL_BS="${EVAL_BS:-2}"
TASKS="${TASKS:-mmlu}"
BASE_LOG="${BASE_LOG:-./eval_base_mmlu.log}"
QUANT_LOG="${QUANT_LOG:-./eval_quant_mmlu.log}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-quant_clean311}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

AUTO_ROUND_BIN="${HOME}/miniconda3/envs/${CONDA_ENV_NAME}/bin/auto-round"
if [[ ! -x "${AUTO_ROUND_BIN}" ]]; then
  echo "ERROR: No encuentro auto-round en ${AUTO_ROUND_BIN}" >&2
  exit 1
fi

echo "=== Benchmark MMLU ==="
echo "BASE_MODEL : ${BASE_MODEL}"
echo "QUANT_MODEL: ${QUANT_MODEL}"
echo "TASKS      : ${TASKS}"
echo "EVAL_BS    : ${EVAL_BS}"
echo "CONDA_ENV  : ${CONDA_ENV_NAME}"
echo "AUTO_ROUND : ${AUTO_ROUND_BIN}"
echo

echo "[1/2] Evaluando modelo base..."
"${AUTO_ROUND_BIN}" \
  --model "${BASE_MODEL}" \
  --eval \
  --tasks "${TASKS}" \
  --eval_task_by_task \
  --eval_bs "${EVAL_BS}" \
  2>&1 | tee "${BASE_LOG}"

echo
echo "[2/2] Evaluando modelo cuantizado..."
"${AUTO_ROUND_BIN}" \
  --model "${QUANT_MODEL}" \
  --eval \
  --tasks "${TASKS}" \
  --eval_task_by_task \
  --eval_bs "${EVAL_BS}" \
  2>&1 | tee "${QUANT_LOG}"

echo
echo "Benchmark terminado."
echo "Logs:"
echo "  - ${BASE_LOG}"
echo "  - ${QUANT_LOG}"
