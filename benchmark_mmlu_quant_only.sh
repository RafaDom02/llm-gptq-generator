#!/usr/bin/env bash
set -euo pipefail

QUANT_MODEL="${QUANT_MODEL:-./Qwen3.5-9B-GPTQ-Int4}"
EVAL_BS="${EVAL_BS:-2}"
TASKS="${TASKS:-mmlu}"
QUANT_LOG="${QUANT_LOG:-./eval_quant_mmlu.log}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate quant_clean

echo "=== Benchmark MMLU Cuantizado ==="
echo "QUANT_MODEL: ${QUANT_MODEL}"
echo "TASKS      : ${TASKS}"
echo "EVAL_BS    : ${EVAL_BS}"
echo

auto-round \
  --model "${QUANT_MODEL}" \
  --eval \
  --tasks "${TASKS}" \
  --eval_task_by_task \
  --eval_bs "${EVAL_BS}" \
  2>&1 | tee "${QUANT_LOG}"

echo
echo "Benchmark terminado."
echo "Log: ${QUANT_LOG}"
