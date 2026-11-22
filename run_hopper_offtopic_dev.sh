#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run this script only inside a Slurm GPU job}"
MODEL="${1:?Usage: run_hopper_offtopic_dev.sh qwen2.5:3b|qwen2.5:7b}"
case "$MODEL" in qwen2.5:3b|qwen2.5:7b) ;; *) echo "Unsupported model" >&2; exit 2;; esac
WORK="/scratch/$USER/conceptlens-verifier"
PYTHON="$WORK/.venv/bin/python"
export PATH="/scratch/$USER/ollama/bin:$PATH"
export OLLAMA_MODELS="/scratch/$USER/ollama-models"
export OLLAMA_HOST="127.0.0.1:$((20000 + SLURM_JOB_ID % 20000))"
export OLLAMA_CONTEXT_LENGTH=4096 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NO_CLOUD=1
cd "$WORK"
mkdir -p eval "$OLLAMA_MODELS"
nvidia-smi
ollama serve > "eval/ollama_offtopic_${SLURM_JOB_ID}.log" 2>&1 &
pid=$!
trap 'kill "$pid" 2>/dev/null || true' EXIT
ready=0
for i in $(seq 1 60); do
  if curl -fsS "http://$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then ready=1; break; fi
  if ! kill -0 "$pid" 2>/dev/null; then break; fi
  sleep 1
done
[[ "$ready" == 1 ]] || { tail -n 40 "eval/ollama_offtopic_${SLURM_JOB_ID}.log"; exit 1; }
if ! ollama list | awk -v model="$MODEL" '$1==model{found=1} END{exit !found}'; then ollama pull "$MODEL"; fi
printf 'Reply with OK only.\n' | ollama run "$MODEL" >/dev/null
ollama ps
ollama ps | grep -q '100% GPU' || { echo 'GPU-only inference not confirmed' >&2; exit 1; }
TAG="${MODEL##*:}"
"$PYTHON" -u -m scripts.evaluate_offtopic_fix_dev --model "$MODEL" \
  --output "eval/offtopic_fix_dev_${TAG}_${SLURM_JOB_ID}.jsonl"
