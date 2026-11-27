#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Must run via sbatch}"
: "${SLURM_ARRAY_TASK_ID:?Must run as a Slurm job array}"
WORK="/scratch/$USER/conceptlens-verifier"
PYTHON="$WORK/.venv/bin/python"
export PATH="/scratch/$USER/ollama/bin:$PATH"
export OLLAMA_MODELS="/scratch/$USER/ollama-models"
export OLLAMA_HOST="127.0.0.1:$((20000 + SLURM_JOB_ID % 20000))"
export OLLAMA_CONTEXT_LENGTH=4096 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NO_CLOUD=1
cd "$WORK"
[[ -f eval/phase3_final_contexts.jsonl && -f eval/phase3_final_manifest.json ]] || { echo 'Missing frozen benchmark inputs' >&2; exit 1; }
"$PYTHON" - <<'PY'
from scripts.run_phase3_final import load_manifest,load_contexts
m=load_manifest('eval/phase3_final_manifest.json','eval/phase3_final_contexts.jsonl')
r=load_contexts('eval/phase3_final_contexts.jsonl')
print('Frozen preflight passed:',len(r),'contexts, model',m['model'],'prompt',m['prompt_variant'],'k',m['retrieval_k'])
PY
mkdir -p eval "$OLLAMA_MODELS"
nvidia-smi
ollama serve > "eval/ollama_phase3final_${SLURM_JOB_ID}.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT
ready=0
for i in $(seq 1 60);do
  if curl -fsS "http://$OLLAMA_HOST/api/tags" >/dev/null 2>&1;then ready=1;break;fi
  if ! kill -0 "$server_pid" 2>/dev/null;then break;fi
  sleep 1
done
[[ "$ready" == 1 ]] || { tail -n 40 "eval/ollama_phase3final_${SLURM_JOB_ID}.log";exit 1; }
if ! ollama list | awk '$1=="qwen2.5:7b"{found=1} END{exit !found}';then
  ollama pull qwen2.5:7b
fi
printf 'Reply OK.\n' | ollama run qwen2.5:7b >/dev/null
ollama ps
ollama ps | grep -q '100% GPU' || { echo 'GPU-only inference not confirmed' >&2;exit 1; }
OUT="eval/phase3_final_7b_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.jsonl"
"$PYTHON" -u -m scripts.run_phase3_final --part "$SLURM_ARRAY_TASK_ID" --output "$OUT"
