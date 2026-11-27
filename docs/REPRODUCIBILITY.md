# Reproducibility — ConceptLens Phase 3 and Streamlit demo

## What is reproducible from this repository?

The **archived** final 96-case assessment scores can be checked without Ollama using the versioned case-level results, summary metrics, and `scripts/verify_published_results.py`:

```bash
python -m scripts.verify_published_results
```

This verifies the published arithmetic, class/category counts, and citation-audit totals. **It is not an independent replication** of the model inference, expert validation, or independently sampled student accuracy.

To reproduce Qwen inference end-to-end, you additionally need the **exact frozen** `phase3_final_contexts.jsonl`, `phase3_final_manifest.json`, and original frozen gold-label CSV from the completed experiment. Those were created on the project author's Mac before inference and should be preserved unchanged. We did **not** invent replacement hashes or assert that the draft CSV in the original ZIP is identical to the approved gold file. The frozen `phase3_final_gold_LOCAL_ONLY.csv` is deliberately gitignored; do not copy it to Hopper or publish it accidentally.

## Frozen evaluation specification

| Component | Value |
| --- | --- |
| GPU runtime | GMU Hopper Slurm; `gpuq`, `--qos=gpu`, `--gres=gpu:1g.10gb:1` |
| Model | Ollama `qwen2.5:7b`, temperature 0 |
| Prompt | `prompt_variant="checklist"` |
| Retrieval | MiniLM semantic (`all-MiniLM-L6-v2`), subject filtered, four distinct sources |
| Chunking | 1,100 characters; 100 overlap |
| Corpus | 32 curated, project-authored concept notes |
| Evaluation | 96 project-authored answer examples, 48 per subject, 24 per expected category |
| Recorded Slurm array | `3884168`, tasks `0`, `1`, `2`; all completed successfully |
| Output name pattern | `eval/phase3_final_7b_3884168_{0,1,2}.jsonl` |
| Final exact agreement | 77/96 (80.2%), four abstentions |

The model was given the question, student answer and **frozen retrieved passages**, not the gold label or grading rationale. The frozen source contexts were prepared on the Mac and transferred to Hopper without the gold file.

## Install / rerun the app

From a clean clone (Python 3.11–3.12 suggested):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama pull qwen2.5:7b
python -m scripts.index_knowledge
python -m streamlit run app.py
```

The local app requires Ollama to be running. Compare the script and model digest against the frozen configuration before making strict reproduction claims:

```bash
ollama list
python --version
python -m pip freeze
```

The repository uses dependency ranges, not an exact locked environment; even with the same prompt, generated responses can differ with Ollama/llama.cpp versions, model digest, hardware or library versions. `temperature=0` does not guarantee bit-for-bit reproduction across all runtimes.

## Original experiment protocol (for a new controlled repetition)

If you are running a new evaluation, not claiming the historical 96 examples are untouched:

1. Review all `eval/phase3_final_review.csv` expected labels **before inference**. The original final kit was distributed separately; do not substitute an old draft for the approved CSV.
2. Use `python -m scripts.prepare_phase3_final --approved` to capture retrieved chunks and create hash-protected manifests. It refuses to overwrite existing frozen files.
3. Transfer **only** inference code, frozen contexts and manifest (not local gold labels) to Hopper.
4. Submit three sequential array tasks (`--array=0-2%1`) using `run_hopper_phase3_final.sh`.
5. Copy predictions back and run `python -m scripts.score_phase3_final --results 'eval/phase3_final_7b_ARRAYID_*.jsonl'` once against the frozen local gold.
6. Review actual cited passages; do not count a valid numeric ID as semantically supported evidence.

The detailed original runbook is in the separately distributed Phase 3 evaluation kit. If retaining its original `README.md`, put it under `docs/ARCHIVED_PHASE3_RUNBOOK.md` rather than replacing the repository root README.

**Do not reuse previously inspected final examples as a new untouched accuracy benchmark.** Keep the 80.2% as the recorded final result. Changed models, prompts, retrieval, or knowledge content need a *different* new test set for new claims.

## Published artifacts and interpretation

- `eval/phase3_final_report_summary.json` — exact label agreement, per-group counts, confusion matrix and descriptive interval.
- `eval/phase3_final_report_case_results.csv` — predictions vs expected labels and per-case rationale.
- `eval/phase3_final_report_citation_audit.csv` — model explanations and cited source text; numeric audit fields originally unfilled.
- `eval/ConceptLens_Phase3_Final_Citation_Audit_AI_Reviewed.csv` — assistant's **preliminary**, case-by-case factual and citation ratings.
- `docs/FINAL_EVALUATION.md` — summarized results and known errors.

The 96 example questions are clustered by concept families; the Wilson interval treats cases as though independent and is **descriptive only**. This work does not have an independently sampled real-student benchmark or a blinded second expert rater.

## What to preserve for long-term reproducibility

Retain the frozen manifest and all original prediction JSONL files outside a shared public repo until you have reviewed them for private data; record their SHA-256 values. If you publish them later, mark them as archived historical outputs and never quietly rewrite them. Preserve the frozen evaluated code revision separately from any future UI changes. Do not present the current repository's HEAD as the precise frozen research snapshot unless the actual manifest verifies that match.
