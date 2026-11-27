# ConceptLens — Phase 3 Final Evaluation and Citation Audit

**Historical final evaluation; no post-test tuning was performed on these cases.**

**Evaluation status:** Final frozen candidate tested on 96 project-authored examples. Model judgments were compared with preassigned labels. Citation support was subsequently assessed by an AI-assisted, case-level qualitative review; **this is not independent human or expert validation.**

## Frozen candidate

- Model: Qwen2.5 7B via Ollama on GMU Hopper.
- Assessment prompt: checklist variant.
- Retrieval: MiniLM + Chroma, four distinct semantic retrieval sources (`k=4`).
- Chunking: 1,100 characters, with 100-character overlap.
- Structured output and citation-ID validation: corrected feedback engine.
- Dataset: 96 project-authored question/answer examples; 48 computer science and 48 electronics; 24 in each of four expected categories. Topic-family examples are clustered, so they are not 96 independent real-world students.

## 1. Assessment agreement (primary result)

**77/96 = 80.2% exact label agreement.** This falls below the project aspiration of 85–90%. The descriptive 95% Wilson interval is **71.1%–86.9%**, which is not evidence of generalization to real student responses. Four responses abstained.

| Expected category | Exact matches | Agreement |
| --- | ---: | ---: |
| Correct | 19/24 | 79.2% |
| Incomplete | 18/24 | 75.0% |
| Incorrect | 23/24 | 95.8% |
| Off-topic | 17/24 | 70.8% |
| **Overall** | **77/96** | **80.2%** |

Subject results: computer science **38/48 = 79.2%**; electronics **39/48 = 81.2%**.

**19 disagreements:** five correct answers called incomplete; four incomplete answers called correct; two incomplete answers called incorrect; one incorrect answer called correct; four off-topic answers caused abstention; one off-topic answer was called incomplete; and two off-topic answers were called incorrect.

## 2. Citation-support audit (AI-assisted; provisional)

All 96 supplied rows were reviewed. There were **75 responses with citations** and **21 responses without citations**. Of the uncited responses, 17 were classified off-topic and appropriately required no substantive reference citation; four were `cannot_assess` abstentions. The latter four were expected off-topic answers, so they represent unnecessary withholding rather than a citation failure.

**Scoring rubric:** `2` = the cited passages substantively support the explanation's central claim; `1` = partially supported, irrelevant extra citations, or failure to support an important assessment inference; `0` = unrelated evidence or cited evidence contradicts the feedback's key judgment. No-citation responses receive an empty support rating rather than an artificial zero.

| Citation support | Cases | Percentage of cited responses |
| --- | ---: | ---: |
| Fully supported (2/2) | **52** | **69.3%** |
| Partially supported (1/2) | **19** | **25.3%** |
| Unsupported/contradictory (0/2) | **4** | **5.3%** |
| **Total cited** | **75** | **100%** |

These percentages are **preliminary AI qualitative ratings**, not a blinded independent expert assessment. Citation IDs being within range cannot prove passage entailment.

### Clear citation failures

- **Z03C, Z03I, Z03P (FIFO/queue):** citations refer to process/thread, mutex, race-condition, or deadlock notes rather than the FIFO ordering rule. One of these also wrongly claims the student omitted the FIFO explanation.
- **Z18I (Ohm's law):** the student wrongly says doubling resistance doubles current at fixed voltage. The feedback labels this `correct`, while its *own explanation* says current will **halve**. The cited passage contradicts the label. This is a safety-relevant consistency failure for assessment quality, even though the arithmetic concept in the explanation is correct.

### Partial-support themes

- **Z13I/Z13P, Z14C, Z18C, Z24C:** the correct core evidence is bundled with other irrelevant or tangential citations.
- **Z11P:** correct leakage observation is mislabeled incorrect, and the explanation adds an unjustified requirement about overfitting.
- **Z21P/Z23P:** the evidence supplies missing facts, but the model incorrectly credits those facts to the student's answer and labels incomplete answers correct.
- **Z06I:** cited note correctly distinguishes stack from CPU cache but does not actually explain the CPU cache's storage role.

## 3. Separate factual-feedback rating (AI-assisted; provisional)

Here `2` means the response and grading interpretation are materially sound, `1` means partly useful but contains a misleading judgment, unsupported inference or terminology issue, and `0` means a materially false or contradictory feedback assessment.

| Factual-feedback rating | Responses | Share of all 96 |
| --- | ---: | ---: |
| Fully sound (2/2) | **72** | **75.0%** |
| Partially sound (1/2) | **17** | **17.7%** |
| Material issue (0/2) | **7** | **7.3%** |

This is not an independently established factual-correctness rate; it is an AI-assisted qualitative rubric intended to make manual follow-up possible. The original assessment-label agreement remains **80.2%**, not 75.0%.

Notable problematic feedback includes **Z01C, Z03C, Z09C, Z12C, and Z16C** (claiming a correct answer omitted information the student actually wrote), as well as the **Z18I** contradiction. Off-topic misclassifications and unnecessary abstentions are additional limitations.

## 4. Interpretation and decision

- The final test is **complete** and should **not be retuned** after inspecting its errors. No 85–90% claim is supported.
- Candidate retrieval and chunking improved a particular development-time FIFO citation scenario, but **did not eliminate queue-related citation failures** in the new test.
- **80.2% assessment agreement** and **69.3% preliminarily fully supported citation responses** answer different questions; they should never be conflated.
- The results support an **educational feedback research prototype**, with provisional assessments, visible retrieved passages, and a warning that feedback can be wrong. It should **not be marketed as reliable autonomous grading**.
- Improving citation semantic entailment and student-answer coverage remains future work. This benchmark can be used for reporting mistakes, but **not for further tuning while treating it as an untouched test**.

## 5. Phase completion and next work

**Phase 3 experimental evaluation may be closed** with the accuracy and these explicitly labeled preliminary qualitative findings. If the course requires human-validated citation-quality statistics, the attached 96-row audit should be independently checked before those figures enter the final report as verified results.

**Phase 4 (completed):** The redesigned Streamlit application calls `src.feedback_engine.give_feedback` with an explicit local Ollama `qwen2.5:7b` model at temperature 0, `prompt_variant="checklist"`, and semantic retrieval `k=4` from `.chromadb_1100`. The UI displays provisional assessments, explanations, revision hints, and model-selected citations separately from other retrieved passages; it also offers demonstration examples and Markdown feedback export. The application was smoke-tested locally on a Mac with Ollama GPU inference, including multiple expected assessment categories and an Electronics example. These functional checks are **not** a new quantitative evaluation or independent verification of citation support.

**Phase 5:** Final project report and demo with reproducible run instructions, the frozen configuration, performance table, citation examples, limitations, and evidence provenance.

## Source files

- Original uploaded assessment results: `phase3_final_report_case_results.csv`
- Original uploaded citation responses and cited passage text: `phase3_final_report_citation_audit.csv`
- Original uploaded quantitative summary: `phase3_final_report_summary.json`
- Produced scored audit: `ConceptLens_Phase3_Final_Citation_Audit_AI_Reviewed.csv`
- Produced machine-readable audit metrics: `ConceptLens_Phase3_Final_Citation_Audit_Metrics.json`

The original three uploaded files were not edited. All later citation/factual scores are an assistant-produced preliminary assessment based on their contents.

---

**Publication note:** This report is an AI-assisted analysis of model outputs and their cited text. The reviewed annotations have not been presented as independent expert labels. The final model may behave differently in later runs.
