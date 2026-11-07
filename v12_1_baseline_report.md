# V12.1 full baseline report

Status: **FAIL**  
Run date: 2026-11-07 to 2026-11-08 (Asia/Taipei)  
Label: `baseline_v12_1_post_recovery_full`  
Branch: `feature/research-agent-adaptive-iteration`  
Test commit: `846b0a7`  
Mode: Mode 2 (`gold`), 12 questions

## Frozen execution context

- Project: `projects/boron_bnct`
- Production environment: `G:\Anaconda\envs\llm_env`
- Ollama: 0.40.15; existing local models only; no model download or index rebuild
- Main model: `qwen3.5:27b`
- Planning model: `qwen2.5:14b`
- Synthesis model: `gemma4:31b`
- Verification/judge model: `qwen3.5:35b-a3b`
- Embedding model: `bge-m3`
- Context window: 32768; Stage 3/5 context: 16384
- Retrieval: vector/BM25 top-k 8; reranker disabled
- Plan-and-Execute: disabled
- Answerability, structured fact contract, direct render, Stage 4 validation, query-aware evidence and final translation: enabled
- NVIDIA/NLI path: CUDA on the detected RTX 3090

No production setting, model, PDF, index, embedding, reranker or top-k value was changed for this run.

## Result summary

| Metric | Result | Gate | Verdict |
|---|---:|---:|---|
| Average correctness | 0.833 | >= 0.90 | FAIL |
| Per-question correctness | Q01/Q03/Q06 = 0.50 | every question >= 0.75 | FAIL |
| Q11/Q12 correctness | 1.00 / 1.00 | both = 1.00 | PASS |
| Paper selection recall | 1.000 | 1.000 | PASS |
| Average grounding | 0.791 | >= V11 1.000 | FAIL |
| Average translation fidelity | 0.950 | >= V11 0.950 | PASS at the formal floor; regressed from the prior V12.1 product run's 1.000 |
| Retriever candidate recall | 0.947 | diagnostic | INFO |
| Stage 2 evidence recall | 0.830 | diagnostic | INFO |
| Average total latency | 618.7 s | diagnostic | INFO |

The full run therefore cannot be frozen or promoted as V12.1.

## Per-question metrics

| ID | Correctness | Translation | Grounding | Paper selection | Candidate recall | Stage 2 recall | Wall time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q01 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 0.667 | 443.5 s |
| Q02 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 | 1154.5 s |
| Q03 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 | 471.2 s |
| Q04 | 1.00 | 1.00 | 0.857 | 1.00 | 1.00 | 1.000 | 481.5 s |
| Q05 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 | 496.9 s |
| Q06 | 0.50 | 1.00 | 1.00 | 1.00 | 0.80 | 1.000 | 613.6 s |
| Q07 | 0.75 | 1.00 | 1.00 | 1.00 | 0.667 | 0.667 | 502.2 s |
| Q08 | 1.00 | 1.00 | 0.750 | 1.00 | 1.00 | 1.000 | 613.2 s |
| Q09 | 0.75 | 1.00 | 0.200 | 1.00 | 1.00 | 0.714 | 1373.5 s |
| Q10 | 1.00 | 0.50 | 0.100 | 1.00 | 1.00 | 0.250 | 988.2 s |
| Q11 | 1.00 | N/A | N/A | N/A | N/A | N/A | 143.2 s |
| Q12 | 1.00 | N/A | N/A | N/A | N/A | N/A | 143.0 s |

## Failure evidence

- Q01 omitted the co-incubation-only and combined pre/co-incubation IC50 values.
- Q03 omitted the traditional substrate-pocket statement and the competitive-blocking mechanism.
- Q06 omitted PVA-BPA boronate/NMR evidence, LAT1-mediated endocytosis and the four median-survival values.
- Q07 met the correctness floor but omitted the explicit pH 2/pH 10 hydrogel-stability relation.
- Q08 had two unsupported claims.
- Q09 had one missing required structural-design relation and 20 unsupported claims; grounding was 0.200.
- Q10 had nine unsupported claims, grounding 0.100 and one material translation error.
- Q11/Q12 correctly rejected unavailable or false-premise requests without fabricating values.

There were no pipeline errors, repair attempts, fallbacks or conflicts in the console log. The process exited with code 0. A non-fatal Colorama cleanup callback wrote to an already closed stream after report generation; it did not alter the artifacts or exit code.

## Comparison with the prior V12.1 product run

| Metric | `baseline_v12_1_product_full` | Current | Delta |
|---|---:|---:|---:|
| Correctness | 1.000 | 0.833 | -0.167 |
| Grounding | 0.988 | 0.791 | -0.197 |
| Translation | 1.000 | 0.950 | -0.050 |
| Paper selection | 1.000 | 1.000 | 0.000 |
| Stage 2 recall | 0.882 | 0.830 | -0.052 |
| Average latency | 419.3 s | 618.7 s | +199.4 s |

## Artifact handling

The private JSON, full generated Markdown, console log and per-question debug artifacts remain under `eval/results/` and are intentionally ignored because they contain verbatim or derived paper content. This report contains only the non-private metrics and failure summary needed for the repository checkpoint.

## Next gate

Reproduce the Q01/Q03/Q06 omissions and Q08-Q10 grounding failures from saved artifacts, fix the shared evidence-selection/rendering root causes, run focused regressions, and then repeat all 12 questions with the same frozen configuration. Do not enter the full Research Agent + RAG GPU benchmark or merge PR #10 until that fresh full run passes.
