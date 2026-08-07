# Academic RAG Pipeline

> A local, citation-grounded RAG pipeline for academic paper analysis — built as a personal research thinking partner.

This project is a local RAG pipeline designed for deep academic paper analysis.
Unlike simple retrieve-and-answer systems, it uses a 7-stage pipeline —
sub-question decomposition, hybrid retrieval, knowledge distillation,
logical verification, and sentence-level NLI citation grounding — to produce
citation-grounded answers you can actually trace back to the source.

Built for researchers who want a stable, controllable thinking partner
rather than a black-box AI that changes without notice.

> Powered by [Ollama](https://ollama.com) + [LlamaIndex](https://www.llamaindex.ai) + [ChromaDB](https://www.trychroma.com), with an OpenAI-compatible API for [Open WebUI](https://github.com/open-webui/open-webui) integration.

---

## Table of Contents / 目錄

- [English](#english)
  - [Features](#features)
  - [Development Status](#development-status)
  - [System Architecture](#system-architecture)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Project Structure](#project-structure)
  - [Configuration](#configuration)
- [繁體中文](#繁體中文)
  - [功能特色](#功能特色)
  - [開發狀態](#開發狀態)
  - [系統架構](#系統架構)
  - [環境需求](#環境需求)
  - [安裝步驟](#安裝步驟)
  - [使用方式](#使用方式)
  - [專案結構](#專案結構)
  - [參數設定](#參數設定)

---

# English

## Features

- **7-Stage Pipeline**: Paper pre-filter → ① Sub-question planning → ② Hybrid retrieval → ③ Knowledge distillation → ④ answer rendering/generation → ⑤ Verification & correction → ⑥ NLI citation grounding → ⑦ optional Traditional Chinese translation
- **Hybrid Retrieval**: BM25 sparse + vector dense search, with optional cross-encoder reranking
- **Three-Tier Answers**: Every answer separates `[Direct Paper Evidence]` / `[Cross-Literature Inference]` / `[Knowledge Extension & Speculation]`, each with its own grounding score — epistemic honesty over completeness
- **Knowledge Distillation (Stage 3)**: LLM distills source-bound evidence blocks into a structured fact list or validated comparison JSON
- **Deterministic cross-paper comparison**: Valid atomic comparison JSON is rendered directly into one-source claims, avoiding a second stochastic Stage 4 rewrite
- **Answer Verification (Stage 5)**: A reasoning model verifies logical leaps and unsupported inferences; a corrector LLM rewrites the answer if issues are found
- **NLI Citation Grounding (Stage 6)**: mDeBERTa multilingual NLI checks each sentence against the **raw PDF chunks** (not LLM summaries) and reports unsupported or contradictory claims
- **Answerability gate**: Optional three-way routing distinguishes answerable, partial, and genuinely unanswerable questions before final generation
- **English-first pipeline**: Reasoning, verification and NLI run in English; final Traditional Chinese translation can be enabled separately
- **Contextual chunk summarization**: Each chunk gets an LLM-generated summary header before embedding (Anthropic Contextual-Retrieval style)
- **Vision-Language support**: Extracts and describes figures from PDFs using a VL model — with smart rasterization for fragmented images and vector drawings
- **Multi-project support**: Manage multiple paper collections (e.g., `zvi`, `boron_bnct`) by switching `ACTIVE_PROJECT` in `config.py`
- **Cross-session memory**: ChromaDB stores episodic reasoning results and user preferences across sessions
- **Evaluation harness**: `eval/run_eval.py` reports correctness, paper-selection recall, retrieval coverage, grounding and per-stage latency for labeled A/B regression tests
- **OpenAI-compatible API**: Connect directly to Open WebUI as a custom model — no tool-call needed
- **Streaming output**: Real-time pipeline progress streamed to Open WebUI as blockquote status messages

## Development Status

As of **2026-08-07**, the first full 12-question V12 candidate has completed. `baseline_v12_candidate_full` scored correctness `0.938`, grounding `0.921`, translation fidelity `0.90`, paper selection `100%`, Stage 2 evidence recall `88.2%`, and mean latency `422.3s`. It is a strong candidate but is not yet the product baseline: that run still had a Q09 product failure, Q10 grounding failure, and Q04/Q10 translation-evaluator failures.

The subsequent fixes are deterministic and source-bound rather than prompt-only. The focused stability gate is now complete: Q10's translation false positive was rejected by `baseline_v12_q10_translation_relation_rejudge`; Q09 passed two fresh product runs after binding-site witness extraction; and `baseline_v12_contract_stability_r2` gave Q04/Q09/Q10 correctness, translation, and grounding `1.0`, paper selection `100%`, `C0/U0`, with no unexpected repair or fallback. The next quality gate is the full 12-question V12.1 product regression.

Retrieval remains fast at roughly `5.7s` in the latest focused run; Stage 3 generation and final translation are still the measured latency bottlenecks. A refreshed maintainability audit counts 58 active tracked Python files and 24,302 lines. Behavior-equivalent Maintainability M0 starts only after the V12.1 gate is frozen; see [maintainability_refactor_spec.md](maintainability_refactor_spec.md). [PENDING_TASKS.md](PENDING_TASKS.md) is the canonical master roadmap.

## System Architecture

```
                        ┌─────────────────────────────┐
                        │        User Question         │
                        └──────────────┬──────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │  Paper Pre-filter        │
                          │  keyword match → LLM     │
                          │  selects relevant papers  │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │  Stage 1 · Planning      │
                          │  LLM decomposes question │
                          │  into sub-questions per  │
                          │  paper                   │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │  Stage 2 · Retrieval     │
                          │  BM25 + Vector Search    │
                          │  → evidence blocks       │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │  Stage 3 · Synthesis     │  ← gemma4:31b
                          │  Distills evidence into  │
                          │  facts or validated      │
                          │  comparison JSON         │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │  Stage 4 · Answer        │  ← gemma4:31b
                          │  atomic comparison JSON  │
                          │  render, otherwise LLM   │
                          │  generation              │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │  Stage 5 · Verify &      │
                          │  Correct                 │
                          │  Verifier: qwen3.5:35b   │  ← finds issues
                          │  Corrector: gemma4:31b   │  ← rewrites answer
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │  Stage 6 · Grounding     │
                          │  mDeBERTa NLI: each      │
                          │  sentence vs raw chunks; │
                          │  weak cites → re-cite    │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │  Stage 7 · Translation   │
                          │  EN draft → 繁體中文      │
                          │  (if translation enabled)│
                          └────────────┬────────────┘
                                       │
                        ┌─────────────▼──────────────┐
                        │  Final Answer + Quality     │
                        │  Report (grounding_score)   │
                        └────────────────────────────┘

Memory Layer (ChromaDB):
  episodic_memory   → cross-paper reasoning conclusions
  preference_memory → user preferences & research style
```

## Requirements

### Hardware
- GPU with at least **16 GB VRAM** recommended (for running 31B models via Ollama)
- The pipeline runs fully locally — no internet connection required after setup

### Software
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- [Ollama](https://ollama.com) — for serving local LLMs and embeddings
- [Open WebUI](https://github.com/open-webui/open-webui) *(optional)* — for chat interface

### Ollama Models Required

Pull the following models before running:

```bash
ollama pull gemma4:31b          # Stage 3 synthesis + Stage 4 generation + Stage 5 correction
ollama pull qwen3.5:27b         # Main/fallback LLM and contextual summaries
ollama pull qwen3.5:35b-a3b     # Stage 5 verification (thinking model)
ollama pull qwen2.5:14b         # Paper selection + sub-question planning
ollama pull qwen3-vl:32b        # Vision-language figure analysis
ollama pull bge-m3              # Embedding model
```

The reranker (`BAAI/bge-reranker-v2-m3`) is downloaded automatically from HuggingFace on first run.

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd rag_project
```

### 2. Create conda environment

**Option A — Simplified (recommended, cross-platform):**
```bash
conda env create -f environment.yml
conda activate llm_env
```

**Option B — Full exact environment (Windows, guaranteed reproducible):**
```bash
conda env create -f environment_full.yml
conda activate llm_env
```

> If Option A fails due to version conflicts, use Option B which mirrors the exact environment used during development.

### 3. Add your papers

Place your PDF files into the appropriate project folder:

```
projects/
  zvi/
    papers/        ← put your ZVI-related PDFs here
  boron_bnct/
    papers/        ← put your BNCT-related PDFs here
```

> **Note**: PDF files are excluded from this repository due to copyright. The folder structure is preserved with `.gitkeep` files.

### 4. Start Ollama

```bash
ollama serve
```

## Usage

### Terminal test (no API server needed)

```bash
conda activate llm_env
cd rag_project
python scripts/test_query.py
```

Edit the `questions` list in [scripts/test_query.py](scripts/test_query.py) to change your queries.

### VL preprocessing (optional, recommended for papers with complex figures)

Before running the main pipeline, you can batch-preprocess all figures using the preprocessing script. This step detects fragmented or vector-drawn figures and rasterizes entire pages at high DPI for better VL analysis quality:

```bash
conda activate llm_env
cd rag_project
python scripts/preprocessing/vl_quality_test-1.py
```

The script:
1. **Extracts all images** from every PDF in the active project, paper by paper
2. **Detects fragmented pages** (≥ 8 embedded images) → rasterizes the whole page at 400 DPI
3. **Detects vector-drawing pages** (≥ 100 drawing commands, 0 embedded images) → rasterizes the whole page
4. **Removes small/decorative images** below 150 × 150 px
5. **Cleans up stale JSON entries** if extraction filenames changed
6. **Runs VL analysis** paper by paper, with checkpoint/resume (skips already-analyzed images)
7. Saves results to `projects/<project>/vl_test_output/<paper_name>/vl_test_result.json`

> If you want to force re-analysis of a specific image, manually delete its entry from `vl_test_result.json` and re-run the script.

### Re-scan failed VL images

If a paper has figures that failed VL analysis (shown as warnings on startup), re-run only those images without affecting other indexes:

```bash
python main.py --rerun-vl <paper_name>
# Example:
python main.py --rerun-vl 41467_2024_Article_45464
```

If any images are fixed, the paper's index is automatically rebuilt. If you manually deleted a problem image file beforehand, it will be marked as `skipped` and removed from the warning list.

### API server + Open WebUI

**Step 1 — Start the API server:**
```bash
conda activate llm_env
cd rag_project
uvicorn api:app --host 0.0.0.0 --port 8000
```

Or create a batch file for convenience (`start_rag.bat`):
```bat
@echo off
call conda activate llm_env
cd /d E:\Projects\rag_project
uvicorn api:app --host 0.0.0.0 --port 8000
```

**Step 2 — Connect Open WebUI:**
1. Open WebUI → Settings → Connections → Add OpenAI API
2. URL: `http://localhost:8000/v1`
3. API Key: `ollama` (any value works)
4. Save → select model `rag-pipeline` to start chatting

The pipeline progress (paper selection, sub-questions, Stage 3/4/5 status) streams directly into the WebUI chat as blockquote messages.

### Evaluation (regression testing)

Before and after any change to chunking, retrieval, prompts or thresholds, run the
evaluation harness to confirm the change actually helped — instead of guessing:

```bash
python eval/run_eval.py --run --label baseline       # run the question set, save results
# ...make a change (e.g. edit RERANK_CANDIDATE_K)...
python eval/run_eval.py --run --label experiment
python eval/run_eval.py --run --label focused --ids Q02,Q08
python eval/run_eval.py --compare baseline experiment # side-by-side metric comparison
```

The question set lives in [eval/eval_set.json](eval/eval_set.json). It runs in two modes:

- **Mode 1** (reference fields empty): reports grounding, unsupported/conflicting claims, per-stage latency, and paper/sub-question counts — works immediately, no labeling needed.
- **Mode 2** (fill `gold_papers`, `gold_spans`, `reference_answer`, and `reference_facts` in `eval_set.json`): additionally reports paper-selection recall, candidate retrieval recall, Stage 2 evidence recall, structured correctness, and translation fidelity against the labeled source-derived contract.

> The harness calls the pipeline directly and does **not** write to ChromaDB memory, so it won't pollute episodic memory. Full `--run` and `--rejudge-existing` commands invoke local AI models and should be run from the project's `llm_env` Anaconda terminal; offline syntax/unit/replay checks do not require that pipeline run.

### Switch project

Edit `config.py`:
```python
ACTIVE_PROJECT = "boron_bnct"   # switch to "zvi" or any new project name
```

Delete the old index if you change chunking parameters:
```bash
rm -rf projects/<project_name>/index_storage/
```

## Project Structure

```
rag_project/
├── main.py                    # Initialization: loads indexes, memory, engines
├── api.py                     # FastAPI server (OpenAI-compatible)
├── config.py                  # All tunable parameters
├── environment.yml            # Conda environment (simplified)
├── environment_full.yml       # Conda environment (full, exact)
│
├── rag/
│   ├── llm_client.py          # LLM & embedding initialization
│   ├── pdf_loader.py          # PDF parsing + VL figure description fusion
│   ├── indexer.py             # Index build / load / config validation
│   ├── retriever.py           # Hybrid retriever (BM25 + vector)
│   ├── reranker.py            # Cross-encoder reranker (bge-reranker-v2-m3)
│   ├── vl_processor.py        # Vision-language figure analysis (auto-triggered)
│   │
│   ├── query_pipeline.py      # 7-stage pipeline entry point
│   ├── query_planning.py      # Stage 1: sub-question decomposition
│   ├── query_retrieval.py     # Stage 2: hybrid retrieval per paper
│   ├── query_translation.py   # Query translation / language handling
│   ├── query_prompts.py       # Centralized prompt templates
│   ├── query_grounding_flow.py # NLI grounding flow orchestration
│   ├── query_types.py         # Shared type definitions
│   ├── query_embedding_guard.py # Embedding consistency guard
│   ├── knowledge_synthesizer.py  # Stage 3: fact list distillation
│   ├── fact_contract.py       # Source-bound requirements and fact validation
│   ├── comparison_json_validator.py # Comparison schema and relation validation
│   ├── answer_verifier.py     # Stage 5: verification + correction
│   ├── answer_processor.py    # Answer post-processing utilities
│   ├── citation_grounding.py  # Grounding score + speculation detection
│   ├── corpus_health.py       # Ingestion duplicate/SI/extraction checks
│   ├── plan_executor.py       # Plan-and-Execute architecture (experimental)
│   ├── task_state.py          # Pipeline task state management
│   ├── chunk_summarizer.py    # Contextual chunk summarization
│   ├── memory.py              # ChromaDB cross-session memory
│   ├── metadata_manager.py    # Auto-generate paper metadata
│   └── chunk_inspector.py     # Chunk quality inspection tool
│
├── scripts/
│   ├── test_query.py          # Terminal Q&A test (no FastAPI needed)
│   ├── test_retrieval.py      # Retrieval quality test
│   ├── test_new_modules.py    # Stage 3/5 module unit tests
│   ├── test_llm_chunks.py     # LLM chunk quality test
│   ├── test_stage5.py         # Stage 5 verifier test
│   ├── test_ab_retrieval.py   # A/B retrieval comparison test
│   ├── test_nli_extensions.py # NLI extension module tests
│   ├── test_embed.py          # Embedding smoke test
│   ├── test_embed_speed.py    # Embedding throughput benchmark
│   ├── test_fulltext.py       # Full-text retrieval test
│   ├── test_refactor.py       # Query pipeline refactor test
│   ├── test_memory_check.py   # ChromaDB memory inspection
│   └── preprocessing/
│       └── vl_quality_test-1.py   # Batch VL preprocessing with smart rasterization
│
├── eval/                        # Tier 0 evaluation harness (regression ruler)
│   ├── eval_set.json            # Gold sources/spans plus answer/fact contracts
│   ├── metrics.py               # Selection recall, retrieval coverage, latency, grounding
│   ├── judge.py                 # Structured correctness and translation judges
│   ├── run_eval.py              # Run the set through the pipeline, compute & compare metrics
│   └── results/                 # (auto-generated) saved metric runs for A/B comparison
│
├── PENDING_TASKS.md            # Current checkpoint and ordered roadmap
├── maintainability_refactor_spec.md # Audited refactor phases and acceptance gates
├── pipeline_v4_task_spec.md    # Staged indexing design
├── ingestion_health_spec.md    # Corpus health MVP and Phase 2
├── memory_redesign_spec.md     # Deferred research-memory redesign
├── api-refactor-spec.md        # Deferred API boundary refactor
│
├── projects/
│   ├── zvi/
│   │   ├── papers/            # (empty — add your PDFs here)
│   │   ├── index_storage/     # (auto-generated)
│   │   └── vl_test_output/    # (auto-generated)
│   └── boron_bnct/
│       ├── papers/            # (empty — add your PDFs here)
│       ├── index_storage/     # (auto-generated)
│       └── vl_test_output/    # (auto-generated)
│
├── memory_db/                 # ChromaDB persistent memory (auto-generated)
└── archive/                   # Old version backups
```

## Configuration

All parameters are centralized in `config.py`:

| Parameter | Current value | Description |
|-----------|---------|-------------|
| `ACTIVE_PROJECT` | `"boron_bnct"` | Active paper collection |
| `LLM_MODEL` | `"qwen3.5:27b"` | Main/fallback model and contextual summaries |
| `PLANNING_LLM_MODEL` | `"qwen2.5:14b"` | Paper selection + planning model |
| `SYNTHESIS_MODEL` | `"gemma4:31b"` | Stage 3 synthesis and normal Stage 4 generation |
| `VERIFY_MODEL` | `"qwen3.5:35b-a3b"` | Stage 5 verifier model |
| `EMBED_MODEL` | `"bge-m3"` | Embedding model |
| `REASONING_MODE` | `"strict"` | `"reasoning"` or `"strict"` |
| `SYNTHESIS_ENABLED` | `True` | Enable Stage 3 knowledge synthesis |
| `VERIFY_ENABLED` | `True` | Enable Stage 5 verification |
| `CHUNK_SIZE` | `1024` | Token size per chunk |
| `SIMILARITY_TOP_K` | `8` | Base candidate count (grounding-flow fallback) |
| `RERANK_CANDIDATE_K` | `24` | Candidates retrieved before reranking (must be > `RERANKER_TOP_N`) |
| `RERANKER_TOP_N` | `8` | Final chunks kept after reranking |
| `RERANK_ENABLED` | `False` | Optional cross-encoder reranking A/B |
| `GROUNDING_TOP_K` | `20` | Chunks retrieved for Stage 6 NLI grounding |
| `STAGE2_LLM_SUBANSWERS_ENABLED` | `False` | Feed retrieval evidence blocks directly to Stage 3 |
| `STAGE2_QUERY_AWARE_EVIDENCE_ENABLED` | `True` | Select source-bound Stage 2 evidence against query requirements |
| `STRUCTURED_FACT_CONTRACT_ENABLED` | `True` | Validate non-comparison facts against explicit source requirements |
| `METHOD_FACT_LIST_DIRECT_RENDER_ENABLED` | `True` | Deterministically render validated method facts |
| `COMPARISON_JSON_DIRECT_RENDER_ENABLED` | `True` | Render validated atomic comparison JSON without Stage 4 LLM prose |
| `STRUCTURED_JUDGE_STABLE_PROTOCOL_ENABLED` | `True` | Require the stable structured judge response contract |
| `ANSWERABILITY_GATE_ENABLED` | `True` | Route answerable, partial, and unsupported questions |
| `EN_DRAFT_PIPELINE` | `True` | Reason, verify and run NLI in English |
| `FINAL_TRANSLATION_ENABLED` | `True` | Translate the final English draft to Traditional Chinese |
| `VL_AUTO_RUN` | `True` | Auto-run VL analysis on new PDFs |
| `CONTEXT_SUMMARY_ENABLED` | `True` | Generate LLM summary header per chunk |
| `NLI_TRANSLATE_TO_EN` | `True` | Translate non-English hypotheses before NLI (normally a no-op for the English draft) |
| `NLI_CONTRADICTION_ENABLED` | `True` | Enable contradiction detection |
| `NLI_DEVICE` | `"cuda"` | Run the NLI model on the GPU |
| `PLAN_EXECUTE_ENABLED` | `False` | Plan-and-Execute architecture (experimental) |

> ⚠️ If you change `CHUNK_SIZE`, `CHUNK_OVERLAP`, or `EMBED_MODEL`, delete `projects/<project>/index_storage/` and re-run to rebuild the index.

---

# 繁體中文

> 一套本地運行、有引用根據的學術論文 RAG Pipeline——為個人研究工作流打造的思考夥伴。

這套系統是專為深度學術論文分析設計的本地 RAG Pipeline。
與單純的「檢索 + 回答」系統不同，它採用 7 階段流程——
子問題分解、混合檢索、知識蒸餾、邏輯自洽驗證、逐句 NLI 引用比對——
產出每一句都能追溯到原始論文的有根據答案。

為那些需要穩定、可控的思考夥伴，而不是一個隨時會悄悄改變行為的黑盒 AI 的研究者而設計。

## 功能特色

- **7 階段 Pipeline**：論文預篩 → ① 子問題規劃 → ② 混合檢索 → ③ 知識蒸餾 → ④ 答案渲染/生成 → ⑤ 驗證與修正 → ⑥ NLI 引用比對 → ⑦ 選用的繁體中文翻譯
- **混合檢索**：BM25 稀疏 + 向量稠密搜尋，可選擇啟用 Cross-encoder Reranker
- **三層答案結構**：每則回答分為【論文直接依據】/【跨文獻推論】/【知識延伸與推測】，各層各算 grounding 分數——認知誠實優先於答案完整
- **知識蒸餾（Stage 3）**：LLM 將具來源歸屬的 evidence blocks 蒸餾成事實清單或經驗證的 comparison JSON
- **確定性跨文獻比較**：atomic comparison JSON 通過驗證後，直接渲染成單一來源的比較陳述，避免 Stage 4 再次隨機改寫
- **答案驗證（Stage 5）**：推理模型驗證推論跳躍與無依據推論；若發現問題，由修正 LLM 重寫答案
- **NLI 引用比對（Stage 6）**：mDeBERTa 多語言 NLI 逐句比對 **PDF 原文 chunk**（非 LLM 摘要），回報無依據或矛盾陳述
- **可答性 gate**：選用的三分路由在最終生成前區分可回答、部分可回答與確實不可回答
- **English-first pipeline**：推理、驗證、NLI 全程用英文，最終繁體中文翻譯可獨立開關
- **情境式 chunk 摘要**：建索引時為每個 chunk 加上 LLM 生成的摘要標頭（Anthropic Contextual-Retrieval 風格）
- **視覺語言支援**：使用 VL 模型自動擷取並描述 PDF 圖表，支援碎片圖偵測與向量圖光柵化
- **多專案支援**：在 `config.py` 切換 `ACTIVE_PROJECT` 即可管理多個論文資料庫
- **跨 session 記憶**：ChromaDB 儲存推論結論與使用者偏好，跨對話保留
- **評估骨架**：`eval/run_eval.py` 回報 correctness、論文選擇命中率、檢索覆蓋率、grounding 與各階段延遲，用於有標籤的 A/B 回歸測試
- **OpenAI 相容 API**：直接在 Open WebUI 當成自訂模型使用，無需工具呼叫設定
- **串流輸出**：Pipeline 進度即時串流至 Open WebUI，以 blockquote 格式顯示

## 開發狀態

截至 **2026-08-07**，第一輪完整 12 題 V12 candidate 已完成。`baseline_v12_candidate_full` 的 correctness `0.938`、grounding `0.921`、translation fidelity `0.90`、paper selection `100%`、Stage 2 evidence recall `88.2%`、平均延遲 `422.3s`。這是一個很強的 candidate，但尚未升格為產品 baseline：該輪仍有 Q09 產品輸出、Q10 grounding，以及 Q04/Q10 translation evaluator 的缺口。

後續修正均採 source-bound deterministic contract，而不是繼續靠 prompt 勸模型。Focused stability gate 已完成：Q10 的翻譯假錯誤由 `baseline_v12_q10_translation_relation_rejudge` 排除；Q09 加入 binding-site witness 後連續兩次 fresh product run 通過；`baseline_v12_contract_stability_r2` 的 Q04/Q09/Q10 correctness、translation、grounding 全為 `1.0`，paper selection `100%`、`C0/U0`，且沒有非預期 repair/fallback。下一道品質 gate 是完整 12 題 V12.1 product regression。

最新 focused run 的 retrieval 約 `5.7s`，主要延遲瓶頸依然是 Stage 3 生成與最後翻譯。最新維護性盤點為 58 個 active tracked Python 檔、24,302 行；V12.1 gate 凍結後才啟動等價重構 Maintainability M0，詳見 [maintainability_refactor_spec.md](maintainability_refactor_spec.md)。[PENDING_TASKS.md](PENDING_TASKS.md) 是唯一 master roadmap。

## 系統架構

```
                        ┌─────────────────────────────┐
                        │           使用者問題          │
                        └──────────────┬──────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │  論文預篩選               │
                          │  關鍵字比對 → LLM 選出    │
                          │  最相關論文               │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │  Stage 1 · 子問題規劃    │
                          │  LLM 將問題拆解為        │
                          │  針對各論文的子問題       │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │  Stage 2 · 混合檢索      │
                          │  BM25 + 向量搜尋         │
                          │  → evidence blocks       │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │  Stage 3 · 知識蒸餾      │  ← gemma4:31b
                          │  將證據蒸餾成             │
                          │  事實清單或經驗證的        │
                          │  comparison JSON         │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │  Stage 4 · 答案處理      │  ← gemma4:31b
                          │  比較題確定性渲染；        │
                          │  其他題型由 LLM 生成      │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │  Stage 5 · 驗證與修正    │
                          │  Verifier: qwen3.5:35b   │  ← 找出問題
                          │  Corrector: gemma4:31b   │  ← 重寫答案
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │  Stage 6 · 引用比對      │
                          │  mDeBERTa NLI：逐句對    │
                          │  PDF 原文 chunk 比對；    │
                          │  依據不足 → 重新引用      │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │  Stage 7 · 翻譯          │
                          │  英文初稿 → 繁體中文      │
                          │  （翻譯開啟時）           │
                          └────────────┬────────────┘
                                       │
                        ┌─────────────▼──────────────┐
                        │  最終答案 + 品質報告         │
                        │  （grounding_score）        │
                        └────────────────────────────┘

記憶層（ChromaDB）：
  episodic_memory   → 跨文獻推論結論
  preference_memory → 使用者偏好與研究風格
```

## 環境需求

### 硬體
- 建議 GPU **VRAM ≥ 16 GB**（執行 31B 模型所需）
- 本系統完全本機運行，設定完成後無需網路連線

### 軟體
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 或 Anaconda
- [Ollama](https://ollama.com) — 本機 LLM 服務
- [Open WebUI](https://github.com/open-webui/open-webui) *(選用)* — 聊天介面

### 所需 Ollama 模型

執行前請先下載以下模型：

```bash
ollama pull gemma4:31b          # Stage 3 蒸餾 + Stage 4 生成 + Stage 5 修正
ollama pull qwen3.5:27b         # 主/備援 LLM 與情境式摘要
ollama pull qwen3.5:35b-a3b     # Stage 5 驗證（思考型模型）
ollama pull qwen2.5:14b         # 論文篩選 + 子問題規劃
ollama pull qwen3-vl:32b        # 視覺語言圖表分析
ollama pull bge-m3              # Embedding 模型
```

Reranker（`BAAI/bge-reranker-v2-m3`）第一次執行時會自動從 HuggingFace 下載。

## 安裝步驟

### 1. 下載專案

```bash
git clone <repo-url>
cd rag_project
```

### 2. 建立 conda 虛擬環境

**方案 A — 精簡版（建議，跨平台相容）：**
```bash
conda env create -f environment.yml
conda activate llm_env
```

**方案 B — 完整版（Windows，與開發環境完全一致）：**
```bash
conda env create -f environment_full.yml
conda activate llm_env
```

> 若方案 A 出現版本衝突，請改用方案 B，這份是從開發環境直接匯出的完整環境。

### 3. 加入論文

將 PDF 放入對應的專案資料夾：

```
projects/
  zvi/
    papers/        ← 放入相關 PDF
  boron_bnct/
    papers/        ← 放入相關 PDF
```

> **注意**：因版權問題，PDF 不包含在本 repo 中，資料夾結構以 `.gitkeep` 佔位保留。

### 4. 啟動 Ollama

```bash
ollama serve
```

## 使用方式

### 終端機測試（不需要 API server）

```bash
conda activate llm_env
cd rag_project
python scripts/test_query.py
```

在 [scripts/test_query.py](scripts/test_query.py) 中修改 `questions` 清單來更換測試問題。

### VL 預處理（選用，建議於論文含複雜圖表時使用）

在執行主 Pipeline 前，可先用預處理腳本批次處理所有圖表。此步驟能偵測細碎嵌入圖或向量繪製圖，改以高 DPI 光柵化整頁截圖送入 VL 分析，有效提升圖表描述品質：

```bash
conda activate llm_env
cd rag_project
python scripts/preprocessing/vl_quality_test-1.py
```

腳本執行流程：
1. **逐篇抽取圖片**：掃描所有 PDF，按篇逐頁抽取嵌入圖片
2. **偵測碎片頁**（單頁 ≥ 8 張嵌入圖）→ 光柵化整頁（400 DPI）
3. **偵測向量圖頁**（0 張嵌入圖 + ≥ 100 個 drawing 命令）→ 光柵化整頁
4. **過濾小圖**（寬或高 < 150 px）
5. **清除過期 JSON 條目**（重新抽取後舊檔名不存在時自動清理）
6. **逐篇 VL 分析**：有 checkpoint 功能，已分析的圖片自動跳過
7. 結果存入 `projects/<project>/vl_test_output/<paper_name>/vl_test_result.json`

> 若要強制重跑特定圖片的 VL 分析，手動刪除 `vl_test_result.json` 中對應的條目後再執行腳本即可。

### 重新掃描失敗的 VL 圖片

若某篇論文有 VL 分析失敗的圖片（啟動時會顯示警告），可只重新掃描失敗的圖片，不影響其他已建好的索引：

```bash
python main.py --rerun-vl <論文名稱>
# 範例：
python main.py --rerun-vl 41467_2024_Article_45464
```

若有圖片修復成功，該論文的索引會自動重建。若事先手動刪除了有問題的圖片檔案，系統會將其標記為 `skipped`，從警告清單中移除。

### API server + Open WebUI

**步驟一 — 啟動 API server：**
```bash
conda activate llm_env
cd rag_project
uvicorn api:app --host 0.0.0.0 --port 8000
```

或建立批次檔方便啟動（`start_rag.bat`）：
```bat
@echo off
call conda activate llm_env
cd /d E:\Projects\rag_project
uvicorn api:app --host 0.0.0.0 --port 8000
```

**步驟二 — 連接 Open WebUI：**
1. Open WebUI → Settings → Connections → 新增 OpenAI API
2. URL：`http://localhost:8000/v1`
3. API Key：`ollama`（隨意填，server 不驗證）
4. 儲存後選擇 `rag-pipeline` 模型開始對話

Pipeline 進度（論文篩選、子問題、Stage 3/4/5 狀態）會即時串流到 WebUI 對話視窗中，以 blockquote 格式呈現。

### 評估（回歸測試）

在改動 chunking、檢索、prompt 或任何門檻**之前與之後**，都跑一次評估骨架，
用數據確認改動真的有效，而不是靠感覺：

```bash
python eval/run_eval.py --run --label baseline       # 跑題組，存結果
# ...做一個改動（例如改 RERANK_CANDIDATE_K）...
python eval/run_eval.py --run --label experiment
python eval/run_eval.py --run --label focused --ids Q02,Q08
python eval/run_eval.py --compare baseline experiment # 並排比較彙總指標
```

題組放在 [eval/eval_set.json](eval/eval_set.json)，有兩種模式：

- **Mode 1**（reference 欄位留空）：回報 grounding、unsupported/conflicting claims、各階段延遲與論文/子問題數，不需人工標註。
- **Mode 2**（在 `eval_set.json` 填好 `gold_papers`、`gold_spans`、`reference_answer` 與 `reference_facts`）：額外回報論文選擇命中率、candidate retrieval recall、Stage 2 evidence recall、結構化 correctness 與 translation fidelity，並與忠於原文的標註 contract 比對。

> 評估骨架直接呼叫 pipeline，**不會**寫入 ChromaDB 記憶，因此不會污染 episodic memory。完整 `--run` 與 `--rejudge-existing` 會呼叫本地 AI 模型，請統一在專案的 `llm_env` Anaconda terminal 執行；離線語法、unit 與 artifact replay 則不需啟動 AI pipeline。

### 切換專案

修改 `config.py`：
```python
ACTIVE_PROJECT = "boron_bnct"   # 改成 "zvi" 或其他專案名稱
```

如果修改了 chunking 參數，需刪除舊索引重新建立：
```bash
rm -rf projects/<project_name>/index_storage/
```

## 專案結構

```
rag_project/
├── main.py                    # 初始化：載入索引、記憶、查詢引擎
├── api.py                     # FastAPI server（OpenAI 相容介面）
├── config.py                  # 所有可調整參數集中管理
├── environment.yml            # Conda 環境設定（精簡版）
├── environment_full.yml       # Conda 環境設定（完整版）
│
├── rag/
│   ├── llm_client.py          # LLM 與 Embedding 初始化
│   ├── pdf_loader.py          # PDF 解析 + VL 圖表描述融合
│   ├── indexer.py             # 索引建立 / 載入 / 設定檢查
│   ├── retriever.py           # 混合檢索（BM25 + 向量）
│   ├── reranker.py            # Cross-encoder Reranker
│   ├── vl_processor.py        # 視覺語言圖表分析（自動觸發）
│   │
│   ├── query_pipeline.py      # 7 階段 Pipeline 主入口
│   ├── query_planning.py      # Stage 1：子問題拆解
│   ├── query_retrieval.py     # Stage 2：逐篇混合檢索
│   ├── query_translation.py   # 查詢翻譯 / 語言處理
│   ├── query_prompts.py       # 集中管理 Prompt 模板
│   ├── query_grounding_flow.py # NLI Grounding 流程協調
│   ├── query_types.py         # 共用型別定義
│   ├── query_embedding_guard.py # Embedding 一致性守衛
│   ├── knowledge_synthesizer.py  # Stage 3：知識蒸餾
│   ├── fact_contract.py       # source-bound requirements 與事實驗證
│   ├── comparison_json_validator.py # 比較 schema 與關係驗證
│   ├── answer_verifier.py     # Stage 5：答案驗證與修正
│   ├── answer_processor.py    # 答案後處理工具
│   ├── citation_grounding.py  # Grounding score + 推測語氣偵測
│   ├── corpus_health.py       # 匯入重複/SI/抽取健康檢查
│   ├── plan_executor.py       # Plan-and-Execute 架構（實驗性）
│   ├── task_state.py          # Pipeline 任務狀態管理
│   ├── chunk_summarizer.py    # 情境式 Chunk 摘要
│   ├── memory.py              # ChromaDB 跨 session 記憶
│   ├── metadata_manager.py    # 論文 metadata 自動生成
│   └── chunk_inspector.py     # Chunk 品質檢查工具
│
├── scripts/
│   ├── test_query.py          # 終端機問答測試
│   ├── test_retrieval.py      # 檢索品質測試
│   ├── test_new_modules.py    # Stage 3/5 模組單元測試
│   ├── test_llm_chunks.py     # LLM Chunk 品質測試
│   ├── test_stage5.py         # Stage 5 驗證模組測試
│   ├── test_ab_retrieval.py   # A/B 檢索對比測試
│   ├── test_nli_extensions.py # NLI 擴展模組測試
│   ├── test_embed.py          # Embedding 基本測試
│   ├── test_embed_speed.py    # Embedding 速度基準測試
│   ├── test_fulltext.py       # 全文檢索測試
│   ├── test_refactor.py       # Query Pipeline 重構測試
│   ├── test_memory_check.py   # ChromaDB 記憶檢查
│   └── preprocessing/
│       └── vl_quality_test-1.py   # 批次 VL 預處理（含智慧光柵化）
│
├── eval/                        # Tier 0 評估骨架（回歸量尺）
│   ├── eval_set.json            # gold 來源/span + answer/fact contracts
│   ├── metrics.py               # 選擇命中率、檢索覆蓋率、延遲、grounding
│   ├── judge.py                 # 結構化 correctness 與 translation judges
│   ├── run_eval.py              # 跑題組過 pipeline、計算與比較指標
│   └── results/                 # （自動生成）存放各次指標結果供 A/B 比較
│
├── PENDING_TASKS.md            # 目前 checkpoint 與排序後 roadmap
├── maintainability_refactor_spec.md # 維護性稽核、拆分階段與驗收 gate
├── pipeline_v4_task_spec.md    # 分階段索引設計
├── ingestion_health_spec.md    # 語料健檢 MVP 與 Phase 2
├── memory_redesign_spec.md     # 延後的研究記憶層重設計
├── api-refactor-spec.md        # 延後的 API 邊界重構
│
├── projects/
│   ├── zvi/
│   │   ├── papers/            # （空白 — 放入 PDF）
│   │   ├── index_storage/     # （自動生成）
│   │   └── vl_test_output/    # （自動生成）
│   └── boron_bnct/
│       ├── papers/            # （空白 — 放入 PDF）
│       ├── index_storage/     # （自動生成）
│       └── vl_test_output/    # （自動生成）
│
├── memory_db/                 # ChromaDB 長期記憶（自動生成）
└── archive/                   # 舊版備份
```

## 參數設定

所有參數集中在 `config.py`：

| 參數 | 目前值 | 說明 |
|------|--------|------|
| `ACTIVE_PROJECT` | `"boron_bnct"` | 目前使用的論文專案 |
| `LLM_MODEL` | `"qwen3.5:27b"` | 主/備援模型與情境式摘要 |
| `PLANNING_LLM_MODEL` | `"qwen2.5:14b"` | 論文篩選與規劃模型 |
| `SYNTHESIS_MODEL` | `"gemma4:31b"` | Stage 3 蒸餾與一般 Stage 4 生成 |
| `VERIFY_MODEL` | `"qwen3.5:35b-a3b"` | Stage 5 驗證模型 |
| `EMBED_MODEL` | `"bge-m3"` | Embedding 模型 |
| `REASONING_MODE` | `"strict"` | `"reasoning"` 或 `"strict"` |
| `SYNTHESIS_ENABLED` | `True` | 啟用 Stage 3 知識蒸餾 |
| `VERIFY_ENABLED` | `True` | 啟用 Stage 5 驗證 |
| `CHUNK_SIZE` | `1024` | 每個 chunk 的 token 數 |
| `SIMILARITY_TOP_K` | `8` | 基礎候選數（grounding flow 的 fallback） |
| `RERANK_CANDIDATE_K` | `24` | 進 reranker 前的候選數（須 > `RERANKER_TOP_N`） |
| `RERANKER_TOP_N` | `8` | Rerank 後保留的 chunk 數 |
| `RERANK_ENABLED` | `False` | 選用的 Cross-encoder reranking A/B |
| `GROUNDING_TOP_K` | `20` | Stage 6 NLI 比對用的 chunk 數 |
| `STAGE2_LLM_SUBANSWERS_ENABLED` | `False` | 將 retrieval evidence blocks 直接交給 Stage 3 |
| `STAGE2_QUERY_AWARE_EVIDENCE_ENABLED` | `True` | 依 query requirements 選取 source-bound Stage 2 證據 |
| `STRUCTURED_FACT_CONTRACT_ENABLED` | `True` | 以明確來源要件驗證非比較題 facts |
| `METHOD_FACT_LIST_DIRECT_RENDER_ENABLED` | `True` | 確定性渲染已驗證的 method facts |
| `COMPARISON_JSON_DIRECT_RENDER_ENABLED` | `True` | atomic comparison JSON 通過驗證後跳過 Stage 4 LLM prose |
| `STRUCTURED_JUDGE_STABLE_PROTOCOL_ENABLED` | `True` | 強制 stable structured judge response contract |
| `ANSWERABILITY_GATE_ENABLED` | `True` | 路由可回答、部分可回答與缺乏依據問題 |
| `EN_DRAFT_PIPELINE` | `True` | 推理、驗證與 NLI 全程英文 |
| `FINAL_TRANSLATION_ENABLED` | `True` | 將最終英文 draft 翻譯為繁體中文 |
| `VL_AUTO_RUN` | `True` | 新增 PDF 時自動執行 VL 圖表分析 |
| `CONTEXT_SUMMARY_ENABLED` | `True` | 為每個 chunk 生成 LLM 摘要標頭 |
| `NLI_TRANSLATE_TO_EN` | `True` | 非英文 hypothesis 才先翻譯；英文 draft 通常不需轉換 |
| `NLI_CONTRADICTION_ENABLED` | `True` | 啟用矛盾偵測 |
| `NLI_DEVICE` | `"cuda"` | 在 GPU 執行 NLI 模型 |
| `PLAN_EXECUTE_ENABLED` | `False` | Plan-and-Execute 架構（實驗性） |

> ⚠️ 若修改 `CHUNK_SIZE`、`CHUNK_OVERLAP` 或 `EMBED_MODEL`，請刪除 `projects/<project>/index_storage/` 後重新執行以重建索引。
