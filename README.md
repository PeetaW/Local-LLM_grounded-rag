# Academic RAG Pipeline

> A local, citation-grounded RAG pipeline for academic paper analysis — built as a personal research thinking partner.

This project is a local RAG pipeline designed for deep academic paper analysis.
Unlike simple retrieve-and-answer systems, it uses a 5-stage pipeline —
sub-question decomposition, hybrid retrieval, knowledge synthesis,
and logical verification — to produce citation-grounded answers
you can actually trace back to the source.

Built for researchers who want a stable, controllable thinking partner
rather than a black-box AI that changes without notice.

> Powered by [Ollama](https://ollama.com) + [LlamaIndex](https://www.llamaindex.ai) + [ChromaDB](https://www.trychroma.com), with an OpenAI-compatible API for [Open WebUI](https://github.com/open-webui/open-webui) integration.

---

## Table of Contents / 目錄

- [English](#english)
  - [Features](#features)
  - [System Architecture](#system-architecture)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Project Structure](#project-structure)
  - [Configuration](#configuration)
- [繁體中文](#繁體中文)
  - [功能特色](#功能特色)
  - [系統架構](#系統架構)
  - [環境需求](#環境需求)
  - [安裝步驟](#安裝步驟)
  - [使用方式](#使用方式)
  - [專案結構](#專案結構)
  - [參數設定](#參數設定)

---

# English

## Features

- **5-Stage Pipeline**: Paper selection → Sub-question planning → Hybrid retrieval → Knowledge synthesis → Answer verification & correction
- **Hybrid Retrieval**: BM25 sparse search + vector dense search + cross-encoder reranking
- **Knowledge Synthesis (Stage 3)**: LLM distills retrieved chunks into a structured fact list before answer generation
- **Answer Verification (Stage 5)**: A second LLM verifies hallucinations, citation gaps, and unsupported inferences; a corrector LLM rewrites the answer if issues are found
- **Multi-project support**: Manage multiple paper collections (e.g., `zvi`, `boron_bnct`) by switching `ACTIVE_PROJECT` in `config.py`
- **Vision-Language support**: Extracts and describes figures/tables from PDFs using a VL model
- **Cross-session memory**: ChromaDB stores episodic reasoning results and user preferences across sessions
- **OpenAI-compatible API**: Connect directly to Open WebUI as a custom model — no tool-call needed
- **Streaming output**: Real-time pipeline progress streamed to Open WebUI as blockquote status messages

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
                          │  → Cross-encoder Rerank  │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │  Stage 3 · Synthesis     │  ← gemma4:31b
                          │  Distills sub-answers    │
                          │  into structured fact    │
                          │  list [Fact 1][Fact 2]…  │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │  Stage 4 · Generation    │  ← gemma4:31b
                          │  LLM writes answer from  │
                          │  fact list with citation │
                          │  and reasoning labels    │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │  Stage 5 · Verify &      │
                          │  Correct                 │
                          │  Verifier: qwen3.5:35b   │  ← finds issues
                          │  Corrector: gemma4:31b   │  ← rewrites answer
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
ollama pull qwen3.5:35b-a3b     # Stage 5 verification (thinking model)
ollama pull qwen2.5:14b         # Paper selection + sub-question planning
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

### Switch project

Edit `config.py`:
```python
ACTIVE_PROJECT = "zvi"        # switch to "boron_bnct" or any new project name
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
│   ├── query_engine.py        # Full 5-stage pipeline orchestration
│   ├── knowledge_synthesizer.py  # Stage 3: fact list distillation
│   ├── answer_verifier.py     # Stage 5: verification + correction
│   ├── citation_grounding.py  # Grounding score + speculation detection
│   ├── chunk_summarizer.py    # Contextual chunk summarization
│   ├── memory.py              # ChromaDB cross-session memory
│   ├── metadata_manager.py    # Auto-generate paper metadata
│   ├── vl_processor.py        # Vision-language figure analysis
│   └── chunk_inspector.py     # Chunk quality inspection tool
│
├── scripts/
│   ├── test_query.py          # Terminal Q&A test (no FastAPI needed)
│   ├── test_retrieval.py      # Retrieval quality test
│   ├── test_new_modules.py    # Stage 3/5 module unit tests
│   └── test_llm_chunks.py     # LLM chunk quality test
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
├── preprocessing/             # VL quality test scripts
└── archive/                   # Old version backups
```

## Configuration

All parameters are centralized in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ACTIVE_PROJECT` | `"zvi"` | Active paper collection |
| `LLM_MODEL` | `"gemma4:31b"` | Main generation model |
| `PLANNING_LLM_MODEL` | `"qwen2.5:14b"` | Paper selection + planning model |
| `VERIFY_MODEL` | `"qwen3.5:35b-a3b"` | Stage 5 verifier model |
| `EMBED_MODEL` | `"bge-m3"` | Embedding model |
| `REASONING_MODE` | `"reasoning"` | `"reasoning"` or `"strict"` |
| `SYNTHESIS_ENABLED` | `True` | Enable Stage 3 knowledge synthesis |
| `VERIFY_ENABLED` | `True` | Enable Stage 5 verification |
| `CHUNK_SIZE` | `1024` | Token size per chunk |
| `SIMILARITY_TOP_K` | `8` | Candidates per retrieval method |
| `RERANKER_TOP_N` | `8` | Final chunks after reranking |

> ⚠️ If you change `CHUNK_SIZE`, `CHUNK_OVERLAP`, or `EMBED_MODEL`, delete `projects/<project>/index_storage/` and re-run to rebuild the index.

---

# 繁體中文

> 一套本地運行、有引用根據的學術論文 RAG Pipeline——為個人研究工作流打造的思考夥伴。

這套系統是專為深度學術論文分析設計的本地 RAG Pipeline。
與單純的「檢索 + 回答」系統不同，它採用 5 階段流程——
子問題分解、混合檢索、知識蒸餾、邏輯自洽驗證——
產出每一句都能追溯到原始論文的有根據答案。

為那些需要穩定、可控的思考夥伴，而不是一個隨時會悄悄改變行為的黑盒 AI 的研究者而設計。

## 功能特色

- **5 階段 Pipeline**：論文篩選 → 子問題規劃 → 混合檢索 → 知識蒸餾 → 答案驗證與修正
- **混合檢索**：BM25 稀疏搜尋 + 向量稠密搜尋 + Cross-encoder Reranker
- **知識蒸餾（Stage 3）**：LLM 在生成答案前，先將檢索結果蒸餾成結構化事實清單
- **答案驗證（Stage 5）**：第二個 LLM 驗證幻覺、引用缺漏與無依據推論；若發現問題，由修正 LLM 重寫答案
- **多專案支援**：在 `config.py` 切換 `ACTIVE_PROJECT` 即可管理多個論文資料庫
- **視覺語言支援**：使用 VL 模型自動擷取並描述 PDF 內的圖表
- **跨 session 記憶**：ChromaDB 儲存推論結論與使用者偏好，跨對話保留
- **OpenAI 相容 API**：直接在 Open WebUI 當成自訂模型使用，無需工具呼叫設定
- **串流輸出**：Pipeline 進度即時串流至 Open WebUI，以 blockquote 格式顯示

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
                          │  → Cross-encoder Rerank  │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │  Stage 3 · 知識蒸餾      │  ← gemma4:31b
                          │  將子答案蒸餾成           │
                          │  結構化事實清單            │
                          │  [事實1][事實2]…         │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │  Stage 4 · 答案生成      │  ← gemma4:31b
                          │  從事實清單生成完整回答   │
                          │  含引用標注與推論層次      │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │  Stage 5 · 驗證與修正    │
                          │  Verifier: qwen3.5:35b   │  ← 找出問題
                          │  Corrector: gemma4:31b   │  ← 重寫答案
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
ollama pull qwen3.5:35b-a3b     # Stage 5 驗證（思考型模型）
ollama pull qwen2.5:14b         # 論文篩選 + 子問題規劃
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

### 切換專案

修改 `config.py`：
```python
ACTIVE_PROJECT = "zvi"        # 改成 "boron_bnct" 或其他專案名稱
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
│   ├── query_engine.py        # 5 階段 Pipeline 主流程
│   ├── knowledge_synthesizer.py  # Stage 3：知識蒸餾
│   ├── answer_verifier.py     # Stage 5：答案驗證與修正
│   ├── citation_grounding.py  # Grounding score + 推測語氣偵測
│   ├── chunk_summarizer.py    # 情境式 Chunk 摘要
│   ├── memory.py              # ChromaDB 跨 session 記憶
│   ├── metadata_manager.py    # 論文 metadata 自動生成
│   ├── vl_processor.py        # 視覺語言圖表分析
│   └── chunk_inspector.py     # Chunk 品質檢查工具
│
├── scripts/
│   ├── test_query.py          # 終端機問答測試
│   ├── test_retrieval.py      # 檢索品質測試
│   ├── test_new_modules.py    # Stage 3/5 模組單元測試
│   └── test_llm_chunks.py     # LLM Chunk 品質測試
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
├── preprocessing/             # VL 品質測試腳本
└── archive/                   # 舊版備份
```

## 參數設定

所有參數集中在 `config.py`：

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `ACTIVE_PROJECT` | `"zvi"` | 目前使用的論文專案 |
| `LLM_MODEL` | `"gemma4:31b"` | 主生成模型 |
| `PLANNING_LLM_MODEL` | `"qwen2.5:14b"` | 論文篩選與規劃模型 |
| `VERIFY_MODEL` | `"qwen3.5:35b-a3b"` | Stage 5 驗證模型 |
| `EMBED_MODEL` | `"bge-m3"` | Embedding 模型 |
| `REASONING_MODE` | `"reasoning"` | `"reasoning"` 或 `"strict"` |
| `SYNTHESIS_ENABLED` | `True` | 啟用 Stage 3 知識蒸餾 |
| `VERIFY_ENABLED` | `True` | 啟用 Stage 5 驗證 |
| `CHUNK_SIZE` | `1024` | 每個 chunk 的 token 數 |
| `SIMILARITY_TOP_K` | `8` | 每種檢索方法的候選數量 |
| `RERANKER_TOP_N` | `8` | Rerank 後保留的 chunk 數 |

> ⚠️ 若修改 `CHUNK_SIZE`、`CHUNK_OVERLAP` 或 `EMBED_MODEL`，請刪除 `projects/<project>/index_storage/` 後重新執行以重建索引。
