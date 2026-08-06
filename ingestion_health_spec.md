# 匯入健檢 Ingestion Health Check — Spec

> 狀態：MVP 已完成（2026-06-26；2026-08-06 重新核對）。Phase 2 等待
> `pipeline_v4_task_spec.md` 的 per-paper state 與 staged indexing 地基。
> 起因：語料審計發現 31 篇索引項有 4 項污染（~13%）。

## 目標
論文匯入時自動抓出語料污染，**零 gold 標籤**，可推廣到任何使用者匯入的新論文。

## 審計實際發現（boron_bnct，31 篇）
- 完全相同：`41467_2024_Article_45464` ≡ `s41467-024-45464-z (1)`（同 md5）
- 內容重複（改名，差 1 byte）：`a-method-...boronate-esters` ≡ `Pinacol Deprotection  a-method-...`
- SI 當獨立論文：`LAT1 ChemComm 2026SI`、`Supplement_info_ ligand-enabled-c-h-hydroxylation-...`

## MVP 範圍
三類檢查，零 gold 標籤：

| 檢查 | 訊號 | 處置 |
|---|---|---|
| 重複（精確+改名） | **正規化全文 sha**（小寫、收斂空白）→ 同 sha 即同篇。一招抓 byte-dup 與 1-byte-diff 改名 | 報告 + ingestion 跳過冗餘份（保留排序首位；orphan cleanup 自動清其索引） |
| SI 當獨立論文 | 檔名 regex：大寫 `SI` 後綴（非字母邊界，避開 `synthesis`）或 `supplement/supporting info`（不分大小寫） | 報告旗標（綁主文＝Phase 2） |
| 抽取健康 | `text_len < 500`（疑掃描檔/空）或 `garbage_ratio > 0.05`（亂碼/控制字元，疑 OCR 壞） | 被動報告旗標 |

## 入口
- **`python main.py --health`**：印審計報告（重複組 / SI / 抽取警示）。重度初始化前攔截。
- **ingestion hook**：`indexer.load_all_papers` 建索引前算各篇 sha，跳過冗餘份；既有冗餘索引由 `_cleanup_orphan_indexes` 順手清掉（不刪 PDF）。

## 刻意不做（Phase 2）
- SI → 主文出處綁定（牽動 citation attribution，風險高，需獨立驗證）
- 自我查詢可答性（用 title/short_desc 生 query 跑檢索+grounding；每篇 LLM，貴；審計未發現抽取壞→ YAGNI）
- `--fix` 自動刪冗餘 PDF（自動刪檔風險；MVP 只報告，手動刪）
- health 欄位寫入 `papers_metadata.json`（目前無人讀；留待 pipeline_v4 per-paper 狀態需要時做。兩者共用地基）

## ponytail 註記
- 去重用全文 sha 非 md5：md5 抓不到 1-byte-diff 改名；全文 sha 一招抓兩種，且本來就抽全文。
- 抽取健康用 `text_len` 非 `chunk_count`：避免耦合 llama-index docstore 格式；text_len 是等價代理，不夠用再換。
- ingestion hook 每次啟動重抽全文算 sha（~10-30s/31 篇）：可接受；要快再把 sha 快取進 metadata（即 v4 地基）。
