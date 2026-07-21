# Eval Report — `q04_fact_contract_a`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-21 23:06
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 1.0 |
| Correctness judge 覆蓋 | 1/1（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 0.5 |
| Translation judge 覆蓋 | 1/1（N/A 0） |
| 平均 grounding 分數 | 1.0 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 100.0% |
| 平均 Stage 2 evidence 覆蓋率 | 100.0% |
| 平均總延遲 | 665.0s |
| 平均 planning 延遲 | 6.1s |
| 平均 retrieval 延遲 | 4.7s |
| └ Phase A embed/vector/BM25 | 4.7s |
| └ Phase B 子答案生成 | 0.1s |
| 平均 grounding 延遲 | 20.0s |
| └ 其中 NLI | 2.6s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ⚠️ | Q04 | table_lookup | 1.0 | 0.5 | 100.0% | 100.0% | 100.0% | 1.0 | 665.0s | C0/U0 |

## 逐題細節

### ⚠️ Q04 · table_lookup

**問題**：What are the main degradation products and impurities of BPA identified by HPLC, and under which storage conditions do they form?

- detected_paper：`None`
- 選出論文：['1-s2.0-S0731708511003451-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water']
- gold_papers：['1-s2.0-S0731708511003451-main']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/7
- translation fidelity：0.5　raw：3/5　judge：`translation_fidelity_v2`
- translation reason：1 material and 0 minor semantic errors; Source S3 states 'Neither BrPD nor a FBBA/BDPA combination were observed'. Target T17 translates this as '未觀察到 BrPD 或 FBBA/BDPA 組合' (did not observe BrPD or the combination), which semantically implies that at least one of them might have been absent but allows for ambiguity in logical negation scop
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：1.0
- 延遲：665.0s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ### 已識別的雜質與 HPLC 檢測參數
>
> Tyrosine 和 phenylalanine 已被報導為 boron phenylalanine (BPA) 的雜質 【1-s2.0-S0731708511003451-main】。合成中間體包括 2-(4-Bromophenyl)-[1,3]dioxane (BrPD)、4-Formylbenzeneboronic acid (FBBA) 以及 2-tert-Butoxycarbonylamino-3-[4-(5,5-dimethyl-[1,3,2]dioxaborinan-2-yl)-phenyl]acrylic acid methyl ester (BDPA) 【1-s2.0-S0731708511003451-main】。在現有的 BPA 樣本中均未觀察到 BrPD 或 FBBA/BDPA 組合 【1-s2.0-S0731708511003451-main】。在 HPLC 分析中，BPA 和 tyrosine 在 230 nm 處進行定量，其保留時間 (retention times) 分別為 5.3 和 4.5 min 【1-s2.0-S0731708511003451-main】。在 HPLC 分析中，phenylalanine 在 11.0 min 洗脫 (elutes) 並在 256 nm 處進行定量 【1-s2.0-S0731708511003451-main】。BrPD 和 FBBA 在 256 nm 處被檢測到，其保留時間分別為 17.3 和 23.7 min 【1-s2.0-S0731708511003451-main】。BDPA 在 270 nm 處被檢測到，但與 FBBA 共洗脫 (co-elutes) 【1-s2.0-S0731708511003451-main】。
>
> ### 降解產物及其形成條 …（完整內容見 JSON）

---
