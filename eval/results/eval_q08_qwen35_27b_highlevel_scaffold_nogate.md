# Eval Report — `q08_qwen35_27b_highlevel_scaffold_nogate`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-05 11:08
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.75 |
| 平均 grounding 分數 | 0.333 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 1996.3s |
| 平均 planning 延遲 | 11.0s |
| 平均 retrieval 延遲 | 1174.0s |
| 平均 grounding 延遲 | 40.4s |
| └ 其中 NLI | 16.2s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q08 | cross_paper | 100.0% | 100.0% | 0.333 | 1996.3s | C0/U6 |

## 逐題細節

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.333
- 延遲：1996.3s　問題標記：{'conflicts': 0, 'unsupported': 6}

**答案預覽**：

> ### 比較框架 (Comparison scaffold)：
> 
> | 來源角色 (source role) | 項目/路徑 (item/route) | 來源論文 (source paper(s)) | 定義性證據 (defining evidence) | 相關比較維度 (relevant comparison dimensions) | 注意事項 (caveats) |
> | :--- | :--- | :--- | :--- | :--- | :--- |
> | 評論/比較來源 (review/comparison source) | 通用策略與試劑優化（直接功能化 Direct Functionalization） | 【CMDC-20-e202500059】 | 比較「手性優先 (Chirality First)」與「硼優先 (Boron First)」；評估 n-BuLi、格氏試劑 (Grignard reagents) 以及硼酸酯 ($\text{B}(\text{OBu})_3$) 相對於 $\text{B}_2\text{pin}_2$ 的效果。 | 同位素富集成本 ($^{10}\text{B}$)、引入硼後的步驟數/產率、安全性 (Pd/$\text{NaIO}_4$)。 | Morita 專利產率高但使用 Pd；$\text{NaIO}_4$ 去保護反應在大規模生產時具有毒性風險。 |
> | 路徑 (route) | Bis Lactimether 混合製程 (Bis Lactimether Hybrid Process) | 【bbb0683】 | 通過鋰化 bis lactimether 加成物隨後進行酶促水解 (enzymatic hydrolysis, chymotrypsin) 來合成光學純的 L-BPA。 | 光學純度 (e.e.)、產率、操作 …（完整內容見 JSON）

---
