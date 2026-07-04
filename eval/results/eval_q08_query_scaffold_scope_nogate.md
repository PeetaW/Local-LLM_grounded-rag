# Eval Report — `q08_query_scaffold_scope_nogate`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-04 00:31
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.5 |
| 平均 grounding 分數 | 0.842 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 2097.5s |
| 平均 planning 延遲 | 10.2s |
| 平均 retrieval 延遲 | 945.1s |
| 平均 grounding 延遲 | 65.1s |
| └ 其中 NLI | 36.6s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 0.842 | 2097.5s | C0/U5 |

## 逐題細節

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.842
- 延遲：2097.5s　問題標記：{'conflicts': 0, 'unsupported': 5}

**答案預覽**：

> **比較框架 (Comparison scaffold)：**
> 
> | 項目/路徑 | 來源論文 | 定義性證據 | 相關比較維度 | 注意事項 |
> | :--- | :--- | :--- | :--- | :--- |
> | 混合製程 (Hybrid Process) | bbb0683 | 對映選擇性烷基化 (enantioselective alkylation) 後接酶促水解 (enzymatic hydrolysis) | 光學純度 (optical purity, e.e.) | 需要特定酶以達到 100% e.e. |
> | 非 Pd 路徑 (Non-Pd Routes) | CMDC-20-e202500059 | 使用 n-BuLi 或 i-PrMgCl 與 $\text{B}(\text{OBu})_3$ 進行金屬-鹵素交換 (metal-halogen exchange) | 同位素富集成本、避免使用 Pd、產率 (yield) | 需要低溫（部分步驟為 -78°C） |
> | Morita 路徑 (Morita Route) | CMDC-20-e202500059 | 硼化後步驟較少的高產率序列 | 可擴展性 (scalability, 產率/步驟) | 在倒數第二步使用 Pd 催化劑 (Pd catalyst) |
> | 先前方法 (Previous Methods) | bbb0683 | 消旋前體 (racemic precursors) 的酶促拆分 (enzymatic resolution) 或對映選擇性氫化 (enantioselective hydrogenation) | 光學純度 (optical purity, e.e.) | e.e. 低於混合製程 |
> 
> **路徑圖 (Route Map)：**
> 1. **混合製程 (Hybrid P …（完整內容見 JSON）

---
