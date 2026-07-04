# Eval Report — `q08_query_scaffold_nogate`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-03 05:41
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.5 |
| 平均 grounding 分數 | 0.929 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 1008.4s |
| 平均 planning 延遲 | 8.0s |
| 平均 retrieval 延遲 | 299.1s |
| 平均 grounding 延遲 | 49.7s |
| └ 其中 NLI | 27.4s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 0.929 | 1008.4s | C0/U2 |

## 逐題細節

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.929
- 延遲：1008.4s　問題標記：{'conflicts': 0, 'unsupported': 2}

**答案預覽**：

> **比較框架 (Comparison scaffold)：**
> 
> | 項目/路徑 | 來源論文 | 定義性證據 | 相關比較維度 | 注意事項 |
> | :--- | :--- | :--- | :--- | :--- |
> | **混合製程 (Hybrid Process)** | bbb0683 | 結合對映選擇性烷基化 (enantioselective alkylation) 與酶水解 (enzymatic hydrolysis) | 光學純度 (optical purity)、操作簡便性 (operational simplicity)、可擴展性 (scalability)（透過減少步驟） | 未提供具體的成本或同位素富集值 |
> | **學術與工業策略 (Academic vs. Industrial Strategies)** | CMDC-20-e202500059 | 比較合成策略的綜合綜述 | 可擴展性 (scalability)、成本效益 (cost-effectiveness)、$^{10}\text{B}$ 同位素富集成本 ($^{10}\text{B}$ isotope enrichment costs) | 所提供的事實中未詳細列出這些路徑的具體化學步驟 |
> | **水溶性衍生物 (級聯多元醇, Cascade Polyols)** | synthesis-and-biological-properties... | 使用級聯型多元醇 (cascade type polyols) 作為增溶元件 ($\text{BPA(OH)}_n$) | 溶解度 vs. 細胞攝取/細胞毒性 (cellular uptake/cytotoxicity) | 側重於 BPA 的修飾而非 L-BPA 的初步合成 |
> 
> **路徑圖 (Route Map)：**
> 1. **混 …（完整內容見 JSON）

---
