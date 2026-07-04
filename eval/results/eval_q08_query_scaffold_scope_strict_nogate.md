# Eval Report — `q08_query_scaffold_scope_strict_nogate`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-04 18:20
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.75 |
| 平均 grounding 分數 | 0.889 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 1525.4s |
| 平均 planning 延遲 | 10.2s |
| 平均 retrieval 延遲 | 884.5s |
| 平均 grounding 延遲 | 46.3s |
| └ 其中 NLI | 25.5s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 0.889 | 1525.4s | C0/U2 |

## 逐題細節

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.889
- 延遲：1525.4s　問題標記：{'conflicts': 0, 'unsupported': 2}

**答案預覽**：

> ### 比較框架 (Comparison Scaffold)
> 
> | 路徑 | 來源論文 | 定義性證據 | 相關比較維度 | 注意事項 |
> | :--- | :--- | :--- | :--- | :--- |
> | **混合製程 (Hybrid Process)** | 【bbb0683】 | 使用手性輔助劑 (chiral auxiliary) 的不對稱烷基化 (asymmetric alkylation)，隨後進行胰凝乳蛋白酶 (chymotrypsin) 酶水解 | 對映體純度 (Enantiopurity) (100% e.e.) | - |
> | **Pd 催化交叉耦合 (Pd-catalyzed cross-coupling)** | 【CMDC-20-e202500059】 | 使用 $\text{B}_2\text{pin}_2$ 或 HBpin 與 Pd 催化劑 | 成本效益 (Cost-effectiveness)、產率 (Yield) ($\le 48\%$) | 高 $^{10}\text{B}$ 浪費使其成本過高 |
> | **n-BuLi 路徑 (n-BuLi route)** | 【CMDC-20-e202500059】 | 在 THF 中將 $\text{B}(\text{OBu})_3$ 與 Boc 保護的 4-iodophenylalanine 耦合 | 同位素富集產率 (Isotopic enrichment yield) ($^{10}\text{B}$ 為 62%) | - |
> | **i-PrMgCl / Turbo Grignard** | 【CMDC-20-e202500059】 | 使用 i-PrMgCl 或 i-PrMgCl/LiCl 與硼酸酯 (borate esters) ($\text{B}(\text{OBu})_3 …（完整內容見 JSON）

---
