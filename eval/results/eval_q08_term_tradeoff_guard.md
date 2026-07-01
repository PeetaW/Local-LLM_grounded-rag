# Eval Report — `q08_term_tradeoff_guard`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-01 21:08
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.5 |
| 平均 grounding 分數 | 0.778 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 2026.8s |
| 平均 planning 延遲 | 10.8s |
| 平均 retrieval 延遲 | 1044.4s |
| 平均 grounding 延遲 | 79.6s |
| └ 其中 NLI | 50.0s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q08 | cross_paper | 100.0% | 100.0% | 0.778 | 2026.8s | C0/U8 |

## 逐題細節

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.778
- 延遲：2026.8s　問題標記：{'conflicts': 0, 'unsupported': 8}

**答案預覽**：

> ## 【論文直接依據】
>
> ### Pd催化交叉耦合路徑 (Pd-Catalyzed Cross-Coupling Route)
> * **同位素富集與成本效益 (Isotopic Enrichment & Cost-effectiveness)**：較不傾向使用 $\text{B}_2\text{pin}_2$，因為它會導致 $^{10}\text{B}$ 原子核的浪費且成本極高 (【CMDC-20-e202500059】)。
> * **同位素富集與成本效益 (Isotopic Enrichment & Cost-effectiveness)**：$\text{B}_2\text{pin}_2$ 結合 DMSO 和 $\text{KOAc}$ 優於 HBpin 結合 1,4-dioxane 和 $\text{NEt}_3$ (【CMDC-20-e202500059】)。
> * **成本效益 (Cost-effectiveness)**：Pd交叉耦合 (Pd cross-coupling) 的總產率不超過 48% (【CMDC-20-e202500059】)。
> * **可擴展性與安全性 (Scalability & Safety)**：此路徑存在金屬污染 (metal contamination) 的風險 (【CMDC-20-e202500059】)。
> * **可擴展性與安全性 (Scalability & Safety)**：在克級生產中，於 $\text{HCl}$ 去保護過程中使用 $\text{NaIO}_4$ 以避免殘留硼酸酯 (boronic acid ester) 會引入毒性與製程安全風險 (【CMDC-20-e202500059】)。
>
> ### 有機金屬路徑 (Organometallic Routes) (n-BuLi / i-PrMgCl)
> * **成本效益 ( …（完整內容見 JSON）

---
