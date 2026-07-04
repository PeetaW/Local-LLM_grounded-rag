# Eval Report — `q08_tradeoff_v3_nogate`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-02 21:21
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.5 |
| 平均 grounding 分數 | 0.812 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 2165.9s |
| 平均 planning 延遲 | 10.8s |
| 平均 retrieval 延遲 | 1192.1s |
| 平均 grounding 延遲 | 68.7s |
| └ 其中 NLI | 44.3s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 0.812 | 2165.9s | C0/U6 |

## 逐題細節

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.812
- 延遲：2165.9s　問題標記：{'conflicts': 0, 'unsupported': 6}

**答案預覽**：

> ## 【論文直接依據】
> 
> ### 同位素富集與成本效益因素
> * $^{10}\text{B}$ 通常透過化學交換蒸餾 (chemical exchange distillation) 從 $^{11}\text{B}$ 中分離 (CMDC-20-e202500059)。
> * $^{10}\text{B}$ 的成本比普通硼酸高出 1000 倍以上 (CMDC-20-e202500059)。
> * L-BPA 合成的經濟效率是根據引入硼之後的總步驟數和產率來評估的 (CMDC-20-e202500059)。
> * 依賴 $\text{B}_2\text{pin}_2$ 的合成路徑成本極高，因為它們會浪費 $^{10}\text{B}$ 原子 (CMDC-20-e202500059)。
> 
> ### Pd 催化與直接功能化路徑
> * 二鹵代苯 (Dihalobenzenes) 的修飾方法因步驟過多，已在很大程度上被直接氨基酸功能化 (direct amino acid functionalization) 所取代 (CMDC-20-e202500059)。
> * 在 Pd 催化路徑中，$\text{B}_2\text{pin}_2/\text{DMSO}/\text{KOAc}$ 的組合比 $\text{HBpin}/1,4\text{-dioxane}/\text{NEt}_3$ 更有效，因為後者的交叉耦合 (cross-coupling) 和脫保護 (deprotection) 總產率不超過 48% (CMDC-20-e202500059)。
> * 森田製藥公司 (Morita Pharmaceutical Company, Kamekawa et al.) 的專利方法具有步驟少且產率極高的特點，但在倒數第二步需要<0xE9><0x88><0x80>催化劑 (palladium catal …（完整內容見 JSON）

---
