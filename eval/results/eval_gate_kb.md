# Eval Report — `gate_kb`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-06-25 02:02
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.5 |
| 平均 grounding 分數 | 0.727 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 1865.0s |
| 平均 planning 延遲 | 9.5s |
| 平均 retrieval 延遲 | 855.0s |
| 平均 grounding 延遲 | 62.4s |
| └ 其中 NLI | 39.1s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q08 | cross_paper | 100.0% | 100.0% | 0.727 | 1865.0s | C0/U7 |

## 逐題細節

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.727
- 延遲：1865.0s　問題標記：{'conflicts': 0, 'unsupported': 7}

**答案預覽**：

> ## 【論文直接依據】
> 
> **同位素富集與成本效益因素**
> * $^{10}\text{B}$ 的天然豐度為 19.9% (CMDC-20-e202500059)。
> * 純化形式的 $^{10}\text{B}$ 估計成本比普通硼酸高出 1000 倍以上 (CMDC-20-e202500059)。
> * 純化 $^{10}\text{B}$ 的成本主導了 L-BPA 合成的經濟效率 (CMDC-20-e202500059)。
> * 高純度且同位素富集 ($^{10}\text{B}$) 材料的合成仍然具有挑戰性 (CMDC-20-e202500059)。
> * 評估 L-BPA 合成方法最相關的參數是產率以及引入硼原子後的總步驟數 (CMDC-20-e202500059)。
> 
> **路徑 1：透過有機金屬試劑引入硼 (CMDC-20-e202500059)**
> * 使用 $\text{B}_2\text{pin}_2$ 的路徑並不流行，因為它會導致 $^{10}\text{B}$ 原子核的浪費 (CMDC-20-e202500059)。
> * 將 $\text{B}_2\text{pin}_2$ 與 $\text{DMSO}$ 和 $\text{KOAc}$ 結合使用優於使用 $\text{HBpin}$ 搭配 1,4-dioxane 和 $\text{NEt}_3$，因為後者的產率 $\le 48\%$ (CMDC-20-e202500059)。
> * 在 $\text{THF}$ 中利用 $\text{n-BuLi}$ 和 $\text{B}(\text{OBu})_3$，隨後在丙酮/水中使用 $\text{HCl}$ 進行脫保護的路徑，總產率為 53–58% (CMDC-20-e202500059)。
> * 在與 $\text{B}(\text{OBu})_3$ 反應時將 $\te …（完整內容見 JSON）

---
