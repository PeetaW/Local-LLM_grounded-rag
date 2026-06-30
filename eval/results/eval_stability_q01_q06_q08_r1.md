# Eval Report — `stability_q01_q06_q08_r1`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-01 00:45
- 題數：3

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.75 |
| 平均 grounding 分數 | 0.901 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 91.7% |
| 平均總延遲 | 2428.8s |
| 平均 planning 延遲 | 9.4s |
| 平均 retrieval 延遲 | 536.2s |
| 平均 grounding 延遲 | 54.6s |
| └ 其中 NLI | 34.4s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q01 | single_paper | 100.0% | 100.0% | 1.0 | 4423.9s | C0/U0 |
| ✅ | Q06 | multi_chunk | 100.0% | 75.0% | 0.882 | 1143.8s | C0/U5 |
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 0.822 | 1718.6s | C0/U8 |

## 逐題細節

### ✅ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：4423.9s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
> * 使用 LAT1 陽性的人類結腸癌 HT-29 細胞中的亮氨酸（leucine）攝取分析，對 JPH203 對於 LAT1 的預孵育抑制效果進行了表徵 (1-s2.0-S1347861320300633-main)。
> * JPH203 在 HT-29 細胞中的預孵育抑制效果具有濃度依賴性 (1-s2.0-S1347861320300633-main)。
> * JPH203 在 HT-29 細胞中的預孵育抑制效果具有時間依賴性 (1-s2.0-S1347861320300633-main)。
> * 在暴露於 $0.03\text{ mM}$ JPH203 120 分鐘後，亮氨酸攝取量下降至對照組的 $63\%$ (1-s2.0-S1347861320300633-main)。
> * 在暴露於 $0.1\text{ mM}$ JPH203 120 分鐘後，亮氨酸攝取量下降至對照組的 $51\%$ (1-s2.0-S1347861320300633-main)。
> * 在暴露於 $0.3\text{ mM}$ JPH203 120 分鐘後，亮氨酸攝取量下降至對照組的 $42\%$ (1-s2.0-S1347861320300633-main)。
> * 在暴露於 $30\text{ mM}$ JPH203 120 分鐘後，亮氨酸攝取量下降至對照組的 $25\%$ (1-s2.0-S1347861320300633-main)。
> * 此預孵育抑制效果的 $\text{IC}_{50}$ 值為 $193 \pm 50\text{ nM}$ (1-s2.0-S1347861320300633-main)。
> * 當 HT-29 細胞與 $10\text{ mM}$ JPH203 預孵育 30 分鐘時，攝取活性下降至對照組的 $43\%$ (1-s2.0-S134786132 …（完整內容見 JSON）

---

### ✅ Q06 · multi_chunk

**問題**：How does poly(vinyl alcohol) enhance the therapeutic effect of 4-L-boronophenylalanine in neutron capture therapy of thoracic tumors? Explain the proposed mechanism with supporting data.

- detected_paper：`None`
- 選出論文：['1-s2.0-S0378517325007926-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'CMDC-20-e202500059']
- gold_papers：['1-s2.0-S0378517325007926-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：75.0%　grounding：0.882
- 延遲：1143.8s　問題標記：{'conflicts': 0, 'unsupported': 5}

**答案預覽**：

> ## 【論文直接依據】
> * 聚乙烯醇 (PVA) 透過硼酸酯與多個 4-L-硼苯丙氨酸 (BPA) 分子形成複合物，稱為 PVA-BPA (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 複合物保留了類苯丙氨酸結構，能夠靶向 L 型氨基酸轉運蛋白 1 (LAT1) (1-s2.0-S0378517325007926-main)。
> * 常規 BPA 經由 LAT1 進入細胞質，但當胞外 BPA 濃度降低時，會透過與胞外酪胺 (tyrosine) 交換而流出 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 透過 LAT1 介導的內吞作用被攝入癌細胞中 (1-s2.0-S0378517325007926-main)。
> * 進入細胞後，PVA-BPA 定位於內體/溶酶體 (endo-/lysosomes) 中 (1-s2.0-S0378517325007926-main)。
> * 在內體/溶酶體中的定位增加了 PVA-BPA 的細胞攝取量 (1-s2.0-S0378517325007926-main)。
> * 在內體/溶酶體中的定位延緩了 BPA 從細胞質的流出 (1-s2.0-S0378517325007926-main)。
> * 流出的延緩延長了 PVA-BPA 在腫瘤內的滯留時間 (1-s2.0-S0378517325007926-main)。
> * 透過對 LAT1 的主動靶向，PVA-BPA 能在 1 hour 內高效地積聚並滯留在腫瘤中 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 在正常器官中的積聚有限 (1-s2.0-S0378517325007926-main)。
> * 使用山梨醇 (sorbitol) 作為增溶劑，開發了一種 PVA-sorbitol-BPA  …（完整內容見 JSON）

---

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.822
- 延遲：1718.6s　問題標記：{'conflicts': 0, 'unsupported': 8}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
> 
> ## 【論文直接依據】
> 
> ### 直接硼化路徑
> **Pd 催化直接硼化**
> * $\text{B}_2\text{pin}_2$ 與 DMSO 及 KOAc 的組合優於 HBpin 與 1,4-dioxane 及 $\text{NEt}_3$ 的組合 【CMDC-20-e202500059】。
> * Pd 催化直接硼化路徑的總產率未超過 $48\%$ 【CMDC-20-e202500059】。
> * 使用 $\text{B}_2\text{pin}_2$ 的成本極高，因為它會浪費 $^{10}\text{B}$ 原子 【CMDC-20-e202500059】。
> 
> **非 Pd 催化直接硼化**
> * 在 Boc 保護的 4-碘苯丙氨酸上使用 n-BuLi 和 $\text{B}(\text{OBu})_3$，總產率為 $53\text{--}58\%$ 【CMDC-20-e202500059】。
> * n-BuLi 路徑的 $^{10}\text{B}$ 版本產率可達 $62\%$ 【CMDC-20-e202500059】。
> * 使用 i-PrMgCl 和 $\text{B}(\text{OBu})_3$ 的總產率為 $78\%$ 【CMDC-20-e202500059】。
> * 使用 Turbo Grignard (i-PrMgCl/LiCl) 試劑的產率在 $80\text{--}85\%$ 之間 【CMDC-20-e202500059】。
> * Turbo Grignard 路徑的 $^{10}\text{B}$ 版本最高產率可達 $87\%$ 【CMDC-20-e202500059】。
> 
> ### 不對稱氫化與混合製程路徑
> **不對稱氫化路徑**
> * 該過程涉及通過 2-苯基氧唑- …（完整內容見 JSON）

---
