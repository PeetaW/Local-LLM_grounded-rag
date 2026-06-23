# Eval Report — `q08fix`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-06-23 19:58
- 題數：3

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.583 |
| 平均 grounding 分數 | 0.924 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 2009.5s |
| 平均 planning 延遲 | 9.9s |
| 平均 retrieval 延遲 | 1035.5s |
| 平均 grounding 延遲 | 63.6s |
| └ 其中 NLI | 42.9s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q01 | single_paper | 100.0% | 100.0% | 1.0 | 821.4s | C0/U0 |
| ⚠️ | Q08 | cross_paper | 100.0% | 100.0% | 0.771 | 2939.4s | C0/U8 |
| ✅ | Q09 | cross_paper | 100.0% | 100.0% | 1.0 | 2267.8s | C0/U0 |

## 逐題細節

### ✅ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：821.4s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
> * 預孵育（Preincubation）在與共同孵育（co-incubation）結合時，能顯著增強 JPH203 的抑制效力 (Chunk 1, Chunk 3)。
> * 在 HT-29 細胞中，使用 JPH203 預孵育 120 分鐘展現出濃度依賴性的抑制作用 (Chunk 3)。
> * 在預孵育 120 分鐘後，在測試範圍 0.001, 0.03, 0.1, 0.3, 及 30 mM 中，除 0.001 mM 外的其他濃度下，leucine 攝取量分別下降至對照組的 63%, 51%, 42%, 30%, 和 25% (Chunk 3)。
> * JPH203 預孵育的 $\text{IC}_{50}$ 值為 $193 \pm 50\text{ nM}$ (Chunk 3)。
> * 在使用 10 mM 濃度的 HT-29 細胞中，JPH203 的預孵育展現出時間依賴性的抑制作用 (Chunk 3)。
> * 使用 10 mM JPH203 預孵育 30 分鐘後，leucine 攝取活性下降至對照組的 43%（或如摘要中所述為 42%）(Chunk 3)。
> * 使用 10 mM JPH203 預孵育 60 分鐘後，leucine 攝取活性下降至對照組的 32% (Chunk 3)。
> * 使用 10 mM JPH203 預孵育 120 分鐘後，leucine 攝取活性下降至對照組的 27%（或如摘要中所述為 28%）(Chunk 3)。
> * 單獨共同孵育的 $\text{IC}_{50}$ 值為 $99.2 \pm 11.0\text{ nM}$ (Chunk 1, Chunk 3)。
> * 預孵育與共同孵育結合後的 $\text{IC}_{50}$ 值下降至 $34.2 \pm 3.6\text{ nM}$ (Chunk 1, Chunk 3)。
> * 單獨共同孵 …（完整內容見 JSON）

---

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.771
- 延遲：2939.4s　問題標記：{'conflicts': 0, 'unsupported': 8}

**答案預覽**：

> ## 【論文直接依據】
> 
> ### 成本效益與同位素富集
> * $^{10}\text{B}$ 的成本比普通硼酸高出 1000 倍以上，使得硼原子的成本成為 L-BPA 合成經濟的主要因素 (CMDC-20-e202500059)。
> * 為了評估成本，關鍵參數是總步驟數以及引入硼原子後的產率 (CMDC-20-e202500059)。
> * 使用 $\text{B}_2\text{pin}_2$ 的合成路徑並不被青睞，因為它們會導致 $^{10}\text{B}$ 核心的浪費 (CMDC-20-e202500059)。
> 
> ### 可擴展性與 GMP 合規性
> * L-BPA 合成的開發目標是一個可擴展、低成本且符合 GMP 高純度要求的製程 (CMDC-20-e202500059)。
> * 在超過克級規模的反應中，使用 $\text{HCl}$/水/丙酮進行脫保護通常會留下殘餘的硼酸酯 (CMDC-20-e202500059)。
> * 添加 $\text{NaIO}_4$ 可以解決大規模反應中的殘留問題，但 $\text{NaIO}_4$ 具有毒性，其 $\text{LD}_{50}$ 值分別為 $318\text{ mg/kg}$（雌性大鼠）、$741\text{ mg/kg}$（雄性大鼠）以及 $58\text{ mg/kg}$（小鼠腹腔注射） (CMDC-20-e202500059)。
> * $\text{NaIO}_4$ 也可能導致最終產物的污染 (CMDC-20-e202500059)。
> * 依賴 Pd 催化氫化進行最終脫保護的路徑可能無法僅通過過濾完全去除重金屬；FDA 對於靜脈注射攝入的限制為每天 $10\ \mu\text{g}$ (CMDC-20-e202500059)。
> 
> ### 合成路徑與產率之比較
> **Pd 催化直接硼化**
> * 使用 $\text{B}_2\te …（完整內容見 JSON）

---

### ✅ Q09 · cross_paper

**問題**：Several papers target LAT1 for cancer therapy. What different therapeutic strategies targeting LAT1 are described across the literature, and how do they differ in mechanism?

- detected_paper：`None`
- 選出論文：['LAT1 ChemComm 2026', '1-s2.0-S0378517325007926-main', '1-s2.0-S1347861320300633-main', 's41421-024-00697-6', '1-s2.0-S1936523321002047-main']
- gold_papers：['1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026', 's41421-024-00697-6']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：2267.8s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 根據提供的學術事實，針對 LAT1 的治療策略可分為三種主要方法：利用 LAT1 進行藥物遞送（特別是將硼用於中子捕獲療法）、直接抑制 LAT1 以使癌細胞飢餓或增加其對輻射的敏感性，以及基因調控。
> 
> ## 【論文直接依據】
> 
> **策略 1：利用 LAT1 進行硼遞送 (BNCT)**
> * p-硼苯丙氨酸 (BPA) 利用 LAT1 進行交換運輸 (1-s2.0-S0378517325007926-main)。
> * 當細胞外 BPA 濃度較高時，它被輸入到細胞質中，而細胞內氨基酸（例如谷氨酰胺）則被輸出 (1-s2.0-S0378517325007926-main)。
> * 當細胞外 BPA 濃度較低時，它與細胞外的酪氨酸進行交換 (1-s2.0-S0378517325007926-main)。
> * BPA 在腫瘤中的滯留時間較短 (1-s2.0-S0378517325007926-main)。
> * BPA 需要在熱中子照射前或照射期間 30-60 分鐘持續輸注 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 和 PVA-sorbitol-BPA 是聚乙二醇 (PVA) 與多個 BPA 分子的複合物 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 和 PVA-sorbitol-BPA 通過「LAT1 介導的內吞作用」進入癌細胞，並定位於內體/溶酶體中 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 增加了 BPA 的攝取並延緩其從細胞質中的流出，從而延長了腫瘤滯留時間 (1-s2.0-S0378517325007926-main)。
> * PVA-sorbitol-BPA 被優化至 pH 7.6 以減少腎臟副作用 (1-s2.0-S0378517325007926- …（完整內容見 JSON）

---
