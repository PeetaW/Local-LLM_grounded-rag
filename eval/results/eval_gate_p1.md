# Eval Report — `gate_p1`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-06-24 00:03
- 題數：5

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.9 |
| 平均 grounding 分數 | 0.691 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 93.8% |
| 平均總延遲 | 873.8s |
| 平均 planning 延遲 | 7.3s |
| 平均 retrieval 延遲 | 305.6s |
| 平均 grounding 延遲 | 32.5s |
| └ 其中 NLI | 16.6s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q01 | single_paper | 100.0% | 100.0% | 0.944 | 872.9s | C0/U1 |
| ⚠️ | Q05 | single_paper | 100.0% | 100.0% | 0.5 | 735.4s | C0/U12 |
| ✅ | Q06 | multi_chunk | 100.0% | 75.0% | 0.842 | 1064.3s | C0/U3 |
| ⚠️ | Q08 | cross_paper | 100.0% | 100.0% | 0.667 | 921.7s | C0/U6 |
| ⚠️ | Q12 | false_premise | N/A | N/A | 0.5 | 774.5s | C0/U3 |

## 逐題細節

### ✅ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.944
- 延遲：872.9s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> ## 【論文直接依據】
> * 預孵育（Preincubation）在與共同孵育（co-incubation）結合時，能顯著增強 JPH203 的抑制效力 (Chunk 1, Chunk 3)。
> * 單獨共同孵育的 $\text{IC}_{50}$ 值為 $99.2 \pm 11.0\text{ nM}$ (Chunk 1, Chunk 3)。
> * 預孵育結合共同孵育的 $\text{IC}_{50}$ 值為 $34.2 \pm 3.6\text{ nM}$ (Chunk 1, Chunk 3)。
> * 單獨共同孵育的 $\text{K}_i$ 值為 $0.46\text{ mM}$ (Chunk 1, Chunk 3)。
> * 預孵育結合共同孵育的 $\text{K}_i$ 值為 $0.37\text{ mM}$ (Chunk 1, Chunk 3)。
> * 預孵育與共同孵育之間的協同作用導致 $\text{IC}_{50}$ 值降低三倍 (Chunk 1)。
> * 預孵育與共同孵育之間的協同作用導致 $\text{K}_i$ 值降低 $20\%$ (Chunk 1)。
> * 在 HT-29 細胞中，JPH203 預孵育 120 分鐘表現出濃度依賴性 (Chunk 3)。
> * 除 $0.001\text{ mM}$ 濃度外，暴露於 $0.03$、$0.1$、$0.3$ 和 $30\text{ mM}$ 的 JPH203 中 120 分鐘，導致亮氨酸（leucine）攝取量分別下降至對照組的 $63\%$、$51\%$、$42\%$、$30\%$ 和 $25\%$ (Chunk 3)。
> * 專指預孵育效果的 $\text{IC}_{50}$ 值為 $193 \pm 50\text{ nM}$ (Chunk 3)。
> * 在 HT-29 細胞中，使用 $10\text{ mM}$ 濃度時，J …（完整內容見 JSON）

---

### ⚠️ Q05 · single_paper

**問題**：What is the solvent-free green protocol for N-Boc protection of amines using picric acid as a catalyst, and what reaction conditions are used?

- detected_paper：`None`
- 選出論文：['1-s2.0-S2773223124000268-main']
- gold_papers：['1-s2.0-S2773223124000268-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.5
- 延遲：735.4s　問題標記：{'conflicts': 0, 'unsupported': 12}

**答案預覽**：

> ## 【論文直接依據】
> 
> * 苦味酸被用作胺類 N-Boc 保護的 Brønsted 酸催化劑 (【1-s2.0-S2773223124000268-main】)。
> * 二叔丁基二碳酸酯 ((Boc)₂O / Boc anhydride) 被用作保護試劑 (【1-s2.0-S2773223124000268-main】)。
> * 所使用的催化劑量為 2 mol% 或 0.2 當量 (【1-s2.0-S2773223124000268-main】)。
> * 根據表 1 的 Entry 13，最佳催化劑用量為 0.2 當量 (【1-s2.0-S2773223124000268-main】)。
> * 反應在無溶劑 (neat) 條件下進行 (【1-s2.0-S2773223124000268-main】)。
> * 反應在室溫 (rt) 下進行 (【1-s2.0-S2773223124000268-main】)。
> * 優化後的反應時間為 60 分鐘 (【1-s2.0-S2773223124000268-main】)。
> * 此方法達到的最高產率為 98% (【1-s2.0-S2773223124000268-main】)。
> * 對於芳香族一級胺，在室溫下使用 0.2 當量催化劑反應 60 分鐘，產率為 95% (【1-s2.0-S2773223124000268-main】)。
> * 此方法適用於芳香族胺 (【1-s2.0-S2773223124000268-main】)。
> * 此方法適用於雜環芳香族胺 (【1-s2.0-S2773223124000268-main】)。
> * 此方法適用於開鏈脂肪族胺 (【1-s2.0-S2773223124000268-main】)。
> * 此方法適用於環狀脂肪族胺 (【1-s2.0-S2773223124000268-main】)。
> * 在提出的機理中，苦味 …（完整內容見 JSON）

---

### ✅ Q06 · multi_chunk

**問題**：How does poly(vinyl alcohol) enhance the therapeutic effect of 4-L-boronophenylalanine in neutron capture therapy of thoracic tumors? Explain the proposed mechanism with supporting data.

- detected_paper：`None`
- 選出論文：['1-s2.0-S0378517325007926-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'CMDC-20-e202500059']
- gold_papers：['1-s2.0-S0378517325007926-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：75.0%　grounding：0.842
- 延遲：1064.3s　問題標記：{'conflicts': 0, 'unsupported': 3}

**答案預覽**：

> ## 【論文直接依據】
> * 聚乙烯醇 (Poly(vinyl alcohol), PVA) 與多個 4-L-硼苯丙氨酸 (4-L-boronophenylalanine, BPA) 分子透過硼酸酯 (boronate esters) 形成稱為 PVA-BPA 的複合物 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 保留了靶向 LAT1 所需的必需苯丙氨酸樣結構 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 與 LAT1 相互作用 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 透過 LAT1 介導的內吞作用 (endocytosis) 被攝入癌細胞中 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 定位於內體/溶酶體 (endo-/lysosomes) 中 (1-s2.0-S0378517325007926-main)。
> * 定位於內體/溶酶體增加了細胞攝取量 (1-s2.0-S0378517325007926-main)。
> * 定位於內體/溶酶體延緩了 BPA 從細胞質中的外排 (efflux) (1-s2.0-S0378517325007926-main)。
> * 定位於內體/溶酶體延長了 BPA 在腫瘤內的滯留時間 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 在 1 hour 內透過 LAT1 主動靶向高效地積聚在腫瘤中 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 在腫瘤中實現了長期滯留 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 限制了在正常器官中的積聚 (1-s2.0-S037851732 …（完整內容見 JSON）

---

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.667
- 延遲：921.7s　問題標記：{'conflicts': 0, 'unsupported': 6}

**答案預覽**：

> ## 【論文直接依據】
> 
> **一般考量與同位素富集**
> * $^{10}\text{B}$ 的同位素富集成本昂貴 (【CMDC-20-e202500059】)。
> * 目前缺乏對 L-BPA 合成路徑優缺點的批判性分析，導致新的生產基地難以確定最合適的路徑 (【CMDC-20-e202500059】)。
> 
> **路徑 1：酶促拆分或不對稱氫化**
> * 對外消旋前體使用酶促拆分被認為並不理想，因為需要額外的分離步驟 (【bbb0683】)。
> * 使用手性膦進行不對稱氫化被認為並不理想，因為底物難以獲取且需要額外的分離步驟 (【bbb0683】)。
> 
> **路徑 2：混合合成方案**
> * 此路徑由兩個步驟組成 (【bbb0683】)。
> * 第一步涉及使用 (2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine 和受保護的 4-bromomethylbenzeneboronic acid 衍生物進行對映選擇性烷基化 (【bbb0683】)。
> * 第一步在 THF 中於 -78°C 下進行 (【bbb0683】)。
> * 第二步涉及使用胰蛋白酶 (trypsin) 在 pH 5.0 下水解 L-BPA 甲酯 (【bbb0683】)。
> * 此路徑產生的 L-BPA 光學純度為 100% (【bbb0683】)。
> * 該方案的優點包括反應步驟少 (【bbb0683】)。
> * 該方案的優點包括操作簡單 (【bbb0683】)。
> * 該方案的優點包括後處理方便 (【bbb0683】)。
> 
> ## 【跨文獻推論】
> 
> **可擴展性與成本效益之比較**
> * 跨論文推論（基於 【bbb0683】）：混合合成方案比酶促拆分或不對稱氫化更具可擴展性和成本效益，因為它避免了底物獲取困難和需要額外分離步驟等「並不理想」的障礙，同時提供了更簡單的操作和更少的步驟。 …（完整內容見 JSON）

---

### ⚠️ Q12 · false_premise

**問題**：Since BPA is administered orally in BNCT, what oral bioavailability values are reported in these papers?

- detected_paper：`None`
- 選出論文：['cas0106-0279', 'Chem Biol Drug Des - 2013 - Achilli - Folic Acid‐Conjugated 4‐Amino‐Phenylboronate  a Boron‐Containing Compound Designed']
- gold_papers：[]
- 論文選擇命中率：N/A　檢索覆蓋率：N/A　grounding：0.5
- 延遲：774.5s　問題標記：{'conflicts': 0, 'unsupported': 3}

**答案預覽**：

> ## 【論文直接依據】
> 
> * BNCT 的硼載體透過靜脈輸注給藥（【Chem Biol Drug Des - 2013 - Achilli - Folic Acid‐Conjugated 4‐Amino‐Phenylboronate a Boron‐Containing Compound Designed】）。
> * 提供之文本中未報告 BPA 的口服生物利用度數值（【Chem Biol Drug Des - 2013 - Achilli - Folic Acid‐Conjugated 4‐Amino‐Phenylboronate a Boron‐Containing Compound Designed】; cas0106-0279）。
> 
> ## 【跨文獻推論】
> 
> * 文獻依據不足：提供的文件不包含關於 BPA 口服給藥或生物利用度的數據，因此無法推導出這些數值。
> 
> ## 【知識延伸與推測】
> 
> * 模型推測（文獻依據不足）：問題的前提與提供的證據之間存在差異；雖然問題暗示 BPA 是口服給藥，但 Fact 4 明確指出 BNCT 的硼載體是透過靜脈輸注遞送。在臨床實踐中，BPA (p-boronophenylalanine) 通常採取靜脈注射，以確保對血液濃度的精確控制，並達到有效治療所需的腫瘤與血液硼比率（如 Fact 9 所述）。口服生物利用度可能會受到首過代謝或胃腸道中氨基酸轉運蛋白（例如 Fact 6 中提到的 LAT1/2）不一致吸收的影響。
> 
> ---
> 📋 **答案品質報告**
> 
> 📊 **分段論文依據率：**
> 
>   ⚠️ 【論文直接依據】：50.0%（1/2 句）
>   ❌ 【跨文獻推論】：0.0%（0/1 句）  ← 跨論文推論，低分為預期範圍
>   ❌ 【知識延伸推測】：0.0%（0/1 句）  ← 知識延伸推測，低分為預期範圍
> 
> ⚠️ **直引依據率**：50.0% …（完整內容見 JSON）

---
