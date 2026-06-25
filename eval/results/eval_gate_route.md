# Eval Report — `gate_route`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-06-25 09:37
- 題數：5

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.8 |
| 平均 grounding 分數 | 0.809 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 93.8% |
| 平均總延遲 | 1178.3s |
| 平均 planning 延遲 | 8.4s |
| 平均 retrieval 延遲 | 530.8s |
| 平均 grounding 延遲 | 47.8s |
| └ 其中 NLI | 29.7s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q01 | single_paper | 100.0% | 100.0% | 1.0 | 747.8s | C0/U1 |
| ✅ | Q06 | multi_chunk | 100.0% | 75.0% | 0.882 | 1084.1s | C0/U3 |
| ⚠️ | Q08 | cross_paper | 100.0% | 100.0% | 0.794 | 2147.5s | C0/U7 |
| ⚠️ | Q10 | contradiction_check | 100.0% | 100.0% | 0.56 | 1423.7s | C0/U15 |
| ✅ | Q12 | false_premise | N/A | N/A | — | 488.2s | C0/U0 |

## 逐題細節

### ✅ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：747.8s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> ## 【論文直接依據】
> * 預孵育（Preincubation）在與共同孵育（co-incubation）結合時，能顯著增強 JPH203 的抑制效力 (1-s2.0-S1347861320300633-main)。
> * 預孵育與共同孵育產生協同效應 (1-s2.0-S1347861320300633-main)。
> * 單獨進行共同孵育抑制分析的 $\text{IC}_{50}$ 值為 $99.2 \pm 11.0\text{ nM}$ (1-s2.0-S1347861320300633-main)。
> * 預孵育與共同孵育結合處理的 $\text{IC}_{50}$ 值為 $34.2 \pm 3.6\text{ nM}$ (1-s2.0-S1347861320300633-main)。
> * 預孵育與共同孵育的協同效應導致 $\text{IC}_{50}$ 值降低三倍 (1-s2.0-S1347861320300633-main)。
> * 單獨預孵育效果的 $\text{IC}_{50}$ 值為 $193 \pm 50\text{ nM}$ (1-s2.0-S1347861320300633-main)。
> * 共同孵育分析中的 $\text{K}_i$ 值為 $0.46\text{ mM}$ (1-s2.0-S1347861320300633-main)。
> * 預孵育與共同孵育結合分析中的 $\text{K}_i$ 值為 $0.37\text{ mM}$ (1-s2.0-S1347861320300633-main)。
> * 預孵育與共同孵育的協同效應導致 $\text{K}_i$ 值降低 20% (1-s2.0-S1347861320300633-main)。
> 
> ## 【跨文獻推論】
> * 跨論文推論（基於 1-s2.0-S1347861320300633-main）：雖然單獨預 …（完整內容見 JSON）

---

### ✅ Q06 · multi_chunk

**問題**：How does poly(vinyl alcohol) enhance the therapeutic effect of 4-L-boronophenylalanine in neutron capture therapy of thoracic tumors? Explain the proposed mechanism with supporting data.

- detected_paper：`None`
- 選出論文：['1-s2.0-S0378517325007926-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'CMDC-20-e202500059']
- gold_papers：['1-s2.0-S0378517325007926-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：75.0%　grounding：0.882
- 延遲：1084.1s　問題標記：{'conflicts': 0, 'unsupported': 3}

**答案預覽**：

> ## 【論文直接依據】
> * 聚乙烯醇 (PVA) 通過硼酸酯與多個 4-L-硼苯丙氨酸 (BPA) 分子結合，形成稱為 PVA-BPA 的複合物 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 複合物保留了類苯丙氨酸結構，這對於與 L 型氨基酸轉運蛋白 1 (LAT1) 的相互作用至關重要 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 通過 LAT1 介導的胞吞作用被攝入癌細胞中 (1-s2.0-S0378517325007926-main)。
> * 一旦進入細胞內部，PVA-BPA 會定位於內體/溶酶體 (endo-/lysosomes) 中 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 的攝取機制增加了細胞攝取的 BPA 總量 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 的攝取機制延緩了 BPA 從細胞質中的外排 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 的攝取機制延長了 BPA 在腫瘤內的滯留時間 (1-s2.0-S0378517325007926-main)。
> * 通過對 LAT1 的主動靶向，PVA-BPA 能在 1 hour 內高效地積聚並留在腫瘤中 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 在正常器官中的積聚有限 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 提供的機制解決了傳統 BPA 滯留時間短的問題 (1-s2.0-S0378517325007926-main)。
> * 山梨醇被用作溶劑/增溶劑，將配方優化為 PVA-sorbitol-BPA (1-s2.0-S0378517 …（完整內容見 JSON）

---

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.794
- 延遲：2147.5s　問題標記：{'conflicts': 0, 'unsupported': 7}

**答案預覽**：

> ## 【論文直接依據】
> 
> ### 一般經濟與可擴展性因素
> * L-BPA 合成的經濟性主要由 $^{10}\text{B}$ 原子的成本決定，其價格比普通硼酸高出 1000 倍以上，且天然豐度僅為 19.9% (CMDC-20-e202500059)。
> * 比較 L-BPA 合成路徑最相關的評估參數是總步驟數以及引入硼後的產率 (CMDC-20-e202500059)。
> * 使用 $\text{B}_2\text{pin}_2$ 的合成路徑並不理想，因為成本高且會浪費 $^{10}\text{B}$ 核心 (CMDC-20-e202500059)。
> * 在過去十年中，直接氨基酸功能化在很大程度上取代了改良的二鹵代苯合成法，因為後者需要過多步驟 (CMDC-20-e202500059)。
> * 森田製藥公司（Morita Pharmaceutical Company）的專利方法被認為非常有前景，因為其報告的產率極高，但它在倒數第二步需要使用<0xE9><0x88><0x80> (Pd) 催化劑 (CMDC-20-e202500059)。
> * 殘留的 Pd 對於臨床應用構成了安全風險，因為 FDA 限制 Pd 的靜脈注射量為每天 $10\ \mu\text{g}$ (CMDC-20-e202500059)。
> * 在克級反應中用作脫保護試劑的 $\text{NaIO}_4$ 具有毒性，其 $\text{LD}_{50}$ 分別為 $58\ \text{mg}/\text{kg}$（小鼠腹腔注射）、$318\ \text{mg}/\text{kg}$（大鼠口服）和 $741\ \text{mg}/\text{kg}$（大鼠口服）(CMDC-20-e202500059)。
> * 在大規模生產中，$\text{NaIO}_4$ 帶來了製程安全風險和產品污染風險 (CMDC-20-e2025 …（完整內容見 JSON）

---

### ⚠️ Q10 · contradiction_check

**問題**：Do the papers agree on the hydrolytic stability of boroxine structures? Describe any agreements or differences between them.

- detected_paper：`None`
- 選出論文：['Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation', '41467_2024_Article_45464', 's41467-024-45464-z (1)', 'a-method-for-the-deprotection-of-alkylpinacolyl-boronate-esters']
- gold_papers：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.56
- 延遲：1423.7s　問題標記：{'conflicts': 0, 'unsupported': 15}

**答案預覽**：

> ## 【論文直接依據】
> 
> **關於硼氧環（boroxine）的一般穩定性與傳統策略：**
> * 傳統認知認為，硼氧環在水相介質中具有水解不穩定性 (41467_2024_Article_45464, s41467-024-45464-z (1))。
> * 硼氧環在接觸水後通常會迅速水解為硼酸，導致功能喪失 (41467_2024_Article_45464, s41467-024-45464-z (1))。
> * 提高水解穩定性的傳統策略包括引入電子捐贈基團 (41467_2024_Article_45464, s41467-024-45464-z (1))。
> * 提高水解穩定性的傳統策略包括引入大體積基團 (41467_2024_Article_45464, s41467-024-45464-z (1))。
> * 提高水解穩定性的傳統策略包括與 N-供體配體（N-donor ligands）形成加合物 (41467_2024_Article_45464, s41467-024-45464-z (1))。
> * 提高水解穩定性的傳統策略包括一種熵穩定化策略，將三個硼酸單元整合到一個柔性大環中 (41467_2024_Article_45464, s41467-024-45464-z (1))。
> * 硼氧環中傳統的 B–O 鍵交換反應需要在甲苯溶液中於 $60\text{ }^\circ\text{C}$ 加熱 8 小時 (41467_2024_Article_45464, s41467-024-45464-z (1))。
> 
> **關於由 HO-PBA 衍生的硼氧環：**
> * 2-羥基苯硼酸（HO-PBA）可以形成在水中穩定的硼氧環結構 (41467_2024_Article_45464, s41467-024-45464-z (1))。
> * HO-PBA 在室溫下會自發脫水形成二聚體  …（完整內容見 JSON）

---

### ✅ Q12 · false_premise

**問題**：Since BPA is administered orally in BNCT, what oral bioavailability values are reported in these papers?

- detected_paper：`None`
- 選出論文：['cas0106-0279', 'Chem Biol Drug Des - 2013 - Achilli - Folic Acid‐Conjugated 4‐Amino‐Phenylboronate  a Boron‐Containing Compound Designed']
- gold_papers：[]
- 論文選擇命中率：N/A　檢索覆蓋率：N/A　grounding：—
- 延遲：488.2s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **誠實棄答**：檢索到的內容與此問題相關，但並不包含可直接回答的資訊。為避免編造不存在的數據，本系統選擇不作答。建議換個問法，或確認該主題是否涵蓋於文獻庫中。

---
