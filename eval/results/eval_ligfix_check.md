# Eval Report — `ligfix_check`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-06-29 21:21
- 題數：3

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.75 |
| 平均 grounding 分數 | 0.0 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 1657.3s |
| 平均 planning 延遲 | 7.7s |
| 平均 retrieval 延遲 | 460.5s |
| 平均 grounding 延遲 | 626.8s |
| └ 其中 NLI | 0.0s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ❌ | Q05 | single_paper | 100.0% | 100.0% | 0.0 | 1271.9s | C0/U23 |
| ❌ | Q07 | figure_dependent | 100.0% | 100.0% | 0.0 | 1837.7s | C0/U33 |
| ❌ | Q10 | contradiction_check | 100.0% | 100.0% | 0.0 | 1862.2s | C0/U26 |

## 逐題細節

### ❌ Q05 · single_paper

**問題**：What is the solvent-free green protocol for N-Boc protection of amines using picric acid as a catalyst, and what reaction conditions are used?

- detected_paper：`None`
- 選出論文：['1-s2.0-S2773223124000268-main']
- gold_papers：['1-s2.0-S2773223124000268-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.0
- 延遲：1271.9s　問題標記：{'conflicts': 0, 'unsupported': 23}

**答案預覽**：

> ## 【論文直接依據】
> 
> * 苦味酸被用作 Brønsted 酸催化劑或有機催化劑 (Chunk 1, Chunk 2)。
> * 所使用的苦味酸催化劑量為 2 mol% (Chunk 1, Chunk 2)。
> * 所使用的苦味酸催化劑量為 0.2 當量 (Chunk 1, Chunk 2)。
> * 二叔丁基二碳酸酯 ((Boc)₂O / Boc anhydride) 被用作保護劑 (Chunk 1, Chunk 2)。
> * 反應在無溶劑（neat）條件下進行 (Chunk 1, Chunk 2)。
> * 反應在室溫 (rt) 下進行 (Chunk 1, Chunk 2)。
> * 最佳反應時間為 60 分鐘 (Chunk 1, Chunk 2)。
> * 此方法達到的最高產率為 98% (Chunk 1, Chunk 2)。
> * 對於芳香族一級胺，在室溫下使用 0.2 當量催化劑反應 60 分鐘可獲得 95% 的產率 (Chunk 1, Chunk 2)。
> * 適用底物包括芳香族胺，例如 2-aminophenol (Chunk 1)。
> * 適用底物包括雜環芳香族胺，例如 quinolin-4-amine (Chunk 1)。
> * 適用底物包括雜環芳香族胺，例如 benzo[d]thiazol-2-amine (Chunk 1)。
> * 適用底物包括雜環芳香族胺，例如 1H-imidazole (Chunk 1)。
> * 適用底物包括脂肪族胺，例如 piperidine (Chunk 1)。
> * 適用底物包括脂肪族胺，例如 morpholine (Chunk 1)。
> * 適用底物包括脂肪族胺，例如 pyrrolidine (Chunk 1)。
> * 適用底物包括脂肪族胺，例如 cyclohexanamine (Chunk 1)。
> * 擬議反應機制的第一步是苦味酸對 Boc anhydride  …（完整內容見 JSON）

---

### ❌ Q07 · figure_dependent

**問題**：What is the water-stable boroxine structure reported, and what role do the dynamic covalent bonds play in its fluoride binding and hydrogel formation?

- detected_paper：`None`
- 選出論文：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- gold_papers：['41467_2024_Article_45464']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.0
- 延遲：1837.7s　問題標記：{'conflicts': 0, 'unsupported': 33}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
> 
> ## 【論文直接依據】
> * 由 2-羥基苯硼酸 (HO-PBA) 形成一種水穩定的 boroxine 結構 (41467_2024_Article_45464)。
> * 此 boroxine 結構在寬廣的 pH 範圍內保持穩定 (41467_2024_Article_45464)。
> * HO-PBA boroxine 結構中的 B–O 鍵是水兼容的動態共價鍵 (DCBs) (41467_2024_Article_45464)。
> * 這些 DCBs 在室溫的水相介質中可以進行快速交換 (41467_2024_Article_45464)。
> * HO-PBA 在環境條件下會自發地發生脫水反應以形成二聚體 (41467_2024_Article_45464)。
> * 在室溫下接觸水時，HO-PBA 二聚體會迅速轉化為 boroxine 結構，具體為 HO-PBA 三聚體–$\text{H}_2\text{O}$ 錯合物 (41467_2024_Article_45464)。
> * 從二聚體到三聚體的轉化是由焓驅動的，其焓變 $\Delta H = -20.10 \text{ kJ mol}^{-1}$ 且熵變 $\Delta S = -6.46 \text{ J mol}^{-1} \text{ K}^{-1}$ (41467_2024_Article_45464)。
> * 一個 HO-PBA 三聚體可結合兩個 $\text{D}_2\text{O}$ 分子 (41467_2024_Article_45464)。
> * HO-PBA 三聚體的穩定 pH 範圍為 $2.4 < \text{pH} < 9$ (41467_2024_Article_45464)。
> * $\text{CH}_3\t …（完整內容見 JSON）

---

### ❌ Q10 · contradiction_check

**問題**：Do the papers agree on the hydrolytic stability of boroxine structures? Describe any agreements or differences between them.

- detected_paper：`None`
- 選出論文：['Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation', '41467_2024_Article_45464', 'a-method-for-the-deprotection-of-alkylpinacolyl-boronate-esters']
- gold_papers：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.0
- 延遲：1862.2s　問題標記：{'conflicts': 0, 'unsupported': 26}

**答案預覽**：

> ## 【論文直接依據】
> 
> **一般穩定性與策略**
> * Boroxines 在水相介質中傳統上具有水解不穩定性，且會迅速水解為硼酸 (boronic acids)，導致功能喪失 (41467_2024_Article_45464)。
> * 提高 boroxine 穩定性的策略包括引入電子捐贈基團 (electron-donating groups) (41467_2024_Article_45464)。
> * 提高 boroxine 穩定性的策略包括添加大尺寸基團 (41467_2024_Article_45464)。
> * 提高 boroxine 穩定性的策略包括與 N-供體配體 (N-donor ligands) 形成加合物 (41467_2024_Article_45464)。
> * 提高 boroxine 穩定性的策略包括將三個硼酸單元整合到一個柔性大環中以實現熵穩定化 (entropic stabilization) (41467_2024_Article_45464)。
> * 將三個硼酸單元整合到柔性大環中是增強 boroxines 水解穩定性的有效方法 (【Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation】)。
> 
> **HO-PBA 衍生的 Boroxines**
> * 2-羥基苯硼酸 (2-hydroxyphenylboronic acid, HO-PBA) 會形成一種在水中穩定且具有水兼容動態共價鍵 (dynamic covalent bonds, DCBs) 的 boroxine 結構 (41467_2024_Article_45464)。
> * HO-PBA 在環境條件下會自發脫水形成二聚體，其脫水自由能 …（完整內容見 JSON）

---
