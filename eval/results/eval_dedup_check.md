# Eval Report — `dedup_check`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-06-26 10:07
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.75 |
| 平均 grounding 分數 | 0.653 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 1156.2s |
| 平均 planning 延遲 | 6.9s |
| 平均 retrieval 延遲 | 515.7s |
| 平均 grounding 延遲 | 42.2s |
| └ 其中 NLI | 22.5s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q07 | figure_dependent | 100.0% | 100.0% | 0.667 | 1105.4s | C0/U13 |
| ⚠️ | Q10 | contradiction_check | 100.0% | 100.0% | 0.64 | 1207.0s | C0/U11 |

## 逐題細節

### ⚠️ Q07 · figure_dependent

**問題**：What is the water-stable boroxine structure reported, and what role do the dynamic covalent bonds play in its fluoride binding and hydrogel formation?

- detected_paper：`None`
- 選出論文：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- gold_papers：['41467_2024_Article_45464']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.667
- 延遲：1105.4s　問題標記：{'conflicts': 0, 'unsupported': 13}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
> 
> ## 【論文直接依據】
> 
> **水穩定硼氧環結構 (Water-Stable Boroxine Structure)**
> * 發現了一種源自 2-羥基苯硼酸 (HO-PBA) 的水穩定硼氧環結構 (Chunk 1, Chunk 2, Chunk 3)。
> * HO-PBA 在室溫下會自發性脫水，形成具有動態共價鍵 (DCBs) 和聚集誘導發光 (AIEE) 活性的二聚體 (Chunk 1, Chunk 2, Chunk 3)。
> * 與水接觸後，這些二聚體在室溫下會迅速轉化為被鑑定為 HO-PBA 三聚體–$\text{H}_2\text{O}$ 複合物的硼氧環結構 (Chunk 1, Chunk 2, Chunk 3)。
> * HO-PBA 二聚體向硼氧環結構的轉化已透過核磁共振 (NMR)、質譜 (MS)、UV-Raman 光譜以及理論計算得到驗證 (Chunk 2)。
> * 此硼氧環結構在室溫下可保持穩定 7 天 (Chunk 2)。
> * HO-PBA 三聚體的 pH 穩定範圍介於 $2.4 < \text{pH} < 9$ 之間 (Chunk 1, Chunk 2)。
> * $\text{CH}_3\text{-HO-PBA}$ 三聚體在 $2.4 < \text{pH} < 10$ 的 pH 範圍內保持穩定 (Chunk 1, Chunk 2)。
> 
> **動態共價鍵與氟離子結合 (Dynamic Covalent Bonds and Fluoride Binding)**
> * 硼氧環結構包含水兼容的動態共價 B–O 鍵，這些鍵在室溫的水相介質中能快速交換 (Chunk 1, Chunk 2, Chunk 3)。
> * 傳統的硼氧環需要在甲苯溶液中加熱至 $60\text{ °C}$ …（完整內容見 JSON）

---

### ⚠️ Q10 · contradiction_check

**問題**：Do the papers agree on the hydrolytic stability of boroxine structures? Describe any agreements or differences between them.

- detected_paper：`None`
- 選出論文：['Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation', '41467_2024_Article_45464', 'a-method-for-the-deprotection-of-alkylpinacolyl-boronate-esters']
- gold_papers：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.64
- 延遲：1207.0s　問題標記：{'conflicts': 0, 'unsupported': 11}

**答案預覽**：

> ## 【論文直接依據】
> 
> **硼氧烷的一般穩定性與傳統方法**
> * 一般的硼氧烷在接觸水或潮濕環境時會迅速水解為硼酸，導致功能喪失 (【41467_2024_Article_45464】)。
> * 提高水解穩定性的傳統方法包括引入電子捐贈基團 (【41467_2024_Article_45464】)。
> * 提高水解穩定性的傳統方法包括引入大體積基團 (【41467_2024_Article_45464】)。
> * 提高水解穩定性的傳統方法包括與 N-供體配體形成加合物 (【41467_2024_Article_45464】)。
> * 提高水解穩定性的傳統方法包括將三個硼酸單元整合到柔性大環中以實現熵穩定化 (【41467_2024_Article_45464】)。
> * 這些傳統方法未能從根本上解決水解不穩定性的問題 (【41467_2024_Article_45464】)。
> 
> **HO-PBA 衍生的硼氧烷**
> * 2-羥基苯硼酸 (HO-PBA) 可以形成在水中穩定的硼氧烷結構 (【41467_2024_Article_45464】)。
> * HO-PBA 在室溫下會自發脫水形成二聚體 (【41467_2024_Article_45464】)。
> * 接觸水後，HO-PBA 二聚體在室溫下迅速轉化為包含水穩定硼氧烷結構的 HO-PBA 三聚體–H₂O 複合物 (【41467_2024_Article_45464】)。
> * 衍生自 HO-PBA 的水穩定硼氧烷結構在寬 pH 範圍內保持穩定 (【41467_2024_Article_45464】)。
> * 使用這些結構開發的水凝膠在 pH=2 的酸性溶液中穩定 (【41467_2024_Article_45464】)。
> * 使用這些結構開發的水凝膠在 pH=10 的鹼性溶液中穩定 (【41467_2024_Article_45464】) …（完整內容見 JSON）

---
