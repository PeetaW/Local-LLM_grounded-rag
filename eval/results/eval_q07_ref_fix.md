# Eval Report — `q07_ref_fix`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-06-26 21:19
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 1.0 |
| 平均 grounding 分數 | 0.87 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 1070.3s |
| 平均 planning 延遲 | 7.1s |
| 平均 retrieval 延遲 | 313.9s |
| 平均 grounding 延遲 | 46.9s |
| └ 其中 NLI | 19.3s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q07 | figure_dependent | 100.0% | 100.0% | 0.87 | 1070.3s | C0/U5 |

## 逐題細節

### ✅ Q07 · figure_dependent

**問題**：What is the water-stable boroxine structure reported, and what role do the dynamic covalent bonds play in its fluoride binding and hydrogel formation?

- detected_paper：`None`
- 選出論文：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- gold_papers：['41467_2024_Article_45464']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.87
- 延遲：1070.3s　問題標記：{'conflicts': 0, 'unsupported': 5}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
>
> ## 【論文直接依據】
>
> **水穩定硼氧環結構及其形成**
> * 報導的水穩定硼氧環結構為 HO-PBA 三聚體–H$_2$O 複合物 (41467_2024_Article_45464)。
> * 2-羥基苯硼酸 (HO-PBA) 在室溫下會自發脫水形成二聚體 (41467_2024_Article_45464)。
> * 此 HO-PBA 二聚體在室溫下接觸水後，會迅速轉化為三聚體硼氧環結構 (41467_2024_Article_45464)。
> * 在 THF-d$_8$ 溶液中，二聚體向三聚體的轉化透過 $^1\text{H NMR}$ 確認，表現為二聚體芳香質子訊號的消失以及三聚體芳香質子訊號的出現 (41467_2024_Article_45464)。
> * ESI-Q-TOF MS 測量結果表明，$\text{H}_2^{18}\text{O}$ 參與了從二聚體到三聚體的轉化過程 (41467_2024_Article_45464)。
> * 由 ESI-Q-TOF MS 和 UV-Raman 測量確認，所得的三聚體在環境條件下可穩定維持 7 天 (41467_2024_Article_45464)。
> * HO-PBA 三聚體的 pH 穩定範圍為 $2.4 < \text{pH} < 9$ (41467_2024_Article_45464)。
> * $\text{CH}_3\text{-HO-PBA}$ 三聚體的 pH 穩定範圍為 $2.4 < \text{pH} < 10$ (41467_2024_Article_45464)。
>
> **動態共價鍵與氟化物結合**
> * 此水穩定硼氧環結構的 B–O 鍵在室溫的水相中會發生快速交換 (41467_2024_Article_4546 …（完整內容見 JSON）

---
