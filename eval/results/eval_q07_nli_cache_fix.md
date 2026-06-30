# Eval Report — `q07_nli_cache_fix`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-06-29 21:48
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 1.0 |
| 平均 grounding 分數 | 0.773 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 1005.7s |
| 平均 planning 延遲 | 7.6s |
| 平均 retrieval 延遲 | 385.2s |
| 平均 grounding 延遲 | 36.2s |
| └ 其中 NLI | 16.1s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q07 | figure_dependent | 100.0% | 100.0% | 0.773 | 1005.7s | C0/U7 |

## 逐題細節

### ⚠️ Q07 · figure_dependent

**問題**：What is the water-stable boroxine structure reported, and what role do the dynamic covalent bonds play in its fluoride binding and hydrogel formation?

- detected_paper：`None`
- 選出論文：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- gold_papers：['41467_2024_Article_45464']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.773
- 延遲：1005.7s　問題標記：{'conflicts': 0, 'unsupported': 7}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
> 
> ## 【論文直接依據】
> 
> **水穩定硼氧環結構**
> * 所報導的水穩定硼氧環結構是一種 HO-PBA 三聚體–$\text{H}_2\text{O}$ 錯合物 (41467_2024_Article_45464)。
> * 2-羥基苯硼酸 (HO-PBA) 在環境條件下會自發脫水形成二聚體 (41467_2024_Article_45464)。
> * HO-PBA 在室溫下接觸水後，會迅速轉化為水穩定硼氧環結構 (HO-PBA 三聚體–$\text{H}_2\text{O}$ 錯合物) (41467_2024_Article_45464)。
> * 此硼氧環結構在水中具有穩定性 (41467_2024_Article_45464)。
> * 此硼氧環結構表現出優異的 pH 穩定性 (41467_2024_Article_45464)。
> 
> **動態共價鍵 (DCBs) 的作用**
> * 硼氧環結構中的 B–O 鍵是水兼容的動態共價鍵 (DCBs) (41467_2024_Article_45464)。
> * 這些 DCBs 在室溫的水相介質中可以進行快速交換 (41467_2024_Article_45464)。
> 
> **氟離子結合**
> * 硼氧環結構在水相介質中對氟離子 ($\text{F}^-$) 具有選擇性識別和強結合能力 (41467_2024_Article_45464)。
> * 硼氧環結構對 $\text{F}^-$ 的結合親和力顯著高於廣泛使用的苯硼酸 (PBA) 氟受體 (41467_2024_Article_45464)。
> * 硼氧環結構作為陰離子受體的有效性歸因於其獨特的環狀結構和缺電子特性 (41467_2024_Article_45464)。
> * 在室溫下，使用 $\text …（完整內容見 JSON）

---
