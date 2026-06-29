# Eval Report — `q07_ligfix`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-06-29 01:14
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 1.0 |
| 平均 grounding 分數 | 0.65 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 1396.9s |
| 平均 planning 延遲 | 8.4s |
| 平均 retrieval 延遲 | 506.1s |
| 平均 grounding 延遲 | 54.4s |
| └ 其中 NLI | 28.8s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q07 | figure_dependent | 100.0% | 100.0% | 0.65 | 1396.9s | C0/U17 |

## 逐題細節

### ⚠️ Q07 · figure_dependent

**問題**：What is the water-stable boroxine structure reported, and what role do the dynamic covalent bonds play in its fluoride binding and hydrogel formation?

- detected_paper：`None`
- 選出論文：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- gold_papers：['41467_2024_Article_45464']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.65
- 延遲：1396.9s　問題標記：{'conflicts': 0, 'unsupported': 17}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
>
> ## 【論文直接依據】
>
> **結構與穩定性**
> * 報導的水穩定硼氧環 (boroxine) 結構衍生自 2-羥基苯硼酸 (HO-PBA) (41467_2024_Article_45464)。
> * 此硼氧環結構在 $\text{pH}=2$ 的酸性溶液中保持穩定 (41467_2024_Article_45464)。
> * 此硼氧環結構在 $\text{pH}=10$ 的鹼性溶液中保持穩定 (41467_2024_Article_45464)。
> * 該結構具有水兼容的動態共價鍵 (DCBs) (41467_2024_Article_45464)。
> * 這些 DCBs 在室溫水相介質中可以進行快速交換 (41467_2024_Article_45464)。
> * HO-PBA 在環境溫度下會自發地發生脫水反應以形成二聚體 (41467_2024_Article_45464)。
> * HO-PBA 二聚體內的 B–O 鍵為動態共價鍵 (41467_2024_Article_45464)。
> * 在室溫下接觸水時，HO-PBA 二聚體會迅速轉化為 $\text{HO-PBA}$ 三聚體-$\text{H}_2\text{O}$ 複合物 (41467_2024_Article_45464)。
> * 此 $\text{HO-PBA}$ 三聚體-$\text{H}_2\text{O}$ 複合物即為水穩定硼氧環結構 (41467_2024_Article_45464)。
> * 從二聚體到三聚體的轉化已使用 $^1\text{H NMR}$ 驗證 (41467_2024_Article_45464)。
> * 從二聚體到三聚體的轉化已使用 $\text{H}_2^{18}\text{O}$ 的 ESI-Q …（完整內容見 JSON）

---
