# Eval Report — `nli_cache_check`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-06-30 01:02
- 題數：3

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 1.0 |
| 平均 grounding 分數 | 0.665 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 1295.3s |
| 平均 planning 延遲 | 7.8s |
| 平均 retrieval 延遲 | 519.1s |
| 平均 grounding 延遲 | 37.0s |
| └ 其中 NLI | 19.1s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q05 | single_paper | 100.0% | 100.0% | 0.533 | 868.0s | C0/U10 |
| ✅ | Q07 | figure_dependent | 100.0% | 100.0% | 0.833 | 1759.7s | C0/U9 |
| ⚠️ | Q10 | contradiction_check | 100.0% | 100.0% | 0.63 | 1258.1s | C0/U14 |

## 逐題細節

### ⚠️ Q05 · single_paper

**問題**：What is the solvent-free green protocol for N-Boc protection of amines using picric acid as a catalyst, and what reaction conditions are used?

- detected_paper：`None`
- 選出論文：['1-s2.0-S2773223124000268-main']
- gold_papers：['1-s2.0-S2773223124000268-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.533
- 延遲：868.0s　問題標記：{'conflicts': 0, 'unsupported': 10}

**答案預覽**：

> ## 【論文直接依據】
> * 苦味酸被用作 Brønsted 酸催化劑 (1-s2.0-S2773223124000268-main)。
> * 所使用的苦味酸催化劑量為 0.2 當量或 2 mol% (1-s2.0-S2773223124000268-main)。
> * 二叔丁基二碳酸酯 ($(Boc)_2O$ / Boc 酐) 被用作反應試劑 (1-s2.0-S2773223124000268-main)。
> * 反應在無溶劑（neat）條件下進行 (1-s2.0-S2773223124000268-main)。
> * 反應在室溫 (rt) 下進行 (1-s2.0-S2773223124000268-main)。
> * 反應時間為 60 分鐘 (1-s2.0-S2773223124000268-main)。
> * 此方法達到的最高產率為 98% (1-s2.0-S2773223124000268-main)。
> * 在優化條件下，芳香族一級胺的產率為 95% (1-s2.0-S2773223124000268-main)。
> * 適用底物包括芳香族胺 (1-s2.0-S2773223124000268-main)。
> * 適用底物包括雜環芳香族胺 (1-s2.0-S2773223124000268-main)。
> * 適用底物包括開鏈脂肪族胺 (1-s2.0-S2773223124000268-main)。
> * 適用底物包括環狀脂肪族胺 (1-s2.0-S2773223124000268-main)。
> * 在反應機制中，苦味酸首先活化 Boc 酐的羰基氧原子 (1-s2.0-S2773223124000268-main)。
> * 隨後，胺的氮原子提供一對孤對電子攻擊 Boc 酐的羰基碳 (1-s2.0-S2773223124000268-main)。
> * 最後，通過質子取代以及消除二氧化碳 ($ …（完整內容見 JSON）

---

### ✅ Q07 · figure_dependent

**問題**：What is the water-stable boroxine structure reported, and what role do the dynamic covalent bonds play in its fluoride binding and hydrogel formation?

- detected_paper：`None`
- 選出論文：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- gold_papers：['41467_2024_Article_45464']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.833
- 延遲：1759.7s　問題標記：{'conflicts': 0, 'unsupported': 9}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
> 
> ## 【論文直接依據】
> 
> **水穩定硼氧環結構及其合成**
> * 一種水穩定的硼氧環（boroxine）結構源自於 2-羥基苯硼酸 (HO-PBA) 【41467_2024_Article_45464】。
> * HO-PBA 在環境條件下會自發地進行脫水反應以形成二聚體 【41467_2024_Article_45464】。
> * 該二聚體在室溫下接觸水後，會迅速轉化為 HO-PBA 三聚體–$\text{H}_2\text{O}$ 複合物 【41467_2024_Article_45464】。
> * 此結構中的 B–O 鍵是水兼容的動態共價鍵 (DCBs) 【41467_2024_Article_45464】。
> * 這些 DCBs 在室溫的水相介質中可以進行快速交換 【41467_2024_Article_45464】。
> * 該硼氧環結構在水中是穩定的 【41467_2024_Article_45464】。
> * 該結構在 $\text{pH}=2$ 的酸性溶液中保持穩定 【41467_2024_Article_45464】。
> * 該結構在 $\text{pH}=10$ 的鹼性溶液中保持穩定 【41467_2024_Article_45464】。
> 
> **氟離子結合特性**
> * HO-PBA 三聚體在水相介質中能選擇性地識別氟離子 ($\text{F}^-$) 【41467_2024_Article_45464】。
> * HO-PBA 三聚體對 $\text{F}^-$ 的結合親和力顯著強於苯硼酸 (PBA) 【41467_2024_Article_45464】。
> * 在向 HO-PBA 三聚體中加入 $\text{F}^-$ 後，$^1\text{H NMR}$ 譜圖顯示峰組數從四組增 …（完整內容見 JSON）

---

### ⚠️ Q10 · contradiction_check

**問題**：Do the papers agree on the hydrolytic stability of boroxine structures? Describe any agreements or differences between them.

- detected_paper：`None`
- 選出論文：['Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation', '41467_2024_Article_45464', 'a-method-for-the-deprotection-of-alkylpinacolyl-boronate-esters']
- gold_papers：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.63
- 延遲：1258.1s　問題標記：{'conflicts': 0, 'unsupported': 14}

**答案預覽**：

> ## 【論文直接依據】
> 
> **關於傳統硼氧環（boroxines）及其通用穩定化策略：**
> * 傳統硼氧環在接觸水或潮濕條件時會迅速水解為硼酸，導致功能喪失 (【41467_2024_Article_45464】)。
> * 提高傳統硼氧環穩定性的策略包括引入電子捐贈基團 (【41467_2024_Article_45464】)。
> * 通過添加龐大基團（bulky groups）可以提高穩定性 (【41467_2024_Article_45464】)。
> * 通過與 N-供體配體形成加合物可以提高穩定性 (【41467_2024_Article_45464】)。
> * 可以通過熵穩定化策略（將三個硼酸單元整合到一個柔性大環中）來提高穩定性 (【41467_2024_Article_45464】)。
> 
> **關於由 2-羥基苯硼酸 (HO-PBA) 衍生的硼氧環：**
> * 已發現一種源自 2-羥基苯硼酸 (HO-PBA) 的水穩定硼氧環結構 (【41467_2024_Article_45464】)。
> * HO-PBA 在室溫下會自發脫水形成二聚體，其自由能為 –21.4 kJ mol⁻¹ (【41467_2024_Article_45464】)。
> * 在室溫下接觸水時，HO-PBA 二聚體會迅速轉化為水穩定的 HO-PBA 三聚體–$\text{H}_2\text{O}$ 複合物 (【41467_2024_Article_45464】)。
> * HO-PBA 的硼氧環結構在寬 pH 範圍內保持穩定 (【41467_2024_Article_45464】)。
> * HO-PBA 硼氧環中的 B–O 鍵是水兼容的動態共價鍵 (DCBs)，在室溫水相介質中可快速交換 (【41467_2024_Article_45464】)。
> * HO-PBA 三聚體的穩定範圍為 $2.4 < \text{pH} …（完整內容見 JSON）

---
