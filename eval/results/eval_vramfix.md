# Eval Report — `vramfix`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-06-18 21:00
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均 grounding 分數 | 0.8 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 2016.2s |
| 平均 planning 延遲 | 8.3s |
| 平均 retrieval 延遲 | 660.7s |
| 平均 grounding 延遲 | 127.0s |
| └ 其中 NLI | 5.1s |
| └ 其中 gemma4 | 96.9s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q01 | single_paper | 100.0% | 100.0% | 0.8 | 2016.2s | C0/U2 |

## 逐題細節

### ✅ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.8
- 延遲：2016.2s　問題標記：{'conflicts': 0, 'unsupported': 2}

**答案預覽**：

> ## 【論文直接依據】
> * 預孵育（Preincubation）在與共同孵育（co-incubation）結合時，能顯著增強 JPH203 的抑制效力 [1-s2.0-S1347861320300633-main]。
> * 報告的抑制效力數值如下：
>     * **僅共同孵育：** $\text{IC}_{50} = 99.2 \pm 11.0\text{ nM}$ 且 $\text{K}_i = 0.46\text{ mM}$ [1-s2.0-S1347861320300633-main]。
>     * **預孵育結合共同孵育：** $\text{IC}_{50} = 34.2 \pm 3.6\text{ nM}$ 且 $\text{K}_i = 0.37\text{ mM}$ [1-s2.0-S1347861320300633-main]。
>     * **僅預孵育：** $\text{IC}_{50} = 193 \pm 50\text{ nM}$ [1-s2.0-S1347861320300633-main]。
> * 預孵育與共同孵育之間的協同作用導致 $\text{IC}_{50}$ 值降低三倍，且 $\text{K}_i$ 值降低 20% [1-s2.0-S1347861320300633-main]。
> 
> ## 【跨文獻推論】
> * 跨論文推論（基於 [1-s2.0-S1347861320300633-main]）：雖然僅預孵育的效力低於僅共同孵育（比較 $\text{IC}_{50}$ 值 $193 \pm 50\text{ nM}$ 與 $99.2 \pm 11.0\text{ nM}$），但兩者的結合產生了協同效應，使 JPH203 的效力顯著高於單獨使用任何一種方法（將 $\text{IC}_{50}$ 進一步降低至 $34.2 \pm 3.6\text{ n …（完整內容見 JSON）

---
