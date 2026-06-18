# Eval Report — `vramtest2`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-06-18 01:54
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均 grounding 分數 | 0.5 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 2300.6s |
| 平均 planning 延遲 | 7.9s |
| 平均 retrieval 延遲 | 231.0s |
| 平均 grounding 延遲 | 820.1s |
| └ 其中 NLI | 13.2s |
| └ 其中 gemma4 | 780.6s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q01 | single_paper | 100.0% | 100.0% | 0.5 | 2300.6s | C0/U5 |

## 逐題細節

### ⚠️ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.5
- 延遲：2300.6s　問題標記：{'conflicts': 0, 'unsupported': 5}

**答案預覽**：

> ## 【論文直接依據】
> * 預孵育（Preincubation）在與共同孵育（co-incubation）結合時，能顯著增強 JPH203 的抑制效力 【1-s2.0-S1347861320300633-main】。
> * 在僅進行共同孵育分析時，$\text{IC}_{50}$ 值為 $99.2 \pm 11.0\text{ nM}$ 【1-s2.0-S1347861320300633-main】。
> * 當加入預孵育（預加共同孵育，pre-plus-co-incubation）時，$\text{IC}_{50}$ 值降低至 $34.2 \pm 3.6\text{ nM}$，這代表由於協同效應而產生了三倍的降低 【1-s2.0-S1347861320300633-main】。
> * 在共同孵育分析中，$K_i$ 值為 $0.46\text{ mM}$ 【1-s2.0-S1347861320300633-main】。
> * 在預加共同孵育分析中，$K_i$ 值降低至 $0.37\text{ mM}$，這是由於協同效應導致的 20% 降低 【1-s2.0-S1347861320300633-main】。
> * 僅進行預孵育（預孵育對 HT-29 細胞中 LAT1 功能的抑制效果）的 $\text{IC}_{50}$ 值為 $193 \pm 50\text{ nM}$ 【1-s2.0-S1347861320300633-main】。
> 
> ## 【跨文獻推論】
> * 跨論文推論（基於 【1-s2.0-S1347861320300633-main】）：雖然僅進行預孵育的效力低於僅進行共同孵育（比較預孵育的 $\text{IC}_{50}$ 為 $193 \pm 50\text{ nM}$ 與共同孵育的 $99.2 \pm 11.0\text{ nM}$），但預孵育的過程「啟動」了系統或轉運蛋白， …（完整內容見 JSON）

---
