# Eval Report — `stage2_r3_numeric_contract_rejudge`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-19 23:31
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.625 |
| Correctness judge 覆蓋 | 2/2（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 0.5 |
| Translation judge 覆蓋 | 2/2（N/A 0） |
| 平均 grounding 分數 | 0.834 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 100.0% |
| 平均 Stage 2 evidence 覆蓋率 | 80.0% |
| 平均總延遲 | 702.4s |
| 平均 planning 延遲 | 6.6s |
| 平均 retrieval 延遲 | 4.8s |
| └ Phase A embed/vector/BM25 | 4.7s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 15.9s |
| └ 其中 NLI | 2.1s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ⚠️ | Q01 | single_paper | 0.75 | 0.5 | 100.0% | 100.0% | 100.0% | 1.0 | 798.5s | C0/U0 |
| ⚠️ | Q04 | table_lookup | 0.5 | 0.5 | 100.0% | 100.0% | 60.0% | 0.667 | 606.3s | C0/U4 |

## 逐題細節

### ⚠️ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- correctness candidate：`answer_for_judge`
- correctness：0.75　raw：4/5　judge：`structured_fact_audit_v1`
- judge reason：covered 4/5; missing F1
- translation fidelity：0.5　raw：3/5　judge：`translation_fidelity_v2`
- translation reason：1 material and 0 minor semantic errors; The source states the unique property is 'significant augmented effects to co-incubation inhibition potency' (meaning preincubation augments the potency of co-incubation), but the translation renders it as '能顯著增強共孵育的抑制效能' which implies preincubation enhances its own effect or creates a general enhan
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：1.0
- 延遲：798.5s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 在涉及 HT-29 細胞 (HT-29 cells) 的研究中，預孵育 (preincubation) 協同增強了 JPH203 對 LAT1 功能 (LAT1 function) 的共孵育抑制效果 (co-incubation inhibitory effects) 【1-s2.0-S1347861320300633-main】。在共孵育中加入預孵育導致 $\text{IC}_{50}$ 值降低了三倍，且 $K_i$ 值降低了 20% 【1-s2.0-S1347861320300633-main】。
>
> 關於抑制效能值 (inhibitory potency values)，JPH203 在共孵育抑制分析 (co-incubation inhibition assays) 中的 $\text{IC}_{50}$ 為 $99.2 \pm 11.0\text{ nM}$ 【1-s2.0-S1347861320300633-main】。當加入預孵育 (preincubation) 時，JPH203 在共孵育抑制分析中的 $\text{IC}_{50}$ 值降低至 $34.2 \pm 3.6\text{ nM}$ 【1-s2.0-S1347861320300633-main】。作為對比，單獨預孵育 (preincubation alone) 的 JPH203 $\text{IC}_{50}$ 值為 $193 \pm 50$ nM 【1-s2.0-S1347861320300633-main】，這表明預孵育與共孵育的結合所產生的 $\text{IC}_{50}$ 值遠低於單獨預孵育 【1-s2.0-S1347861320300633-main】。
>
> 在抑制常數 (inhibition constants) 方面，JPH203 共孵育抑制分析中的 $K_i$ 值為 0.46 mM 【1 …（完整內容見 JSON）

---

### ⚠️ Q04 · table_lookup

**問題**：What are the main degradation products and impurities of BPA identified by HPLC, and under which storage conditions do they form?

- detected_paper：`None`
- 選出論文：['1-s2.0-S0731708511003451-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water']
- gold_papers：['1-s2.0-S0731708511003451-main']
- correctness candidate：`answer_for_judge`
- correctness：0.5　raw：3/5　judge：`structured_fact_audit_v1`
- judge reason：covered 4/7; missing F2, F4, F7
- translation fidelity：0.5　raw：3/5　judge：`translation_fidelity_v2`
- translation reason：1 material and 0 minor semantic errors; The source sentence states 'Tyrosine and phenylalanine' (two compounds), but the target translation only lists 'phenylalanine', omitting tyrosine from the subject list.
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：60.0%　grounding：0.667
- 延遲：606.3s　問題標記：{'conflicts': 0, 'unsupported': 4}

**答案預覽**：

> **透過 HPLC 鑑定之雜質與降解產物**
>
> Tyrosine 與 phenylalanine 已被報導為 boron phenylalanine (BPA) 的雜質 [1-s2.0-S0731708511003451-main]。在本研究中，這些化合物使用特定波長的 HPLC 分析進行鑑定，以將其與 BPA 及合成中間體 (synthetic intermediates) 區分開來 [1-s2.0-S0731708511003451-main]。在 230 nm 檢測下，tyrosine 的保留時間 (retention time) 為 4.5 min，而 BPA 為 5.3 min [1-s2.0-S0731708511003451-main]。當在 256 nm 分析時，phenylalanine 在 11.0 min 洗脫 (elutes) [1-s2.0-S0731708511003451-main]。BPA 的氧化降解 (oxidative degradation) 會導致 tyrosine 的形成，證實其為氧化壓力下相關的降解產物 [1-s2.0-S0731708511003451-main]。
>
> 合成中間體 (synthetic intermediates)，例如 2-(4-Bromophenyl)-[1,3]dioxane (BrPD)、4-Formylbenzeneboronic acid (FBBA) 以及 2-tert-Butoxycarbonylamino-3-[4-(5,5-dimethyl-[1,3,2]dioxaborinan-2-yl)-phenyl]acrylic acid methyl ester (BDPA)，被評估為潛在雜質 [1-s2.0-S0731708511003451-main]。在 256 nm 的 HPLC 分析顯示  …（完整內容見 JSON）

---
