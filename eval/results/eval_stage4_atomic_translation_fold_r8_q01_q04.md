# Eval Report — `stage4_atomic_translation_fold_r8_q01_q04`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-21 01:53
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 1.0 |
| Correctness judge 覆蓋 | 2/2（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 0.75 |
| Translation judge 覆蓋 | 2/2（N/A 0） |
| 平均 grounding 分數 | 0.974 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 100.0% |
| 平均 Stage 2 evidence 覆蓋率 | 100.0% |
| 平均總延遲 | 604.2s |
| 平均 planning 延遲 | 6.7s |
| 平均 retrieval 延遲 | 4.8s |
| └ Phase A embed/vector/BM25 | 4.7s |
| └ Phase B 子答案生成 | 0.1s |
| 平均 grounding 延遲 | 16.8s |
| └ 其中 NLI | 2.6s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ✅ | Q01 | single_paper | 1.0 | 1.0 | 100.0% | 100.0% | 100.0% | 1.0 | 620.1s | C0/U0 |
| ⚠️ | Q04 | table_lookup | 1.0 | 0.5 | 100.0% | 100.0% | 100.0% | 0.947 | 588.2s | C0/U1 |

## 逐題細節

### ✅ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 5/5
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：1.0
- 延遲：620.1s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 使用 JPH203 進行 preincubation (預孵育) 能協同增強對 LAT1 功能的 co-incubation (共同孵育) 抑制效果 [Source: 1-s2.0-S1347861320300633-main]。與單獨進行 co-incubation (共同孵育) 相比，增加 preincubation (預孵育) 顯著增強了其抑制效能 [Source: 1-s2.0-S1347861320300633-main]。
>
> 報告的 JPH203 抑制效果之 IC50 值如下：
> *   在 HT-29 細胞中，對 LAT1 功能之 preincubation (預孵育) 抑制效果的 IC50 值測定為 193 ± 50 nM [Source: 1-s2.0-S1347861320300633-main]。
> *   單獨進行 co-incubation (共同孵育) 抑制的 IC50 值為 99.2 ± 11.0 nM [Source: 1-s2.0-S1347861320300633-main]。
> *   preincubation (預孵育) 與 co-incubation (共同孵育) 效果結合後的 IC50 值較低，為 34.2 ± 3.6 nM [Source: 1-s2.0-S1347861320300633-main]。
>
> 動力學分析 (kinetic analyses) 報告的 Ki 值如下：
> *   在 co-incubation (共同孵育) 抑制分析中的 Ki 值為 0.46 mM [Source: 1-s2.0-S1347861320300633-main]。
> *   在 pre-plus co-incubation (預孵育加共同孵育) 抑制分析中的 Ki 值為 0.37 mM，低於 co-incubation (共同孵育) 抑制分析中的 …（完整內容見 JSON）

---

### ⚠️ Q04 · table_lookup

**問題**：What are the main degradation products and impurities of BPA identified by HPLC, and under which storage conditions do they form?

- detected_paper：`None`
- 選出論文：['1-s2.0-S0731708511003451-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water']
- gold_papers：['1-s2.0-S0731708511003451-main']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/7
- translation fidelity：0.5　raw：3/5　judge：`translation_fidelity_v2`
- translation reason：1 material and 0 minor semantic errors; The source lists four distinct degradation conditions: NaOH, HCl, FeCl3 (all incubated at 55°C for 24h), and a separate condition of an H2O2 solution. The target sentence incorrectly groups the H2O2 solution into the list of reagents that were all 'incubated at 55 °C for 24 h', implying it underwent
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：0.947
- 延遲：588.2s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> ### 透過 HPLC 鑑定之雜質與降解產物
>
> Boron phenylalanine (BPA) 的定量分析在 230 nm 下進行，保留時間為 5.3 min [1-s2.0-S0731708511003451-main]。Tyrosine (酪胺酸) 被鑑定為雜質與降解產物，在 230 nm 下定量，保留時間為 4.5 min [1-s2.0-S0731708511003451-main]。Phenylalanine (苯丙胺酸) 在 11.0 min 洗脫，並在 256 nm 下進行定量 [1-s2.0-S0731708511003451-main]。先前研究者已將 Tyrosine 與 Phenylalanine 報告為雜質 [1-s2.0-S0731708511003451-main]。
>
> 三種合成雜質被鑑定以供檢測：BrPD (2-(4-Bromophenyl)-[1,3]dioxane) 與 FBBA (4-Formylbenzeneboronic acid) 在 256 nm 下被檢測到，保留時間分別為 17.3 min 與 23.7 min [1-s2.0-S0731708511003451-main]。合成雜質 BDPA (2-tert-Butoxycarbonylamino-3-[4-(5,5-dimethyl-[1,3,2]dioxaborinan-2-yl)-phenyl]acrylic acid methyl ester) 在 270 nm 下被檢測到，且與 FBBA 共洗脫 (co-elutes) [1-s2.0-S0731708511003451-main]。這三種合成雜質在濃度為 0.5 µg/ml（或 BPA NWC 的 0.1%）時，無論是在存在或不存在 BPA 的情況下均可檢測到 [1-s2.0-S0731708511003451- …（完整內容見 JSON）

---
