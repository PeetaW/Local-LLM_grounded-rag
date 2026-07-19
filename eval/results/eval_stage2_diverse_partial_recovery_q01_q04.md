# Eval Report — `stage2_diverse_partial_recovery_q01_q04`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-19 00:24
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.625 |
| Correctness judge 覆蓋 | 2/2（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 0.5 |
| Translation judge 覆蓋 | 2/2（N/A 0） |
| 平均 grounding 分數 | 0.833 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 100.0% |
| 平均 Stage 2 evidence 覆蓋率 | 30.0% |
| 平均總延遲 | 729.5s |
| 平均 planning 延遲 | 6.8s |
| 平均 retrieval 延遲 | 4.8s |
| └ Phase A embed/vector/BM25 | 4.8s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 16.2s |
| └ 其中 NLI | 2.2s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ❌ | Q01 | single_paper | 0.75 | 0.5 | 100.0% | 100.0% | 0.0% | 0.833 | 604.4s | C0/U1 |
| ⚠️ | Q04 | table_lookup | 0.5 | 0.5 | 100.0% | 100.0% | 60.0% | 0.833 | 854.5s | C0/U2 |

## 逐題細節

### ❌ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- correctness candidate：`answer_for_judge`
- correctness：0.75　raw：4/5　judge：`structured_fact_audit_v1`
- judge reason：covered 4/5; missing F2
- translation fidelity：0.5　raw：3/5　judge：`translation_fidelity_v2`
- translation reason：1 material and 0 minor semantic errors; The source states a 'three-fold IC50 value decrease' (meaning the new value is 1/3 of the original), but the translation says '降低三倍' which in Chinese typically implies reducing by three times (resulting in negative or zero) or confusingly implying multiplication. The correct scientific phrasing for
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：0.0%　grounding：0.833
- 延遲：604.4s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> 在 JPH203 的研究中，預孵育 (preincubation) 顯著增強了其對 LAT1 的抑制效能 【1-s2.0-S1347861320300633-main】。在 HT-29 cells 中，同時孵育抑制分析 (co-incubation inhibition assays) 的 $\text{IC}_{50}$ 值為 $99.2 \pm 11.0\text{ nM}$ 【1-s2.0-S1347861320300633-main】。加入預孵育 (preincubation) 將此效能提升至 $\text{IC}_{50}$ 為 $34.2 \pm 3.6\text{ nM}$ 【1-s2.0-S1347861320300633-main】。在 LAT1 抑制分析 (LAT1 inhibition assays) 中，預孵育加同時孵育抑制分析 (pre-plus co-incubation inhibition assays) 的 $K_i$ 值為 $0.37\text{ mM}$ 【1-s2.0-S1347861320300633-main】。這低於在同時孵育抑制分析 (co-incubation inhibition assays) 中觀察到的 $0.46\text{ mM}$ 【1-s2.0-S1347861320300633-main】。此交互作用的協同效應程度對應於 $\text{IC}_{50}$ 值降低三倍以及 $K_i$ 值降低 20% 【1-s2.0-S1347861320300633-main】。
>
> ---
> 📋 **答案品質報告**
>
> ✅ **整體論文依據率**：83.3%　高（答案高度忠實於論文內容）
>
> ⚠️  **以下 1 個陳述未找到明確論文依據，請謹慎參考：**
>
> - In HT-29 cells, the $\text{IC}_{50 …（完整內容見 JSON）

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
- translation reason：1 material and 0 minor semantic errors; The English term 'phenylalanine' (苯丙胺酸 in the translation) is a specific amino acid. The Chinese word used, 苯丙胺酸，is chemically incorrect for phenylalanine; it refers to amphetamine or implies an error with tyramine/tyrosine naming conventions where 酪胺酸 was correctly used earlier in the sentence. Phe
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：60.0%　grounding：0.833
- 延遲：854.5s　問題標記：{'conflicts': 0, 'unsupported': 2}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
>
> **透過 HPLC 鑑定之雜質與降解產物**
>
> 之前的研究人員報導了酪胺酸 (tyrosine) 與苯丙胺酸 (phenylalanine) 為硼苯丙胺酸 (boron phenylalanine, BPA) 的雜質 [1-s2.0-S0731708511003451-main]。BPA 的合成雜質包括 2-(4-Bromophenyl)-[1,3]dioxane (BrPD)、4-Formylbenzeneboronic acid (FBBA) 以及 2-tert-Butoxycarbonylamino-3-[4-(5,5-dimethyl-[1,3,2]dioxaborinan-2-yl)-phenyl]acrylic acid methyl ester (BDPA) [1-s2.0-S0731708511003451-main]。然而，在現有的 BPA 樣本中未觀察到 BrPD 或 FBBA/BDPA 的組合 [1-s2.0-S0731708511003451-main]。BPA 的氧化降解 (oxidative degradation) 會產生酪胺酸 (tyrosine) [1-s2.0-S0731708511003451-main]。
>
> 在 HPLC 分析中，BPA 與酪胺酸 (tyrosine) 在 230 nm 處進行定量，其保留時間 (retention times) 分別為 5.3 與 4.5 min [1-s2.0-S0731708511003451-main]。苯丙胺酸 (phenylalanine) 在 11.0 min 洗脫 (elutes)，並在 256 nm 處進行定量 [1-s2.0-S0731708511003451-main]。BrPD 與 …（完整內容見 JSON）

---
