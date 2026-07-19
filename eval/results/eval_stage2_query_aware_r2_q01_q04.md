# Eval Report — `stage2_query_aware_r2_q01_q04`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-18 23:28
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.5 |
| Correctness judge 覆蓋 | 2/2（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 0.375 |
| Translation judge 覆蓋 | 2/2（N/A 0） |
| 平均 grounding 分數 | 0.959 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 100.0% |
| 平均 Stage 2 evidence 覆蓋率 | 30.0% |
| 平均總延遲 | 568.1s |
| 平均 planning 延遲 | 6.9s |
| 平均 retrieval 延遲 | 4.9s |
| └ Phase A embed/vector/BM25 | 4.8s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 18.6s |
| └ 其中 NLI | 2.2s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ❌ | Q01 | single_paper | 0.5 | 0.5 | 100.0% | 100.0% | 0.0% | 1.0 | 545.5s | C0/U0 |
| ❌ | Q04 | table_lookup | 0.5 | 0.25 | 100.0% | 100.0% | 60.0% | 0.917 | 590.6s | C0/U1 |

## 逐題細節

### ❌ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- correctness candidate：`answer_for_judge`
- correctness：0.5　raw：3/5　judge：`structured_fact_audit_v1`
- judge reason：covered 4/5; contradicted F2
- translation fidelity：0.5　raw：3/5　judge：`translation_fidelity_v2`
- translation reason：1 material and 0 minor semantic errors; The source states a 'three-fold ... decrease' (implying the value becomes 1/3 of original), but the target translates this as '降低三倍', which in Chinese typically implies reducing by three times (resulting in negative or zero) rather than becoming one-third. The correct translation for a fold-decrease
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：0.0%　grounding：1.0
- 延遲：545.5s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 在 HT-29 細胞中，與單獨共同孵育 (co-incubation) 相比，增加預孵育 (preincubation) 顯著增強了 JPH203 對 LAT1 的抑制效能 (inhibitory potency) 【1-s2.0-S1347861320300633-main】。共同孵育抑制 (co-incubation inhibition) 的 $\text{IC}_{50}$ 值为 $99.2 \pm 11.0\text{ nM}$，而預孵育 (preincubation) 將 $\text{IC}_{50}$ 降低至 $34.2 \pm 3.6\text{ nM}$ 【1-s2.0-S1347861320300633-main】。關於抑制常數 (inhibition constants, $K_i$)，預孵育加共同孵育分析 (pre-plus co-incubation assays) 得到的數值為 $0.37\text{ mM}$，低於在共同孵育抑制分析 (co-incubation inhibition assays) 中觀察到的 $0.46\text{ mM}$ 【1-s2.0-S1347861320300633-main】。此交互作用 (interaction) 的協同效應 (cooperative effects) 程度特徵為 $\text{IC}_{50}$ 值降低三倍，且 $K_i$ 值降低 20% 【1-s2.0-S1347861320300633-main】。
>
> ---
> 📋 **答案品質報告**
>
> ✅ **整體論文依據率**：100.0%　高（答案高度忠實於論文內容）
>
> ✅ **所有陳述均有論文依據**
>
>
> <!-- grounding_score=1.000 -->
> ---

---

### ❌ Q04 · table_lookup

**問題**：What are the main degradation products and impurities of BPA identified by HPLC, and under which storage conditions do they form?

- detected_paper：`None`
- 選出論文：['1-s2.0-S0731708511003451-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water']
- gold_papers：['1-s2.0-S0731708511003451-main']
- correctness candidate：`answer_for_judge`
- correctness：0.5　raw：3/5　judge：`structured_fact_audit_v1`
- judge reason：covered 4/7; missing F2, F4, F7
- translation fidelity：0.25　raw：2/5　judge：`translation_fidelity_v2`
- translation reason：2 material and 0 minor semantic errors; Source S1 (Title: Impurities and Degradation Products Identified by HPLC) is completely missing from the target sentences.
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：60.0%　grounding：0.917
- 延遲：590.6s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
>
> **透過 HPLC 鑑定之雜質與降解產物**
>
> Tyrosine 與 phenylalanine 已被報導為 boron phenylalanine (BPA) 的雜質 [1-s2.0-S0731708511003451-main]。合成雜質或中間體包括 2-(4-Bromophenyl)-[1,3]dioxane (BrPD)、4-Formylbenzeneboronic acid (FBBA) 以及 2-tert-Butoxycarbonylamino- 3-[4-(5,5-dimethyl-[1,3,2]dioxaborinan-2-yl)-phenyl]acrylic acid methyl ester (BDPA) [1-s2.0-S0731708511003451-main]。Tyrosine 被明確鑑定為 BPA 的氧化降解產物 (oxidative degradation product) [1-s2.0-S0731708511003451-main]。在 HPLC 分析中，這些化合物透過保留時間 (retention times) 與定量波長 (quantification wavelengths) 進行區分：BPA 為 230 nm (5.3 min)、tyrosine 為 230 nm (4.5 min)、phenylalanine 為 256 nm (11.0 min) [1-s2.0-S0731708511003451-main]。BrPD 在 256 nm 處被檢出且於 17.3 min 洗脫 (elutes)，而 FBBA 在 256 nm 處被檢出且於 23.7 min 洗脫 [1-s2.0-S0731708511003451-main]。BD …（完整內容見 JSON）

---
