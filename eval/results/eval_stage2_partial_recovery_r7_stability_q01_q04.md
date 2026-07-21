# Eval Report — `stage2_partial_recovery_r7_stability_q01_q04`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-21 00:41
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 1.0 |
| Correctness judge 覆蓋 | 2/2（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 0.5 |
| Translation judge 覆蓋 | 2/2（N/A 0） |
| 平均 grounding 分數 | 1.0 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 100.0% |
| 平均 Stage 2 evidence 覆蓋率 | 100.0% |
| 平均總延遲 | 695.2s |
| 平均 planning 延遲 | 6.6s |
| 平均 retrieval 延遲 | 4.8s |
| └ Phase A embed/vector/BM25 | 4.8s |
| └ Phase B 子答案生成 | 0.1s |
| 平均 grounding 延遲 | 18.0s |
| └ 其中 NLI | 1.2s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ⚠️ | Q01 | single_paper | 1.0 | 0.5 | 100.0% | 100.0% | 100.0% | 1.0 | 744.8s | C0/U0 |
| ⚠️ | Q04 | table_lookup | 1.0 | 0.5 | 100.0% | 100.0% | 100.0% | 1.0 | 645.6s | C0/U0 |

## 逐題細節

### ⚠️ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 5/5
- translation fidelity：0.5　raw：3/5　judge：`translation_fidelity_v2`
- translation reason：1 material and 0 minor semantic errors; The source states a 'three-fold IC50 value decrease' (meaning the new value is 1/3 of the original), but the target translates this as 'IC50 values decreased by three times', which in Chinese scientific context typically implies multiplication or an incorrect magnitude, failing to convey the specifi
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：1.0
- 延遲：744.8s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 預孵育 (preincubation) 協同增強了 JPH203 對 LAT1 功能的共孵育抑制效果 (co-incubation inhibitory effects) 【1-s2.0-S1347861320300633-main】。在 HT-29 細胞中，單獨共孵育抑制的 IC50 值為 $99.2 \pm 11.0$ nM 【1-s2.0-S1347861320300633-main】。預孵育與共孵育效果的結合顯著增強了抑制效能 (inhibitory potency)，導致較低的 IC50 值 $34.2 \pm 3.6$ nM 【1-s2.0-S1347861320300633-main】。這種程度的協同效應 (cooperative effect) 的特徵是 IC50 值降低了三倍 【1-s2.0-S1347861320300633-main】。
>
> 關於動力學參數 (kinetic parameters)，預孵育加共孵育抑制分析 (pre-plus co-incubation inhibition assays) 中的 Ki 值為 0.37 mM，低於在共孵育抑制分析中觀察到的 Ki 值 (0.46 mM) 【1-s2.0-S1347861320300633-main】。這對應於協同效應導致的 Ki 值降低 20% 【1-s2.0-S1347861320300633-main】。此外，在 HT-29 細胞中，單獨預孵育 (preincubation) 的 IC50 值被測定為 $193 \pm 50$ nM 【1-s2.0-S1347861320300633-main】。
>
> - 這些結果表明，JPH203 以濃度與時間依賴的方式 (concentration- and time-dependent manner) 對 HT-29 細胞中的 LAT1 功能產生預孵 …（完整內容見 JSON）

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
- translation reason：1 material and 0 minor semantic errors; The source states BPA is stable in acidic and FeCl3 solutions, including forced degradation tests using NaOH (a base), HCl (an acid) or FeCl3. The translation incorrectly lists the test conditions as 'using 100 mM NaOH... performed' under the clause describing stability in acidic/FeCl3 solutions, im
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：1.0
- 延遲：645.6s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ### 透過 HPLC 鑑定之雜質 (Impurities Identified by HPLC)
>
> Tyrosine (酪胺酸) 與 phenylalanine (苯丙胺酸) 已被報導為 boron phenylalanine (BPA, 硼苯丙氨酸) 的雜質 [1-s2.0-S0731708511003451-main]。合成中間體 (synthetic intermediates) 包括 2-(4-Bromophenyl)-[1,3]dioxane (BrPD)、4-Formylbenzeneboronic acid (FBBA) 以及 2-tert-Butoxycarbonylamino-3-[4-(5,5-dimethyl-[1,3,2]dioxaborinan-2-yl)-phenyl]acrylic acid methyl ester (BDPA) [1-s2.0-S0731708511003451-main]。然而，在現有的 BPA 樣品中未觀察到 BrPD 或 FBBA/BDPA 組合 [1-s2.0-S0731708511003451-main]。
>
> 在 HPLC 分析 (HPLC analysis) 中，這些化合物的特定檢測參數如下：
> *   **BPA 與 Tyrosine：** 於 230 nm 定量，保留時間 (retention times) 分別為 5.3 min 與 4.5 min [1-s2.0-S0731708511003451-main]。
> *   **Phenylalanine：** 於 256 nm 定量，保留時間為 11.0 min [1-s2.0-S0731708511003451-main]。
> *   **BrPD 與 FBBA：** 於 256 nm 檢測，保留時間分別為 17.3 min 與 23.7 min [1-s2 …（完整內容見 JSON）

---
