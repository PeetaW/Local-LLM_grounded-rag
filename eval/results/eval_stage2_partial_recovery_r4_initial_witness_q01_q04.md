# Eval Report — `stage2_partial_recovery_r4_initial_witness_q01_q04`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-20 08:21
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.5 |
| Correctness judge 覆蓋 | 2/2（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 0.375 |
| Translation judge 覆蓋 | 2/2（N/A 0） |
| 平均 grounding 分數 | 0.875 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 100.0% |
| 平均 Stage 2 evidence 覆蓋率 | 30.0% |
| 平均總延遲 | 15406.1s |
| 平均 planning 延遲 | 6.8s |
| 平均 retrieval 延遲 | 4.8s |
| └ Phase A embed/vector/BM25 | 4.8s |
| └ Phase B 子答案生成 | 0.1s |
| 平均 grounding 延遲 | 19.0s |
| └ 其中 NLI | 1.8s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ❌ | Q01 | single_paper | 0.5 | 0.5 | 100.0% | 100.0% | 0.0% | 0.75 | 973.0s | C0/U2 |
| ❌ | Q04 | table_lookup | 0.5 | 0.25 | 100.0% | 100.0% | 60.0% | 1.0 | 29839.3s | C0/U0 |

## 逐題細節

### ❌ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- correctness candidate：`answer_for_judge`
- correctness：0.5　raw：3/5　judge：`structured_fact_audit_v1`
- judge reason：covered 3/5; missing F1; contradicted F2
- translation fidelity：0.5　raw：3/5　judge：`translation_fidelity_v2`
- translation reason：1 material and 0 minor semantic errors; The source states a 'three-fold decrease' (meaning the value became 1/3 of original), but the target translates this as '降低了三倍', which in Chinese typically implies reducing by three times (resulting in negative or zero) rather than becoming one-third. The correct phrasing for a reduction to one-thir
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：0.0%　grounding：0.75
- 延遲：973.0s　問題標記：{'conflicts': 0, 'unsupported': 2}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
>
> 在 JPH203 對 LAT1 的預孵育 (preincubation) 抑制效果研究中，與單獨進行同時孵育 (co-incubation) 相比，預孵育顯著增強了抑制效能 【1-s2.0-S1347861320300633-main】。在 HT-29 cells 中，此變化的報告數值如下：
>
> *   **$\text{IC}_{50}$ 值：** 在同時孵育 (co-incubation) 抑制分析中，$\text{IC}_{50}$ 值為 $99.2 \pm 11.0\text{ nM}$ 【1-s2.0-S1347861320300633-main】。加入預孵育 (preincubation) 後，抑制效能增加，$\text{IC}_{50}$ 變為 $34.2 \pm 3.6\text{ nM}$ 【1-s2.0-S1347861320300633-main】。這代表 $\text{IC}_{50}$ 值降低了三倍，顯示效能有所增強 【1-s2.0-S1347861320300633-main】。
> *   **$K_i$ 值：** 在預孵育加同時孵育 (pre-plus co-incubation) 抑制分析中，報告的 $K_i$ 值為 $0.37\text{ mM}$ 【1-s2.0-S1347861320300633-main】。這低於單獨在同時孵育 (co-incubation) 抑制分析中觀察到的 $0.46\text{ mM}$ 【1-s2.0-S1347861320300633-main】，對應於 $K_i$ 值降低了 20% 【1-s2.0-S1347861320300633-main】。
>
> 這些協同抑制效果被描述為處於輕微至中度水平 【1-s2.0-S …（完整內容見 JSON）

---

### ❌ Q04 · table_lookup

**問題**：What are the main degradation products and impurities of BPA identified by HPLC, and under which storage conditions do they form?

- detected_paper：`None`
- 選出論文：['1-s2.0-S0731708511003451-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water']
- gold_papers：['1-s2.0-S0731708511003451-main']
- correctness candidate：`answer_for_judge`
- correctness：0.5　raw：3/5　judge：`structured_fact_audit_v1`
- judge reason：covered 5/7; missing F2, F4
- translation fidelity：0.25　raw：2/5　judge：`translation_fidelity_v2`
- translation reason：2 material and 0 minor semantic errors; The source states the pH was adjusted to 8.0 ± 0.1 in WFI before freeze drying, implying the solution containing BPA and mannitol had its pH adjusted. The target translation incorrectly structures this as adjusting the pH of 'WFI' (injectable water) itself after mixing with Mannitol but ambiguously
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：60.0%　grounding：1.0
- 延遲：29839.3s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ### 以 HPLC 鑑定之雜質與降解產物
>
> BPA 的合成雜質包括 2-(4-Bromophenyl)-[1,3]dioxane (BrPD)、4-Formylbenzeneboronic acid (FBBA) 以及 2-tert-Butoxycarbonylamino-3-[4-(5,5-dimethyl-[1,3,2]dioxaborinan-2-yl)-phenyl]acrylic acid methyl ester (BDPA) 【Source: 1-s2.0-S0731708511003451-main】。先前研究者曾報導 Tyrosine (酪胺酸) 與 Phenylalanine (苯丙胺酸) 為雜質 【Source: 1-s2.0-S0731708511003451-main】。
>
> 關於 HPLC (高效液相層析) 鑑定，BPA 與 Tyrosine (酪胺酸) 在 230 nm 下進行定量，其保留時間 (retention times) 分別為 5.3 與 4.5 min 【Source: 1-s2.0-S0731708511003451-main】。Phenylalanine (苯丙胺酸) 在 11.0 min 洗脫 (elutes)，並在 256 nm 下進行定量 【Source: 1-s2.0-S0731708511003451-main】。BrPD 與 FBBA 在 256 nm 下被檢出，且分別在 17.3 與 23.7 min 洗脫 (elute) 【Source: 1-s2.0-S0731708511003451-main】。BDPA 在 270 nm 下被檢出，但與 FBBA 共洗脫 (co-elutes) 【Source: 1-s2.0-S0731708511003451-main】。
>
> ### 儲存條件與雜質的形成
>
> 作為原藥粉末， …（完整內容見 JSON）

---
