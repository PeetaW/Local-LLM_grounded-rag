# Eval Report — `stage2_r2_evaluator_contract_rejudge`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-19 22:30
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.5 |
| Correctness judge 覆蓋 | 2/2（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 0.75 |
| Translation judge 覆蓋 | 2/2（N/A 0） |
| 平均 grounding 分數 | 0.778 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 100.0% |
| 平均 Stage 2 evidence 覆蓋率 | 30.0% |
| 平均總延遲 | 672.9s |
| 平均 planning 延遲 | 6.6s |
| 平均 retrieval 延遲 | 4.8s |
| └ Phase A embed/vector/BM25 | 4.8s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 17.9s |
| └ 其中 NLI | 2.5s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ❌ | Q01 | single_paper | 0.5 | 1.0 | 100.0% | 100.0% | 0.0% | 0.667 | 594.3s | C0/U2 |
| ⚠️ | Q04 | table_lookup | 0.5 | 0.5 | 100.0% | 100.0% | 60.0% | 0.889 | 751.5s | C0/U1 |

## 逐題細節

### ❌ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- correctness candidate：`answer_for_judge`
- correctness：0.5　raw：3/5　judge：`structured_fact_audit_v1`
- judge reason：covered 3/5; missing F1, F2
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：0.0%　grounding：0.667
- 延遲：594.3s　問題標記：{'conflicts': 0, 'unsupported': 2}

**答案預覽**：

> 在關於 JPH203 對 LAT1 的預孵育 (preincubation) 抑制效果的研究中，與單獨的同時孵育 (co-incubation) 相比，預孵育 (preincubation) 顯著增強了抑制效能 【1-s2.0-S1347861320300633-main】。JPH203 與 LAT1 交互作用的協同效應程度，其特徵為 IC50 值降低三倍以及 Ki 值減少 20% 【1-s2.0-S1347861320300633-main】。
>
> 關於在 HT-29 cells 中報告的具體數值，JPH203 在同時孵育抑制分析 (co-incubation inhibition assays) 中的 IC50 值為 $99.2 \pm 11.0\text{ nM}$ 【1-s2.0-S1347861320300633-main】。在 HT-29 cells 中，加入預孵育 (preincubation) 顯著增強了抑制效能，使其 IC50 達到 $34.2 \pm 3.6\text{ nM}$ 【1-s2.0-S1347861320300633-main】。此外，在預孵育加同時孵育抑制分析 (pre-plus co-incubation inhibition assays) 中的 Ki 值為 $0.37\text{ mM}$，低於在同時孵育抑制分析 (co-incubation inhibition assays) 中觀察到的 $0.46\text{ mM}$ 【1-s2.0-S1347861320300633-main】。預孵育與同時孵育組合 (pre- and co-incubation combination) 的 IC50 值遠低於單獨預孵育 (preincubation alone)，表明聯合暴露可增強效能 【1-s2.0-S1347861320300633- …（完整內容見 JSON）

---

### ⚠️ Q04 · table_lookup

**問題**：What are the main degradation products and impurities of BPA identified by HPLC, and under which storage conditions do they form?

- detected_paper：`None`
- 選出論文：['1-s2.0-S0731708511003451-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water']
- gold_papers：['1-s2.0-S0731708511003451-main']
- correctness candidate：`answer_for_judge`
- correctness：0.5　raw：3/5　judge：`structured_fact_audit_v1`
- judge reason：covered 4/7; missing F2, F3, F4
- translation fidelity：0.5　raw：3/5　judge：`translation_fidelity_v2`
- translation reason：1 material and 0 minor semantic errors; The source states that tyrosine and phenylalanine were reported as impurities OF BPA (meaning they are contaminants found within the substance). The translation '將...報告為BPA的雜質' implies these amino acids were identified AS being the specific chemical entity of BPA itself, rather than foreign substanc
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：60.0%　grounding：0.889
- 延遲：751.5s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
>
> 根據來自單一來源 【1-s2.0-S0731708511003451-main】 的提供數據，以下資訊探討了 boron phenylalanine (BPA) 的降解產物、雜質以及儲存穩定性。
>
> **已鑑定之雜質與合成中間體**
> 先前的研究人員將 tyrosine (酪胺酸) 與 phenylalanine (苯丙氨酸) 報告為 BPA 的雜質 【1-s2.0-S0731708511003451-main】。在 HPLC 分析中，這些氨基酸可與主化合物區分；tyrosine 在 230 nm 下定量且保留時間 (retention time) 為 4.5 min，而 phenylalanine 則在 11.0 min 洗脫並在 256 nm 下定量 【1-s2.0-S0731708511003451-main】。合成中間體如 (2-(4-Bromophenyl)-[1,3]dioxane (BrPD)、4-Formylbenzeneboronic acid (FBBA) 以及 BDPA 亦針對 HPLC 檢測進行了表徵 【1-s2.0-S0731708511003451-main】。然而，在現有的 BPA 樣本中未觀察到 BrPD 或 FBBA/BDPA 的組合 【1-s2.0-S0731708511003451-main】。
>
> **儲存條件與穩定性**
> 關於可能產生降解產物的儲存條件，BPA 作為原粉 (raw powder) 時，在 55 °C 下儲存 6 months 或在 40 °C 下儲存 12 months 均未產生可檢測到的降解 【1-s2.0-S0731708511003451-main】。此外，觀察到 BPA 在酸性溶液與 FeCl3 溶液中具有穩定性，此 …（完整內容見 JSON）

---
