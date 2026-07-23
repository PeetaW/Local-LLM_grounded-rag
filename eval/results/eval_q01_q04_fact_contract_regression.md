# Eval Report — `q01_q04_fact_contract_regression`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-22 01:09
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.75 |
| Correctness judge 覆蓋 | 2/2（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 0.75 |
| Translation judge 覆蓋 | 2/2（N/A 0） |
| 平均 grounding 分數 | 1.0 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 100.0% |
| 平均 Stage 2 evidence 覆蓋率 | 100.0% |
| 平均總延遲 | 366.4s |
| 平均 planning 延遲 | 6.6s |
| 平均 retrieval 延遲 | 4.8s |
| └ Phase A embed/vector/BM25 | 4.7s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 14.6s |
| └ 其中 NLI | 0.9s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ⚠️ | Q01 | single_paper | 0.5 | 0.5 | 100.0% | 100.0% | 100.0% | 1.0 | 309.2s | C0/U0 |
| ✅ | Q04 | table_lookup | 1.0 | 1.0 | 100.0% | 100.0% | 100.0% | 1.0 | 423.7s | C0/U0 |

## 逐題細節

### ⚠️ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- correctness candidate：`answer_for_judge`
- correctness：0.5　raw：3/5　judge：`structured_fact_audit_v1`
- judge reason：covered 3/5; missing F3, F5
- translation fidelity：0.5　raw：3/5　judge：`translation_fidelity_v2`
- translation reason：1 material and 0 minor semantic errors; The target sentence omits the comparative value for 'co-incubation inhibition alone' which is present in the source (implied by 'vs [Source...]' indicating a missing comparison value that should follow). The translation cuts off mid-sentence without providing the necessary data point to complete the
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：1.0
- 延遲：309.2s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - 總結而言，我們發現 JPH203 能透過其預孵育效應 (preincubation effect) 抑制 LAT1 功能，且該預孵育效應能協同增強共孵育抑制效果 (co-incubation inhibitory effects) [Source: 1-s2.0-S1347861320300633-main]。
> - 此外，預孵育與共孵育效應組合的 IC50 值低於單獨共孵育抑制的效果 (34.2 ± 3.6 nM vs [Source: 1-s2.0-S1347861320300633-main]。
> - 此外，預孵育能協同增強共孵育抑制效果 (co-incubation inhibitory effects) [Source: 1-s2.0-S1347861320300633-main]。
> - 在 OATP2B1 與 simeprevir 中也報導了類似的協同效應。另一方面，就 tacrolimus 而言，其預孵育對 OATP1B1 功能的共孵育抑制缺乏顯著的協同效應。使用 CsA 的預孵育僅輕微影響 OATP1B3 功能的共孵育抑制效果 (IC50 值降至原來約 1/1.5)。關於 JPH203-LAT1 的交互作用，根據上述報導，其協同效應的程度 (IC50 值降至原來約三分之一且 Ki 值降低 20%) 似乎處於輕微至中度的水平 [Source: 1-s2.0-S1347861320300633-main]。
> - 這些結果表明，JPH203 在 HT-29 細胞中對 LAT1 功能產生的預孵育抑制效應是以濃度與時間依賴的方式進行的 [Source: 1-s2.0-S1347861320300633-main]。
> - 根據結果，IC50 值被確定為 193 ± 50 (nM) [Source: 1-s2.0-S134786132030063 …（完整內容見 JSON）

---

### ✅ Q04 · table_lookup

**問題**：What are the main degradation products and impurities of BPA identified by HPLC, and under which storage conditions do they form?

- detected_paper：`None`
- 選出論文：['1-s2.0-S0731708511003451-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water']
- gold_papers：['1-s2.0-S0731708511003451-main']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/7
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：1.0
- 延遲：423.7s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - BPA 與 tyrosine 的分離在 230 nm 處定量，保留時間分別為 5.3 和 4.5 min [Source: 1-s2.0-S0731708511003451-main]。
> - Phenylalanine 在 11.0 min 洗脫，並在 256 nm 處定量 [Source: 1-s2.0-S0731708511003451-main]。
> - BrPD 與 FBBA 在 256 nm 處檢出，洗脫時間分別為 17.3 和 23.7 min [Source: 1-s2.0-S0731708511003451-main]。
> - BDPA 在 270 nm 處檢出，但與 FBBA 共洗脫 (co-elutes) [Source: 1-s2.0-S0731708511003451-main]。
> - 所有三種合成雜質 (synthetic impurities) 在 BPA 存在或不存在的情況下，在濃度為 0.5 µg/ml（或 BPA NWC 的 0.1%）時均可檢出，且在現有的 BPA 樣本中未觀察到 BrPD 或 FBBA/BDPA 組合 [Source: 1-s2.0-S0731708511003451-main]。
> - 降解 (Degradation)：作為原料粉末 (raw powder)，BPA 是穩定的，在 55 °C 下儲存 6 個月或在 40 °C 下儲存 12 個月時，未產生可檢出的降解 [Source: 1-s2.0-S0731708511003451-main]。
> - BPA 在酸性溶液和 FeCl3 溶液中也被觀察到是穩定的（後者用於測試金屬催化降解 (metal-catalysed degradation)）[Source: 1-s2.0-S0731708511003451-main]。
> - BPA 及其合成 …（完整內容見 JSON）

---
