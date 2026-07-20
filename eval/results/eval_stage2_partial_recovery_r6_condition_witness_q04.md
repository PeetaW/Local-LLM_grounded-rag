# Eval Report — `stage2_partial_recovery_r6_condition_witness_q04`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-20 21:37
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 1.0 |
| Correctness judge 覆蓋 | 1/1（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 0.0 |
| Translation judge 覆蓋 | 1/1（N/A 0） |
| 平均 grounding 分數 | 1.0 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 100.0% |
| 平均 Stage 2 evidence 覆蓋率 | 100.0% |
| 平均總延遲 | 619.7s |
| 平均 planning 延遲 | 6.2s |
| 平均 retrieval 延遲 | 4.8s |
| └ Phase A embed/vector/BM25 | 4.8s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 24.1s |
| └ 其中 NLI | 3.4s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ❌ | Q04 | table_lookup | 1.0 | 0.0 | 100.0% | 100.0% | 100.0% | 1.0 | 619.7s | C0/U0 |

## 逐題細節

### ❌ Q04 · table_lookup

**問題**：What are the main degradation products and impurities of BPA identified by HPLC, and under which storage conditions do they form?

- detected_paper：`None`
- 選出論文：['1-s2.0-S0731708511003451-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water']
- gold_papers：['1-s2.0-S0731708511003451-main']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/7
- translation fidelity：0.0　raw：1/5　judge：`translation_fidelity_v2`
- translation reason：5 material and 0 minor semantic errors; The English term 'Tyrosine' (酪胺酸 in the target) is a different chemical compound from phenylalanine. Tyrosine should be translated as 酪氨酸, while tyramine is 酪胺酸.
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：1.0
- 延遲：619.7s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> tyrosine (酪胺酸) 與 phenylalanine (苯丙胺酸) 已被報導為 boron phenylalanine (BPA, 硼苯丙胺酸) 的雜質 [Source: 1-s2.0-S0731708511003451-main]。可透過 HPLC (高效液相層析) 檢測到的合成中間體包括 (2-(4-Bromophenyl)-[1,3]dioxane (BrPD)、4-Formylbenzeneboronic acid (FBBA) 以及 2-tert-Butoxycarbonylamino-3-[4-(5,5-dimethyl-[1,3,2]dioxaborinan-2-yl)-phenyl]acrylic acid methyl ester (BDPA) [Source: 1-s2.0-S0731708511003451-main]。然而，儘管在 0.5 µg/ml 的濃度下可被檢測到，但在現有的 BPA 樣本中均未觀察到 BrPD 或 FBBA/BDPA 組合 [Source: 1-s2.0-S0731708511003451-main]。在 HPLC 分析中，tyrosine (酪胺酸) 在 230 nm 下進行定量，其保留時間 (retention time) 為 4.5 min，而 phenylalanine (苯丙胺酸) 的洗脫時間為 11.0 min [Source: 1-s2.0-S0731708511003451-main]。
>
> BPA 分解為 tyrosine (酪胺酸) 發生於鹼性與氧化條件下，且氧化過程發生得極其迅速 [Source: 1-s2.0-S0731708511003451-main]。相反地，觀察到 BPA 在酸性及 FeCl3 溶液中具有穩定性 [Source: 1-s2.0-S0731708511003451-main] …（完整內容見 JSON）

---
