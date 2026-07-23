# Eval Report — `q04_fact_contract_clean_r3`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-22 00:50
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 1.0 |
| Correctness judge 覆蓋 | 1/1（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 1.0 |
| Translation judge 覆蓋 | 1/1（N/A 0） |
| 平均 grounding 分數 | 1.0 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 100.0% |
| 平均 Stage 2 evidence 覆蓋率 | 100.0% |
| 平均總延遲 | 349.4s |
| 平均 planning 延遲 | 6.3s |
| 平均 retrieval 延遲 | 4.9s |
| └ Phase A embed/vector/BM25 | 4.8s |
| └ Phase B 子答案生成 | 0.1s |
| 平均 grounding 延遲 | 19.4s |
| └ 其中 NLI | 2.0s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ✅ | Q04 | table_lookup | 1.0 | 1.0 | 100.0% | 100.0% | 100.0% | 1.0 | 349.4s | C0/U0 |

## 逐題細節

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
- 延遲：349.4s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - BPA 與 tyrosine 的分離在 230 nm 處定量，保留時間 (retention times) 分別為 5.3 和 4.5 min [Source: 1-s2.0-S0731708511003451-main]。
> - Phenylalanine 在 11.0 min 洗脫 (elutes)，並在 256 nm 處定量 [Source: 1-s2.0-S0731708511003451-main]。
> - BrPD 與 FBBA 在 256 nm 處檢測，洗脫時間分別為 17.3 和 23.7 min [Source: 1-s2.0-S0731708511003451-main]。
> - BDPA 在 270 nm 處檢測，但與 FBBA 共洗脫 (co-elutes) [Source: 1-s2.0-S0731708511003451-main]。
> - 無論是在存在或不存在 BPA 的情況下，三種合成雜質 (synthetic impurities) 在濃度為 0.5 µg/ml（或 BPA NWC 的 0.1%）時均可檢測到，且在現有的 BPA 樣品中未觀察到 BrPD 或 FBBA/BDPA 組合 [Source: 1-s2.0-S0731708511003451-main]。
> - 降解 (Degradation)：作為原料粉末 (raw powder)，BPA 是穩定的，在 55 °C 下儲存 6 個月或在 40 °C 下儲存 12 個月時，未產生可檢測到的降解 [Source: 1-s2.0-S0731708511003451-main]。
> - 亦觀察到 BPA 在酸性溶液和 FeCl3 溶液（後者用於測試金屬催化降解 metal-catalysed degradation）中具有穩定性 [Source: 1-s2.0-S073 …（完整內容見 JSON）

---
