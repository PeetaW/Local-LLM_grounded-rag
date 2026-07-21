# Eval Report — `q04_fact_contract_b`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-21 22:45
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 1.0 |
| Correctness judge 覆蓋 | 1/1（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | N/A |
| Translation judge 覆蓋 | 0/1（N/A 1） |
| 平均 grounding 分數 | 1.0 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 100.0% |
| 平均 Stage 2 evidence 覆蓋率 | 100.0% |
| 平均總延遲 | 424.1s |
| 平均 planning 延遲 | 6.1s |
| 平均 retrieval 延遲 | 4.8s |
| └ Phase A embed/vector/BM25 | 4.7s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 19.3s |
| └ 其中 NLI | 2.0s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ⚠️ | Q04 | table_lookup | 1.0 | N/A | 100.0% | 100.0% | 100.0% | 1.0 | 424.1s | C0/U0 |

## 逐題細節

### ⚠️ Q04 · table_lookup

**問題**：What are the main degradation products and impurities of BPA identified by HPLC, and under which storage conditions do they form?

- detected_paper：`None`
- 選出論文：['1-s2.0-S0731708511003451-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water']
- gold_papers：['1-s2.0-S0731708511003451-main']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/7
- translation fidelity：N/A　raw：N/A　judge：`translation_fidelity_v2`
- translation reason：invalid translation audit: top-level 'errors' must be a list
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：1.0
- 延遲：424.1s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - 分離出的 BPA 與 tyrosine 在 230 nm 處進行定量，其保留時間（retention times）分別為 5.3 與 4.5 min [Source: 1-s2.0-S0731708511003451-main]。
> - Phenylalanine 在 11.0 min 洗脫（elutes）並在 256 nm 處進行定量 [Source: 1-s2.0-S0731708511003451-main]。
> - （參見 Fig. 2。）BrPD 與 FBBA 在 256 nm 處被檢出，且分別在 17.3 與 23.7 min 洗脫 [Source: 1-s2.0-S0731708511003451-main]。
> - BDPA 在 270 nm 處被檢出，但與 FBBA 共洗脫（co-elutes）[Source: 1-s2.0-S0731708511003451-main]。
> - 在有或沒有 BPA 的情況下，三種合成雜質（synthetic impurities）在濃度為 0.5 $\mu$g/ml（或 BPA NWC 的 0.1%）時均可檢出，且在現有的 BPA 樣本中未觀察到 BrPD 或 FBBA/BDPA 組合 [Source: 1-s2.0-S0731708511003451-main]。
> - tyrosine 與 phenylalanine 均被之前的研究者 [4] 報告為雜質，且兩者皆具有低...（\*通訊作者）[Source: 1-s2.0-S0731708511003451-main]。
> - 在鹼性（alkali）與氧化條件下觀察到 BPA 降解為 tyrosine，且在後者情況下發生速度極快（觀察到質量平衡 mass balance）[Source: 1-s2.0-S0731708511003451-main]。 …（完整內容見 JSON）

---
