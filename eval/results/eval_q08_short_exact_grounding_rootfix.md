# Eval Report — `q08_short_exact_grounding_rootfix`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-29 20:48
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 1.0 |
| Correctness judge 覆蓋 | 1/1（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 1.0 |
| Translation judge 覆蓋 | 1/1（N/A 0） |
| 平均 grounding 分數 | 0.833 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 100.0% |
| 平均 Stage 2 evidence 覆蓋率 | 100.0% |
| 平均總延遲 | 569.3s |
| 平均 planning 延遲 | 32.4s |
| 平均 retrieval 延遲 | 7.6s |
| └ Phase A embed/vector/BM25 | 7.5s |
| └ Phase B 子答案生成 | 0.1s |
| 平均 grounding 延遲 | 31.9s |
| └ 其中 NLI | 3.4s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ✅ | Q08 | cross_paper | 1.0 | 1.0 | 100.0% | 100.0% | 100.0% | 0.833 | 569.3s | C0/U1 |

## 逐題細節

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/7
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：0.833
- 延遲：569.3s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
>
> Comparison scaffold:
> - Route: 【bbb0683】報導了一種涉及對映選擇性烷基化 (enantioselective alkylation) 及隨後酶促水解 (enzymatic hydrolysis) 的混合製程 (hybrid process)，從而產生光學純 L-BPA 【bbb0683】。
> - Review/comparison source: 【CMDC-20-e202500059】報導指出，儘管其結構看似簡單，但目前尚無統一的合成方法——L-BPA 的合成已透過多種路徑進行，這反映了生產高純度、同位素富集材料 (isotopically enriched material) 的挑戰 【CMDC-20-e202500059】。
> - Review dimensions: 該回顧強調了每種方法在可擴展性 (scalability)、成本效益 (cost-effectiveness) 和安全性方面的局限性 【CMDC-20-e202500059】。
>
> Central trade-off (high-purity/isotopic enrichment versus scalability and cost-effectiveness):
> - High-purity/isotopic enrichment: L-BPA 的合成反映了生產高純度、同位素富集材料 (isotopically enriched material) 的挑戰 【CMDC-20-e202500059】。
> - Scalability: 在大規模生產中使用任何氧化劑 (oxidant) 本質上都存在製程安全風險 (process safety risk) 【CMDC-20-e20 …（完整內容見 JSON）

---
