# Eval Report — `q08_review_witness_dimension_clip_rootfix`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-29 09:52
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
| 平均總延遲 | 538.0s |
| 平均 planning 延遲 | 32.4s |
| 平均 retrieval 延遲 | 7.6s |
| └ Phase A embed/vector/BM25 | 7.6s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 24.9s |
| └ 其中 NLI | 2.6s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ✅ | Q08 | cross_paper | 1.0 | 1.0 | 100.0% | 100.0% | 100.0% | 0.833 | 538.0s | C0/U1 |

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
- 延遲：538.0s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
>
> 比較框架：
> - Route: 【bbb0683】報導了一種涉及對映選擇性烷基化 (enantioselective alkylation) 及隨後的酶水解 (enzymatic hydrolysis) 的混合製程，產出光學純的 L-BPA 【bbb0683】。
> - Review/comparison source: 【CMDC-20-e202500059】報導該評論亦強調了每種方法在可擴展性 (scalability)、成本效益 (cost-effectiveness) 及安全性方面的局限性，特別是考慮到同位素富集 10B (isotopically enriched 10B) 的高成本 【CMDC-20-e202500059】。
> - Review dimensions: 該評論強調了每種方法在可擴展性 (scalability)、成本效益 (cost-effectiveness) 及安全性方面的局限性 【CMDC-20-e202500059】。
>
> 核心權衡（高純度/同位素富集與可擴展性及成本效益）：
> - High-purity/isotopic enrichment: 在生產高純度、同位素富集材料 (isotopically enriched material) 方面存在挑戰 【CMDC-20-e202500059】。
> - Scalability: 相較於報導的方法，其優勢在於操作與後處理 (workup) 簡便且反應步驟少 【bbb0683】。
> - Cost-effectiveness: 同位素富集 10B (isotopically enriched 10B) 的成本高昂 【CMDC-20-e202500059】。
>
> ---
> 📋 **答案品質報告**
>
> ✅ **整體論文 …（完整內容見 JSON）

---
