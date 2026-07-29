# Eval Report — `q08_review_role_witness_rootfix`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-29 02:13
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
| 平均 Stage 2 evidence 覆蓋率 | 66.7% |
| 平均總延遲 | 587.0s |
| 平均 planning 延遲 | 33.6s |
| 平均 retrieval 延遲 | 7.7s |
| └ Phase A embed/vector/BM25 | 7.6s |
| └ Phase B 子答案生成 | 0.1s |
| 平均 grounding 延遲 | 30.6s |
| └ 其中 NLI | 3.2s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ⚠️ | Q08 | cross_paper | 1.0 | 1.0 | 100.0% | 100.0% | 66.7% | 0.833 | 587.0s | C0/U1 |

## 逐題細節

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/7
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：66.7%　grounding：0.833
- 延遲：587.0s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
>
> 比較框架：
> - 路徑：【bbb0683】報導了對映選擇性烷基化 (enantioselective alkylation) 以及隨後的酶促水解 (enzymatic hydrolysis)，從而產生光學純 L-BPA 【bbb0683】。
> - 評論/比較來源：【CMDC-20-e202500059】報導 L-BPA 的合成已透過多種路徑實現，這反映了在生產高純度、同位素富集 (isotopically enriched) 物質方面所面臨的挑戰 【CMDC-20-e202500059】。
> - 評論維度：該評論強調了每種方法在成本效益 (cost-effectiveness)、可擴展性 (scalability) 和安全性 (safety) 方面的局限性 【CMDC-20-e202500059】。
>
> 核心權衡（高純度/同位素富集與可擴展性及成本效益）：
> - 高純度/同位素富集：生產高純度、同位素富集 (isotopically enriched) 的物質是 L-BPA 合成中的主要挑戰 【CMDC-20-e202500059】。
> - 可擴展性：在規模化生產中使用任何氧化劑 (oxidant) 本質上是一種製程安全風險 (process safety risk) 【CMDC-20-e202500059】。
> - 成本效益：在製備同位素富集 (isotopically enriched) 的化合物時，主要成本通常來自於同位素起始原料 (isotope starting material) 【CMDC-20-e202500059】。
>
> ---
> 📋 **答案品質報告**
>
> ✅ **整體論文依據率**：83.3%　高（答案高度忠實於論文內容）
>
> ⚠️  **以下 1 個陳述未找到明確論文依據 …（完整內容見 JSON）

---
