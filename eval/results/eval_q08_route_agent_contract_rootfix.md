# Eval Report — `q08_route_agent_contract_rootfix`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-30 08:01
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 1.0 |
| Correctness judge 覆蓋 | 1/1（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 0.5 |
| Translation judge 覆蓋 | 1/1（N/A 0） |
| 平均 grounding 分數 | 0.75 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 100.0% |
| 平均 Stage 2 evidence 覆蓋率 | 100.0% |
| 平均總延遲 | 522.8s |
| 平均 planning 延遲 | 32.5s |
| 平均 retrieval 延遲 | 7.6s |
| └ Phase A embed/vector/BM25 | 7.6s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 30.5s |
| └ 其中 NLI | 4.1s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ⚠️ | Q08 | cross_paper | 1.0 | 0.5 | 100.0% | 100.0% | 100.0% | 0.75 | 522.8s | C0/U2 |

## 逐題細節

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/7
- translation fidelity：0.5　raw：3/5　judge：`translation_fidelity_v2`
- translation reason：1 material and 0 minor semantic errors; The source states that the review 'reports' (i.e., presents findings) about methods being examined. The target mistranslates this as '報導，對...進行了...' which reads like a direct report of an action rather than stating what the review does, creating ambiguity and altering the grammatical subject-verb关系.
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：0.75
- 延遲：522.8s　問題標記：{'conflicts': 0, 'unsupported': 2}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
>
> 比較框架：
> - 路徑：【bbb0683】報導了一種涉及對映選擇性烷基化 (enantioselective alkylation) 及隨後酶水解 (enzymatic hydrolysis) 的混合製程，產出光學純的 L-BPA 【bbb0683】。
> - 路徑細節：使用 chymotrypsin 進行水解以提供光學純的 L-BPA 【bbb0683】。
> - 評論/比較來源：【CMDC-20-e202500059】報導，對學術文獻和專利文獻中用於合成 L-BPA 的報導方法進行了全面且批判性的檢查與比較。該評論還強調了每種方法在可擴展性 (scalability)、成本效益 (cost-effectiveness) 和安全性方面的局限性 【CMDC-20-e202500059】。
> - 評論維度：該評論強調了每種方法在可擴展性 (scalability)、成本效益 (cost-effectiveness) 和安全性方面的局限性 【CMDC-20-e202500059】。
>
> 核心權衡（高純度/同位素富集與可擴展性及成本效益）：
> - 高純度/同位素富集：在 L-BPA 合成中，生產高純度、同位素富集的材料是一項挑戰 【CMDC-20-e202500059】。
> - 可擴展性：在大規模生產中使用任何氧化劑 (oxidant) 本質上都存在製程安全風險 【CMDC-20-e202500059】。
> - 成本效益：同位素富集的 10B 成本很高；在製備同位素富集化合物時，主要成本通常來自於同位素起始原料 【CMDC-20-e202500059】。
>
> ---
> 📋 **答案品質報告**
>
> ⚠️ **整體論文依據率**：75.0%　中（部分陳述需要確認）
>
> ⚠️  **以下 2 個陳述未找到明確論文 …（完整內容見 JSON）

---
