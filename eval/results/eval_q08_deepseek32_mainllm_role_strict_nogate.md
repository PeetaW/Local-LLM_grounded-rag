# Eval Report — `q08_deepseek32_mainllm_role_strict_nogate`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-04 23:38
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.5 |
| 平均 grounding 分數 | 0.909 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 921.4s |
| 平均 planning 延遲 | 10.0s |
| 平均 retrieval 延遲 | 478.4s |
| 平均 grounding 延遲 | 35.1s |
| └ 其中 NLI | 15.8s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 0.909 | 921.4s | C0/U1 |

## 逐題細節

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.909
- 延遲：921.4s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> ### 4-Borono-L-phenylalanine (L-BPA) 的合成路徑
> 
> #### 路徑比較：
> 
> 1. **化學-酶法路徑 (chemo-enzymatic route)**：
>    - **關鍵步驟**：對映選擇性烷基化 (enantioselective alkylation)，隨後在 pH 5.0 下使用 chymotrypsin 進行酶促水解 (enzymatic hydrolysis)。
>    - **光學純度 (Optical Purity)**：L-BPA 的 ee 達到 100%。
>    - **同位素富集 (Isotopic Enrichment)**：由於純度高，適合生產同位素富集材料。
>    - **可擴展性與成本效益 (Scalability and Cost-Effectiveness)**：由於使用酶可能會增加成本，因此具有潛在挑戰。然而，它避免了額外的消旋分解步驟 (resolution steps)。
> 
> 2. **鹼水解路徑 (alkaline hydrolysis route)**：
>    - **關鍵步驟**：使用 sodium hydroxide 進行鹼處理。
>    - **光學純度 (Optical Purity)**：L-BPA 的 ee 達到 86%。
>    - **同位素富集 (Isotopic Enrichment)**：由於光學純度較低，與化學-酶法相比效率較低。
>    - **可擴展性與成本效益 (Scalability and Cost-Effectiveness)**：由於避免了酶促過程，因此更具成本效益且更具可擴展性，適合大規模生產。
> 
> #### 核心權衡 (Central Trade-off)：
> 在實現高純度/對映純 (enantiopure) 的同位素富集 L-BPA 與可擴展性/成本效益之間的權衡 …（完整內容見 JSON）

---
