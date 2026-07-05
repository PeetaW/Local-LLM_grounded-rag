# Eval Report — `q08_qwen35_27b_concise_scaffold_nogate`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-05 12:03
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.75 |
| 平均 grounding 分數 | 0.5 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 1331.8s |
| 平均 planning 延遲 | 10.7s |
| 平均 retrieval 延遲 | 674.9s |
| 平均 grounding 延遲 | 23.9s |
| └ 其中 NLI | 8.1s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q08 | cross_paper | 100.0% | 100.0% | 0.5 | 1331.8s | C0/U2 |

## 逐題細節

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.5
- 延遲：1331.8s　問題標記：{'conflicts': 0, 'unsupported': 2}

**答案預覽**：

> Comparison scaffold:
> | 來源角色 | 項目/路徑 | 來源論文 | 定義性證據 | 相關比較維度 | 注意事項 |
> | :--- | :--- | :--- | :--- | :--- | :--- |
> | 評論/比較來源 | 手性優先與硼優先策略及其經濟評估 (Chirality First vs Boron First Strategies & Economic Evaluation) | 【CMDC-20-e202500059】 | 將路徑分為「手性優先」(Chirality First) 與「硼優先」(Boron First)；評估 $^{10}\text{B}$ 成本（高於常規 1000 倍以上）、步驟數、產率；指出 $\text{B}_2\text{pin}_2$ 的浪費是不理想的。 | 同位素富集 ($^{10}\text{B}$ 豐度/成本)、可擴展性 (步驟數/產率)、成本效益。 | 側重於經濟參數，而非所有引用方法的完整實驗方案。 |
> | 路徑 | 使用手性輔助劑的對映選擇性烷基化與酶促水解 (Enantioselective Alkylation with Chiral Auxiliary & Enzymatic Hydrolysis) | 【bbb0683】 | 透過 bis lactim ether 的 n-BuLi 烷基化，隨後進行 chymotrypsin 水解來合成 L-BPA，以達到 e.e. 100%。 | 可擴展性 (商業輔助劑優勢)、對映體純度。 | 與評論來源相比，未明確討論同位素富集成本或大規模經濟指標。 |
> 
> 【bbb0683】中詳細描述的特定合成路徑採用了一種手性輔助劑 (chiral auxiliary) 策略，包括在 $-78^\circ\text{C}$ 下使用 n-BuLi 對 bis lacti …（完整內容見 JSON）

---
