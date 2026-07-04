# Eval Report — `q08_qwen35_27b_mainllm_role_strict_nogate`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-05 00:05
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.5 |
| 平均 grounding 分數 | 0.778 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 1371.1s |
| 平均 planning 延遲 | 9.9s |
| 平均 retrieval 延遲 | 625.3s |
| 平均 grounding 延遲 | 31.0s |
| └ 其中 NLI | 14.5s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q08 | cross_paper | 100.0% | 100.0% | 0.778 | 1371.1s | C0/U2 |

## 逐題細節

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.778
- 延遲：1371.1s　問題標記：{'conflicts': 0, 'unsupported': 2}

**答案預覽**：

> ### 路徑圖：4-硼基-L-苯丙氨酸 (4-Borono-L-Phenylalanine, L-BPA) 的合成路徑
> 
> | 路徑名稱 | 定義步驟與試劑 | 來源論文 |
> | :--- | :--- | :--- |
> | **化學-酶雜合路徑 (Hybrid Chemo-Enzymatic Route)** | **(1)** 對映選擇性烷基化 (Enantioselective alkylation)：使用 $(2\text{R})(-)-2,5\text{-dihydro-2-isopropyl-3,6-dimethoxypyrazine}$ 與受保護的 4-bromomethylbenzeneboronic acid 衍生物，在 $-78^\circ\text{C}$ 的 THF 中使用 $\text{n-BuLi}$。<br>**(2)** 酶水解 (Enzymatic hydrolysis)：中間體 L-BPA methyl ester 使用 **chymotrypsin** (pH 5.0, 室溫) 處理，與使用氫氧化鈉 (sodium hydroxide) 的化學水解進行對比。 | 【bbb0683】 |
> 
> ---
> 
> ### 比較框架 (Comparison Scaffold)
> 
> | 來源角色 | 項目/路徑 | 來源論文 | 定義性證據 | 相關比較維度 | 注意事項 |
> | :--- | :--- | :--- | :--- | :--- | :--- |
> | **route** | L-BPA 的化學-酶雜合合成 (Hybrid chemo-enzymatic synthesis) | 【bbb0683】 | 對映選擇性烷基化隨後進行酶水解可獲得 $100\% \text{ ee}$；化學水解 (NaOH) 則獲得 $86\% \text{ ee}$。 …（完整內容見 JSON）

---
