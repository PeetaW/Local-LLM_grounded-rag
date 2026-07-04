# Eval Report — `q08_query_scaffold_role_strict_nogate`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-04 23:15
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.5 |
| 平均 grounding 分數 | 0.714 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 1663.0s |
| 平均 planning 延遲 | 10.0s |
| 平均 retrieval 延遲 | 763.0s |
| 平均 grounding 延遲 | 41.5s |
| └ 其中 NLI | 20.6s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q08 | cross_paper | 100.0% | 100.0% | 0.714 | 1663.0s | C0/U4 |

## 逐題細節

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.714
- 延遲：1663.0s　問題標記：{'conflicts': 0, 'unsupported': 4}

**答案預覽**：

> **比較框架 (Comparison scaffold)：**
> 
> | 來源角色 | 項目/路徑 | 來源論文 | 定義性證據 | 相關比較維度 | 注意事項 |
> | :--- | :--- | :--- | :--- | :--- | :--- |
> | route | Pd催化偶聯 (Pd-catalyzed coupling) | 【CMDC-20-e202500059】 | 使用 $\text{B}_2\text{pin}_2$ 或 HBpin 引入硼 | 成本效益、產率 | 總產率 $\le 48\%$；對於 $^{10}\text{B}$ 而言，$\text{B}_2\text{pin}_2$ 成本高且浪費 |
> | route | 有機金屬偶聯 (Organometallic coupling) (n-BuLi, i-PrMgCl, Turbo Grignard) | 【CMDC-20-e202500059】 | $\text{B}(\text{OBu})_3$ 與 4-碘苯丙氨酸衍生物的偶聯 | 同位素富集 ($^{10}\text{B}$)、產率 | 高產率 (Turbo Grignard 最高可達 $87\%$) |
> | route | 森田製藥專利 (Morita Pharmaceutical patent) | 【CMDC-20-e202500059】 | 硼引入後步驟少且產率高 | 可擴展性、成本效益 | 在倒數第二步需要<0xE9><0x88><0x80>催化劑 (palladium catalyst) |
> | route | 混合製程 (Hybrid process) | 【bbb0683】 | 對映選擇性烷基化後接酶促水解 | 光學純度、可擴展性、後處理便捷性 | 需要 *chymotrypsin* 以達到 $100\%$ 光學純度 |
> | rou …（完整內容見 JSON）

---
