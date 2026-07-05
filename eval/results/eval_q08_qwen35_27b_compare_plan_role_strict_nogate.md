# Eval Report — `q08_qwen35_27b_compare_plan_role_strict_nogate`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-05 00:31
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.5 |
| 平均 grounding 分數 | 0.6 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 1212.6s |
| 平均 planning 延遲 | 10.6s |
| 平均 retrieval 延遲 | 567.5s |
| 平均 grounding 延遲 | 39.9s |
| └ 其中 NLI | 16.4s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q08 | cross_paper | 100.0% | 100.0% | 0.6 | 1212.6s | C0/U4 |

## 逐題細節

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.6
- 延遲：1212.6s　問題標記：{'conflicts': 0, 'unsupported': 4}

**答案預覽**：

> ### Comparison Scaffold:
> 
> | 來源角色 | 項目/路徑 | 來源論文 | 定義性證據 | 相關比較維度 | 注意事項 |
> | :--- | :--- | :--- | :--- | :--- | :--- |
> | review/comparison source | 通用 L-BPA 合成概況與同位素/成本限制 | 【CMDC-20-e202500059】 | Facts 1, 3, 4, 6, 7, 8 | 成本 ($^{10}\text{B}$ >1000倍於常規), 安全性 (Pd/NaIO$_4$ 毒性/純化) | Morita 方法儘管產率高但仍使用 Pd 催化劑 【CMDC-20-e202500059】 |
> | route | 直接氨基酸功能化 (Direct amino acid functionalization) (4-Iodo-L-phenylalanine 衍生物) | 【CMDC-20-e202500059】 | Facts 9, 10 | 產率 (總產率 75%-88%), 去保護基產率 (91%) | 需要特定的保護基 (N-Boc 或 ─COO-t-Bu) 【CMDC-20-e202500059】 |
> | route | 傳統 Cbz–Tyr–OBn 路徑 (Traditional Cbz–Tyr–OBn route) | 【CMDC-20-e202500059】 | Fact 11 | 步驟 (5), 產率 (40%-53%) | 較舊的方法 (1998, 2000) 【CMDC-20-e202500059】 |
> | route | p-甲苯硼酸法 (p-Tolylboronic acid method) | 【CMDC-20-e202500059】 | Fact 12 | 總產率 (15%) | 與新路徑相比效率較 …（完整內容見 JSON）

---
