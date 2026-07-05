# Eval Report — `q08_qwen35_27b_reference_level_nogate`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-05 12:54
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.5 |
| 平均 grounding 分數 | 0.25 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 1712.4s |
| 平均 planning 延遲 | 11.1s |
| 平均 retrieval 延遲 | 1063.0s |
| 平均 grounding 延遲 | 22.4s |
| └ 其中 NLI | 6.6s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ❌ | Q08 | cross_paper | 100.0% | 100.0% | 0.25 | 1712.4s | C0/U3 |

## 逐題細節

### ❌ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.25
- 延遲：1712.4s　問題標記：{'conflicts': 0, 'unsupported': 3}

**答案預覽**：

> Comparison scaffold:
> 
> | 來源角色 | 項目/路徑 | 來源論文 | 定義性證據 | 相關比較維度 | 注意事項 |
> | :--- | :--- | :--- | :--- | :--- | :--- |
> | review/comparison source | 通用 L-BPA 合成策略（手性優先 (Chirality First) 與 硼優先 (Boron First)；二鹵代苯修改 (Dihalobenzene modification) 與 直接氨基酸功能化 (Direct amino acid functionalization)） | 【CMDC-20-e202500059】 | 評估經濟參數，如總步驟數和產率；指出 $^{10}\text{B}$ 的同位素富集成本顯著高於普通硼酸；記錄了重金屬 (Pd) 和有毒氧化劑 ($\text{NaIO}_4$) 的純化挑戰。 | 同位素富集成本、可擴展性（步驟數）、安全性/純化。 | 部分路徑已基本被放棄；有前景的專利在用於注射劑 (parenteral use) 時仍面臨催化劑去除問題 【CMDC-20-e202500059】。 |
> | route | 混合製程（對映選擇性烷基化 (enantioselective alkylation) + 酶水解 (enzymatic hydrolysis)） | 【bbb0683】 | 使用手性吡嗪鋰鹽 (chiral pyrazine lithium salt) 和受保護的硼酸衍生物，隨後進行酶或鹼水解；比較了酶法與化學方法之間的對映體純度。 | 對映體純度、透過保護基 (protection groups) 進行產率優化。 | 需要特定的低溫條件；在水解步驟中，光學純度與反應速率/產率之間存在權衡 【bbb0683】。 |
> | background | …（完整內容見 JSON）

---
