# Eval Report — `q02_q08_query_scaffold_scope_strict_nogate`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-04 19:10
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.625 |
| 平均 grounding 分數 | 0.625 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 1024.8s |
| 平均 planning 延遲 | 8.2s |
| 平均 retrieval 延遲 | 571.4s |
| 平均 grounding 延遲 | 27.4s |
| └ 其中 NLI | 14.3s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q02 | single_paper | 100.0% | 100.0% | 0.625 | 760.3s | C0/U3 |
| ⚠️ | Q08 | cross_paper | 100.0% | 100.0% | 0.625 | 1289.2s | C0/U6 |

## 逐題細節

### ⚠️ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.625
- 延遲：760.3s　問題標記：{'conflicts': 0, 'unsupported': 3}

**答案預覽**：

> **路線圖 (Route Map)**
> *   **混合製程 (Hybrid Process)**：對映選擇性烷基化 (Enantioselective alkylation) $\rightarrow$ 酶水解 (Enzymatic hydrolysis) [bbb0683]。
> 
> **比較框架 (Comparison scaffold)**
> | 項目/路線 | 來源論文 | 定義依據 | 相關比較維度 | 注意事項 |
> | :--- | :--- | :--- | :--- | :--- |
> | 混合製程 (化學-酶法) (Hybrid Process (Chemo-enzymatic)) | bbb0683 | 對映選擇性烷基化，隨後使用糜蛋白酶 (chymotrypsin) 進行酶水解 | 光學純度 (100% e.e.)、操作簡便性、產率 | 需要特定的 pH 控制和酶 |
> | 非酶路線 (Non-enzymatic Route) | bbb0683 | 使用 0.5 N 氫氧化鈉 (sodium hydroxide) 對 L-BPA methyl ester 進行化學水解 | 產率 vs. 光學純度 | 對映體純度顯著較低 (86%) |
> 
> **核心權衡 (Central trade-off)：** 合成最後一步的主要權衡在於，透過酶水解 (Enzymatic hydrolysis) 實現絕對的光學純度 (100% e.e.)，以及透過非酶水解 (Non-enzymatic hydrolysis) 獲得較高的化學產率，但後者會導致對映體純度顯著降低 (86%) [bbb0683]。
> 
> ---
> 
> **L-p-boronophenylalanine (L-BPA) 混合製程的詳細合成**
> 
> 為了將光學純的 L-p-boronophenylalanine (L-BPA) …（完整內容見 JSON）

---

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.625
- 延遲：1289.2s　問題標記：{'conflicts': 0, 'unsupported': 6}

**答案預覽**：

> ### 比較框架：
> 
> | 路徑 | 來源論文 | 定義性證據 | 相關比較維度 | 注意事項 |
> | :--- | :--- | :--- | :--- | :--- |
> | Pd 催化路徑 (Pd-catalyzed routes) | 【CMDC-20-e202500059】 | 使用 $\text{B}_2\text{pin}_2$ 或 $\text{HBpin}$ 引入硼 | 成本、產率、安全性/純度 | 重金屬殘留風險 (FDA 限制) |
> | 有機金屬路徑 (Organometallic routes) | 【CMDC-20-e202500059】 | 使用 $\text{n-BuLi}$、$\text{i-PrMgCl}$ 或 Turbo Grignard 與硼酸酯 ($\text{B(OBu)}_3, \text{B(OMe)}_3, \text{B(OEt)}_3$) 反應 | 同位素富集成本、產率 | - |
> | Kamekawa 專利路徑 (Kamekawa Patent route) | 【CMDC-20-e202500059】 | 為森田製藥公司 (Morita Pharmaceutical Company) 優化了總步驟和產率 | 可擴展性、效率 | 在倒數第二步需要<0xE9><0x88><0x80>催化劑 (palladium catalyst) |
> | 混合化學-酶路徑 (Hybrid chemo-enzymatic route) | 【bbb0683】 | 使用鋰化吡嗪 (lithiated pyrazine) 的對映選擇性烷基化，隨後進行 chymotrypsin 水解 | 對映純度、產率 | - |
> | 分辨/氫化路徑 (Resolution/Hydrogenation routes) | 【bbb0683】 | 外消旋前體的酶法分辨 …（完整內容見 JSON）

---
