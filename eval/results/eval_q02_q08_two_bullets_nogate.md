# Eval Report — `q02_q08_two_bullets_nogate`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-05 16:22
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.625 |
| 平均 grounding 分數 | 0.785 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 1087.9s |
| 平均 planning 延遲 | 8.4s |
| 平均 retrieval 延遲 | 513.3s |
| 平均 grounding 延遲 | 18.7s |
| └ 其中 NLI | 6.3s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q02 | single_paper | 100.0% | 100.0% | 0.571 | 998.2s | C0/U3 |
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 1.0 | 1177.6s | C0/U0 |

## 逐題細節

### ⚠️ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.571
- 延遲：998.2s　問題標記：{'conflicts': 0, 'unsupported': 3}

**答案預覽**：

> ### 概述與背景
> L-p-硼苯丙氨酸 (L-p-boronophenylalanine, L-BPA) 被認為是最有效的硼中子捕獲治療 (Boroneutron capture therapy, BNCT) 腫瘤靶向硼載體之一 [bbb0683]。先前利用胰凝乳蛋白酶 (chymotrypsin) 或氨基醯基酶 (aminoacylase) 對外消旋前驅體進行酶促拆分 (enzymatic resolution)，或使用手性膦催化不對稱氫化 (chiral phosphine catalyzed asymmetric hydrogenation) 的合成方法，被發現存在底物可用性以及需要進一步分離步驟以獲得光學純 L-BPA 等限制 [bbb0683]。為了縮小這些問題，開發了一種結合對映選擇性烷基化 (enantioselective alkylation) 與隨後酶促水解 (enzymatic hydrolysis) 的混合製程，用以合成光學純的 L-p-硼苯丙氨酸 (L-BPA) [bbb0683]。
> 
> ### 混合製程之關鍵步驟
> 該合成包含四個主要階段：保護、不對稱烷基化、去保護/酯化以及最終水解。
> 
> 1.  **保護 (Protection)：** 製程始於環狀硼酸酯衍生物 (cyclic borinate derivative, 化合物 2) 的製備。這是透過將 4-硼甲基苯硼酸 (4-boromethylbenzene-bornic acid，由商業化的 4-溴甲苯製備) 與 2,2-二甲基-1,3-丙二醇 (2,2-dimethyl-1,3-propanediol) 在乾燥的四氫呋喃 (tetrahydrofuran) 中於室溫下反應 10 分鐘而實現 [bbb0683]。此步驟產生的環狀硼酸酯衍生物產率為 79% [bbb0683]。
> 2.  **不對稱 …（完整內容見 JSON）

---

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：1177.6s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 比較框架：
> - 綜述/比較來源：【CMDC-20-e202500059】透過權衡同位素富集成本 (isotopic enrichment costs) 與步驟數量效率 (step count efficiency)，評估 L-BPA 的合成策略 【CMDC-20-e202500059】。
> - 路徑：【bbb0683】描述了一條利用對映選擇性烷基化 (enantioselective alkylation) 與酶水解 (enzymatic hydrolysis) 來生產光學純 L-BPA 的特定合成路徑 【bbb0683】。
> 
> * 直接路徑證據表明，將 chymotrypsin 介導的水解與手性輔助劑 (chiral auxiliaries) 相結合可實現高光學純度，同時在後處理簡便性 (workup simplicity) 和減少反應步驟方面具有優勢 【bbb0683】。
> * 綜述/比較來源證據強調，在硼引入 (boron incorporation) 後盡量減少合成操作對於成本效益至關重要，因為 $^{10}\text{B}$ 同位素比標準材料昂貴得多，且必須將浪費降至最低 【CMDC-20-e202500059】。
> 
> 核心權衡：L-BPA 的合成在透過專門的酶法或手性方案 (enzymatic or chiral protocols) 實現高對映純度 (enantiopurity) 的要求與經濟可擴展性 (economic scalability) 之間取得平衡，其中盡量減少涉及昂貴 $^{10}\text{B}$ 同位素的步驟對於減輕材料浪費至關重要 【CMDC-20-e202500059】。
> 
> ---
> 📋 **答案品質報告**
> 
> ✅ **整體論文依據率**：100.0%　高（答案高度忠實於論文內容）
> 
> ✅ **所有陳述均有論文依據**
> 
> 
> <!-- groundin …（完整內容見 JSON）

---
