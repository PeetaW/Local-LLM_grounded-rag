# Eval Report — `q02_q08_stage2_evidence_rolehint`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-06 00:33
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.625 |
| 平均 grounding 分數 | 0.65 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 556.3s |
| 平均 planning 延遲 | 8.3s |
| 平均 retrieval 延遲 | 6.4s |
| └ Phase A embed/vector/BM25 | 6.3s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 20.5s |
| └ 其中 NLI | 6.1s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q02 | single_paper | 100.0% | 100.0% | 0.5 | 513.8s | C0/U2 |
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 0.8 | 598.7s | C0/U1 |

## 逐題細節

### ⚠️ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.5
- 延遲：513.8s　問題標記：{'conflicts': 0, 'unsupported': 2}

**答案預覽**：

> 光學純的 L-p-boronophenylalanine (L-BPA) 的不對稱合成 (Asymmetric Synthesis) 採用了一種將對映選擇性烷基化 (Enantioselective Alkylation) 與隨後的酵素水解 (Enzymatic Hydrolysis) 相結合的混合製程 (Hybrid Process) [bbb0683]。根據提供的數據，定義此化學-酵素路徑 (Chemo-enzymatic Route) 的核心關鍵步驟如下：
> 
> *   **對映選擇性烷基化 (Enantioselective Alkylation)：** 立體中心 (Stereocentre) 是透過鋰化的 (2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine 與 4-bromomethylbenzeneboronate (2) 之間的反應建立的 [bbb0683]。此對映選擇性烷基化在 -78°C 的 THF 中進行，以產生加成物 4 (Adduct 4) [bbb0683]。
> *   **酸性水解/去保護 (Acidic Hydrolysis/Deprotection)：** 加成物 4 (Adduct 4) 經過鹽酸 (Hydrochloric Acid) 的逐步處理，將中間體轉化為 L-BPA methyl ester 6 [bbb0683]。此非酵素性酸處理 (Non-enzymatic Acidic Treatment) 用於去保護並產生氨基酸甲酯前驅物 (Amino-acid Methyl Ester Precursor)。
> *   **酵素水解 (Enzymatic Hydrolysis)：** 最後的轉化涉及由 chymotrypsin 催化的 L-BPA methyl ester 6 的酵素水解 …（完整內容見 JSON）

---

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.8
- 延遲：598.7s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> 比較框架：
> *   【CMDC-20-e202500059】提供了一種利用二乙醇胺保護 (diethanolamine protection) 和不對稱氫化 (asymmetric hydrogenation) 來生產 L-BPA 的化學合成路徑 (chemical synthetic route)。
> *   【bbb0683】描述了一種混合化學-酶法製程 (hybrid chemo-enzymatic process)，涉及對映選擇性烷基化 (enantioselective alkylation) 隨後進行酶促水解 (enzymatic hydrolysis)，以合成光學純 L-BPA。
> 
> 證據要點：
> *   直接路徑證據表明，使用 Rh 催化劑 (Rh catalysts) 的化學方法可實現高產率（最高達 95%）和 ee 值（結晶後 >96%）【CMDC-20-e202500059】，而混合製程則利用胰凝乳蛋白酶 (chymotrypsin) 催化的酶促水解 (enzymatic hydrolysis)，從甲酯 (methyl esters) 中獲取光學純 L-BPA 【bbb0683】。
> *   文獻集中的策略比較強調，在丙二酸酯化 (malonation) 後引入硼酸 (boronic acid) 可透過消除保護基來縮短合成路徑，這與需要二乙醇胺保護 (diethanolamine protection) 及隨後脫保護步驟的路徑形成對比 【CMDC-20-e202500059】。
> 
> 核心權衡：現有數據優先考慮透過酶促或不對稱催化步驟實現的對映純度 (enantiopurity)，而非透過減少步驟策略實現的合成效率 (synthetic efficiency)，儘管在提供的摘錄中未報告同位素富集 (isotopic enrichment) 和成本效益 (cost- …（完整內容見 JSON）

---
