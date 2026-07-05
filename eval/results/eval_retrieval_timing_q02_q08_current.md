# Eval Report — `retrieval_timing_q02_q08_current`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-05 22:04
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.625 |
| 平均 grounding 分數 | 0.875 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 1304.0s |
| 平均 planning 延遲 | 8.1s |
| 平均 retrieval 延遲 | 807.1s |
| └ Phase A embed/vector/BM25 | 7.6s |
| └ Phase B 子答案生成 | 799.5s |
| 平均 grounding 延遲 | 16.6s |
| └ 其中 NLI | 4.5s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q02 | single_paper | 100.0% | 100.0% | 1.0 | 928.0s | C0/U0 |
| ⚠️ | Q08 | cross_paper | 100.0% | 100.0% | 0.75 | 1680.0s | C0/U1 |

## 逐題細節

### ✅ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：928.0s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 用於光學純 L-p-boronophenylalanine (L-BPA) 不對稱合成 (asymmetric synthesis) 的混合製程，旨在作為硼中子捕獲治療 (Boron Neutron Capture Therapy (BNCT)) 的腫瘤靶向硼載體 [bbb0683]。此方法將對映選擇性烷基化 (enantioselective alkylation) 與隨後的酶促水解 (enzymatic hydrolysis) 相結合，以實現高光學純度，而無需傳統方法中所需的額外分離步驟 (resolution steps) [bbb0683]。
> 
> ### 混合製程的關鍵步驟 (Key Steps of the Hybrid Process)
> 核心合成過程分為三個定義階段，不包括初始起始原料的保護：
> 
> 1.  **對映選擇性烷基化 (Enantioselective Alkylation)：** 此步驟使用手性輔助劑 (chiral auxiliary)，具體為 (2R)-(- )-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine (bislactimether) 以及 n-BuLi 與受保護的硼酸衍生物 (protected boronic acid derivative) 進行反應以建立立體中心 (stereocentre) [bbb0683]。根據研究中不同章節引用的評估方法，該反應產生的加成物 (adduct) 之非對映體過量 (diastereomeric excess) 為 72% 或對映體過量 (enantiomeric excess) 為 74% [bbb0683]。
> 2.  **酸性水解與去保護 (Acidic Hydrolysis and Deprotection)：** 使用鹽酸 (hydrochloric …（完整內容見 JSON）

---

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.75
- 延遲：1680.0s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> 比較框架：
> - 【CMDC-20-e202500059】作為一項評論/比較來源，評估了影響可擴展性 (scalability) 的合成策略 (synthesis strategies)、同位素富集經濟性 (isotopic enrichment economics) 以及安全風險 (safety risks) 【CMDC-20-e202500059】。
> - 【bbb0683】提出了一種針對 L-BPA 的特定雜合化學-酶法路徑 (hybrid chemo-enzymatic route)，重點在於對映選擇性烷基化 (enantioselective alkylation) 和酶促水解步驟 (enzymatic hydrolysis steps) 【bbb0683】。
> 
> 與外消旋拆分 (racemic resolution) 或不對稱氫化法 (asymmetric hydrogenation methods) 相比，該雜合製程 (hybrid process) 以較少的操作步驟實現了高光學純度 (optical purity)，在後處理簡便性 (workup simplicity) 方面具有優勢 【bbb0683】【CMDC-20-e202500059】。
> 由於硼化路徑 (borylation routes) 中的前驅體成本 (precursor costs) 和廢物考量 (waste concerns)，同位素富集 (isotopic enrichment) 帶來了顯著的經濟限制 (economic constraints)，而來自氧化劑 (oxidants) 或重金屬 (heavy metals) 的安全風險則限制了可擴展製造的選擇 【CMDC-20-e202500059】。
> 
> 核心權衡 (Central trade-off)：其核心權衡在於同位素富集材料的高昂成本 (p …（完整內容見 JSON）

---
