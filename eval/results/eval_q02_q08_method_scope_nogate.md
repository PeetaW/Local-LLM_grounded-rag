# Eval Report — `q02_q08_method_scope_nogate`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-05 20:45
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.75 |
| 平均 grounding 分數 | 0.75 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 997.2s |
| 平均 planning 延遲 | 8.3s |
| 平均 retrieval 延遲 | 482.7s |
| 平均 grounding 延遲 | 15.8s |
| └ 其中 NLI | 3.8s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q02 | single_paper | 100.0% | 100.0% | 0.5 | 980.4s | C0/U1 |
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 1.0 | 1013.9s | C0/U0 |

## 逐題細節

### ⚠️ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.5
- 延遲：980.4s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> 用於光學純 L-p-boronophenylalanine (L-BPA) 不對稱合成的混合製程 (hybrid process) 結合了對映選擇性烷基化 (enantioselective alkylation) 與隨後的酵素水解 (enzymatic hydrolysis) [bbb0683]。此方法旨在生產 L-BPA，這是一種腫瘤靶向氨基酸且為硼中子捕獲治療 (Boron Neutron Capture Therapy, BNCT) 的有效硼載體 (boron carrier)，且不需要額外的拆分步驟 (resolution steps) [bbb0683]。
> 
> **混合製程的關鍵步驟：**
> 
> 1.  **對映選擇性烷基化 (Enantioselective Alkylation)：** 立體中心 (stereocenter) 是透過使用 bislactimether 輔助基 (bislactimether auxiliary)，特別是 (2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine 的對映選擇性烷基化步驟來建立的 [bbb0683]。此反應涉及使用 n-BuLi 對輔助基進行鋰化 (lithiation)，隨後加入受保護的硼酸衍生物 (protected boronic acid derivative) 以產生具有確定立體化學 (stereochemistry) 的加成物 (adduct)（報告為 74% e.e. 或 72% 非對映異構體過量值 diastereomeric excess）[bbb0683]。
> 2.  **酸性水解與去保護 (Acidic Hydrolysis and Deprotection)：** 手性輔助基 (chiral auxiliary) 被移除，且硼保護基被脫除，透過在  …（完整內容見 JSON）

---

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：1013.9s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 比較框架：
> - 【CMDC-20-e202500059】：評估 L-BPA 生產之合成策略、同位素富集成本（isotopic enrichment costs）及催化劑經濟性的評論/比較來源。
> - 【bbb0683】：描述一種涉及對映選擇性烷基化（enantioselective alkylation）與酶促水解（enzymatic hydrolysis）以生產光學純 L-p-boronophenylalanine 的雜合化學-酶法過程（hybrid chemo-enzymatic process）。
> 
> 該雜合過程（hybrid process）利用對映選擇性烷基化（enantioselective alkylation），隨後進行酸性脫保護（acidic deprotection）以及由 chymotrypsin 催化的酶促水解（enzymatic hydrolysis）；與使用手性膦（chiral phosphines）進行氫化（hydrogenation）的方法相比，其步驟較少且操作更簡便 【bbb0683】。
> 一般合成策略強調，由於 10B 富集的成本高昂，盡量減少含硼步驟的數量至關重要；而經濟型路徑則傾向於使用如 i-PrMgCl 的格氏試劑（Grignard reagents）來取代昂貴的 Pd 或 Rh 催化劑 【CMDC-20-e202500059】。
> 核心權衡（Central trade-off）：為了臨床使用而實現高對映純度（enantiopurity）和同位素富集，通常需要昂貴的貴金屬催化劑或複雜的保護方案（protection schemes）；而具成本效益的可擴展性則優先考慮減少硼源步驟並利用較便宜的有機金屬試劑（organometallic reagents），儘管在光學純度控制方面可能存在挑戰。
> 
> ---
> 📋 **答案品質報告**
> 
> ✅ **整 …（完整內容見 JSON）

---
