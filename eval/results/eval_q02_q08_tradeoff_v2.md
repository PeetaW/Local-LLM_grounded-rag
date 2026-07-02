# Eval Report — `q02_q08_tradeoff_v2`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-02 20:12
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.75 |
| 平均 grounding 分數 | 0.674 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 1078.4s |
| 平均 planning 延遲 | 7.4s |
| 平均 retrieval 延遲 | 341.3s |
| 平均 grounding 延遲 | 47.6s |
| └ 其中 NLI | 24.4s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q02 | single_paper | 100.0% | 100.0% | 0.766 | 1259.1s | C0/U13 |
| ⚠️ | Q08 | cross_paper | 100.0% | 100.0% | 0.583 | 897.6s | C0/U6 |

## 逐題細節

### ⚠️ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.766
- 延遲：1259.1s　問題標記：{'conflicts': 0, 'unsupported': 13}

**答案預覽**：

> ## 【論文直接依據】
>
> **總體流程與目標**
> * 研究目標是探索一種更高效的對映純 L-p-硼苯丙氨酸 (L-p-boronophenylalanine, L-BPA) 合成方法，它是硼中子捕獲治療 (boron neutron capture therapy, BNCT) 中最有效的硼載體之一 【bbb0683】。
> * 合成對映純 L-BPA 的混合流程結合了對映選擇性烷基化 (enantioselective alkylation) 與隨後的酶促水解 (enzymatic hydrolysis) 【bbb0683】。
> * 該混合流程的優點包括反應步驟較少，且操作與後處理簡單 【bbb0683】。
> * 此流程將立體純度從加成物 4 (adduct 4, 74% e.e.) 提高到最終的 L-BPA 產物 (100% e.e.)，避免了額外的分離步驟 【bbb0683】。
>
> **步驟 1：化合物 2 (2-(4-Bromomethyl)phenyl-5,5-dimethyl-1,3,2-dioxaborane) 的合成**
> * 使用的試劑為 4-溴甲基苯硼酸 (4-bromomethylbenzeneboronic acid, 1) (5.99 g, 27.9 mmol) 和 2,2-二甲基-1,3-丙二醇 (2,2-dimethyl-1,3-propanediol) (2.91 g, 27.9 mmol) 【bbb0683】。
> * 反應在乾燥 THF (30 ml) 中進行 【bbb0683】。
> * 混合物在室溫下攪拌 10 分鐘 【bbb0683】。
> * 此步驟將二羥基硼基保護為環狀硼酸酯 (cyclic borinate, 2) 【bbb0683】。
> * 此步驟的產率為 79% (6.24 g) 【bbb0683】。
>
> **步驟 2：不對稱烷基化以製備加成物 4  …（完整內容見 JSON）

---

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.583
- 延遲：897.6s　問題標記：{'conflicts': 0, 'unsupported': 6}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
>
> ## 【論文直接依據】
>
> **CMDC-20-e202500059 中回顧的方法**
> * 此回顧比較了合成 $^{10}\text{B}$ 富集 4-Borono-L-phenylalanine (L-BPA) 的各種方法 (CMDC-20-e202500059)。
> * 評估重點在於可擴展性 (scalability)、成本效益 (cost-effectiveness) 和安全性方面的限制 (CMDC-20-e202500059)。
> * 此回顧考慮了 $^{10}\text{B}$ 同位素富集的高成本 (CMDC-20-e202500059)。
> * 這是首篇整合並比較學術界與工業界合成策略的全面回顧，包括專利和經過同行評審的方法 (peer-reviewed methods) (CMDC-20-e202500059)。
>
> **bbb0683 中的混合製程 (hybrid process)**
> * 此路徑使用結合對映選擇性烷基化 (enantioselective alkylation) 和酶水解 (enzymatic hydrolysis) 的「混合製程」(hybrid process) 來合成光學純的 L-p-boronophenylalanine (L-BPA) (bbb0683)。
> * 該方法利用 (2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine 作為手性輔助劑 (chiral auxiliary)，在 -78°C 的 THF 中與 4-bromomethylbenzeneboronate 反應 (bbb0683)。
> * 它在 pH 5.0 時使用 chymotrypsin 對 L-BPA methyl est …（完整內容見 JSON）

---
