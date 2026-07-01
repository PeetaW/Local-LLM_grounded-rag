# Eval Report — `term_guard_q02_q08`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-01 08:20
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.75 |
| 平均 grounding 分數 | 0.718 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 989.0s |
| 平均 planning 延遲 | 7.3s |
| 平均 retrieval 延遲 | 330.4s |
| 平均 grounding 延遲 | 41.5s |
| └ 其中 NLI | 21.1s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q02 | single_paper | 100.0% | 100.0% | 0.735 | 1143.8s | C0/U10 |
| ⚠️ | Q08 | cross_paper | 100.0% | 100.0% | 0.7 | 834.1s | C0/U5 |

## 逐題細節

### ⚠️ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.735
- 延遲：1143.8s　問題標記：{'conflicts': 0, 'unsupported': 10}

**答案預覽**：

> ## 【論文直接依據】
>
> **一般流程與目的 (General Process and Purpose)**
> * 本研究的目的在於探索一種合成光學純 L-p-boronophenylalanine (L-BPA) 的高效方法 (bbb0683)。
> * L-BPA 是硼中子捕獲治療 (boron neutron capture therapy, BNCT) 中最有效的腫瘤靶向硼載體 (tumor targeting boron carriers) 之一 (bbb0683)。
> * 合成光學純 L-BPA 的混合製程 (hybrid process) 將「對稱烷基化」(symmetric alkylation) 或「對映選擇性烷基化」(enantioselective alkylation) 與隨後的「酶水解」(enzymatic hydrolysis) 相結合 (bbb0683)。
> * 與使用 $\alpha$-benzamidocinnamic acid 衍生物作為底物或需要額外分離純化的方法相比，此混合製程的反應步驟較少 (bbb0683)。
> * 與上述替代方法相比，此混合製程的操作和後處理 (workup) 更簡單 (bbb0683)。
>
> **化合物 2 的合成 (Synthesis of Compound 2)**
> * 合成始於商業可得的 4-bromotoluene，用以製備 4-bromomethylbenzeneboronic acid (1) (bbb0683)。
> * 化合物 1 的二羥基硼基 (dihydroxyboryl group) 在室溫下使用 2,2-dimethyl-1,3-propanediol 在乾燥的 THF (dry THF) 中保護 10 分鐘，以產生環狀硼酸酯 (cyclic borinate, compound 2) (bbb0683) …（完整內容見 JSON）

---

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.7
- 延遲：834.1s　問題標記：{'conflicts': 0, 'unsupported': 5}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
>
> 根據提供的文本，關於 4-borono-L-phenylalanine (L-BPA) 合成路徑在同位素富集 (isotopic enrichment)、可擴展性 (scalability) 和成本效益 (cost-effectiveness) 方面的比較如下：
>
> ## 【論文直接依據】
>
> **通用合成概況（學術界 vs. 工業界）**
> * 目前對於 L-BPA 的標準合成方法尚未達成共識 (CMDC-20-e202500059)。
> * 在生產兼具高純度且同位素富集 (isotopically enriched) 的材料方面存在挑戰 (CMDC-20-e202500059)。
> * 存在一篇綜合綜述，比較了學術界與工業界的 L-BPA 合成策略，特別分析了在可擴展性 (scalability)、成本效益 (cost-effectiveness)、安全性以及與 $^{10}\text{B}$ 同位素富集相關成本方面的限制 (CMDC-20-e202500059)。
>
> **混合製程路徑 (bbb0683)**
> * 一種合成光學純 L-BPA 的混合製程 (hybrid process) 將對映選擇性烷基化 (enantioselective alkylation) 與酶水解 (enzymatic hydrolysis) 相結合 (bbb0683)。
> * 此路徑涉及使手性輔助劑 (chiral auxiliary) (2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine 與受保護的 4-bromomethylbenzeneboronic acid 衍生物 (2) 在 THF 中於 -78°C 下反應，以產生對映體過量值 (e.e. …（完整內容見 JSON）

---
