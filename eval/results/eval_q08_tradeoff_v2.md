# Eval Report — `q08_tradeoff_v2`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-02 02:29
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.5 |
| 平均 grounding 分數 | 0.765 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 856.4s |
| 平均 planning 延遲 | 8.1s |
| 平均 retrieval 延遲 | 259.2s |
| 平均 grounding 延遲 | 54.1s |
| └ 其中 NLI | 31.5s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q08 | cross_paper | 100.0% | 100.0% | 0.765 | 856.4s | C0/U6 |

## 逐題細節

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.765
- 延遲：856.4s　問題標記：{'conflicts': 0, 'unsupported': 6}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
>
> ## 【論文直接依據】
>
> **一般合成概況（學術與工業）**
> * 目前尚無 L-BPA 合成的共識方法 (CMDC-20-e202500059)。
> * 在生產高純度材料方面存在挑戰 (CMDC-20-e202500059)。
> * 在生產同位素富集 ($^{10}\text{B}$) 材料方面存在挑戰 (CMDC-20-e202500059)。
> * 學術與工業策略在可擴展性 (scalability) 方面存在限制 (CMDC-20-e202500059)。
> * 學術與工業策略在成本效益 (cost-effectiveness) 方面存在限制 (CMDC-20-e202500059)。
> * 學術與工業策略在安全性方面存在限制 (CMDC-20-e202500059)。
>
> **「混合製程」(Hybrid Process) (bbb0683)**
> * 該混合製程 (hybrid process) 由兩個步驟組成 (bbb0683)。
> * 第一步涉及使用手性輔助劑 (chiral auxiliary) (2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine 和受保護的 4-bromomethylbenzeneboronic acid 衍生物進行對映選擇性烷基化 (enantioselective alkylation) (bbb0683)。
> * 第二步利用 chymotrypsin 進行酶水解 (enzymatic hydrolysis)，將中間體 L-BPA methyl ester (6) 轉化為 L-BPA (bbb0683)。
> * 此製程實現了 100% e.e. 的光學純度 (optical purity) (bbb06 …（完整內容見 JSON）

---
