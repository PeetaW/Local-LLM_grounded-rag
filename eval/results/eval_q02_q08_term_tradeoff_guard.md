# Eval Report — `q02_q08_term_tradeoff_guard`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-02 01:12
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.75 |
| 平均 grounding 分數 | 0.831 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 1406.0s |
| 平均 planning 延遲 | 8.3s |
| 平均 retrieval 延遲 | 586.8s |
| 平均 grounding 延遲 | 56.7s |
| └ 其中 NLI | 35.2s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q02 | single_paper | 100.0% | 100.0% | 0.8 | 1146.0s | C0/U10 |
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 0.862 | 1666.0s | C0/U7 |

## 逐題細節

### ✅ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.8
- 延遲：1146.0s　問題標記：{'conflicts': 0, 'unsupported': 10}

**答案預覽**：

> ## 【論文直接依據】
>
> **通用流程與目的**
> * 合成光學純 L-p-硼苯丙氨酸 (L-p-boronophenylalanine, L-BPA) 的目的是將其用作硼中子捕獲治療 (boron neutron capture therapy, BNCT) 中的腫瘤靶向硼載體 (bbb0683)。
> * 用於 L-BPA 不對稱合成的混合製程結合了「對映選擇性烷基化 (enantioselective alkylation)」以及隨後的「酶促水解 (enzymatic hydrolysis)」(bbb0683)。
> * 與其他已報導的方法相比，該混合製程的特點是反應步驟較少，操作和後處理 (workup) 更簡單 (bbb0683)。
>
> **步驟 1：合成保護的硼酸衍生物（化合物 2）**
> * 合成涉及 4-溴甲基苯硼酸 (4-bromomethylbenzeneboronic acid, 1) 與 2,2-二甲基-1,3-丙二醇 (2,2-dimethyl-1,3-propanediol) 的反應 (bbb0683)。
> * 反應在室溫 (ambient temperature) 下的乾燥 THF (dry THF) 中進行 (bbb0683)。
> * 混合物攪拌 10 分鐘 (bbb0683)。
> * 此過程將二羥基硼基 (dihydroxyboryl group) 保護為環狀硼酸酯 (cyclic borinate) (bbb0683)。
> * 此步驟的產率為 79% (bbb0683)。
> * 使用的具體數量為：5.99 g (27.9 mmol) 的化合物 1 和 2.91 g (27.9 mmol) 的 2,2-二甲基-1,3-丙二醇，溶於 30 ml 的乾燥 THF 中 (bbb0683)。
>
> **步驟 2：對映選擇性烷基化（合成化合物 4）**
> * 此步驟涉及溴化物 2 …（完整內容見 JSON）

---

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.862
- 延遲：1666.0s　問題標記：{'conflicts': 0, 'unsupported': 7}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
>
> 4-硼代-L-苯丙氨酸 (4-borono-L-phenylalanine, L-BPA) 的合成路徑在手性 (chirality) 和硼引入 (boron introduction) 的方法上存在顯著差異，且高度強調同位素富集 (isotopic enrichment) 的經濟影響以及臨床使用的純度要求。
>
> ## 【論文直接依據】
>
> **同位素富集與經濟限制**
> * $^{10}\text{B}$ 通常透過化學交換蒸餾 (chemical exchange distillation) 從 $^{11}\text{B}$ 中分離 (CMDC-20-e202500059)。
> * 分離出的 $^{10}\text{B}$ 隨後被轉化為硼酸酯 (boronic esters) 或水解為 $^{10}\text{B}$ 硼酸 (boric acid) (CMDC-20-e202500059)。
> * $^{10}\text{B}$ 的成本比普通硼酸高出 1000 倍以上，這主導了合成的經濟效益 (CMDC-20-e202500059)。
> * 依賴 $\text{B}_2\text{pin}_2$ 的合成路徑並不流行，因為該試劑導致 $^{10}\text{B}$ 原子的浪費成本過高 (CMDC-20-e202500059)。
>
> **非 Pd 催化路徑（側重於成本與產率）**
> * 在使用 n-BuLi 和 $\text{B}(\text{OBu})_3$ 對 Boc 保護的 4-碘苯丙氨酸 (4-iodophenylalanine) 進行處理的非 Pd 路徑中，兩步總產率為 $53\text{--}58\%$ (CMDC-20-e202500059)。
> * 使用 $^{10}\text{ …（完整內容見 JSON）

---
