# Eval Report — `stage2_partial_recovery_r5_literal_contract_q01_q04`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-20 19:49
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.875 |
| Correctness judge 覆蓋 | 2/2（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 0.5 |
| Translation judge 覆蓋 | 2/2（N/A 0） |
| 平均 grounding 分數 | 0.969 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 100.0% |
| 平均 Stage 2 evidence 覆蓋率 | 100.0% |
| 平均總延遲 | 783.3s |
| 平均 planning 延遲 | 7.0s |
| 平均 retrieval 延遲 | 4.8s |
| └ Phase A embed/vector/BM25 | 4.7s |
| └ Phase B 子答案生成 | 0.1s |
| 平均 grounding 延遲 | 21.4s |
| └ 其中 NLI | 3.4s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ⚠️ | Q01 | single_paper | 1.0 | 0.5 | 100.0% | 100.0% | 100.0% | 1.0 | 743.9s | C0/U0 |
| ⚠️ | Q04 | table_lookup | 0.75 | 0.5 | 100.0% | 100.0% | 100.0% | 0.938 | 822.7s | C0/U1 |

## 逐題細節

### ⚠️ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 5/5
- translation fidelity：0.5　raw：3/5　judge：`translation_fidelity_v2`
- translation reason：1 material and 0 minor semantic errors; The sentence begins with the number '5' which appears to be a fragment or typo from the source text (likely referencing Figure 5), but it is not translated, retained as English, or explained in the target. The translation starts directly with '5，', leaving this element untranslated and potentially c
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：1.0
- 延遲：743.9s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 使用 JPH203 進行預孵育 (preincubation) 能協同增強對 L-type amino acid transporter 1 (LAT1) 功能的共孵育 (co-incubation) 抑制效果 [1-s2.0-S1347861320300633-main]。在 HT-29 細胞中，預孵育與共孵育組合的 $\text{IC}_{50}$ 值為 $34.2 \pm 3.6\text{ nM}$ [1-s2.0-S1347861320300633-main]。與僅進行共孵育抑制（其 $\text{IC}_{50}$ 值為 $99.2 \pm 11.0\text{ nM}$）相比，這代表效力顯著提高 [1-s2.0-S1347861320300633-main]。JPH203 與 LAT1 交互作用的協同效應程度，其特徵為 $\text{IC}_{50}$ 值降低三倍以及 $\text{K}_i$ 值減少 20% [1-s2.0-S1347861320300633-main]。具體而言，在預孵育加共孵育 (pre-plus co-incubation) 抑制分析中，$\text{K}_i$ 值為 $0.37\text{ mM}$，低於共孵育抑制分析中的 $\text{K}_i$ 值 $0.46\text{ mM}$ [1-s2.0-S1347861320300633-main]。此外，JPH203 對 LAT1 功能的預孵育抑制效果呈濃度與時間依賴性 (concentration- and time-dependent manner) [1-s2.0-S1347861320300633-main]。
>
> - JPH203 在 HT-29 細胞中對 LAT1 功能的預孵育抑制效果。根據結果，其 $\text{IC}_{50}$ 值被確定為 $193 \pm 50\te …（完整內容見 JSON）

---

### ⚠️ Q04 · table_lookup

**問題**：What are the main degradation products and impurities of BPA identified by HPLC, and under which storage conditions do they form?

- detected_paper：`None`
- 選出論文：['1-s2.0-S0731708511003451-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water']
- gold_papers：['1-s2.0-S0731708511003451-main']
- correctness candidate：`answer_for_judge`
- correctness：0.75　raw：4/5　judge：`structured_fact_audit_v1`
- judge reason：covered 6/7; missing F7
- translation fidelity：0.5　raw：3/5　judge：`translation_fidelity_v2`
- translation reason：1 material and 0 minor semantic errors; The source states degradation 'to phenylalanine' (BPA converts into phenylalanine), but the target translates this as 'phenylalanine degradation' (phenylalanine degrades). This reverses the direction of the chemical reaction.
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：0.938
- 延遲：822.7s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> 根據單一來源文件【1-s2.0-S0731708511003451-main】提供的數據，透過 HPLC (高效液相層析) 鑑定出的 BPA 主要降解產物與雜質及其形成條件總結如下：
>
> **降解產物與形成條件 (Degradation Products and Formation Conditions)**
> *   **Tyrosine (酪胺酸)：** 此化合物在 230 nm 下進行定量，其保留時間 (retention time) 為 4.5 min 【1-s2.0-S0731708511003451-main】。BPA 降解為 tyrosine 發生於鹼性 (alkali) 與氧化條件下，特別是在後者（觀察到質量平衡 mass balance 的情況）中發生速度極快 【1-s2.0-S0731708511003451-main】。硼酸 (boronic acids) 在水相過氧化氫 ($\text{H}_2\text{O}_2$) 中的氧化降解促進了 tyrosine 與 boric acid (硼酸) 的釋放 【1-s2.0-S0731708511003451-main】。
> *   **Phenylalanine (苯丙胺酸)：** 此化合物在 11.0 min 洗脫 (elutes)，並在 256 nm 下進行定量 【1-s2.0-S0731708511003451-main】。在 BPA/mannitol (甘露醇) 凍乾藥品中，透過一個 mannitol 介導的途徑 (mannitol-mediated pathway)，會發生緩慢且隨溫度變化的 phenylalanine 降解 【1-s2.0-S0731708511003451-main】。在 40 °C 下存放 6 個月，此過程會產生約 1% 的 phenylalanine (相對於 BPA 質量) 【 …（完整內容見 JSON）

---
