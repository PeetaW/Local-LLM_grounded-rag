# Eval Report — `stage2_query_aware_q01_q03_q04`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-18 02:59
- 題數：3

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.75 |
| Correctness judge 覆蓋 | 3/3（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 0.5 |
| Translation judge 覆蓋 | 3/3（N/A 0） |
| 平均 grounding 分數 | 0.739 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 100.0% |
| 平均 Stage 2 evidence 覆蓋率 | 66.7% |
| 平均總延遲 | 656.7s |
| 平均 planning 延遲 | 6.5s |
| 平均 retrieval 延遲 | 4.8s |
| └ Phase A embed/vector/BM25 | 4.8s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 17.4s |
| └ 其中 NLI | 3.5s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ⚠️ | Q01 | single_paper | 1.0 | 0.5 | 100.0% | 100.0% | 100.0% | 0.75 | 734.4s | C0/U3 |
| ⚠️ | Q03 | figure_dependent | 1.0 | 0.5 | 100.0% | 100.0% | 60.0% | 0.867 | 647.2s | C0/U2 |
| ❌ | Q04 | table_lookup | 0.25 | 0.5 | 100.0% | 100.0% | 40.0% | 0.6 | 588.4s | C0/U4 |

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
- translation reason：1 material and 0 minor semantic errors; The source text specifies the Ki value as '0.37 mM' and '0.46 mM'. The target translation retains these values with the unit 'mM', which is likely a typo in the original English source (as IC50 values are in nM, Ki should logically be in µM or similar for consistency), but strictly following the ins
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：0.75
- 延遲：734.4s　問題標記：{'conflicts': 0, 'unsupported': 3}

**答案預覽**：

> JPH203 的預孵育 (preincubation) 與僅進行同時孵育 (co-incubation) 相比，顯著增強了其對 LAT1 功能的抑制效能 【1-s2.0-S1347861320300633-main】。在 HT-29 細胞 (HT-29 cells) 中，同時孵育抑制的 IC50 值為 $99.2 \pm 11.0$ nM，而預孵育與同時孵育的組合則導致較低的 IC50 值，$34.2 \pm 3.6$ nM 【1-s2.0-S1347861320300633-main】。這種協同效應 (cooperative effect) 的特徵是 IC50 值降低了三倍，且 Ki 值降低了 20% 【1-s2.0-S1347861320300633-main】。具體而言，預孵育加同時孵育 (pre-plus co-incubation) 抑制分析的 Ki 值報告為 0.37 mM，低於在同時孵育抑制分析中觀察到的 0.46 mM 【1-s2.0-S1347861320300633-main】。
>
> 僅進行預孵育 (preincubation) 也表現出抑制能力，在 HT-29 細胞 (HT-29 cells) 的 LAT1 功能中 IC50 值為 $193 \pm 50$ nM 【1-s2.0-S1347861320300633-main】。預孵育期間的抑制程度具有時間依賴性 (time-dependent)；使用 10 mM JPH203 時，亮氨酸 (leucine) 攝取活性在 30 min 時降低至約 42% 【1-s2.0-S1347861320300633-main】。在預孵育 60 min 後，活性進一步降低至對照組水平 (control levels) 的 32% 【1-s2.0-S1347861320300633-main】。到預孵育 120 min  …（完整內容見 JSON）

---

### ⚠️ Q03 · figure_dependent

**問題**：According to the cryo-EM structure, how does JPH203 bind to and inhibit the LAT1-4F2hc complex?

- detected_paper：`None`
- 選出論文：['s41421-024-00697-6', '1-s2.0-S1347861320300633-main']
- gold_papers：['s41421-024-00697-6']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 6/6
- translation fidelity：0.5　raw：3/5　judge：`translation_fidelity_v2`
- translation reason：1 material and 0 minor semantic errors; The source states that binding affinity 'decreases' (meaning the Kd value increases, indicating weaker binding) in specific mutations. The target translation uses '有所下降', which implies a decrease in magnitude/value. In scientific contexts regarding dissociation constants ($K_d$), an increase in $K_d
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：60.0%　grounding：0.867
- 延遲：647.2s　問題標記：{'conflicts': 0, 'unsupported': 2}

**答案預覽**：

> ### 結構解析與配體組成
> 與 JPH203 結合的 LAT1–4F2hc 複合體的向外面向 (outward-facing) 結構以 3.30 Å 的分辨率被揭示，且透過聚焦精修 (focused refinement) 將 JPH203 的密度提升至 3.25 Å 【s41421-024-00697-6】。在結構上，JPH203 由兩個部分組成：一個具有苯丙氨酸 (phenylalanine) 骨架的親水性頭部基團 (hydrophilic head group)，以及一個具有 5-amino-2-(3-aminophenyl) benzoxazole 骨架的疏水性尾部基團 (hydrophobic tail group) 【s41421-024-00697-6】。JPH203 結合在 LAT1 的傳統底物結合口袋 (substrate-binding pocket) 內，與抑制劑 Diiodo-Tyr、JX-075、JX-078 及 JX-119 相似 【s41421-024-00697-6】。
>
> ### 結合交互作用
> JPH203 頭部基團的 $\alpha$-氨基 ($\alpha$-amino group) 與 $\alpha$-羧基 ($\alpha$-carboxyl group) 與 LAT1 中 TM1 和 TM6 展開區域 (unwound region) 的主鏈原子形成氫鍵網絡 【s41421-024-00697-6】。JPH203 尾部的氨基與 TM10 上 Phe400 的主鏈形成氫鍵 【s41421-024-00697-6】。JPH203 尾部基團的苯環與 Phe252 的側鏈產生 T 型 $\pi$–$\pi$ 交互作用 ($\pi$–$\pi$ interaction) 【s41421-024-00697-6】。在 Ile63、Ile64、 …（完整內容見 JSON）

---

### ❌ Q04 · table_lookup

**問題**：What are the main degradation products and impurities of BPA identified by HPLC, and under which storage conditions do they form?

- detected_paper：`None`
- 選出論文：['1-s2.0-S0731708511003451-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water']
- gold_papers：['1-s2.0-S0731708511003451-main']
- correctness candidate：`answer_for_judge`
- correctness：0.25　raw：2/5　judge：`structured_fact_audit_v1`
- judge reason：covered 3/7; missing F1, F2, F5, F6
- translation fidelity：0.5　raw：3/5　judge：`translation_fidelity_v2`
- translation reason：1 material and 0 minor semantic errors; The source states degradation 'to phenylalanine' (BPA converts into phenylalanine), but the target translates this as 'phenylalanine 降解' (degradation of phenylalanine), implying phenylalanine is degrading rather than being formed.
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：40.0%　grounding：0.6
- 延遲：588.4s　問題標記：{'conflicts': 0, 'unsupported': 4}

**答案預覽**：

> 先前研究人員報導 Tyrosine (酪胺酸) 和 phenylalanine (苯丙胺酸) 為 boron phenylalanine (BPA, 硼苯丙胺酸) 的雜質 [Chunk 1], [Chunk 2]。在鹼性 (alkali) 和氧化條件 (oxidative conditions) 下，觀察到 BPA 降解為 Tyrosine [Chunk 1]。在氧化條件 (oxidative conditions) 下，BPA 降解為 Tyrosine 的速度極其迅速 [Chunk 1]。相比之下，一種 BPA/mannitol (甘露醇) 凍乾藥品 (lyophilised drug product) 顯示出緩慢且具有溫度依賴性的 phenylalanine 降解 [Chunk 1]。在 $40^\circ\text{C}$ 下存放 6 個月，BPA/mannitol 凍乾藥品產生了約 1% 的 phenylalanine (相對於 BPA 質量) [Chunk 1]。當儲存時間較長時，凍乾的 BPA/mannitol 樣本會部分降解為 phenylalanine [Chunk 1], [Chunk 2]。凍乾 BPA/mannitol 的降解被認為與關於在 fructose (果糖) 中儲存的文獻先例 (literature precedent) 一致 [Chunk 1], [Chunk 2]。在脫水條件 (dehydrating conditions) 下，BPA 容易與 fructose 形成熱力學上較有利的 (thermodynamically favoured) borate esters (硼酸酯) [Chunk 1], [Chunk 2]。就 mannitol (一種 polyol 多元醇或 reduced sugar 還原糖) 而言，游離羥基位點 (fr …（完整內容見 JSON）

---
