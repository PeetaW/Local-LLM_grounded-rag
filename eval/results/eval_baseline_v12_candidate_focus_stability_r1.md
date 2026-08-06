# Eval Report — `baseline_v12_candidate_focus_stability_r1`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-08-04 00:46
- 題數：5

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 1.0 |
| Correctness judge 覆蓋 | 5/5（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 0.875 |
| Translation judge 覆蓋 | 4/5（N/A 1） |
| 平均 grounding 分數 | 1.0 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 95.0% |
| 平均 Stage 2 evidence 覆蓋率 | 89.3% |
| 平均總延遲 | 457.5s |
| 平均 planning 延遲 | 29.2s |
| 平均 retrieval 延遲 | 6.3s |
| └ Phase A embed/vector/BM25 | 6.2s |
| └ Phase B 子答案生成 | 0.1s |
| 平均 grounding 延遲 | 19.3s |
| └ 其中 NLI | 0.8s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ✅ | Q02 | single_paper | 1.0 | 1.0 | 100.0% | 100.0% | 100.0% | 1.0 | 351.6s | C0/U0 |
| ✅ | Q03 | figure_dependent | 1.0 | 1.0 | 100.0% | 100.0% | 100.0% | 1.0 | 334.1s | C0/U0 |
| ✅ | Q06 | multi_chunk | 1.0 | 1.0 | 100.0% | 80.0% | 100.0% | 1.0 | 406.0s | C0/U0 |
| ⚠️ | Q09 | cross_paper | 1.0 | 0.5 | 100.0% | 100.0% | 57.1% | 1.0 | 1029.2s | C0/U0 |
| ✅ | Q12 | false_premise | 1.0 | N/A | N/A | N/A | N/A | — | 166.7s | C0/U0 |

## 逐題細節

### ✅ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 6/6
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：1.0
- 延遲：351.6s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 方法證據：
> - 鋰化的 (lithiated) (2R)-( - )-2,5-dihydro- 2-isoprop}'1-3,6-dimethoxypyrazine 與 4-bromo- methylbenzeneboronate (2) 反應，以 74% e.e 產出加成物 4 (adduct 4) 【bbb0683】。
> - 將 4 分步處理 (stepwise treatment) 以鹽酸 (hydrochloric acid)，得到 L-BP A methyl ester 6，隨後 hydrolyzed with chymotrypsin (使用 chymotrypsin 水解) 以提供光學純的 L-BPA (optically pure L-BPA) 【bbb0683】。
> - 在 - 78°C 的 THF 中，將鋰化的 3 (lithiated 3) 與溴化物 2 (bromide 2) 進行對映選擇性烷基化 (enantioselective alkylation) 【bbb0683】。
> - 將 4 以 0.1 N 鹽酸 (hydrochloric acid) 處理，得到 D-valine methyl ester (D-Val-OMe) 與具有所需立體化學 (desired stereochemistry) 的氨基酸甲酯 5 (amino acid methyl ester 5) 之等莫量混合物 (equimolar mixture)，隨後透過蒸餾 (distillation) 去除 D-Val-OMe 以獲得粗產物 5 (crude 5)，接著進一步使用 1 N 鹽酸 (hydrochloric acid) 處理以進行去保護 (deprotected)，從 4 得到 L-BPA methyl ester 6，產率為 85% 【bbb0683】。
>
> ---
> 📋  …（完整內容見 JSON）

---

### ✅ Q03 · figure_dependent

**問題**：According to the cryo-EM structure, how does JPH203 bind to and inhibit the LAT1-4F2hc complex?

- detected_paper：`None`
- 選出論文：['s41421-024-00697-6', '1-s2.0-S1347861320300633-main']
- gold_papers：['s41421-024-00697-6']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 6/6
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：1.0
- 延遲：334.1s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - 在我們的結構分析中，我們觀察到 JPH203 結合在傳統的基質結合口袋 (substrate-binding pocket) 內，與先前確定的 LAT1 抑制劑如 Diiodo-Tyr 或 JX-075、JX-078 以及 JX-119 相似 (Fig. 1b, c)13 [Source: s41421-024-00697-6]。
> - 頭部的 $\alpha$-氨基 ($\alpha$-amino group) 和 $\alpha$-羧基 ($\alpha$-carboxyl group) 在穩定分子方面起著關鍵作用，與 LAT1 中 TM1 和 TM6 展開區域 (unwound region) 的主鏈原子形成了氫鍵網絡 (hydrogen bond network) (Fig. 1b) [Source: s41421-024-00697-6]。
> - 此外，JPH203 雙鹵代酪氨酸核心 (bi-halogenated tyrosine core) 上的氯原子與 Tyr259 形成了鹵鍵 (halogen bond) [Source: s41421-024-00697-6]。
> - 此外，尾部基團 (tail group) 的苯環與 Phe252 的側鏈發生了 T 型 $\pi$-$\pi$ 交互作用 (T-shaped $\pi$-$\pi$ interaction) (Fig. 1c) [Source: s41421-024-00697-6]。
> - 疏水性尾部部分 (hydrophobic tail moiety) 緊密地貼合在一個由特定氨基酸構成的疏水口袋 (hydrophobic pocket) 中，這些氨基酸包括：TM1 上的 Ile63、Ile64 和 Ile68；TM3 上的 Ile140、Ile147、Ile148；TM6 上的 …（完整內容見 JSON）

---

### ✅ Q06 · multi_chunk

**問題**：How does poly(vinyl alcohol) enhance the therapeutic effect of 4-L-boronophenylalanine in neutron capture therapy of thoracic tumors? Explain the proposed mechanism with supporting data.

- detected_paper：`None`
- 選出論文：['1-s2.0-S0378517325007926-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'CMDC-20-e202500059']
- gold_papers：['1-s2.0-S0378517325007926-main']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/7
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：80.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：1.0
- 延遲：406.0s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - PVA 透過 boronate esters 與多個 BPA 分子形成複合物 (complex)，保留了針對 LAT1 的關鍵 phenylalanine-like structure（類苯丙氨酸結構）(Fig. 1C)，且該複合物的形成已由 NMR 確認 [Source: 1-s2.0-S0378517325007926-main]。
> - 被稱為 PVA-BPA 的複合物與 LAT1 相互作用，並透過 LAT1-mediated endocytosis（LAT1 介導的胞吞作用）被攝入癌細胞中，且定位於 endo-/lysosomes（內體/溶酶體），從而增加細胞攝取量並延緩 BPA 從 cytosol（細胞質）的流出 (Fig. 1D) [Source: 1-s2.0-S0378517325007926-main]。
> - 與 PVA-BPA complexes 的原型相比，被稱為 PVA-sorbitol-BPA 的優化配方將可能的副作用大幅降低至可忽略的程度 [Source: 1-s2.0-S0378517325007926-main]。
> - 對照組 (cold)、對照組 (hot)、sorbitol-BPA 組以及 PVA-sorbitol-BPA 組在治療日後的中位生存時間及其範圍分別為 23 (range 3–39), 16 (range 16–58), 46 (range 37–82), 以及 85 (62–85 < ) 天 [Source: 1-s2.0-S0378517325007926-main]。
> - 重要的是，intravenously administered (靜脈注射) 的 PVA-sorbitol-BPA 在模擬 malignant pleural mesothelioma (MPM, 惡性胸膜間皮瘤) 的 th …（完整內容見 JSON）

---

### ⚠️ Q09 · cross_paper

**問題**：Several papers target LAT1 for cancer therapy. What different therapeutic strategies targeting LAT1 are described across the literature, and how do they differ in mechanism?

- detected_paper：`None`
- 選出論文：['LAT1 ChemComm 2026', '1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6', 'cas0106-0279']
- gold_papers：['1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026', 's41421-024-00697-6']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/7
- translation fidelity：0.5　raw：3/5　judge：`translation_fidelity_v2`
- translation reason：1 material and 0 minor semantic errors; The source states 'conjugated to', meaning chemically linked, but the target translates this as '結合' (binding/combining), which implies a non-covalent interaction or general association rather than chemical conjugation.
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：57.1%　grounding：1.0
- 延遲：1029.2s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> Comparison scaffold:
> - 策略：【1-s2.0-S1347861320300633-main】報告了 JPH203 單藥治療 (monotherapy)，其透過共孵育 (co-incubation) 與預孵育 (preincubation) 效應抑制 LAT1 功能 [1-s2.0-S1347861320300633-main]。
> - 策略：【1-s2.0-S1936523321002047-main】報告了 JPH203 聯合放射治療 (radiation therapy)，其透過 mTOR 下調 (downregulation) 與增強細胞衰老 (cellular senescence) 使癌細胞產生放射增敏作用 (radiosensitization) [1-s2.0-S1936523321002047-main]。
> - 策略：【LAT1 ChemComm 2026】報告了 LffVLKK-4Phe 自組裝肽 (self-assembling peptide) 治療，其透過多價交互作用 (multivalent interaction) 與抑制 LAT1 介導的氨基酸運輸來抑制增殖 [LAT1 ChemComm 2026]。
> - 策略依據：這可能與報告顯示 JPH203 作為非底物阻斷劑 (non-substrate blocker),14 有關，儘管作者稱其抑制模式為競爭性 (competitive) [1-s2.0-S1347861320300633-main]。
> - 策略依據：我們表明 X 射線照射 (X-irradiation) 增加了 A549 與 MIA Paca-2 細胞中透過 LAT1 的細胞中性氨基酸攝取 (neutral amino acid uptake)，且 LAT1 抑制劑 JPH203 在最低毒性濃度下，顯著增強了癌細胞 …（完整內容見 JSON）

---

### ✅ Q12 · false_premise

**問題**：Since BPA is administered orally in BNCT, what oral bioavailability values are reported in these papers?

- detected_paper：`None`
- 選出論文：['cas0106-0279', 'Chem Biol Drug Des - 2013 - Achilli - Folic Acid‐Conjugated 4‐Amino‐Phenylboronate  a Boron‐Containing Compound Designed']
- gold_papers：[]
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 3/3
- translation fidelity：N/A　raw：N/A　judge：`N/A`
- 論文選擇命中率：N/A　candidate 覆蓋率：N/A　Stage 2 evidence 覆蓋率：N/A　grounding：—
- 延遲：166.7s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **前提更正**：檢索文獻沒有報告可供回答的 `oral bioavailability` 數值，因此不能把問題中的前提視為已成立，也不會臆測數值。
>
> 文獻實際報告的是不同的給藥途徑或 regimen：
>
> - Clinically, a high-dose and longer-infusion regimen (900 mg BPA ⁄kg, 6-h infusion) is reported to extend patients’ survival.(1,30) In addition to the transporter kinetics, expression levels of LAT1 and ATB0,+ in tumor tissues are thought to impact on BPA accumulation [cas0106-0279]

---
