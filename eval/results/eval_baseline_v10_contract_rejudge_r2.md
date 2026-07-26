# Eval Report — `baseline_v10_contract_rejudge_r2`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-27 02:52
- 題數：3

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.917 |
| Correctness judge 覆蓋 | 3/3（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 0.75 |
| Translation judge 覆蓋 | 2/3（N/A 1） |
| 平均 grounding 分數 | 0.952 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 93.3% |
| 平均 Stage 2 evidence 覆蓋率 | 85.7% |
| 平均總延遲 | 596.7s |
| 平均 planning 延遲 | 28.5s |
| 平均 retrieval 延遲 | 5.8s |
| └ Phase A embed/vector/BM25 | 5.7s |
| └ Phase B 子答案生成 | 0.1s |
| 平均 grounding 延遲 | 36.3s |
| └ 其中 NLI | 0.8s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ⚠️ | Q02 | single_paper | 1.0 | 0.5 | 100.0% | 100.0% | 100.0% | 1.0 | 527.3s | C0/U0 |
| ⚠️ | Q06 | multi_chunk | 1.0 | N/A | 100.0% | 80.0% | 100.0% | 1.0 | 420.5s | C0/U0 |
| ⚠️ | Q09 | cross_paper | 0.75 | 1.0 | 100.0% | 100.0% | 57.1% | 0.857 | 842.4s | C0/U1 |

## 逐題細節

### ⚠️ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 6/6
- translation fidelity：0.5　raw：3/5　judge：`translation_fidelity_v2`
- translation reason：1 material and 0 minor semantic errors; The target translation fails to translate or represent the chemical reaction scheme diagram (the ASCII art showing reagents and structures) present in S8, reducing a complex visual data element to just text description of reactants.
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：1.0
- 延遲：527.3s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> Method evidence:
> - 鋰化的 (Lithiated) (2R)-( - )-2,5-dihydro- 2-isoprop}'1-3,6-dimethoxypyrazine 與 4-bromo-methylbenzeneboronate (2) 反應，以 74% e.e. 產出加成物 (adduct) 4 【bbb0683】。
> - 將 4 以鹽酸 (hydrochloric acid) 逐步處理得到 L-BP A methyl ester 6，隨後使用 chymotrypsin 進行水解以製得光學純的 (optically pure) L-BPA 【bbb0683】。
> - 將 4 以 0.1 N 鹽酸 (hydrochloric acid) 處理，得到 D-valine methyl ester (D-Val-OMe) 與具有所需立體化學 (stereochemistry) 的 amino acid methyl ester 5 之等莫量混合物 (equimolar mixture)，隨後透過蒸餾 (distillation) 去除 D-Val-OMe 以獲得粗產物 (crude) 5，接著進一步以 1 N 鹽酸 (hydrochloric acid) 處理進行脫保護 (deprotected)，從 4 出發以 85% 的產率得到 L-BPA methyl ester 6 【bbb0683】。
> - 另一方面，在 pH 5.0 下使用 chymo- trypsin 5 ) 對 6 進行酶水解 (enzymatic hydrolysis)，以 79% 的產率製得光學純的 L-BPA，mp 275-283-'C (decomp.)，[(X]~8 - 8.6 (c 0.49, 0.1 N HCI) [lit.,41 [tx]53 - 8.2( (c 0.7, 0.1  …（完整內容見 JSON）

---

### ⚠️ Q06 · multi_chunk

**問題**：How does poly(vinyl alcohol) enhance the therapeutic effect of 4-L-boronophenylalanine in neutron capture therapy of thoracic tumors? Explain the proposed mechanism with supporting data.

- detected_paper：`None`
- 選出論文：['1-s2.0-S0378517325007926-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'CMDC-20-e202500059']
- gold_papers：['1-s2.0-S0378517325007926-main']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/7
- translation fidelity：N/A　raw：N/A　judge：`translation_fidelity_v2`
- translation reason：invalid translation audit: top-level 'errors' must be a list
- 論文選擇命中率：100.0%　candidate 覆蓋率：80.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：1.0
- 延遲：420.5s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - 在此，我們報告由聚乙烯醇 (poly(vinyl alcohol) (PVA)) 與 4-L-硼苯丙氨酸 (4-L-boronophenylalanine (BPA))（BNCT 中唯一經臨床批准的藥物）組成的複合物，對小鼠體內模擬難治性惡性胸膜間皮瘤 (malignant pleural mesothelioma (MPM)) 的胸腔腫瘤可顯示出極高的治療效率 [Source: 1-s2.0-S0378517325007926-main]。
> - 系統性給藥的 PVA-sorbitol-BPA 在目標腫瘤內顯示出高效的累積與長期滯留 (long-term retention) [Source: 1-s2.0-S0378517325007926-main]。
> - 與傳統 BPA 相比，使用 PVA-sorbitol-BPA 的 BNCT 顯著提高了生存率 [Source: 1-s2.0-S0378517325007926-main]。
> - 在這方面，我們之前報告了一項使用聚乙烯醇 (poly(vinyl alcohol) (PVA)) 來延長 BPA 在腫瘤內滯留時間 (intratumoral retention time) 的簡單技術 (Nomoto et al., 2020) [Source: 1-s2.0-S0378517325007926-main]。
> - PVA 通過硼酸酯 (boronate esters) 與多個 BPA 分子形成複合物，保留了用於靶向 LAT1 的必需類苯丙氨酸結構 (phenylalanine-like structure)（圖 1C），且該複合物的形成已通過 NMR 確認 [Source: 1-s2.0-S0378517325007926-main]。
> - 被稱為 PVA-BPA 的複合物與 LAT1 相互作 …（完整內容見 JSON）

---

### ⚠️ Q09 · cross_paper

**問題**：Several papers target LAT1 for cancer therapy. What different therapeutic strategies targeting LAT1 are described across the literature, and how do they differ in mechanism?

- detected_paper：`None`
- 選出論文：['LAT1 ChemComm 2026', '1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6', 'cas0106-0279']
- gold_papers：['1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026', 's41421-024-00697-6']
- correctness candidate：`answer_for_judge`
- correctness：0.75　raw：4/5　judge：`structured_fact_audit_v1`
- judge reason：covered 6/7; missing F4
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：57.1%　grounding：0.857
- 延遲：842.4s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> Comparison scaffold:
> - 策略 (Strategy)：【1-s2.0-S1936523321002047-main】報告了 JPH203 與放射治療 (radiation therapy) 的聯合應用，透過 mTOR 下調 (downregulation) 和細胞衰老 (cellular senescence) 增強放射敏感性 [1-s2.0-S1936523321002047-main]。
> - 策略 (Strategy)：【LAT1 ChemComm 2026】報告了靶向 LAT1 的自組裝肽 (self-assembling peptide) LffVLKK-4Phe，透過多價交互作用 (multivalent interaction) 及隨後的膜破壞 (membrane disruption) 抑制癌細胞增殖 [LAT1 ChemComm 2026]。
> - 策略 (Strategy)：【cas0106-0279】報告了用於 BNCT 的 p-硼苯丙氨酸 (p-Boronophenylalanine, BPA) 的選擇性遞送，使 10B 定位於惡性細胞，以便在中子照射 (neutron irradiation) 時發生核捕獲與裂變反應 [cas0106-0279]。
> - 機制 (Mechanism)：【s41421-024-00697-6】報告 JPH203 的結合由氫鍵網絡 (hydrogen bond network)、與 Tyr259 的鹵鍵 (halogen bond) 以及尾部基團 (tail moiety) 的疏水交互作用 (hydrophobic interactions) 所穩定 [s41421-024-00697-6]。
> - 機制 (Mechanism)：【1-s2.0-S1347861320300633-main】報告 JPH20 …（完整內容見 JSON）

---
