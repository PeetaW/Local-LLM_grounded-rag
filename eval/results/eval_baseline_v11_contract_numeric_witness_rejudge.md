# Eval Report — `baseline_v11_contract_numeric_witness_rejudge`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-08-03 08:35
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 1.0 |
| Correctness judge 覆蓋 | 2/2（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 0.75 |
| Translation judge 覆蓋 | 1/2（N/A 1） |
| 平均 grounding 分數 | 1.0 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 100.0% |
| 平均 Stage 2 evidence 覆蓋率 | 100.0% |
| 平均總延遲 | 290.9s |
| 平均 planning 延遲 | 29.5s |
| 平均 retrieval 延遲 | 7.0s |
| └ Phase A embed/vector/BM25 | 7.0s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 12.9s |
| └ 其中 NLI | 0.0s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ✅ | Q02 | single_paper | 1.0 | 0.75 | 100.0% | 100.0% | 100.0% | 1.0 | 425.0s | C0/U0 |
| ✅ | Q12 | false_premise | 1.0 | N/A | N/A | N/A | N/A | — | 156.8s | C0/U0 |

## 逐題細節

### ✅ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 6/6
- translation fidelity：0.75　raw：4/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 1 minor semantic errors; Target contains an unresolved encoding artifact: <0xE7><0xBA><0x88>
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：1.0
- 延遲：425.0s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 方法依據：
> - 鋰化的 (lithiated) (2R)-( - )-2,5-dihydro- 2-isoprop}'1-3,6-dimethoxypyrazine 與 4-溴-甲基苯硼酸酯 (4-bromo-methylbenzeneboronate) (2) 反應，以 74% e.e 得到加成物 4 (adduct 4) 【bbb0683】。
> - 將 4 分步處理 (stepwise treatment) 於鹽酸 (hydrochloric acid)，得到 L-BPA 甲基酯 (L-BPA methyl ester) 6，隨後使用 chymotrypsin 水解以獲得光學純的 L-BPA 【bbb0683】。
> - 在 - 78°C 的 THF 中，將鋰化的 3 (lithiated 3) 與溴化物 2 (bromide 2) 進行對映選擇性烷基化 (enantioselective alkylation) 【bbb0683】。
> - 使用 0.1 N 鹽酸 (hydrochloric acid) 處理 4，得到 D-<0xE7><0xBA><0x88>胺酸甲基酯 (D-valine methyl ester, D-Val-OMe) 與具有所需立體化學 (desired stereochemistry) 的胺基酸甲基酯 5 (amino acid methyl ester 5) 的等莫量混合物 (equimolar mixture)，隨後通過蒸餾 (distillation) 除去 D-Val-OMe 以獲得粗產物 5 (crude 5)，接著進一步使用 1 N 鹽酸 (hydrochloric acid) 處理以去保護 (deprotected)，從 4 得到 L-BPA 甲基酯 (L-BPA methyl ester) 6，產率為 85% 【bbb0683】。
>
> -- …（完整內容見 JSON）

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
- 延遲：156.8s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **前提更正**：檢索文獻沒有報告可供回答的 `oral bioavailability` 數值，因此不能把問題中的前提視為已成立，也不會臆測數值。
>
> 文獻實際報告的是不同的給藥途徑或 regimen：
>
> - Clinically, a high-dose and longer-infusion regimen (900 mg BPA ⁄kg, 6-h infusion) is reported to extend patients’ survival.(1,30) In addition to the transporter kinetics, expression levels of LAT1 and ATB0,+ in tumor tissues are thought to impact on BPA accumulation [cas0106-0279]

---
