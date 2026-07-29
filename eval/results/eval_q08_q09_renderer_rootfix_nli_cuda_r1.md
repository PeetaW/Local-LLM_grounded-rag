# Eval Report — `q08_q09_renderer_rootfix_nli_cuda_r1`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-29 01:19
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 1.0 |
| Correctness judge 覆蓋 | 2/2（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 0.625 |
| Translation judge 覆蓋 | 2/2（N/A 0） |
| 平均 grounding 分數 | 0.916 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 100.0% |
| 平均 Stage 2 evidence 覆蓋率 | 61.9% |
| 平均總延遲 | 1052.5s |
| 平均 planning 延遲 | 30.6s |
| 平均 retrieval 延遲 | 7.5s |
| └ Phase A embed/vector/BM25 | 7.5s |
| └ Phase B 子答案生成 | 0.1s |
| 平均 grounding 延遲 | 26.7s |
| └ 其中 NLI | 2.4s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ⚠️ | Q08 | cross_paper | 1.0 | 1.0 | 100.0% | 100.0% | 66.7% | 0.833 | 635.9s | C0/U1 |
| ❌ | Q09 | cross_paper | 1.0 | 0.25 | 100.0% | 100.0% | 57.1% | 1.0 | 1469.0s | C0/U0 |

## 逐題細節

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/7
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：66.7%　grounding：0.833
- 延遲：635.9s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
>
> 比較框架：
> - 路徑：【bbb0683】報導了一種涉及對映選擇性烷基化 (enantioselective alkylation) 及隨後的酶促水解 (enzymatic hydrolysis) 的混合製程，產出光學純的 L-BPA 【bbb0683】。
> - 評論/比較來源：【CMDC-20-e202500059】報導 L-BPA 的合成已透過多種路徑進行，反映出在生產高純度、同位素富集 (isotopically enriched) 材料方面的挑戰 【CMDC-20-e202500059】。
> - 評論維度：該評論強調了每種方法在成本效益 (cost-effectiveness)、可擴展性 (scalability) 和安全性方面的局限性 【CMDC-20-e202500059】。
>
> 核心權衡（高純度/同位素富集與可擴展性及成本效益）：
> - 高純度/同位素富集：生產高純度、同位素富集 (isotopically enriched) 的材料是 L-BPA 合成中的一項挑戰 【CMDC-20-e202500059】。
> - 可擴展性：在大規模生產中使用任何氧化劑 (oxidant) 本質上都存在製程安全風險 【CMDC-20-e202500059】。
> - 成本效益：在製備同位素富集 (isotopically enriched) 化合物時，主要成本通常來自於同位素起始原料 【CMDC-20-e202500059】。
>
> ---
> 📋 **答案品質報告**
>
> ✅ **整體論文依據率**：83.3%　高（答案高度忠實於論文內容）
>
> ⚠️  **以下 1 個陳述未找到明確論文依據，請謹慎參考：**
>
> - Review/comparison source: `CMDC-20-e20250005 …（完整內容見 JSON）

---

### ❌ Q09 · cross_paper

**問題**：Several papers target LAT1 for cancer therapy. What different therapeutic strategies targeting LAT1 are described across the literature, and how do they differ in mechanism?

- detected_paper：`None`
- 選出論文：['LAT1 ChemComm 2026', '1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6', 'cas0106-0279']
- gold_papers：['1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026', 's41421-024-00697-6']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/7
- translation fidelity：0.25　raw：2/5　judge：`translation_fidelity_v2`
- translation reason：3 material and 0 minor semantic errors; The source specifies the inhibition was observed in HT-29 cells, but this detail is missing from T1.
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：57.1%　grounding：1.0
- 延遲：1469.0s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> Comparison scaffold:
> - 策略：【1-s2.0-S1347861320300633-main】報告了 JPH203 的預孵育 (preincubation) 與共同孵育 (co-incubation) 抑制作用，在 HT-29 細胞中 leucine (亮氨酸) 攝取顯著降低（使用 30 mM JPH203 處理 120 min 後，降至對照組的 25%） 【1-s2.0-S1347861320300633-main】。
> - 策略：【1-s2.0-S1936523321002047-main】報告了 JPH203 與 X-irradiation (X 射線照射) 的聯合應用，在 A549 與 MIA Paca-2 細胞中透過 mTOR 下調 (downregulation) 增強細胞衰老 (cellular senescence) 【1-s2.0-S1936523321002047-main】。
> - 策略：【LAT1 ChemComm 2026】報告了 LffVLKK-4Phe 自組裝肽 (self-assembling peptide) 處理，透過多價交互作用 (multivalent interactions) 以及對 LAT1 介導的氨基酸運輸之早期抑制，抑制 MCF-7 與 MDA-MB-231 細胞增殖 【LAT1 ChemComm 2026】。
> - 策略：【cas0106-0279】報告了用於 BNCT 的 p-Boronophenylalanine (BPA) 遞送，boron-10 (10B) 在惡性細胞中選擇性累積，隨後由 neutron beam irradiation (中子束照射) 將其殺死 【cas0106-0279】。
> - 機制：【s41421-024-00697-6】報告此外，JPH203 的雙鹵化酪氨酸 (bi-halog …（完整內容見 JSON）

---
