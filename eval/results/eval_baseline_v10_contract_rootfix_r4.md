# Eval Report — `baseline_v10_contract_rootfix_r4`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-27 22:16
- 題數：3

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.75 |
| Correctness judge 覆蓋 | 3/3（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 0.833 |
| Translation judge 覆蓋 | 3/3（N/A 0） |
| 平均 grounding 分數 | 0.849 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 88.9% |
| 平均 Stage 2 evidence 覆蓋率 | 57.9% |
| 平均總延遲 | 532.3s |
| 平均 planning 延遲 | 29.6s |
| 平均 retrieval 延遲 | 6.7s |
| └ Phase A embed/vector/BM25 | 6.7s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 22.8s |
| └ 其中 NLI | 1.7s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ⚠️ | Q07 | figure_dependent | 0.75 | 1.0 | 100.0% | 66.7% | 50.0% | 1.0 | 412.2s | C0/U0 |
| ⚠️ | Q08 | cross_paper | 1.0 | 1.0 | 100.0% | 100.0% | 66.7% | 0.833 | 511.7s | C0/U1 |
| ⚠️ | Q09 | cross_paper | 0.5 | 0.5 | 100.0% | 100.0% | 57.1% | 0.714 | 673.1s | C0/U2 |

## 逐題細節

### ⚠️ Q07 · figure_dependent

**問題**：What is the water-stable boroxine structure reported, and what role do the dynamic covalent bonds play in its fluoride binding and hydrogel formation?

- detected_paper：`None`
- 選出論文：['41467_2024_Article_45464']
- gold_papers：['41467_2024_Article_45464']
- correctness candidate：`answer_for_judge`
- correctness：0.75　raw：4/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/8; missing F6
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：66.7%　Stage 2 evidence 覆蓋率：50.0%　grounding：1.0
- 延遲：412.2s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - 有趣的是，在接觸水後，該二聚體（dimer）迅速轉化為具有優異 pH 穩定性和水兼容動態共價鍵（water-compatible dynamic covalent bonds）的 boroxine 結構 [Source: 41467_2024_Article_45464]。
> - 更令人感興趣的是，在室溫下接觸水後，HO‑PBA 二聚體（dimers）迅速轉化為 HO‑PBA 三聚體–H2O 複合物（trimer–H2O complexes）（圖 1b），其中包含水穩定 boroxine 結構，這已透過核磁共振光譜 (nuclear magnetic resonance (NMR) spectroscopy)、質譜 (mass spectrometry (MS))、UV–Raman 光譜 (UV–Raman spectrometry) 以及理論計算（theoretical calculations）所證實 [Source: 41467_2024_Article_45464]。
> - 這些發現使得 boroxines 在水相介質（aqueous media）中能選擇性地識別氟離子 (fluoride ions (F−))，且其親和力顯著強於廣泛使用的 F- 受體 (F−receptor) 苯硼酸 (phenylboronic acid (PBA))25–27 [Source: 41467_2024_Article_45464]。
> - 水穩定 boroxine 結構中 B–O 鍵的動態特性（dynamic nature），可由各種 HO‑PBA 三聚體結構（trimeric structures）之間的快速交換來證明 [Source: 41467_2024_Article_45464]。
> - 此外，我們開發了一種由水穩定 boroxines 交聯（cr …（完整內容見 JSON）

---

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
- 延遲：511.7s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
>
> 比較框架：
> - 路徑 (Route)：【bbb0683】報導了一種 hybrid process involving enantioselective alkylation and subsequent enzymatic hydrolysis（涉及對映選擇性烷基化及隨後酶水解的混合製程），從而產生光學純 L-BPA 【bbb0683】。
> - 評論/比較來源：【CMDC-20-e202500059】報導 L-BPA 的合成已透過多種路徑進行，反映了生產高純度、同位素富集 (isotopically enriched) 物質的挑戰 【CMDC-20-e202500059】。
> - 評論維度：該評論強調了每種方法在可擴展性 (scalability)、成本效益 (cost-effectiveness) 和安全性 (safety) 方面的局限性 【CMDC-20-e202500059】。
>
> 核心權衡（高純度/同位素富集與可擴展性及成本效益）：
> - 高純度/同位素富集：生產高純度、同位素富集 (isotopically enriched) 的物質是 L-BPA 合成中的重大挑戰 【CMDC-20-e202500059】。
> - 可擴展性：大規模使用任何氧化劑 (oxidant) 本質上都存在製程安全風險 (process safety risk) 【CMDC-20-e202500059】。
> - 成本效益：在製備同位素富集 (isotopically enriched) 的化合物時，主要成本通常來自於同位素起始原料 (isotope starting material) 【CMDC-20-e202500059】。
>
> ---
> 📋 **答案品質報告**
>
> ✅ **整體論文依據率**：83.3% …（完整內容見 JSON）

---

### ⚠️ Q09 · cross_paper

**問題**：Several papers target LAT1 for cancer therapy. What different therapeutic strategies targeting LAT1 are described across the literature, and how do they differ in mechanism?

- detected_paper：`None`
- 選出論文：['LAT1 ChemComm 2026', '1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6', 'cas0106-0279']
- gold_papers：['1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026', 's41421-024-00697-6']
- correctness candidate：`answer_for_judge`
- correctness：0.5　raw：3/5　judge：`structured_fact_audit_v1`
- judge reason：covered 5/7; missing F4, F5
- translation fidelity：0.5　raw：3/5　judge：`translation_fidelity_v2`
- translation reason：1 material and 0 minor semantic errors; The source sentence S7 explicitly states that the α-amino and α-carboxyl groups are of 'the head' (of a molecule), but T7 omits this critical semantic detail, referring only to 'top/upper end' (頂端) which changes or loses the specific structural context implied by 'head group'. Additionally, S7 is an
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：57.1%　grounding：0.714
- 延遲：673.1s　問題標記：{'conflicts': 0, 'unsupported': 2}

**答案預覽**：

> 比較框架：
> - 策略：【1-s2.0-S1347861320300633-main】報告了 JPH203 的抑制作用（共同孵育 co-incubation 與預孵育 preincubation），可抑制 HT-29 細胞中不依賴 Na+ 的 leucine 攝取活性 【1-s2.0-S1347861320300633-main】。
> - 策略：【1-s2.0-S1936523321002047-main】報告了 JPH203 與 X-irradiation (X 射線照射) 的聯合應用，透過 mTOR 下調 (downregulation) 與細胞衰老 (cellular senescence) 增強放射敏感性 【1-s2.0-S1936523321002047-main】。
> - 策略：【LAT1 ChemComm 2026】報告了 LffVLKK-4Phe 自組裝肽 (self-assembling peptide) 處理，透過多價 LAT1 交互作用 (multivalent LAT1 interaction) 及隨後的細胞膜破壞 (membrane disruption) 抑制癌細胞增殖 【LAT1 ChemComm 2026】。
> - 策略：【cas0106-0279】報告了 p-Boronophenylalanine (BPA) 遞送隨後進行中子束照射 (neutron beam irradiation)，透過核捕獲 (nuclear capture) 與裂變反應 (fission reactions) 選擇性殺傷含有 10B 的惡性細胞 【cas0106-0279】。
> - 機制：【s41421-024-00697-6】報告了與 JX075 或 Diiodo-Tyr 相比，JPH203 誘導 LAT1 產生不同的構象變化 (conformational changes …（完整內容見 JSON）

---
