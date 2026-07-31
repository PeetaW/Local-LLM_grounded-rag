# Eval Report — `q07_q09_required_contract_rejudge`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-08-01 01:48
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.5 |
| Correctness judge 覆蓋 | 2/2（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 0.75 |
| Translation judge 覆蓋 | 2/2（N/A 0） |
| 平均 grounding 分數 | 1.0 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 83.4% |
| 平均 Stage 2 evidence 覆蓋率 | 53.5% |
| 平均總延遲 | 589.9s |
| 平均 planning 延遲 | 28.2s |
| 平均 retrieval 延遲 | 6.3s |
| └ Phase A embed/vector/BM25 | 6.2s |
| └ Phase B 子答案生成 | 0.1s |
| 平均 grounding 延遲 | 19.7s |
| └ 其中 NLI | 1.1s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ⚠️ | Q07 | figure_dependent | 0.75 | 1.0 | 100.0% | 66.7% | 50.0% | 1.0 | 430.4s | C0/U0 |
| ❌ | Q09 | cross_paper | 0.25 | 0.5 | 100.0% | 100.0% | 57.1% | 1.0 | 749.3s | C0/U0 |

## 逐題細節

### ⚠️ Q07 · figure_dependent

**問題**：What is the water-stable boroxine structure reported, and what role do the dynamic covalent bonds play in its fluoride binding and hydrogel formation?

- detected_paper：`None`
- 選出論文：['41467_2024_Article_45464']
- gold_papers：['41467_2024_Article_45464']
- correctness candidate：`answer_for_judge`
- correctness：0.75　raw：4/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/8; missing F1
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：66.7%　Stage 2 evidence 覆蓋率：50.0%　grounding：1.0
- 延遲：430.4s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - 有趣的是，在接觸水後，該二聚體（dimer）迅速轉化為具有優異 pH 穩定性和水兼容動態共價鍵（water-compatible dynamic covalent bonds）的 boroxine 結構 【Source: 41467_2024_Article_45464】。
> - 更令人感興趣的是，在室溫下接觸水後，HO‑PBA 二聚體（dimers）迅速轉化為 HO‑PBA 三聚體–H2O 複合物（trimer–H2O complexes）（圖 1b），其中包含水穩定 boroxine 結構，這一點已通過核磁共振光譜 (nuclear magnetic resonance (NMR) spectroscopy)、質譜 (mass spectrometry (MS))、UV-Raman 光譜 (UV–Raman spectrometry) 以及理論計算（theoretical calculations）得到證實 【Source: 41467_2024_Article_45464】。
> - 這一推測得到了以下事實的支持：如圖 7b 所示，該水凝膠（hydrogel）在酸性 (pH=2) 和鹼性 (pH=10) 溶液中均保持穩定 【Source: 41467_2024_Article_45464】。
> - 這些發現使得 boroxines 在水相介質（aqueous media）中能選擇性地識別氟離子 (fluoride ions (F−))，且其親和力顯著強於廣泛使用的 F- 受體（F−receptor）苯硼酸 (phenylboronic acid (PBA)) 25–27 【Source: 41467_2024_Article_45464】。
> - 水穩定 boroxine 結構中 B–O 鍵的動態特性（dynamic nature）通過各種 HO‑ …（完整內容見 JSON）

---

### ❌ Q09 · cross_paper

**問題**：Several papers target LAT1 for cancer therapy. What different therapeutic strategies targeting LAT1 are described across the literature, and how do they differ in mechanism?

- detected_paper：`None`
- 選出論文：['LAT1 ChemComm 2026', '1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6', 'cas0106-0279']
- gold_papers：['1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026', 's41421-024-00697-6']
- correctness candidate：`answer_for_judge`
- correctness：0.25　raw：2/5　judge：`structured_fact_audit_v1`
- judge reason：covered 2/7; missing F1, F2, F5, F6, F7
- translation fidelity：0.5　raw：3/5　judge：`translation_fidelity_v2`
- translation reason：1 material and 0 minor semantic errors; The source sentence begins with 'Furthermore,' indicating a logical transition, but the target translation adds an explicit subject '報告此外' (reports furthermore) which alters the grammatical structure and implies a new reporting action rather than just continuing the previous statement. The phrase 'F
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：57.1%　grounding：1.0
- 延遲：749.3s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 比較框架：
> - 策略：【1-s2.0-S1347861320300633-main】報告了 JPH203 給藥，透過共孵育 (co-incubation) 與預孵育 (preincubation) 效應抑制 LAT1 功能 [1-s2.0-S1347861320300633-main]。
> - 策略：【1-s2.0-S1936523321002047-main】報告了 JPH203 與放射治療 (radiation therapy) 的聯合應用，透過 mTOR 下調 (downregulation) 與細胞衰老 (cellular senescence) 增強放射敏感性 (radiosensitivity) [1-s2.0-S1936523321002047-main]。
> - 策略：【LAT1 ChemComm 2026】報告了 LffVLKK-4Phe 自組裝肽 (self-assembling peptide) 治療，抑制 LAT1 介導的氨基酸運輸並隨後導致細胞膜破裂 (cell membrane disruption) [LAT1 ChemComm 2026]。
> - 策略：【cas0106-0279】報告了用於 BNCT 的 p-Boronophenylalanine (BPA) 遞送，在中子照射 (neutron irradiation) 下透過 10B 核捕獲 (nuclear capture) 與裂變反應 (fission reactions) 實現選擇性細胞殺傷 [cas0106-0279]。
> - 機制：【s41421-024-00697-6】報告此外，JPH203 的雙鹵代酪氨酸 (bi-halogenated tyrosine) 核心上的氯原子與 Tyr259 形成鹵鍵 (halogen bond) [s41421-024-00697-6]。
> - 機制：【 …（完整內容見 JSON）

---
