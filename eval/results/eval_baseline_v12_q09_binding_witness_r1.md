# Eval Report — `baseline_v12_q09_binding_witness_r1`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-08-07 02:23
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 1.0 |
| Correctness judge 覆蓋 | 1/1（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 1.0 |
| Translation judge 覆蓋 | 1/1（N/A 0） |
| 平均 grounding 分數 | 1.0 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 100.0% |
| 平均 Stage 2 evidence 覆蓋率 | 57.1% |
| 平均總延遲 | 810.4s |
| 平均 planning 延遲 | 29.2s |
| 平均 retrieval 延遲 | 7.4s |
| └ Phase A embed/vector/BM25 | 7.4s |
| └ Phase B 子答案生成 | 0.1s |
| 平均 grounding 延遲 | 30.5s |
| └ 其中 NLI | 3.0s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ⚠️ | Q09 | cross_paper | 1.0 | 1.0 | 100.0% | 100.0% | 57.1% | 1.0 | 810.4s | C0/U0 |

## 逐題細節

### ⚠️ Q09 · cross_paper

**問題**：Several papers target LAT1 for cancer therapy. What different therapeutic strategies targeting LAT1 are described across the literature, and how do they differ in mechanism?

- detected_paper：`None`
- 選出論文：['LAT1 ChemComm 2026', '1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6', 'cas0106-0279']
- gold_papers：['1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026', 's41421-024-00697-6']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/7
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：57.1%　grounding：1.0
- 延遲：810.4s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 比較框架：
> - 策略：`1-s2.0-S1347861320300633-main` 報告了 JPH203 單藥療法 (monotherapy)（預孵育 preincubation 與共同孵育 co-incubation），以及對 LAT1 功能的協同抑制作用 【1-s2.0-S1347861320300633-main】。
> - 策略：`1-s2.0-S1936523321002047-main` 報告了 JPH203 與放射治療 (radiation therapy) 的聯合應用，透過 mTOR 下調 (downregulation) 與細胞衰老 (cellular senescence) 增強放射敏感性 (radiosensitivity) 【1-s2.0-S1936523321002047-main】。
> - 策略：`LAT1 ChemComm 2026` 報告了 LffVLKK-4Phe 自組裝肽 (self-assembling peptide) 治療，透過多價抑制氨基酸運輸及隨後的細胞膜破壞來抑制增殖 【LAT1 ChemComm 2026】。
> - 策略依據：這可能與報告顯示 JPH203 作為非底物阻斷劑 (non-substrate blocker) 的結果相關，儘管作者表示其抑制模式為競爭性 (competitive) 【1-s2.0-S1347861320300633-main】。
> - 策略依據：我們證明 X 射線照射 (X-irradiation) 增加了 A549 和 MIA Paca-2 細胞中透過 LAT1 的細胞中性氨基酸攝取 (neutral amino acid uptake)，且 LAT1 抑制劑 JPH203 在極低毒性濃度 (minimally toxic concentrations) 下，能顯著增加癌細胞對放射線的敏感性 【1-s2 …（完整內容見 JSON）

---
