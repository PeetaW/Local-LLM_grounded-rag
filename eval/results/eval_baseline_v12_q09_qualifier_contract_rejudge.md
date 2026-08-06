# Eval Report — `baseline_v12_q09_qualifier_contract_rejudge`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-08-06 21:08
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.75 |
| Correctness judge 覆蓋 | 1/1（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 1.0 |
| Translation judge 覆蓋 | 1/1（N/A 0） |
| 平均 grounding 分數 | 1.0 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 100.0% |
| 平均 Stage 2 evidence 覆蓋率 | 57.1% |
| 平均總延遲 | 716.8s |
| 平均 planning 延遲 | 27.6s |
| 平均 retrieval 延遲 | 7.5s |
| └ Phase A embed/vector/BM25 | 7.5s |
| └ Phase B 子答案生成 | 0.1s |
| 平均 grounding 延遲 | 31.7s |
| └ 其中 NLI | 2.0s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ⚠️ | Q09 | cross_paper | 0.75 | 1.0 | 100.0% | 100.0% | 57.1% | 1.0 | 716.8s | C0/U0 |

## 逐題細節

### ⚠️ Q09 · cross_paper

**問題**：Several papers target LAT1 for cancer therapy. What different therapeutic strategies targeting LAT1 are described across the literature, and how do they differ in mechanism?

- detected_paper：`None`
- 選出論文：['LAT1 ChemComm 2026', '1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6', 'cas0106-0279']
- gold_papers：['1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026', 's41421-024-00697-6']
- correctness candidate：`answer_for_judge`
- correctness：0.75　raw：4/5　judge：`structured_fact_audit_v1`
- judge reason：covered 6/7; missing F1
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：57.1%　grounding：1.0
- 延遲：716.8s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 比較框架：
> - 策略：【1-s2.0-S1936523321002047-main】報導了 JPH203 與放射治療 (radiation therapy) 的聯合應用，通過下調 mTOR 來增強細胞衰老 (cellular senescence)，從而使癌細胞對放射治療更為敏感 【1-s2.0-S1936523321002047-main】。
> - 策略：【LAT1 ChemComm 2026】報導了自組裝肽 (self-assembling peptide) LffVLKK-4Phe 的處理，其在早期抑制氨基酸轉運活性，並在後期誘導細胞膜破裂 (cell membrane disruption) 【LAT1 ChemComm 2026】。
> - 機制證據：這可能與報導顯示 JPH203 作為非底物阻斷劑 (non-substrate blocker) 的結果相關，儘管作者表示其抑制模式是競爭性的 (competitive) 【1-s2.0-S1347861320300633-main】。
> - 策略證據：我們證明了 X 射線照射 (X-irradiation) 增加了 A549 和 MIA Paca-2 細胞中通過 LAT1 的細胞中性氨基酸攝取，且 LAT1 抑制劑 (inhibitor) JPH203 在最低毒性濃度下，顯著地使癌細胞對放射治療更為敏感 【1-s2.0-S1936523321002047-main】。
> - 策略證據：我們設計了一種偶聯至 L-苯丙氨酸 (L-phenylalanine, Phe) 基序的自組裝肽，作為一種 LAT1 配體 (ligand) (LffVLKK-4Phe) 【LAT1 ChemComm 2026】。
> - 策略證據：這些結果表明，需要針對 LAT1 (LAT1 targeting) 才能抑制 MCF-7 細胞的增殖 【LAT1 C …（完整內容見 JSON）

---
