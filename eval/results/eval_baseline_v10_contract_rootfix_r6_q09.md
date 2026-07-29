# Eval Report — `baseline_v10_contract_rootfix_r6_q09`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-28 08:36
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 1.0 |
| Correctness judge 覆蓋 | 1/1（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 1.0 |
| Translation judge 覆蓋 | 1/1（N/A 0） |
| 平均 grounding 分數 | 0.889 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 100.0% |
| 平均 Stage 2 evidence 覆蓋率 | 57.1% |
| 平均總延遲 | 719.1s |
| 平均 planning 延遲 | 28.1s |
| 平均 retrieval 延遲 | 7.6s |
| └ Phase A embed/vector/BM25 | 7.5s |
| └ Phase B 子答案生成 | 0.1s |
| 平均 grounding 延遲 | 34.0s |
| └ 其中 NLI | 4.2s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ⚠️ | Q09 | cross_paper | 1.0 | 1.0 | 100.0% | 100.0% | 57.1% | 0.889 | 719.1s | C0/U1 |

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
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：57.1%　grounding：0.889
- 延遲：719.1s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> 比較框架：
> - 策略：【1-s2.0-S1347861320300633-main】報告了 JPH203 的抑制作用（共孵育 co-incubation 與預孵育 preincubation），可抑制 HT-29 細胞中不依賴 Na+ 的 leucine（亮氨酸）攝取活性 【1-s2.0-S1347861320300633-main】。
> - 策略：【1-s2.0-S1936523321002047-main】報告了 JPH203 與 X 射線 (X-irradiation) 的聯合應用，通過 mTOR 下調與誘導細胞衰老 (cellular senescence) 來增強放射敏感性 (radiosensitivity) 【1-s2.0-S1936523321002047-main】。
> - 策略：【LAT1 ChemComm 2026】報告了 LffVLKK-4Phe 自組裝肽 (self-assembling peptide) 處理，通過抑制 LAT1 介導的氨基酸轉運活性來抑制 MCF-7 和 MDA-MB-231 細胞的增殖 【LAT1 ChemComm 2026】。
> - 策略：【cas0106-0279】報告了使用 p-硼苯丙氨酸 (p-Boronophenylalanine, BPA) 的硼中子捕獲治療 (Boron neutron capture therapy, BNCT)，在中子束照射下，通過核捕獲與裂變反應選擇性地殺死含有硼-10 (10B) 的惡性細胞 【cas0106-0279】。
> - 機制：【s41421-024-00697-6】報告 JPH203 的結構結合涉及一個親水性頭部 (hydrophilic head) 與 TM1 和 TM6 形成氫鍵網絡 (H-bond network)，一個氯原子 (chloride atom) 與 Tyr259 形成 …（完整內容見 JSON）

---
