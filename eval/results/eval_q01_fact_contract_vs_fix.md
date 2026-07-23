# Eval Report — `q01_fact_contract_vs_fix`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-22 01:49
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.75 |
| Correctness judge 覆蓋 | 1/1（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | N/A |
| Translation judge 覆蓋 | 0/1（N/A 1） |
| 平均 grounding 分數 | 1.0 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 100.0% |
| 平均 Stage 2 evidence 覆蓋率 | 100.0% |
| 平均總延遲 | 323.5s |
| 平均 planning 延遲 | 8.7s |
| 平均 retrieval 延遲 | 7.6s |
| └ Phase A embed/vector/BM25 | 7.5s |
| └ Phase B 子答案生成 | 0.1s |
| 平均 grounding 延遲 | 13.3s |
| └ 其中 NLI | 0.0s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ⚠️ | Q01 | single_paper | 0.75 | N/A | 100.0% | 100.0% | 100.0% | 1.0 | 323.5s | C0/U0 |

## 逐題細節

### ⚠️ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- correctness candidate：`answer_for_judge`
- correctness：0.75　raw：4/5　judge：`structured_fact_audit_v1`
- judge reason：covered 4/5; missing F5
- translation fidelity：N/A　raw：N/A　judge：`translation_fidelity_v2`
- translation reason：invalid translation audit: top-level 'errors' must be a list
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：1.0
- 延遲：323.5s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - 總結而言，我們揭示了 JPH203 能夠透過其預孵育效應 (preincubation effect) 抑制 LAT1 的功能，且該預孵育效應能協同增強共孵育抑制效果 (co-incubation inhibitory effects) [Source: 1-s2.0-S1347861320300633-main]。
> - 與 IC50 分析結果一致，預孵育加共孵育抑制分析 (pre-plus co-incubation inhibition assays) 中的 Ki 值為 0.37 mM，低於共孵育抑制分析 (co-incubation inhibition assays) 中的值 (0.46 mM) [Source: 1-s2.0-S1347861320300633-main]。
> - 這些結果表明，JPH203 對 HT-29 細胞中 LAT1 功能的預孵育抑制效果 (preincubation inhibitory effects) 呈濃度與時間依賴性 [Source: 1-s2.0-S1347861320300633-main]。
> - 根據結果，IC50 值被確定為 193 ± 50 (nM) [Source: 1-s2.0-S1347861320300633-main]。
> - 如圖 5 (Fig. 5) 所示，雖然共孵育抑制分析 (co-incubation inhibition assays) 中的 IC50 值為 99.2 ± 11.0 nM，但加入預孵育 (preincubation) 顯著增強了其抑制效能 (IC50 = 34.2 ± 3.6 nM) [Source: 1-s2.0-S1347861320300633-main]。
> - 聚 epsilon-己內酯 (Poly(e-caprolactone), PCL) 因其對藥 …（完整內容見 JSON）

---
