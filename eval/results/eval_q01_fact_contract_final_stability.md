# Eval Report — `q01_fact_contract_final_stability`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-22 23:18
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
| 平均 Stage 2 evidence 覆蓋率 | 100.0% |
| 平均總延遲 | 288.6s |
| 平均 planning 延遲 | 7.0s |
| 平均 retrieval 延遲 | 5.0s |
| └ Phase A embed/vector/BM25 | 4.9s |
| └ Phase B 子答案生成 | 0.1s |
| 平均 grounding 延遲 | 10.8s |
| └ 其中 NLI | 0.0s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ✅ | Q01 | single_paper | 1.0 | 1.0 | 100.0% | 100.0% | 100.0% | 1.0 | 288.6s | C0/U0 |

## 逐題細節

### ✅ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 5/5
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：1.0
- 延遲：288.6s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - 此外，preincubation 與 co-incubation 共同作用的 IC50 值低於單獨 co-incubation 抑制的效果 (34.2 ± 3.6 nM vs. 99.2 ± 11.0 nM) [Source: 1-s2.0-S1347861320300633-main]。
> - 這些結果表明 JPH203 對於 HT-29 細胞中 LAT1 功能的 preincubation 抑制效果呈濃度與時間依賴性 [Source: 1-s2.0-S1347861320300633-main]。
> - 根據結果，IC50 值被確定為 193 ± 50 (nM) [Source: 1-s2.0-S1347861320300633-main]。
> - 如圖 5 所示，雖然 co-incubation 抑制分析中的 IC50 值為 99.2 ± 11.0 nM，但加入 preincubation 顯著增強了其抑制效能 (IC50 = 34.2 ± 3.6 nM) [Source: 1-s2.0-S1347861320300633-main]。
> - 與 IC50 分析結果一致，pre-plus co-incubation 抑制分析中的 Ki 值為 0.37 mM，低於 co-incubation 抑制分析中的值 (0.46 mM) [Source: 1-s2.0-S1347861320300633-main]。
>
> ---
> 📋 **答案品質報告**
>
> 📊 **分段論文依據率：**
>
>   ✅ 【論文直接依據】：100.0%（5/5 句）
>
> ✅ **直引依據率**：100.0%　高（直接引用高度忠實於論文）
>
> ✅ **所有陳述均有論文依據**
>
>
> <!-- grounding_score=1.000 -->
> ---

---
