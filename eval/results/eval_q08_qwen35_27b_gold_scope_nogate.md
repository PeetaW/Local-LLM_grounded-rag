# Eval Report — `q08_qwen35_27b_gold_scope_nogate`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-05 14:39
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.5 |
| 平均 grounding 分數 | 1.0 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 1817.6s |
| 平均 planning 延遲 | 11.1s |
| 平均 retrieval 延遲 | 1261.8s |
| 平均 grounding 延遲 | 22.7s |
| └ 其中 NLI | 6.9s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 1.0 | 1817.6s | C0/U0 |

## 逐題細節

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：1817.6s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 比較框架：
> *   回顧/比較來源：【CMDC-20-e202500059】評估了不同的策略，例如「手性優先」(Chirality First) 與 「硼優先」(Boron First)，權衡同位素富集成本 (isotopic enrichment costs) 與步驟效率 (step efficiency) 及催化劑安全性限制 (catalyst safety constraints) [CMDC-20-e202500059]。
> *   路徑：【bbb0683】詳細介紹了一種利用手性輔助劑 (chiral auxiliaries) 和酶水解 (enzymatic hydrolysis) 以實現光學純度 (optical purity) 的混合製程 (hybrid process)，並在底物可用性 (substrate availability) 方面與對映選擇性氫化法 (enantioselective hydrogenation methods) 進行對比 [bbb0683]。
> 
> 直接路徑證據表明，實現高對映體過量值 (enantiomeric excess) 通常需要複雜的輔助系統 (complex auxiliary systems) 或特定催化劑 (specific catalysts)，而前驅物可獲得性 (precursor accessibility) 在不同方法之間存在顯著差異 [bbb0683]。
> 回顧/比較來源證據強調，利用富集硼前驅物 (enriched boron precursors) 的路徑由於廢棄物產生 (waste generation) 而面臨巨大的經濟障礙 (economic barriers)，而涉及重金屬的催化步驟 (catalytic steps involving heavy metals) 則為藥品可擴展性 (pharmaceu …（完整內容見 JSON）

---
