# Eval Report — `q08_qwen35_27b_two_bullets_nogate`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-05 14:02
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.75 |
| 平均 grounding 分數 | 1.0 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 1754.4s |
| 平均 planning 延遲 | 10.9s |
| 平均 retrieval 延遲 | 1192.5s |
| 平均 grounding 延遲 | 24.2s |
| └ 其中 NLI | 8.4s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 1.0 | 1754.4s | C0/U0 |

## 逐題細節

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：1754.4s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 比較框架：
> - **評論/比較來源 (Review/Comparison Source)：** 【CMDC-20-e202500059】評估了多種 L-BPA 合成策略，權衡同位素成本 (isotopic cost) 與步驟效率，以及催化劑的監管安全限制 (regulatory safety limits)。
> - **路徑 (Route)：** 【bbb0683】詳細介紹了一種特定的混合製程，將對映選擇性烷基化 (enantioselective alkylation) 與酶促水解 (enzymatic hydrolysis) 相結合，以實現高光學純度 (optical purity)。
> 
> *   直接路徑證據：所報導的合成在化學加合物形成 (chemical adduct formation) 後，於酶促拆分 (enzymatic resolution) 步驟中使用 chymotrypsin，優先考慮光學純度 (optical purity) 而非純化學的去保護基方法 (deprotection methods) 【bbb0683】。
> *   評論/比較來源證據：成本效益 (cost-effectiveness) 受到 $^{10}\text{B}$ 的可用性與試劑選擇的嚴重影響，其中使用 $\text{B}_2\text{pin}_2$ 的路徑由於同位素浪費，較硼酸酯 (borate esters) 較不理想 【CMDC-20-e202500059】。
> 
> 核心權衡：最大化同位素富集 (isotopic enrichment) 與對映體純度 (enantiopurity) 通常需要複雜且昂貴的步驟，而這與由昂貴的同位素保存 (isotope conservation) 以及對催化劑殘留物 (catalyst residues) 的嚴格監管限制所驅動的可擴展性要求 (sca …（完整內容見 JSON）

---
