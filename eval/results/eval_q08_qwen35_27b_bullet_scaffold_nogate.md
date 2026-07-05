# Eval Report — `q08_qwen35_27b_bullet_scaffold_nogate`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-05 13:19
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.75 |
| 平均 grounding 分數 | 0.6 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 1103.7s |
| 平均 planning 延遲 | 10.5s |
| 平均 retrieval 延遲 | 631.6s |
| 平均 grounding 延遲 | 25.4s |
| └ 其中 NLI | 9.3s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q08 | cross_paper | 100.0% | 100.0% | 0.6 | 1103.7s | C0/U2 |

## 逐題細節

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.6
- 延遲：1103.7s　問題標記：{'conflicts': 0, 'unsupported': 2}

**答案預覽**：

> Comparison scaffold:
> - review/comparison source: 【CMDC-20-e202500059】(比較 Chirality First 與 Boron First 策略，評估同位素富集成本與步驟數效率)。
> - route: 【bbb0683】(報導了一種結合對映選擇性烷基化 (enantioselective alkylation) 與酶促水解 (enzymatic hydrolysis) 用於 L-BPA 合成的混合製程)。
> 
> 【bbb0683】中詳細描述的特定合成路徑採用了一種混合製程，該製程整合了對映選擇性烷基化 (enantioselective alkylation)，隨後進行酶促水解 (enzymatic hydrolysis)，以生產光學純的 L-p-boronophenylalanine。此方法利用 chymotrypsin 進行最終的分辨步驟 (resolution step)，在確保高光學純度的同時，與現有方法相比，在操作便利性和減少反應步驟方面具有優勢 【bbb0683】。透過避免會導致較低對映體過量 (enantiomeric excess) 的非酶促水解 (non-enzymatic hydrolysis)，此路徑優先考慮適合臨床應用的產品品質，且無需依賴其他方案中常見的複雜保護基操作 (protecting group manipulations) 【bbb0683】。
> 
> 相比之下，評論來源【CMDC-20-e202500059】將 L-BPA 的合成分為 Chirality First 和 Boron First 策略，並強調對於 BNCT therapy 而言，使用 $^{10}$B 的同位素富集 (isotopic enrichment) 是必須的，但與天然硼相比成本高昂。該來源指出，由於富集試劑的 …（完整內容見 JSON）

---
