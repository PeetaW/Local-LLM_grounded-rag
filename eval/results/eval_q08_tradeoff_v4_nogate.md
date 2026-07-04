# Eval Report — `q08_tradeoff_v4_nogate`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-03 04:12
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.5 |
| 平均 grounding 分數 | 0.815 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 1856.9s |
| 平均 planning 延遲 | 14.3s |
| 平均 retrieval 延遲 | 894.3s |
| 平均 grounding 延遲 | 77.7s |
| └ 其中 NLI | 48.8s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 0.815 | 1856.9s | C0/U7 |

## 逐題細節

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.815
- 延遲：1856.9s　問題標記：{'conflicts': 0, 'unsupported': 7}

**答案預覽**：

> **路線圖 (Route Map)**
> *   **<0xE9><0x88><0x80>催化交叉耦合反應 (Pd-Catalyzed Cross-Coupling)**：定義為使用<0xE9><0x88><0x80>催化劑以及硼源，如 $\text{B}_2\text{pin}_2$ 或 $\text{HBpin}$。
> *   **直接氨基酸官能基化 (Direct Amino Acid Functionalization)**：定義為保護氨基酸（例如 4-碘苯丙氨酸 (4-iodophenylalanine)）與有機金屬試劑 ($\text{n-BuLi}$, $\text{i-PrMgCl}$ 或 Turbo Grignard）以及硼酸酯 (borate esters) 的反應。
> *   **混合化學-酶法製程 (Hybrid Chemo-Enzymatic Process)**：結合使用手性輔助劑 ((2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine) 的對映選擇性烷基化 (enantioselective alkylation)，隨後透過 chymotrypsin 進行酶水解 (enzymatic hydrolysis)。
> 
> **核心權衡 (Central trade-off)：** L-BPA 合成的主要矛盾在於：追求絕對對映純度 (absolute enantiopurity) 與同位素效率 (isotopic efficiency)（透過混合化學-酶法或直接官能基化路徑），與工業<0xE9><0x88><0x80>催化方法的高產率之間。後者的缺點在於 $^{10}\text{B}$ 浪費的極高成本以及重金屬污染 (heavy metal contamination) 的風險。
> 
> ## 【論文直接依據】
>  …（完整內容見 JSON）

---
