# Eval Report — `citfix`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-06-23 13:17
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.5 |
| 平均 grounding 分數 | 0.975 |
| 平均論文選擇命中率 | 87.5% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 1284.1s |
| 平均 planning 延遲 | 8.7s |
| 平均 retrieval 延遲 | 516.6s |
| 平均 grounding 延遲 | 74.0s |
| └ 其中 NLI | 49.5s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 0.95 | 1067.4s | C0/U4 |
| ⚠️ | Q09 | cross_paper | 75.0% | 100.0% | 1.0 | 1500.8s | C0/U0 |

## 逐題細節

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.95
- 延遲：1067.4s　問題標記：{'conflicts': 0, 'unsupported': 4}

**答案預覽**：

> ## 【論文直接依據】
> 
> **光學純 L-BPA 的混合合成工藝（來源：bbb0683）**
> * 該混合工藝結合了對映選擇性烷基化與酶促水解 (bbb0683)。
> * 此工藝的特點是反應步驟較少 (bbb0683)。
> * 此工藝具有操作和後處理更簡單的特點 (bbb0683)。
> * 步驟 1 包括使 4-溴甲基苯硼酸與 2,2-二甲基-1,3-丙二醇反應 (bbb0683)。
> * 步驟 1 在室溫下的乾燥 THF 中進行 (bbb0683)。
> * 步驟 1 生成環狀硼酸酯，產率為 79% (bbb0683)。
> * 步驟 2 包括使用 $(2\text{R})(-)-2,5$-二氫-2-異丙基-3,6-二甲氧基吡嗪和 n-BuLi 進行對映選擇性烷基化 (bbb0683)。
> * 步驟 2 在 $-78^\circ\text{C}$ 下進行 (bbb0683)。
> * 步驟 2 生成加成產物，其立體選擇性為 $74\%$ e.e. 或 $72\%$ 非對映異構體過量 (bbb0683)。
> * 步驟 3 包括使用 $0.1\text{ N}$ 鹽酸處理及脫保護，隨後使用 $1\text{ N}$ 鹽酸和 THF 攪拌 (bbb0683)。
> * 步驟 3 利用 ODS 柱色譜法進行純化 (bbb0683)。
> * 步驟 3 生成 L-BPA 甲酯，基於加成產物的產率為 85% (bbb0683)。
> * 步驟 4 包括使用胰蛋白酶進行酶促水解 (bbb0683)。
> * 步驟 4 在 $\text{pH } 5.0$ 和 $25^\circ\text{C}$ 下進行 30 小時 (bbb0683)。
> * 步驟 4 利用超濾膜去除酶，隨後進行重結晶 (bbb0683)。
> * 步驟 4 生成光學純度為 $100\%$ e.e. 且產率為 79% 的 L-BPA (bbb0683)。
> 
> **一般 …（完整內容見 JSON）

---

### ⚠️ Q09 · cross_paper

**問題**：Several papers target LAT1 for cancer therapy. What different therapeutic strategies targeting LAT1 are described across the literature, and how do they differ in mechanism?

- detected_paper：`None`
- 選出論文：['LAT1 ChemComm 2026', '1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026SI', 'cas0106-0279']
- gold_papers：['1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026', 's41421-024-00697-6']
- 論文選擇命中率：75.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：1500.8s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 根據提供的文獻，描述了幾種針對 L-型氨基酸轉運蛋白 1 (LAT1) 的治療策略。這些策略可分為小分子抑制、自組裝肽抑制、放射治療的標靶遞送以及基因敲低（genetic knockdown）。
> 
> ## 【論文直接依據】
> 
> **小分子抑制 (JPH203)**
> * JPH203 是一種酪胺酸衍生物，可競爭性地抑制 LAT1 (1-s2.0-S1347861320300633-main)。
> * JPH203 的 $\text{IC}_{50}$ 值为 $60\text{--}140\text{ nM}$ (1-s2.0-S1347861320300633-main)。
> * JPH203 在多種癌細胞和腫瘤負荷動物模型中表現出抗增殖效果 (1-s2.0-S1347861320300633-main)。
> * JPH203 對於部分結腸癌和膽道癌患者具有療效且耐受性良好 (1-s2.0-S1347861320300633-main)。
> * JPH203 抑制多種癌細胞株的腫瘤生長 (1-s2.0-S1936523321002047-main)。
> * JPH203 對於食道鱗狀細胞癌具有治療潛力 (1-s2.0-S1936523321002047-main)。
> * JPH203 誘導 YD-38 人類口腔癌細胞凋亡 (1-s2.0-S1936523321002047-main)。
> * JPH203 在完全免疫缺陷小鼠模型中降低了未分化甲狀腺癌的生長 (1-s2.0-S1936523321002047-main)。
> * JPH203 已進入針對晚期實體腫瘤患者的 I 期和 II 期臨床試驗 (1-s2.0-S1936523321002047-main)。
> * 當結合預孵育（preincubation）和共同孵育（co-incubation）時，JPH203 的 $\text{IC}_{ …（完整內容見 JSON）

---
