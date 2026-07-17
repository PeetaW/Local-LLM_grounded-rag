# Eval Report — `structured_eval_contract_smoke`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-17 03:38
- 題數：3

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.333 |
| Correctness judge 覆蓋 | 3/3（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 0.583 |
| Translation judge 覆蓋 | 3/3（N/A 0） |
| 平均 grounding 分數 | 0.739 |
| 平均論文選擇命中率 | 91.7% |
| 平均檢索覆蓋率 | 88.9% |
| 平均總延遲 | 775.9s |
| 平均 planning 延遲 | 6.0s |
| 平均 retrieval 延遲 | 5.7s |
| └ Phase A embed/vector/BM25 | 5.7s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 20.1s |
| └ 其中 NLI | 2.7s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|---------|-----------|------|------|
| ⚠️ | Q03 | figure_dependent | 0.5 | 1.0 | 100.0% | 100.0% | 0.917 | 646.2s | C0/U1 |
| ❌ | Q07 | figure_dependent | 0.25 | 0.25 | 100.0% | 66.7% | 0.5 | 575.5s | C0/U3 |
| ❌ | Q09 | cross_paper | 0.25 | 0.5 | 75.0% | 100.0% | 0.8 | 1105.9s | C0/U1 |

## 逐題細節

### ⚠️ Q03 · figure_dependent

**問題**：According to the cryo-EM structure, how does JPH203 bind to and inhibit the LAT1-4F2hc complex?

- detected_paper：`None`
- 選出論文：['s41421-024-00697-6', '1-s2.0-S1347861320300633-main']
- gold_papers：['s41421-024-00697-6']
- correctness candidate：`answer_for_judge`
- correctness：0.5　raw：3/5　judge：`structured_fact_audit_v1`
- judge reason：covered 3/6; missing F3, F4, F5
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v1`
- translation reason：The translation is scientifically faithful, accurately rendering all technical terms (e.g., LAT1-4F2hc complex, competitive inhibition kinetics), numerical values, chemical structures, and logical relationships without omissions or errors.
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.917
- 延遲：646.2s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> 根據提供的數據，JPH203 與 LAT1-4F2hc 複合體結合並抑制其作用的機制，涉及底物結合口袋 (substrate-binding pocket) 內的特定結構交互作用，以及在功能分析中觀察到的競爭性抑制動力學 (competitive inhibition kinetics)。
>
> **結構結合機制 (Structural Binding Mechanism)**
> L-型氨基酸轉運蛋白 1 (L-type amino acid transporter 1, LAT1 或 SLC7A5) 與 4F2hc (或 SLC3A2) 耦合，以一種不依賴於鈉的交換方式促進大中性氨基酸和甲狀腺激素跨細胞膜的轉運 [s41421-024-00697-6]。JPH203 由兩個不同的部分組成：一個具有苯丙氨酸 (phenylalanine) 骨架的親水性頭部基團，以及一個具有 5-amino-2-(3-aminophenyl) benzoxazole 骨架的疏水性尾部基團 [s41421-024-00697-6]。JPH203 結合在 LAT1 的傳統底物結合口袋 (substrate-binding pocket) 中，與抑制劑 Diiodo-Tyr、JX-075、JX-078 和 JX-119 相似 [s41421-024-00697-6]。JPH203 頭部的 $\alpha$-氨基 ($\alpha$-amino group) 和 $\alpha$-羧基 ($\alpha$-carboxyl group) 與 LAT1 中 TM1 和 TM6 解旋區域 (unwound region) 的主鏈原子形成氫鍵網絡 [s41421-024-00697-6]。
>
> **抑制動力學與細胞效應 (Inhibition Kinetics and Cellular Effects)**
> 雖然 …（完整內容見 JSON）

---

### ❌ Q07 · figure_dependent

**問題**：What is the water-stable boroxine structure reported, and what role do the dynamic covalent bonds play in its fluoride binding and hydrogel formation?

- detected_paper：`None`
- 選出論文：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- gold_papers：['41467_2024_Article_45464']
- correctness candidate：`answer_for_judge`
- correctness：0.25　raw：2/5　judge：`structured_fact_audit_v1`
- judge reason：covered 1/8; missing F2, F3, F4, F6, F7, F8; contradicted F5
- translation fidelity：0.25　raw：2/5　judge：`translation_fidelity_v1`
- translation reason：The translation retains excessive English technical terms (e.g., boroxine, anion receptors, fluoride binding) and citation markers within the Chinese text without providing necessary translations or removing redundant references, resulting in a fragmented scientific narrative that fails to meet stan
- 論文選擇命中率：100.0%　檢索覆蓋率：66.7%　grounding：0.5
- 延遲：575.5s　問題標記：{'conflicts': 0, 'unsupported': 3}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
>
> 根據【41467_2024_Article_45464】提供的數據，水穩定 boroxine 結構是從 2-hydroxyphenylboronic acid 中發現的【41467_2024_Article_45464】。在環境條件下 (ambient environments)，該化合物會發生自發性脫水 (spontaneous dehydration)，形成具有動態共價鍵 (dynamic covalent bonds) 和聚集誘導增強發光活性 (aggregation-induced enhanced emission activity) 的二聚體 (dimer)【41467_2024_Article_45464】。
>
> 關於這些結構在結合與材料形成中的作用：
> *   **一般 Boroxine 功能：** Boroxines 在陰離子受體 (anion receptors) 的製備中是重要的結構，儘管提供的文本中並未詳細說明具體的 fluoride 結合【41467_2024_Article_45464】。
> *   **水穩定結構的應用：** 水穩定 boroxine 結構在生物感測 (biosensing)、生物分離 (bioseparation)、藥物設計 (drug design)、親水性 COFs、分子結構 (molecular architectures) 以及可修復的水下黏著劑 (repairable underwater adhesive) 等應用方面具有潛力【41467_2024_Article_45464】。
> *   **材料特性：** Boroxines 在自癒合材料 (self-healing materials) 的製備中也被指出具有重要意義【4 …（完整內容見 JSON）

---

### ❌ Q09 · cross_paper

**問題**：Several papers target LAT1 for cancer therapy. What different therapeutic strategies targeting LAT1 are described across the literature, and how do they differ in mechanism?

- detected_paper：`None`
- 選出論文：['LAT1 ChemComm 2026', '1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026SI', 'cas0106-0279']
- gold_papers：['1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026', 's41421-024-00697-6']
- correctness candidate：`answer_for_judge`
- correctness：0.25　raw：2/5　judge：`structured_fact_audit_v1`
- judge reason：covered 1/7; missing F1, F2, F3, F4, F5, F7
- translation fidelity：0.5　raw：3/5　judge：`translation_fidelity_v1`
- translation reason：The translation retains significant English technical terms and phrases without translating them into Traditional Chinese, including "solid phase peptide synthesis", "cupper catalyzed azide-alkyne cycloaddition" (despite a partial parenthetical), "a-carbon", "tyrosine", "p-Boronophenylalanine", "the
- 論文選擇命中率：75.0%　檢索覆蓋率：100.0%　grounding：0.8
- 延遲：1105.9s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> 比較框架：
> - 路徑：【LAT1 ChemComm 2026】報告了 solid phase peptide synthesis (SPPS) followed by cupper catalyzed azide-alkyne cycloaddition (CuAAC)（固相肽合成隨後進行銅催化疊氮-炔環加成反應），產出 LffVLKK-4Phe 【LAT1 ChemComm 2026】。
> - 路徑：【1-s2.0-S1347861320300633-main】報告其開發，產出 JPH203 (IC50 values of 60e140 nM) 【1-s2.0-S1347861320300633-main】。
> - 路徑：【1-s2.0-S1936523321002047-main】報告其開發，產出 JPH203 【1-s2.0-S1936523321002047-main】。
> - 路徑：【cas0106-0279】報告 tyrosine 中 a-carbon 的甲基化賦予了 LAT1 選擇性，產出 p-Boronophenylalanine (BPA) / LAT1 選擇性 【cas0106-0279】。
>
> 核心權衡：
> - 同位素富集 (Isotopic enrichment)：基於 BNCT 是一種利用硼-10 (10B) 的核捕獲與裂變反應來治療惡性腫瘤的治療方式 (therapeutic modality) 【cas0106-0279】，且利用 p-Boronophenylalanine (BPA) 作為載體 (carrier) 【cas0106-0279】這一事實，可以合理推論其使用了同位素富集的硼-10 (10B) 材料。
>
> ---
> 📋 **答案品質報告**
>
> ✅ **整體論文依據率**：80.0%　高（答案高度忠實於論文內容）
>
> ⚠️  **以下 1 個陳述未找 …（完整內容見 JSON）

---
