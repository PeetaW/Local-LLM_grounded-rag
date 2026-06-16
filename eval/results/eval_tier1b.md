# Eval Report — `tier1b`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-06-17 00:50
- 題數：12

## 彙總

| 指標 | 值 |
|------|-----|
| 平均 grounding 分數 | N/A |
| 平均論文選擇命中率 | 97.5% |
| 平均檢索覆蓋率 | 97.5% |
| 平均總延遲 | N/A |
| 平均 planning 延遲 | N/A |
| 平均 retrieval 延遲 | N/A |
| 平均 grounding 延遲 | N/A |
| └ 其中 NLI | N/A |
| └ 其中 gemma4 | N/A |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q01 | single_paper | 100.0% | 100.0% | — | N/A | C0/U0 |
| ✅ | Q02 | single_paper | 100.0% | 100.0% | — | N/A | C0/U0 |
| ✅ | Q03 | figure_dependent | 100.0% | 100.0% | — | N/A | C0/U0 |
| ✅ | Q04 | table_lookup | 100.0% | 100.0% | — | N/A | C0/U0 |
| ✅ | Q05 | single_paper | 100.0% | 100.0% | — | N/A | C0/U0 |
| ✅ | Q06 | multi_chunk | 100.0% | 75.0% | — | N/A | C0/U0 |
| ✅ | Q07 | figure_dependent | 100.0% | 100.0% | — | N/A | C0/U0 |
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | — | N/A | C0/U0 |
| ⚠️ | Q09 | cross_paper | 75.0% | 100.0% | — | N/A | C0/U0 |
| ✅ | Q10 | contradiction_check | 100.0% | 100.0% | — | N/A | C0/U0 |
| ✅ | Q11 | out_of_scope | N/A | N/A | — | N/A | C0/U0 |
| ✅ | Q12 | false_premise | N/A | N/A | — | N/A | C0/U0 |

## 逐題細節

### ✅ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：—
- 延遲：N/A　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 

---

### ✅ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：—
- 延遲：N/A　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 

---

### ✅ Q03 · figure_dependent

**問題**：According to the cryo-EM structure, how does JPH203 bind to and inhibit the LAT1-4F2hc complex?

- detected_paper：`None`
- 選出論文：['s41421-024-00697-6', '1-s2.0-S1347861320300633-main']
- gold_papers：['s41421-024-00697-6']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：—
- 延遲：N/A　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 

---

### ✅ Q04 · table_lookup

**問題**：What are the main degradation products and impurities of BPA identified by HPLC, and under which storage conditions do they form?

- detected_paper：`None`
- 選出論文：['1-s2.0-S0731708511003451-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water']
- gold_papers：['1-s2.0-S0731708511003451-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：—
- 延遲：N/A　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 

---

### ✅ Q05 · single_paper

**問題**：What is the solvent-free green protocol for N-Boc protection of amines using picric acid as a catalyst, and what reaction conditions are used?

- detected_paper：`None`
- 選出論文：['1-s2.0-S2773223124000268-main']
- gold_papers：['1-s2.0-S2773223124000268-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：—
- 延遲：N/A　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 

---

### ✅ Q06 · multi_chunk

**問題**：How does poly(vinyl alcohol) enhance the therapeutic effect of 4-L-boronophenylalanine in neutron capture therapy of thoracic tumors? Explain the proposed mechanism with supporting data.

- detected_paper：`None`
- 選出論文：['1-s2.0-S0378517325007926-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'CMDC-20-e202500059']
- gold_papers：['1-s2.0-S0378517325007926-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：75.0%　grounding：—
- 延遲：N/A　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 

---

### ✅ Q07 · figure_dependent

**問題**：What is the water-stable boroxine structure reported, and what role do the dynamic covalent bonds play in its fluoride binding and hydrogel formation?

- detected_paper：`None`
- 選出論文：['41467_2024_Article_45464', 's41467-024-45464-z (1)']
- gold_papers：['41467_2024_Article_45464']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：—
- 延遲：N/A　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 

---

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：—
- 延遲：N/A　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 

---

### ⚠️ Q09 · cross_paper

**問題**：Several papers target LAT1 for cancer therapy. What different therapeutic strategies targeting LAT1 are described across the literature, and how do they differ in mechanism?

- detected_paper：`None`
- 選出論文：['LAT1 ChemComm 2026', '1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026SI', 'cas0106-0279']
- gold_papers：['1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026', 's41421-024-00697-6']
- 論文選擇命中率：75.0%　檢索覆蓋率：100.0%　grounding：—
- 延遲：N/A　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 

---

### ✅ Q10 · contradiction_check

**問題**：Do the papers agree on the hydrolytic stability of boroxine structures? Describe any agreements or differences between them.

- detected_paper：`None`
- 選出論文：['Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation', '41467_2024_Article_45464', 's41467-024-45464-z (1)', 'a-method-for-the-deprotection-of-alkylpinacolyl-boronate-esters']
- gold_papers：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：—
- 延遲：N/A　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 

---

### ✅ Q11 · out_of_scope

**問題**：What are the phase III clinical trial overall survival results for BNCT in glioblastoma patients?

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', '1-s2.0-S0378517325007926-main', 'cas0106-0279']
- gold_papers：[]
- 論文選擇命中率：N/A　檢索覆蓋率：N/A　grounding：—
- 延遲：N/A　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 

---

### ✅ Q12 · false_premise

**問題**：Since BPA is administered orally in BNCT, what oral bioavailability values are reported in these papers?

- detected_paper：`None`
- 選出論文：['cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：[]
- 論文選擇命中率：N/A　檢索覆蓋率：N/A　grounding：—
- 延遲：N/A　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 

---
