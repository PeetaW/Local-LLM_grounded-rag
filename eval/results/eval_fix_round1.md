# Eval Report — `fix_round1`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-06-22 08:27
- 題數：4

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.812 |
| 平均 grounding 分數 | 0.933 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 628.0s |
| 平均 planning 延遲 | 5.2s |
| 平均 retrieval 延遲 | 240.3s |
| 平均 grounding 延遲 | 28.3s |
| └ 其中 NLI | 11.8s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q01 | single_paper | 100.0% | 100.0% | 1.0 | 896.1s | C0/U0 |
| ✅ | Q04 | table_lookup | 100.0% | 100.0% | 0.8 | 641.9s | C0/U9 |
| ✅ | Q11 | out_of_scope | N/A | N/A | — | 332.4s | C0/U0 |
| ✅ | Q12 | false_premise | N/A | N/A | 1.0 | 641.5s | C0/U1 |

## 逐題細節

### ✅ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：896.1s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
> * JPH203 對於 HT-29 細胞中 LAT1 的預孵育抑制效果具有濃度依賴性 【1-s2.0-S1347861320300633-main】。
> * JPH203 對於 HT-29 細胞中 LAT1 的預孵育抑制效果具有時間依賴性 【1-s2.0-S1347861320300633-main】。
> * 在 120 分鐘的預孵育實驗中，除 0.001 mM 濃度外，JPH203 濃度為 0.03, 0.1, 0.3, 和 30 mM 時，亮氨酸（leucine）的攝取量分別下降至對照組的 63%, 51%, 42%, 30%, 和 25% 【1-s2.0-S1347861320300633-main】。
> * JPH203 預孵育抑制的 $\text{IC}_{50}$ 值為 $193 \pm 50\text{ nM}$ 【1-s2.0-S1347861320300633-main】。
> * 使用 10 mM JPH203，預孵育 30 分鐘將攝取量降低至對照組的 43% 【1-s2.0-S1347861320300633-main】。
> * 使用 10 mM JPH203，預孵育 60 分鐘將攝取量降低至對照組的 32% 【1-s2.0-S1347861320300633-main】。
> * 使用 10 mM JPH203，預孵育 120 分鐘將攝取量降低至對照組的 27% 【1-s2.0-S1347861320300633-main】。
> * 預孵育與共同孵育（co-incubation）結合時，能顯著增強 JPH203 的抑制效力，產生協同效應 【1-s2.0-S1347861320300633-main】。
> * 單獨共同孵育的 $\text{IC}_{50}$ 值為 $99.2 \pm 11.0\text{ nM}$ 【1-s2.0-S1347861 …（完整內容見 JSON）

---

### ✅ Q04 · table_lookup

**問題**：What are the main degradation products and impurities of BPA identified by HPLC, and under which storage conditions do they form?

- detected_paper：`None`
- 選出論文：['1-s2.0-S0731708511003451-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water']
- gold_papers：['1-s2.0-S0731708511003451-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.8
- 延遲：641.9s　問題標記：{'conflicts': 0, 'unsupported': 9}

**答案預覽**：

> ## 【論文直接依據】
> 
> **降解產物與條件**
> * 在鹼性和氧化條件下，酪胺酸 (Tyrosine) 是 BPA 的降解產物 [Chunk 1], [Chunk 2]。
> * 在含有 $6\text{ mM H}_2\text{O}_2$ 的水溶液中，生成酪胺酸的反應極快 [Chunk 1], [Chunk 2]。
> * 當 BPA 溶解於 $100\text{ mM NaOH}$ 並在 $55^\circ\text{C}$ 下孵育 24 小時時，會降解為酪胺酸 [Chunk 1], [Chunk 2]。
> * 在冷凍乾燥的 BPA/甘露醇 (mannitol) 藥品中，苯丙胺酸 (Phenylalanine) 是 BPA 的降解產物 [Chunk 1], [Chunk 2]。
> * 苯丙胺酸的生成速度較慢且與溫度相關 [Chunk 1], [Chunk 2]。
> * 當冷凍乾燥的 BPA/甘露醇在 $40^\circ\text{C}$ 下儲存 6 個月時，會產生約 1% 的苯丙胺酸（相對於 BPA 質量） [Chunk 1], [Chunk 2]。
> * 在水性鹼性條件下，BPA 經由原脫硼反應 (protodeborylation) 生成母體芳環化合物 (parent arene) [Chunk 2]。
> 
> **HPLC 鑑定數據**
> * 酪胺酸的 HPLC 檢測波長為 230 nm [Chunk 1]。
> * 酪胺酸的 HPLC 保留時間為 4.5 分鐘 [Chunk 1]。
> * 苯丙胺酸的 HPLC 檢測波長為 256 nm [Chunk 1]。
> * 苯丙胺酸的 HPLC 保留時間為 11.0 分鐘 [Chunk 1]。
> 
> **合成雜質**
> * BrPD (2-(4-Bromophenyl)-[1,3]dioxane) 是一種合成雜質 [Chunk 1]。
> * BrPD 的 HP …（完整內容見 JSON）

---

### ✅ Q11 · out_of_scope

**問題**：What are the phase III clinical trial overall survival results for BNCT in glioblastoma patients?

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', '1-s2.0-S0378517325007926-main', 'cas0106-0279']
- gold_papers：[]
- 論文選擇命中率：N/A　檢索覆蓋率：N/A　grounding：—
- 延遲：332.4s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **資料來源說明**：本地學術文獻資料庫中未找到與此問題直接相關的內容。以下回答來自模型自身知識，非論文原文，請謹慎參考並自行查證。
> 
> **此為模型推測，建議查閱原始文獻確認。**
> 
> 針對您詢問關於硼中子捕獲治療（Boron Neutron Capture Therapy, BNCT）在膠質母細胞瘤（Glioblastoma, GBM）患者中的第三期臨床試驗總生存期（Overall Survival, OS）結果，說明如下：
> 
> **1. 數據存在性分析**
> 首先必須指出，目前在國際主流醫學文獻中，**尚未有針對 BNCT 治療膠質母細胞瘤且具備大規模、隨機對照之第三期臨床試驗（Phase III Clinical Trial）所產生的定論性總生存期數據。**
> 
> 大多數關於 BNCT 的研究目前仍處於第一期（安全性評估）或第二期（初步療效探索）臨床試驗階段。由於 BNCT 需要特殊的中子源設備（如加速器中子源）以及特定的硼載體藥物，其試驗規模通常較小，難以在短期內完成符合第三期標準的大樣本量隨機對照試驗。
> 
> **2. 定性說明**
> 雖然缺乏第三期試驗的統計數據，但從現有的早期臨床研究中可以觀察到以下定性趨勢：
> *   **治療機制：** BNCT 旨在透過硼化合物在腫瘤細胞內富集，並利用熱中子誘導核反應產生高能 $\alpha$ 粒子與鋰原子核，實現細胞級別的精準破壞。
> *   **初步觀察：** 在部分小規模研究中，BNCT 被認為對於某些特定亞型的惡性腦腫瘤具有潛在的治療效益，且在降低副作用方面展現出可能性。
> *   **限制因素：** 由於缺乏與標準治療（如 Stupp protocol：手術 + 放療 + 化療）的大規模對比數據，目前無法定論 BNCT 是否能顯著延長 GBM 患者的總生存期。
> 
> **3. 結論**
> 您所尋找的「第三期臨床試驗 OS 結果」在目前 …（完整內容見 JSON）

---

### ✅ Q12 · false_premise

**問題**：Since BPA is administered orally in BNCT, what oral bioavailability values are reported in these papers?

- detected_paper：`None`
- 選出論文：['cas0106-0279', 'Chem Biol Drug Des - 2013 - Achilli - Folic Acid‐Conjugated 4‐Amino‐Phenylboronate  a Boron‐Containing Compound Designed']
- gold_papers：[]
- 論文選擇命中率：N/A　檢索覆蓋率：N/A　grounding：1.0
- 延遲：641.5s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> ## 【論文直接依據】
> * BNCT 的硼載體透過靜脈注射給藥 【Chem Biol Drug Des - 2013 - Achilli - Folic Acid‐Conjugated 4‐Amino‐Phenylboronate a Boron‐Containing Compound Designed】。
> 
> ## 【跨文獻推論】
> * 跨論文推論（基於 【Chem Biol Drug Des - 2013 - Achilli - Folic Acid‐Conjugated 4‐Amino‐Phenylboronate a Boron‐Containing Compound Designed】）：提供的文本中不包含 BPA 的任何口服生物利用度數值；此外，BPA 為口服給藥的假設與 BNCT 硼載體透過靜脈注射給藥的陳述相矛盾。
> 
> ## 【知識延伸與推測】
> * 模型推測（文獻依據不足）：在臨床實踐中，p-boronophenylalanine (BPA) 通常透過靜脈注射給藥，以確保對血硼濃度的精確控制，並避開可能影響所請求之生物利用度數值的胃腸道吸收或首過效應（first-pass metabolism）問題。這些論文中缺乏口服生物利用度數據，與其作為靜脈注射劑的標準用途一致。
> 
> ---
> 📋 **答案品質報告**
> 
> 📊 **分段論文依據率：**
> 
>   ✅ 【論文直接依據】：100.0%（1/1 句）
>   ✅ 【跨文獻推論】：100.0%（1/1 句）  ← 跨論文推論，低分為預期範圍
>   ❌ 【知識延伸推測】：0.0%（0/1 句）  ← 知識延伸推測，低分為預期範圍
> 
> ✅ **直引依據率**：100.0%　高（直接引用高度忠實於論文）
> 
> ⚠️  **以下 1 個陳述未找到明確論文依據，請謹慎參考：**
> 
> - * Model speculation (insufficien …（完整內容見 JSON）

---
