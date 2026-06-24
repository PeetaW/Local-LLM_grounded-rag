# Eval Report — `gate_p2`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-06-25 01:15
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.75 |
| 平均 grounding 分數 | 0.625 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 874.4s |
| 平均 planning 延遲 | 5.7s |
| 平均 retrieval 延遲 | 339.2s |
| 平均 grounding 延遲 | 37.7s |
| └ 其中 NLI | 15.4s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q08 | cross_paper | 100.0% | 100.0% | 0.75 | 918.7s | C0/U5 |
| ⚠️ | Q12 | false_premise | N/A | N/A | 0.5 | 830.2s | C0/U3 |

## 逐題細節

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.75
- 延遲：918.7s　問題標記：{'conflicts': 0, 'unsupported': 5}

**答案預覽**：

> ## 【論文直接依據】
> 
> **綜述目標與範圍**
> * 【CMDC-20-e202500059】旨在回顧並比較合成 4-borono-L-phenylalanine (L-BPA) 的學術與專利方法 (【CMDC-20-e202500059】)。
> * 【CMDC-20-e202500059】的評估重點包括可擴展性、成本效益以及安全性 (【CMDC-20-e202500059】)。
> * 【CMDC-20-e202500059】中的安全性考量特別考慮到了 $^{10}\text{B}$ 同位素富集的成本高昂 (【CMDC-20-e202500059】)。
> 
> **混合合成路徑 (【bbb0683】)**
> * 論文 【bbb0683】提出了一種結合對稱烷基化與酶水解的「混合製程」，用以合成光學純的 L-BPA (【bbb0683】)。
> * 此混合製程的特點是操作較簡單且反應步驟較少 (【bbb0683】)。
> * 在 【bbb0683】路徑的第一步中，$(2R)-(-)$-2-isopropyl-3,6-dimethoxypyrazine (3) 與受保護的 4-bromomethylphenylboronic acid 衍生物 (2) 發生反應 (【bbb0683】)。
> * 此第一步反應在 THF 中於 -78°C 下進行 (【bbb0683】)。
> * 第一步產物的立體選擇性為 74% e.e. (【bbb0683】)。
> * 在第二步中，使用 0.1 N 和 1 N 的鹽酸處理可產生 L-BPA methyl ester (6) (【bbb0683】)。
> * 生產 L-BPA methyl ester 的產率為 85% (【bbb0683】)。
> * 在第三步中，使用胰蛋白酶 (trypsin) 在 pH 5.0 下水解化合物 6 (【bbb0683】)。
> * 此最終步驟產生的 L-B …（完整內容見 JSON）

---

### ⚠️ Q12 · false_premise

**問題**：Since BPA is administered orally in BNCT, what oral bioavailability values are reported in these papers?

- detected_paper：`None`
- 選出論文：['cas0106-0279', 'Chem Biol Drug Des - 2013 - Achilli - Folic Acid‐Conjugated 4‐Amino‐Phenylboronate  a Boron‐Containing Compound Designed']
- gold_papers：[]
- 論文選擇命中率：N/A　檢索覆蓋率：N/A　grounding：0.5
- 延遲：830.2s　問題標記：{'conflicts': 0, 'unsupported': 3}

**答案預覽**：

> ## 【論文直接依據】
> 
> * 提供之文本中未報告 BPA 的口服生物利用度數值。
> * BNCT 的硼載體需透過靜脈輸注給藥 (【Chem Biol Drug Des - 2013 - Achilli - Folic Acid‐Conjugated 4‐Amino‐Phenylboronate a Boron‐Containing Compound Designed】)。
> 
> ## 【跨文獻推論】
> 
> * 在提供的事實中，缺乏足夠的文獻依據來確定 BPA 的口服生物利用度數值。此外，問題的前提（BPA 為口服給藥）與 [Fact 4] 中提供的證據（BNCT 硼載體需要靜脈輸注）之間存在矛盾。
> 
> ## 【知識延伸與推測】
> 
> * 模型推測（文獻依據不足）：在臨床實踐中，p-boronophenylalanine (BPA) 通常採取靜脈給藥而非口服，以確保精確的劑量並將其最佳化地遞送至腫瘤部位，且通常與果糖聯合使用，以增強其透過氨基酸轉運體的攝取。提供文本中缺乏口服生物利用度數據，可能反映了這種標準的靜脈給藥途徑。
> 
> ---
> 📋 **答案品質報告**
> 
> 📊 **分段論文依據率：**
> 
>   ⚠️ 【論文直接依據】：50.0%（1/2 句）
>   ❌ 【跨文獻推論】：0.0%（0/1 句）  ← 跨論文推論，低分為預期範圍
>   ❌ 【知識延伸推測】：0.0%（0/1 句）  ← 知識延伸推測，低分為預期範圍
> 
> ⚠️ **直引依據率**：50.0%　中（部分直引陳述需確認）
> 
> ⚠️  **以下 3 個陳述未找到明確論文依據，請謹慎參考：**
> 
> - * No oral bioavailability values for BPA are reported in the provided texts.（信心度：31.5%，最近似來源：cas0106-0279-d7164215）
> - * The …（完整內容見 JSON）

---
