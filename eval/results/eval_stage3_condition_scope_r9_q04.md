# Eval Report — `stage3_condition_scope_r9_q04`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-21 21:40
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 1.0 |
| Correctness judge 覆蓋 | 1/1（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 0.5 |
| Translation judge 覆蓋 | 1/1（N/A 0） |
| 平均 grounding 分數 | 0.778 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 100.0% |
| 平均 Stage 2 evidence 覆蓋率 | 100.0% |
| 平均總延遲 | 661.5s |
| 平均 planning 延遲 | 6.2s |
| 平均 retrieval 延遲 | 4.8s |
| └ Phase A embed/vector/BM25 | 4.7s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 24.4s |
| └ 其中 NLI | 3.8s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ⚠️ | Q04 | table_lookup | 1.0 | 0.5 | 100.0% | 100.0% | 100.0% | 0.778 | 661.5s | C0/U4 |

## 逐題細節

### ⚠️ Q04 · table_lookup

**問題**：What are the main degradation products and impurities of BPA identified by HPLC, and under which storage conditions do they form?

- detected_paper：`None`
- 選出論文：['1-s2.0-S0731708511003451-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water']
- gold_papers：['1-s2.0-S0731708511003451-main']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/7
- translation fidelity：0.5　raw：3/5　judge：`translation_fidelity_v2`
- translation reason：1 material and 0 minor semantic errors; The source specifies the degradation product is generated 'with respect to BPA mass' (相對於BPA質量), but this specific quantitative basis is omitted in the target translation, which only states 'produced approximately 1% of phenylalanine'. This omission removes a critical scientific constraint defining
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：0.778
- 延遲：661.5s　問題標記：{'conflicts': 0, 'unsupported': 4}

**答案預覽**：

> **HPLC 雜質與降解產物鑑定**
>
> Boron phenylalanine (BPA, 硼苯丙氨酸) 在 230 nm 處定量，保留時間為 5.3 min 【Source: 1-s2.0-S0731708511003451-main】。Tyrosine (酪胺酸) 被鑑定為雜質與降解產物，在 230 nm 處定量，保留時間為 4.5 min 【Source: 1-s2.0-S0731708511003451-main】。Phenylalanine (苯丙氨酸) 在 11.0 min 洗脫並在 256 nm 處定量，先前研究者將其報告為雜質 【Source: 1-s2.0-S0731708511003451-main】。合成雜質 BrPD (2-(4-Bromophenyl)-[1,3]dioxane)、FBBA (4-Formylbenzeneboronic acid) 與 BDPA 在濃度為 0.5 µg/ml 時可檢出，但在現有的 BPA 樣本中未觀察到 【Source: 1-s2.0-S0731708511003451-main】。BrPD 在 256 nm 處檢出，洗脫時間為 17.3 min，而 FBBA 與 BDPA 則在 256 nm 或 270 nm 處檢出 【Source: 1-s2.0-S0731708511003451-main】。
>
> **原粉配方之穩定性**
>
> 作為原粉 (raw powder)，BPA 在 55 °C 下儲存 6 個月或在 40 °C 下儲存 12 個月時，未產生可檢出的降解 【Source: 1-s2.0-S0731708511003451-main】。此穩定性是在不存在 mannitol (甘露醇) 及藥品中發現的其他賦形劑的情況下觀察到的 【Source: 1-s2.0-S0731708511003451-main】。
>
>  …（完整內容見 JSON）

---
