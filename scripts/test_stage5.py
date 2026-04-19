# scripts/test_stage5.py
# 單獨測試 Stage 5（AnswerVerifier）用，跳過 Stage 1-4
#
# 使用方式：
#   conda activate llm_env
#   cd E:\Projects\rag_project
#   python scripts/test_stage5.py
#
# 調參數只需改下方 TEMPERATURE / PRESENCE_PENALTY，不用重跑完整 pipeline

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ══════════════════════════════════════════════════════════════
#  ← 在這裡調整 Stage 5 的參數
# ══════════════════════════════════════════════════════════════
TEMPERATURE      = 0.6
PRESENCE_PENALTY = 1.2
DISABLE_THINKING = False   # True = 關閉 qwen3.5 extended thinking（速度快很多）

# NLI 檢索設定（用於向量索引取 raw chunks）
TEST_QUESTION    = (
    "glycine 修飾 nZVI 的合成步驟中，具體使用了哪些試藥、"
    "比例與反應條件？gelatin aerogel 在其中扮演什麼角色？"
    "請盡量引用論文中的具體數值。"
)
TEST_PAPER_NAMES = ["1-s2.0-S2214714425005100-main"]  # 限定論文；留空 [] 則搜全庫
# ══════════════════════════════════════════════════════════════

import time
import json
import requests
from rag.answer_verifier import AnswerVerifier, VERIFY_SYSTEM_PROMPT, CORRECTION_SYSTEM_PROMPT
import config as cfg


class _TunableVerifier(AnswerVerifier):
    """
    AnswerVerifier 子類，允許在測試時動態注入 temperature / presence_penalty，
    以及全域控制 disable_thinking，不需要改動主程式碼。
    """

    def _call_ollama(self, model, system, prompt,
                     disable_thinking=False, on_status=None):
        options = {
            "temperature":      TEMPERATURE,
            "num_predict":      -1,
            "num_ctx":          cfg.STAGE5_NUM_CTX,
            "presence_penalty": PRESENCE_PENALTY,
        }
        # 全域 DISABLE_THINKING 優先於呼叫端傳入的 disable_thinking
        if DISABLE_THINKING or disable_thinking:
            options["thinking"] = False

        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model":  model,
                "system": system,
                "prompt": prompt,
                "stream": True,
                "options": options,
            },
            timeout=self.timeout,
            stream=True,
        )
        resp.raise_for_status()

        def _status(msg):
            if on_status:
                on_status(msg)
            else:
                print(msg)

        full_response = []
        thinking_printed = False

        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            try:
                chunk = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            thinking_chunk = chunk.get("thinking", "")
            response_chunk = chunk.get("response", "")

            if thinking_chunk:
                if not thinking_printed:
                    _status("\n  💭 [Verifier] === 思考過程開始 ===")
                    thinking_printed = True
                print(thinking_chunk, end="", flush=True)

            if response_chunk:
                if thinking_printed and not full_response:
                    _status("\n  💭 [Verifier] === 思考過程結束，正式輸出 ===\n")
                print(response_chunk, end="", flush=True)
                full_response.append(response_chunk)

            if chunk.get("done"):
                break

        print()
        return "".join(full_response).strip()


# ══════════════════════════════════════════════════════════════
#  測試資料：hardcode 一段 draft_answer + knowledge_base
#  可替換成從完整 pipeline 輸出的真實資料
# ══════════════════════════════════════════════════════════════

# ── 以下資料取自實際 pipeline run 的 test_run.log ──────────────────
# knowledge_base = Stage 3 知識蒸餾輸出（事實1-15）
# draft_answer   = Stage 4 初稿（run 仍進行中時擷取，【知識延伸與推測】段落截斷）
# 若 run 已完成，可將完整 Stage 4 輸出貼入 DRAFT_ANSWER 取代

# ── 以下資料取自實際 pipeline run 的 test_run.log ──────────────────
# knowledge_base = Stage 3 知識蒸餾輸出（事實1-15，含原始 LaTeX）
# draft_answer   = Stage 4 完整初稿（含 LaTeX，Stage 5 LLM 可直接處理）
# LaTeX→plain 轉換只在 citation_grounding.py 的 NLI 路徑執行，不影響此處

KNOWLEDGE_BASE = """
[事實1] 本研究旨在合成一種甘氨酸修飾的明膠碳氣凝膠載載納米零價鐵催化劑（NZVI@G-GEL），用於去除水中的四環素（TC）（來源：1-s2.0-S2214714425005100-main [Chunk 1]）
[事實2] NZVI@G-GEL 的合成在常壓環境下進行，無需氮氣保護（來源：1-s2.0-S2214714425005100-main [Chunk 1, Chunk 2]）
[事實3] 甘氨酸（Glycine）作為表面修飾劑，透過其氨基與明膠的羧基形成酰胺鍵，對 NZVI 產生螯合與錨定作用，以提升 NZVI 的電子選擇性（來源：1-s2.0-S2214714425005100-main [Chunk 1]）
[事實4] 明膠氣凝膠（gelatin aerogels）利用其超高孔隙率、超低質量密度和巨大的比表面積作為載入 NZVI 的基質，用以減輕腐蝕並提高污染物去除效率（來源：1-s2.0-S2214714425005100-main [Chunk 3]）
[事實5] 明膠氣凝膠憑藉生物相容性以及均勻的孔徑和互連性，為 NZVI 提供分散位點以加速質量傳遞速率（來源：1-s2.0-S2214714425005100-main [Chunk 3]）
[事實6] 明膠氣凝膠提供額外的吸附位點，使更多電子能被目標污染物捕捉，從而改善 NZVI 的電子選擇性（來源：1-s2.0-S2214714425005100-main [Chunk 3]）
[事實7] GEL 碳化氣凝膠的製備使用 20 wt% 的明膠（gelatin）（來源：1-s2.0-S2214714425005100-main [Chunk 2, Chunk 4]）
[事實8] GEL 碳化氣凝膠的製備條件為：在 60 °C 恆溫水浴攪拌 1 小時，注入 6 孔培養板冷卻至室溫形成凝膠，冷凍乾燥 24 小時，最後在 250 °C 下煅燒 2 小時（升溫速率 5 °C/min）（來源：1-s2.0-S2214714425005100-main [Chunk 2, Chunk 4]）
[事實9] G-GEL（甘氨酸修飾 GEL 碳化氣凝膠）的合成使用 0.6 wt% 的甘氨酸，其與 GEL 碳化氣凝膠的比例為 1:1（來源：1-s2.0-S2214714425005100-main [Chunk 2, Chunk 4]）
[事實10] G-GEL 的合成條件為：將甘氨酸添加到 GEL 碳化氣凝膠混合物中反應 12 小時，隨後在 85 °C 烘箱中乾燥 12 小時（來源：1-s2.0-S2214714425005100-main [Chunk 2, Chunk 4]）
[事實11] NZVI@G-GEL 的合成試劑與比例為：0.3 g G-GEL 材料、0.45 g $\text{FeSO}_4\cdot 7\text{H}_2\text{O}$、40 mL 水溶液以及過量的 $\text{KBH}_4$ 溶液（還原劑）（來源：1-s2.0-S2214714425005100-main [Chunk 2, Chunk 4]）
[事實12] NZVI@G-GEL 的合成步驟為：將 G-GEL 與 $\text{FeSO}_4\cdot 7\text{H}_2\text{O}$ 加入水溶液中攪拌 30 分鐘，以 5 秒/滴的速度滴加過量 $\text{KBH}_4$ 溶液，隨後靜置 30 分鐘（來源：1-s2.0-S2214714425005100-main [Chunk 2, Chunk 4]）
[事實13] NZVI@G-GEL 的後處理步驟為：透過過濾分離溶液，使用超純水和無水乙醇各清洗 3 次，最後密封於乾燥器中直至完全乾燥（來源：1-s2.0-S2214714425005100-main [Chunk 2]）
[事實14] 論文中未提及合成過程中的具體攪拌速度數值（來源：1-s2.0-S2214714425005100-main [Chunk 4]）
[事實15] NZVI@G-GEL 能在無需添加強氧化劑或外部能量的情況下，活化溶解氧（DO）產生自由基（$\cdot\text{OH}$、$\cdot\text{O}_2^-$）和非自由基（$^1\text{O}_2$）來降解 TC（來源：1-s2.0-S2214714425005100-main [Chunk 1]）
"""

DRAFT_ANSWER = """
根據您提供的已知事實清單，關於甘氨酸（Glycine）修飾的納米零價鐵催化劑（NZVI@G-GEL）之合成步驟、試劑比例及明膠氣凝膠角色的詳細回答如下：

## 【論文直接依據】

**1. 合成步驟與反應條件**
本研究的合成過程分為三個階段：製備 GEL 碳化氣凝膠 $\rightarrow$ 合成 G-GEL（甘氨酸修飾） $\rightarrow$ 載入 NZVI。

*   **第一階段：GEL 碳化氣凝膠的製備**
    *   **試劑與比例**：使用 20 wt% 的明膠（gelatin）【1-s2.0-S2214714425005100-main】。
    *   **操作條件**：在 60 °C 恆溫水浴中攪拌 1 小時 $\rightarrow$ 注入 6 孔培養板冷卻至室溫形成凝膠 $\rightarrow$ 冷凍乾燥 24 小時 $\rightarrow$ 在 250 °C 下煅燒 2 小時（升溫速率為 5 °C/min）【1-s2.0-S2214714425005100-main】。
*   **第二階段：G-GEL（甘氨酸修飾）的合成**
    *   **試劑與比例**：使用 0.6 wt% 的甘氨酸，其與 GEL 碳化氣凝膠的比例為 1:1【1-s2.0-S2214714425005100-main】。
    *   **操作條件**：將甘氨酸添加到 GEL 碳化氣凝膠混合物中反應 12 小時 $\rightarrow$ 在 85 °C 烘箱中乾燥 12 小時【1-s2.0-S2214714425005100-main】。
*   **第三階段：NZVI@G-GEL 的合成**
    *   **試劑與用量**：0.3 g G-GEL 材料、0.45 g $\text{FeSO}_4\cdot 7\text{H}_2\text{O}$、40 mL 水溶液以及過量的 $\text{KBH}_4$ 溶液（還原劑）【1-s2.0-S2214714425005100-main】。
    *   **操作步驟**：將 G-GEL 與 $\text{FeSO}_4\cdot 7\text{H}_2\text{O}$ 加入水溶液中攪拌 30 分鐘 $\rightarrow$ 以 5 秒/滴的速度滴加過量 $\text{KBH}_4$ 溶液 $\rightarrow$ 靜置 30 分鐘【1-s2.0-S2214714425005100-main】。
    *   **後處理**：過濾分離 $\rightarrow$ 使用超純水和無水乙醇各清洗 3 次 $\rightarrow$ 密封於乾燥器中直至完全乾燥【1-s2.0-S2214714425005100-main】。
*   **環境條件**：合成在常壓環境下進行，無需氮氣保護【1-s2.0-S2214714425005100-main】。
*   **缺失資訊**：論文中未提及具體的攪拌速度數值【1-s2.0-S2214714425005100-main】。

**2. 明膠氣凝膠（Gelatin Aerogel）的角色**
*   **作為基質（Matrix）**：利用其超高孔隙率、超低質量密度和巨大的比表面積來載入 NZVI，旨在減輕腐蝕並提高污染物去除效率【1-s2.0-S2214714425005100-main】。
*   **加速質量傳遞**：憑藉生物相容性以及均勻的孔徑和互連性，為 NZVI 提供分散位點，從而加速質量傳遞速率【1-s2.0-S2214714425005100-main】。
*   **改善電子選擇性**：提供額外的吸附位點，使更多電子能被目標污染物捕捉【1-s2.0-S2214714425005100-main】。

**3. 甘氨酸（Glycine）的角色**
*   **表面修飾與錨定**：透過其氨基與明膠的羧基形成酰胺鍵，對 NZVI 產生螯合與錨定作用，以提升 NZVI 的電子選擇性【1-s2.0-S2214714425005100-main】。

---

## 【跨文獻推論】

綜合推論（基於【1-s2.0-S2214714425005100-main】之各項事實）：
**NZVI@G-GEL 的高效能源於「結構支撐 $\rightarrow$ 化學錨定 $\rightarrow$ 活性活化」的協同作用。**
其邏輯推導如下：
1.  **結構層面**：明膠氣凝膠先經過 250 °C 碳化形成碳化氣凝膠，提供了高比表面積的物理骨架，解決了 NZVI 易聚集的問題（事實 4, 8）。
2.  **化學層面**：甘氨酸並非簡單的混合，而是透過酰胺鍵與骨架結合，將 NZVI 牢牢「錨定」在載體上，這不僅防止了活性物質流失，更從化學性質上優化了電子的傳遞路徑（事實 3）。
3.  **功能層面**：這種複合結構使得材料能在無需外部能量的情況下，直接活化水中的溶解氧（DO）產生自由基（$\cdot\text{OH}, \cdot\text{O}_2^-$）與非自由基（$^1\text{O}_2$），將物理吸附（氣凝膠孔隙）與化學降解（NZVI 催化）有機結合，共同實現對四環素（TC）的高效去除（事實 1, 15）。

---

## 【知識延伸與推測】

**1. 關於攪拌速度的推測**
模型推測（文獻依據不足）：雖然論文未提及攪拌速度，但根據納米粒子合成的一般原理，在滴加 $\text{KBH}_4$ 還原劑時，通常需要較高強度的攪拌（如 500-1000 rpm）以確保還原劑迅速分散，避免局部濃度過高導致 NZVI 粒子過度生長或聚集。若實驗中攪拌不足，可能會影響 NZVI 在 G-GEL 上的分散均勻度。

**2. 關於碳化溫度（250 °C）的影響**
模型推測（文獻依據不足）：選擇 250 °C 進行煅燒可能是一個平衡點。過低的溫度可能無法將明膠充分碳化以形成穩定的導電骨架；而過高的溫度可能會導致氣凝膠結構坍塌或孔隙率大幅下降。碳化過程可能引入了部分 $\pi$-電子系統，這與事實 3 和事實 6 提到的「提升電子選擇性」有潛在的關聯，因為碳基底通常能輔助電子從 NZVI 傳遞至污染物。

**3. 關於無需氮氣保護的特殊性**
模型推測（文獻依據不足）：傳統 NZVI 合成通常需要嚴格除氧（如通氮氣）以防止 $\text{Fe}^0$ 立即氧化。本研究能在常壓且無需氮氣保護下進行（事實 2），推測是因為 G-GEL 的孔隙結構或甘氨酸的螯合作用為 NZVI 提供了一定的保護層，或者該研究旨在開發一種更簡便、更具工業可行性的合成路徑，容許一定程度的表面氧化層存在，而該氧化層可能反而有利於後續活化溶解氧（DO）的過程（事實 15）。
"""

# ══════════════════════════════════════════════════════════════

def _extract_section(text: str, keyword_pattern: str) -> str:
    """抽取 ## 標題含 keyword_pattern 的段落，直到下一個 ## 或結尾。"""
    import re
    matches = re.findall(
        rf'(##[^\n]*(?:{keyword_pattern})[^\n]*\n[\s\S]*?)(?=\n##|\Z)',
        text
    )
    return "\n\n".join(m.strip() for m in matches)


def _get_raw_chunks() -> list[dict]:
    """
    從向量索引做 hybrid retrieval，回傳原始 chunk 文字。
    比對對象是 PDF 原文，而非 sub_answers（LLM 生成的摘要）。
    """
    print("  🔍 從向量索引取得原始 chunks（需要 embedding 初始化）...")
    from rag.llm_client import init_llm_and_embedding
    init_llm_and_embedding()

    from main import paper_engines

    filter_set = set(TEST_PAPER_NAMES) if TEST_PAPER_NAMES else None
    chunks = []

    for name, engine in paper_engines.items():
        if filter_set and name not in filter_set:
            continue
        retriever = engine._retriever  # QueryFusionRetriever（vector + BM25）
        nodes = retriever.retrieve(TEST_QUESTION)
        for node_with_score in nodes:
            chunks.append({
                "id":   f"{name[:25]}-{node_with_score.node.node_id[:8]}",
                "text": node_with_score.node.get_content(),
                "source": name,
                "score":  node_with_score.score or 0.0,
            })

    chunks.sort(key=lambda c: c["score"], reverse=True)
    print(f"  → 取得 {len(chunks)} 個 raw chunks")
    return chunks


def _run_nli_grounding(answer: str):
    from rag.citation_grounding import (
        split_into_sentences,
        check_citation_grounding,
        format_grounding_report,
        compute_grounding_score,
    )

    raw_chunks = _get_raw_chunks()

    # ── 各段落分開評分 ──────────────────────────────────────────
    # 段落定義與期望值：
    #   【論文直接依據】→ 直引原文，entailment 應高（目標 ≥ 0.7）
    #   【跨文獻推論】  → 跨文獻合成推論，entailment 預期偏低屬正常，
    #                    重點看 contradiction（推論不應違背原文）
    #   【知識延伸與推測】→ 模型外推，entailment 低是預期行為，僅供參考
    sections = [
        ("【論文直接依據】", "論文直接依據|直接依據|直引",
         "entailment 應 ≥ 0.7；低分代表答案超出原文"),
        ("【跨文獻推論】",   "跨文獻推論|推論",
         "entailment 預期偏低（正常）；contradiction 若高才是問題"),
        ("【知識延伸與推測】", "知識延伸|推測",
         "entailment 低屬預期，僅做 contradiction 偵測"),
    ]

    print(f"\n{'='*65}")
    print("NLI Citation Grounding 審查（對象：PDF raw chunks）")
    print(f"  raw chunks 數量 = {len(raw_chunks)}")
    print(f"{'='*65}")

    t_total = time.time()
    has_any = False

    for section_label, pattern, expectation in sections:
        section_text = _extract_section(answer, pattern)
        if not section_text:
            continue
        sentences = split_into_sentences(section_text)
        if not sentences:
            continue

        has_any = True

        # ── 印出斷句結果（方便 debug split_into_sentences 邏輯）──────
        print(f"\n  {'─'*60}")
        print(f"  {section_label}  ← 斷句結果（共 {len(sentences)} 句）")
        for i, s in enumerate(sentences, 1):
            print(f"    [{i:02d}] {s[:120]}" + ("…" if len(s) > 120 else ""))

        t0 = time.time()
        results = check_citation_grounding(sentences, raw_chunks)
        score = compute_grounding_score(results)
        elapsed = time.time() - t0

        contradictions = [r for r in results if r.get("contradiction_detected")]

        print(f"\n  期望：{expectation}")
        print(f"  grounding score = {score:.1%}　({elapsed:.1f}s)")

        # ── 逐句 NLI 分數 ─────────────────────────────────────────────
        print(f"\n  逐句 NLI 分數：")
        for r in results:
            status = r.get("status", "?")
            ent    = r.get("confidence", 0.0)
            flag   = " ⚠️ CONTRA" if r.get("contradiction_detected") else ""
            print(f"    [{status:<10}] ent={ent:.3f}{flag}")
            print(f"             {r['sentence'][:100]}" + ("…" if len(r['sentence']) > 100 else ""))

        if contradictions:
            print(f"\n  ⚠️  發現 {len(contradictions)} 個 contradiction：")
            for r in contradictions:
                print(f"     - {r['sentence'][:80]}...")
        else:
            print(f"\n  ✅ 無 contradiction 偵測")

    if not has_any:
        # fallback：全文
        print("  （找不到分段標題，改做全文 grounding）")
        sentences = split_into_sentences(answer)
        results = check_citation_grounding(sentences, raw_chunks)
        print(format_grounding_report(results))
        print(f"  全文 grounding score = {compute_grounding_score(results):.1%}")

    print(f"\n  ⏱ NLI 總耗時：{time.time() - t_total:.1f}s")


def main():
    print("=" * 65)
    print(f"Stage 5 單獨測試")
    print(f"  temperature      = {TEMPERATURE}")
    print(f"  presence_penalty = {PRESENCE_PENALTY}")
    print(f"  disable_thinking = {DISABLE_THINKING}")
    print(f"  model            = {cfg.VERIFY_MODEL}")
    print(f"  num_ctx          = {cfg.STAGE5_NUM_CTX}")
    print("=" * 65)

    verifier = _TunableVerifier()

    t0 = time.time()
    final_answer = verifier.verify_and_correct(
        draft_answer=DRAFT_ANSWER,
        knowledge_base=KNOWLEDGE_BASE,
    )
    stage5_elapsed = time.time() - t0

    print(f"\n{'─'*65}")
    print("★ Stage 5 輸出：")
    print(f"{'─'*65}")
    print(final_answer)
    print(f"{'─'*65}")
    minutes = int(stage5_elapsed // 60)
    seconds = int(stage5_elapsed % 60)
    print(f"\n⏱ Stage 5 總耗時：{minutes}分{seconds}秒")

    # ── NLI Citation Grounding ────────────────────────────────
    _run_nli_grounding(final_answer)


if __name__ == "__main__":
    main()
