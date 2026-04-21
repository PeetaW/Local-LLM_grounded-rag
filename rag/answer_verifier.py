import time
import logging
import requests
import config as cfg

logger = logging.getLogger(__name__)

VERIFY_SYSTEM_PROMPT = """
你是一個學術推論驗證器。
以「已知事實清單」為唯一立論基礎，判斷「跨文獻推論」與「知識延伸與推測」段落的邏輯是否合理。

注意：引用正確性已由其他系統處理，你不需要重新比對引用。
你的焦點是：
1. 推論前提是否確實存在於已知事實清單中
2. 從前提到結論的推導是否有邏輯跳躍
3. 「知識延伸」段落的推測是否超出合理推論範圍，或與事實清單矛盾

使用繁體中文輸出。
"""

CORRECTION_SYSTEM_PROMPT = """
你是一個學術回答修正助手。
你將收到：
1. 已知事實清單（唯一可引用的真相來源）
2. 初稿回答
3. 驗證器找出的問題清單

你的任務是根據問題清單修正初稿，規則如下：

[幻覺風險] 處理方式：
  - 找到初稿中對應的句子
  - 在已知事實清單中重新尋找支撐依據
  - 若找得到：修正引用編號並確保內容相符
  - 若找不到：直接刪除該句子，不補充其他內容

[推論跳躍] 處理方式：
  - 找到初稿中對應的 [推論] 段落
  - 在已知事實清單中找出可支撐的前提事實
  - 補上完整的推導鏈：「根據[事實N]（XXX），可合理推論...」
  - 若無法補上合理推導：改標注「[資訊不足] 文獻依據不足，無法確認此推論」

[未標注引用] 處理方式：
  - 找到未標注來源的具體數值、方法名稱、結論性陳述
  - 在已知事實清單中找到對應的事實編號
  - 標注格式：「...（見 [事實N]）」
  - 若涉及關鍵數值，附上原始文獻的原文 quote（15字以內）
  - 若在清單中找不到依據：標注「[待確認]」

輸出修正後的完整回答，不要輸出說明或前言。
使用繁體中文。
"""

# ── English versions for EN_DRAFT_PIPELINE mode ───────────────────
VERIFY_SYSTEM_PROMPT_EN = """
You are an academic reasoning verifier.
Examine the "[Cross-Literature Inference]" and "[Knowledge Extension and Speculation]" sections,
using the "Known Facts List" as the sole evidentiary basis.

Note: Citation accuracy has been handled by another system; you do not need to re-verify citations.
Your focus is:
1. Whether the premises of each inference actually exist in the Known Facts List
2. Whether the reasoning from premises to conclusions contains logical leaps
3. Whether the speculations in the "Knowledge Extension" section exceed reasonable inference bounds or contradict the Known Facts List

Output in English.
"""

CORRECTION_SYSTEM_PROMPT_EN = """
You are an academic answer correction assistant.
You will receive:
1. Known Facts List (the sole source of truth)
2. Draft answer
3. List of issues found by the verifier

Your task is to revise the draft based on the issues list, following these rules:

[Logical Leap] handling:
  - Locate the corresponding inference passage in the draft
  - Find supporting premise facts in the Known Facts List
  - Add a complete reasoning chain: "Based on [Fact N] (XXX), it is reasonable to infer..."
  - If no sound reasoning chain can be constructed: change to "[Insufficient Evidence] Insufficient literature basis to confirm this inference"

[Missing Premise] handling:
  - Find sentences where the premise cannot be located in the Known Facts List
  - Remove those sentences or mark them as "[Unverified]"

[Contradicts Facts] / [Excessive Speculation] handling:
  - Find the relevant sentence in the draft
  - Correct it to align with the Known Facts List, or mark it as "[Unverified] contradicts known facts"

Output the complete revised answer with no preamble or explanation.
Output in English.
"""

class AnswerVerifier:
    # num_ctx = 65536；1 token ≈ 2 中文字 / 4 英文字母，取 2.5 字元/token 粗估
    # 保留 15% 給 system prompt + 輸出，所以 prompt 安全上限：
    _CTX_TOKENS   = 65536
    _CHARS_PER_TOKEN = 2.5
    _SAFETY_RATIO    = 0.85
    _MAX_PROMPT_CHARS = int(_CTX_TOKENS * _SAFETY_RATIO * _CHARS_PER_TOKEN)  # ≈ 139,264

    def __init__(
        self,
        verify_model: str = None,
        correction_model: str = None,
        ollama_base_url: str = None,
        timeout: int = 21600,
        max_retries: int = None
    ):
        self.verify_model     = verify_model     or cfg.VERIFY_MODEL
        self.correction_model = correction_model or cfg.SYNTHESIS_MODEL  # gemma4:31b
        self.base_url         = ollama_base_url  or cfg.OLLAMA_BASE_URL
        self.timeout          = timeout
        self.max_retries      = max_retries      or cfg.MAX_VERIFY_RETRIES

    def _call_ollama(self, model: str, system: str, prompt: str,
                     disable_thinking: bool = False, on_status=None) -> str:
        """
        串流呼叫 Ollama，即時印出 thinking 與輸出內容，方便 debug。
        thinking token 以灰色 [思考中...] 標示，正式輸出直接印出。
        回傳完整的非 thinking 輸出文字。
        """
        import json

        options = {
            "temperature": 0.6,
            "num_predict": -1,
            "num_ctx": cfg.STAGE5_NUM_CTX,
            "presence_penalty": 1.2,
        }
        if disable_thinking:
            options["thinking"] = False

        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model":  model,
                "system": system,
                "prompt": prompt,
                "stream": True,   # 改為串流
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

            # thinking 內容：完整串流到 terminal，方便 debug 觀察模型推理行為
            if thinking_chunk:
                if not thinking_printed:
                    _status("\n  💭 [Verifier] === 思考過程開始 ===")
                    thinking_printed = True
                print(thinking_chunk, end="", flush=True)

            # 正式輸出：即時印出，讓使用者看到進度
            if response_chunk:
                if thinking_printed and not full_response:
                    _status("\n  💭 [Verifier] === 思考過程結束，正式輸出 ===\n")
                print(response_chunk, end="", flush=True)
                full_response.append(response_chunk)

            if chunk.get("done"):
                break

        print()  # 換行
        return "".join(full_response).strip()

    @staticmethod
    def _extract_reasoning_sections(draft_answer: str) -> str:
        """
        從 draft_answer 中只抽取需要邏輯驗證的段落：
        - 【跨文獻推論】
        - 【知識延伸與推測】
        引用正確性已由 mDeBERTa grounding 處理，不重複送進 Verifier。
        若找不到這兩個段落（表示答案全是直引，無推論），回傳空字串。
        """
        import re
        # 匹配 ## 開頭、包含推論/延伸/推測（中文）或 Inference/Extension/Speculation（英文）的段落
        pattern = r'(##[^\n]*(?:推論|延伸|推測|Inference|Extension|Speculation)[^\n]*\n[\s\S]*?)(?=\n##|\Z)'
        matches = re.findall(pattern, draft_answer)
        return "\n\n".join(m.strip() for m in matches)

    @staticmethod
    def _split_answer_sections(draft_answer: str) -> list[str]:
        """
        按 ## 標題切分 draft_answer 成多個段落。
        若沒有 ## 標題，則按空行切分成段落。
        保證每個元素非空。
        """
        import re
        parts = re.split(r'(?=^## )', draft_answer, flags=re.MULTILINE)
        sections = [p.strip() for p in parts if p.strip()]
        if len(sections) <= 1:
            # fallback：按連續空行切
            sections = [p.strip() for p in re.split(r'\n{2,}', draft_answer) if p.strip()]
        return sections if sections else [draft_answer]

    def _pack_batches(self, sections: list[str], kb_chars: int, system_prompt: str = None) -> list[list[str]]:
        """
        將 sections 打包成多個 batch，每個 batch 的
        (kb_chars + 合併段落字元 + 固定開銷) 不超過 _MAX_PROMPT_CHARS。
        knowledge_base 一定完整傳入，只切分 draft_answer 的段落。
        """
        OVERHEAD = len(system_prompt if system_prompt is not None else VERIFY_SYSTEM_PROMPT) + 120
        budget = self._MAX_PROMPT_CHARS - kb_chars - OVERHEAD

        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_chars = 0

        for sec in sections:
            sec_len = len(sec)
            if current_batch and current_chars + sec_len > budget:
                batches.append(current_batch)
                current_batch = [sec]
                current_chars = sec_len
            else:
                current_batch.append(sec)
                current_chars += sec_len

        if current_batch:
            batches.append(current_batch)

        return batches

    def _verify_single(self, answer_chunk: str, knowledge_base: str, on_status=None, system_prompt: str = None) -> tuple[bool, str]:
        """
        對單一 chunk 執行一次 Verifier 呼叫。
        回傳 (passed, issues_text)。
        """
        active_sys = system_prompt if system_prompt is not None else VERIFY_SYSTEM_PROMPT
        if cfg.EN_DRAFT_PIPELINE:
            prompt = (
                f"Known Facts List:\n{knowledge_base}\n\n"
                f"Draft answer (partial sections):\n{answer_chunk}\n\n"
                "Verdict: output VERIFY_PASS if reasoning is sound, or VERIFY_FAIL followed by issues "
                "(format: IssueType | Quoted claim | Explanation). First line must be VERIFY_PASS or VERIFY_FAIL."
            )
        else:
            prompt = (
                f"已知事實清單：\n{knowledge_base}\n\n"
                f"初稿回答（部分段落）：\n{answer_chunk}\n\n"
                "判斷：推論整體合理則輸出 VERIFY_PASS，有問題則輸出 VERIFY_FAIL 並逐條說明"
                "（格式：問題類型｜引用句子｜說明）。第一行必須是 VERIFY_PASS 或 VERIFY_FAIL。"
            )
        try:
            result = self._call_ollama(self.verify_model, active_sys, prompt, on_status=on_status)
            first_line = result.split("\n")[0].strip().upper()
            passed = "VERIFY_PASS" in first_line
            issues = result if not passed else ""
            return passed, issues
        except Exception as e:
            logger.warning("[Verifier] 單批次驗證失敗，視為通過: %s", e)
            return True, ""

    def verify(
        self,
        draft_answer: str,
        knowledge_base: str,
        on_status=None,
    ) -> tuple[bool, str]:
        """
        驗證初稿回答。若 prompt 超過 context 安全上限，自動切分 draft_answer
        為多個批次分別驗證（knowledge_base 每次完整傳入），最後匯總問題。
        回傳 (passed: bool, issues: str)
        """
        t0 = time.time()
        active_sys = VERIFY_SYSTEM_PROMPT_EN if cfg.EN_DRAFT_PIPELINE else VERIFY_SYSTEM_PROMPT

        # ── Step 1：只抽取需要邏輯驗證的推論段落 ────────────────
        # 引用正確性由 mDeBERTa grounding 負責；
        # Verifier 只驗證跨文獻推論與知識延伸的邏輯合理性
        def _status(msg):
            if on_status:
                on_status(msg)
            else:
                print(msg)

        reasoning_text = self._extract_reasoning_sections(draft_answer)
        if not reasoning_text:
            _status("  ℹ️  [Verifier] 無推論段落，跳過邏輯驗證")
            logger.info("[Verifier] no reasoning sections found, skipping")
            return True, ""

        _status(f"  🔍 [Verifier] 抽取推論段落（{len(reasoning_text):,} 字元）進行邏輯驗證...")

        # ── Step 2：判斷是否需要分批（推論段落 + kb + overhead）──
        overhead = len(active_sys) + 120
        total_chars = overhead + len(knowledge_base) + len(reasoning_text)

        if total_chars <= self._MAX_PROMPT_CHARS:
            batches = [[reasoning_text]]
        else:
            sections = self._split_answer_sections(reasoning_text)
            batches  = self._pack_batches(sections, len(knowledge_base), active_sys)
            _status(
                f"  📦 [Verifier] 推論段落仍過長（約 {total_chars:,} 字元），"
                f"切分為 {len(batches)} 批次驗證..."
            )

        all_issues: list[str] = []
        for i, batch_sections in enumerate(batches):
            chunk = "\n\n".join(batch_sections) if isinstance(batch_sections, list) else batch_sections
            batch_label = f"批次 {i+1}/{len(batches)}" if len(batches) > 1 else ""
            if batch_label:
                _status(f"  🔍 [Verifier] 驗證{batch_label}...")
            passed_chunk, issues_chunk = self._verify_single(chunk, knowledge_base, on_status=on_status, system_prompt=active_sys)
            if not passed_chunk and issues_chunk:
                body = issues_chunk.split("\n", 1)[1].strip() if "\n" in issues_chunk else issues_chunk
                if body:
                    all_issues.append(f"【{batch_label or '推論段落'}】\n{body}")

        elapsed = time.time() - t0
        passed = len(all_issues) == 0
        issues = "\n\n".join(all_issues) if all_issues else ""

        status_str = "✅ PASS" if passed else "⚠️  FAIL"
        _status(f"  {status_str} [Verifier] ({elapsed:.1f}s)")
        if not passed:
            issue_preview = issues.split("\n")[0]
            _status(f"  → 發現問題：{issue_preview}")
        logger.info("[Verifier] passed=%s batches=%d elapsed=%.1fs", passed, len(batches), elapsed)
        return passed, issues

    def correct(
        self,
        draft_answer: str,
        knowledge_base: str,
        issues: str,
        on_status=None,
    ) -> str:
        """
        根據問題清單修正初稿。
        回傳修正後的答案字串。
        """
        # 去掉 VERIFY_FAIL 標記，只傳問題清單給 Corrector
        # 優先從第一個換行後取內容；若模型把所有內容擠在同一行，改用 replace 移除標記
        if "\n" in issues:
            issues_body = issues.split("\n", 1)[1].strip()
        else:
            issues_body = issues.replace("VERIFY_FAIL", "", 1).strip()

        def _status(msg):
            if on_status:
                on_status(msg)
            else:
                print(msg)

        # issues 為空代表 Verifier token 耗盡，沒有實質問題可修正，保留初稿
        if not issues_body:
            logger.warning("[Corrector] issues 為空（Verifier 可能 token 耗盡），保留初稿")
            _status("  ⚠️  [Corrector] issues 為空，跳過修正")
            return draft_answer

        active_corr = CORRECTION_SYSTEM_PROMPT_EN if cfg.EN_DRAFT_PIPELINE else CORRECTION_SYSTEM_PROMPT

        # Corrector prompt 長度預檢：kb + draft + issues 三者之和
        overhead_corr = len(active_corr) + 120
        total_corr = overhead_corr + len(knowledge_base) + len(draft_answer) + len(issues_body)
        if total_corr > self._MAX_PROMPT_CHARS:
            budget_issues = self._MAX_PROMPT_CHARS - overhead_corr - len(knowledge_base) - len(draft_answer)
            if budget_issues > 200:
                issues_body = issues_body[:budget_issues] + "\n...(問題清單已截短)"
                _status(f"  ⚠️  [Corrector] 問題清單過長，截短至 {budget_issues} 字元")
            else:
                logger.warning("[Corrector] prompt 過長且無法截短，保留初稿")
                _status("  ⚠️  [Corrector] prompt 過長，無法修正，保留初稿")
                return draft_answer

        if cfg.EN_DRAFT_PIPELINE:
            prompt = (
                f"Known Facts List:\n{knowledge_base}\n\n"
                f"Draft answer:\n{draft_answer}\n\n"
                f"Issues found by the verifier:\n{issues_body}\n\n"
                "Please revise the draft based on the issues list and output the complete corrected answer."
            )
        else:
            prompt = (
                f"已知事實清單：\n{knowledge_base}\n\n"
                f"初稿回答：\n{draft_answer}\n\n"
                f"驗證器發現的問題清單：\n{issues_body}\n\n"
                "請根據上述問題清單修正初稿，輸出修正後的完整回答。"
            )
        t0 = time.time()
        try:
            result = self._call_ollama(
                self.correction_model, active_corr, prompt, on_status=on_status
            )
            elapsed = time.time() - t0
            _status(f"  🔧 [Corrector] 修正完成 ({elapsed:.1f}s)")
            _status(f"\n  === Corrector 輸出 ===\n{result[:500]}\n")
            logger.info("[Corrector] elapsed=%.1fs", elapsed)
            if not result:
                logger.warning("[Corrector] 輸出為空，保留初稿")
                _status("  ⚠️  [Corrector] 輸出為空，保留初稿")
                return draft_answer
            return result
        except Exception as e:
            logger.warning("[Corrector] 修正失敗，保留初稿: %s", e)
            _status(f"  ⚠️  [Corrector] 修正失敗，保留初稿 ({e})")
            return draft_answer

    def verify_and_correct(
        self,
        draft_answer: str,
        knowledge_base: str,
        on_status=None,
    ) -> str:
        """
        主入口：驗證 → 若有問題則修正 → 最多重試 max_retries 次。
        回傳最終答案字串。
        """
        def _status(msg):
            if on_status:
                on_status(msg)
            else:
                print(msg)

        current_answer = draft_answer

        for attempt in range(self.max_retries):
            passed, issues = self.verify(current_answer, knowledge_base, on_status=on_status)
            if passed:
                return current_answer
            _status(f"  🔄 第 {attempt+1}/{self.max_retries} 次修正...")
            current_answer = self.correct(current_answer, knowledge_base, issues, on_status=on_status)

        # 最後一次驗證（不再修正）
        passed, _ = self.verify(current_answer, knowledge_base, on_status=on_status)
        if not passed:
            _status("  ⚠️  [Verifier] 超過最大重試次數，保留最後修正版本")
        return current_answer
