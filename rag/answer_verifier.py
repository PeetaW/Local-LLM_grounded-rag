import time
import logging
import requests
import config as cfg

logger = logging.getLogger(__name__)

VERIFY_SYSTEM_PROMPT = """
你是一個學術回答驗證器。
你的任務是比對「初稿回答」與「已知事實清單」，找出不一致或無依據的陳述。
已知事實清單是唯一的真相來源。

你的輸出格式：
- 若無任何問題，輸出第一行為：VERIFY_PASS
  之後不需要其他內容。
- 若有問題，輸出第一行為：VERIFY_FAIL
  之後逐條列出發現的問題，每個問題標注類型：
  [幻覺風險]、[推論跳躍]、[未標注引用]
  格式：問題類型｜引用或句子內容｜問題說明

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

class AnswerVerifier:
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
                     disable_thinking: bool = False) -> str:
        options = {
            "temperature": 0.1,
            "num_predict": -1,      # 不限輸出長度，由硬體決定上限
            "num_ctx": 65536,
        }
        if disable_thinking:
            # Qwen3 系列支援此選項，關閉後所有 token 預算全給輸出
            options["thinking"] = False

        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model":  model,
                "system": system,
                "prompt": prompt,
                "stream": False,
                "options": options,
            },
            timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    def verify(
        self,
        draft_answer: str,
        knowledge_base: str
    ) -> tuple[bool, str]:
        """
        驗證初稿回答。
        回傳 (passed: bool, issues: str)
        passed=True 代表通過，issues=""
        passed=False 代表有問題，issues 含問題清單
        """
        prompt = (
            f"已知事實清單：\n{knowledge_base}\n\n"
            f"初稿回答：\n{draft_answer}\n\n"
            "請根據驗證規則逐項檢查。"
        )
        t0 = time.time()
        try:
            result = self._call_ollama(
                self.verify_model, VERIFY_SYSTEM_PROMPT, prompt
            )
            elapsed = time.time() - t0
            first_line = result.split("\n")[0].strip().upper()
            passed = "VERIFY_PASS" in first_line
            issues = result if not passed else ""
            status = "✅ PASS" if passed else "⚠️  FAIL"
            print(f"  {status} [Verifier] ({elapsed:.1f}s)")
            if not passed:
                # 只印 issues 的前幾行，避免刷屏
                issue_preview = "\n".join(result.split("\n")[1:4])
                print(f"  → 發現問題：{issue_preview}")
            logger.info("[Verifier] passed=%s elapsed=%.1fs", passed, elapsed)
            return passed, issues
        except Exception as e:
            logger.warning("[Verifier] 驗證失敗，視為通過: %s", e)
            print(f"  ⚠️  [Verifier] 驗證失敗，跳過 ({e})")
            return True, ""  # 驗證失敗時不阻塞 pipeline

    def correct(
        self,
        draft_answer: str,
        knowledge_base: str,
        issues: str
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

        # issues 為空代表 Verifier token 耗盡，沒有實質問題可修正，保留初稿
        if not issues_body:
            logger.warning("[Corrector] issues 為空（Verifier 可能 token 耗盡），保留初稿")
            print("  ⚠️  [Corrector] issues 為空，跳過修正")
            return draft_answer

        prompt = (
            f"已知事實清單：\n{knowledge_base}\n\n"
            f"初稿回答：\n{draft_answer}\n\n"
            f"驗證器發現的問題清單：\n{issues_body}\n\n"
            "請根據上述問題清單修正初稿，輸出修正後的完整回答。"
        )
        t0 = time.time()
        try:
            result = self._call_ollama(
                self.correction_model, CORRECTION_SYSTEM_PROMPT, prompt
            )
            elapsed = time.time() - t0
            print(f"  🔧 [Corrector] 修正完成 ({elapsed:.1f}s)")
            print(f"\n  === Corrector 輸出 ===\n{result[:500]}\n")
            logger.info("[Corrector] elapsed=%.1fs", elapsed)
            # 空回傳保護：模型輸出為空時保留初稿，避免雪崩式失敗
            if not result:
                logger.warning("[Corrector] 輸出為空，保留初稿")
                print("  ⚠️  [Corrector] 輸出為空，保留初稿")
                return draft_answer
            return result
        except Exception as e:
            logger.warning("[Corrector] 修正失敗，保留初稿: %s", e)
            print(f"  ⚠️  [Corrector] 修正失敗，保留初稿 ({e})")
            return draft_answer  # 修正失敗時保留初稿

    def verify_and_correct(
        self,
        draft_answer: str,
        knowledge_base: str
    ) -> str:
        """
        主入口：驗證 → 若有問題則修正 → 最多重試 max_retries 次。
        回傳最終答案字串。
        """
        current_answer = draft_answer

        for attempt in range(self.max_retries):
            passed, issues = self.verify(current_answer, knowledge_base)
            if passed:
                return current_answer
            print(
                f"  🔄 第 {attempt+1}/{self.max_retries} 次修正..."
            )
            current_answer = self.correct(current_answer, knowledge_base, issues)

        # 最後一次驗證（不再修正）
        passed, _ = self.verify(current_answer, knowledge_base)
        if not passed:
            print(f"  ⚠️  [Verifier] 超過最大重試次數，保留最後修正版本")
        return current_answer
