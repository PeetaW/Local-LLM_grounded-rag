# test_new_modules.py
# 測試 KnowledgeSynthesizer 和 AnswerVerifier 是否正常運作
# 使用方式：
#   conda activate llm_env
#   cd E:\Projects\rag_project\rag_project
#   python test_new_modules.py

import sys, os
sys.path.insert(0, ".")

print("=" * 60)
print("  新模組單元測試")
print("=" * 60)

# ══════════════════════════════════════════════════════════
# 測試 1：KnowledgeSynthesizer
# ══════════════════════════════════════════════════════════
print("\n【測試 1】KnowledgeSynthesizer 載入與執行")
print("-" * 60)

from rag.knowledge_synthesizer import KnowledgeSynthesizer

synth = KnowledgeSynthesizer()
fake_chunks = [
    {
        "text": "NZVI was synthesized using FeSO4 (0.278g) and NaBH4. Stirring at 500 rpm.",
        "source": "paper_A"
    },
    {
        "text": "The reaction was conducted under nitrogen atmosphere for 30 minutes.",
        "source": "paper_B"
    },
]

knowledge_base = synth.synthesize(fake_chunks, query="NZVI 合成步驟")

print("\n=== Synthesizer 輸出 ===")
print(knowledge_base)

# 基本檢查
if "[事實1]" in knowledge_base or "[Fact 1]" in knowledge_base or "1." in knowledge_base:
    print("\n✅ Synthesizer：輸出包含編號清單，格式正確")
else:
    print("\n⚠️  Synthesizer：輸出格式可能有問題，請人工確認上方內容")

# ══════════════════════════════════════════════════════════
# 測試 2：AnswerVerifier - 應該 PASS
# ══════════════════════════════════════════════════════════
print("\n【測試 2】AnswerVerifier — 正確引用（預期 PASS）")
print("-" * 60)

from rag.answer_verifier import AnswerVerifier

verifier = AnswerVerifier()

# 使用寫死的 knowledge_base，確保事實編號與 draft_good 對應
fixed_knowledge_base = (
    "[事實1] NZVI 使用 FeSO4 (0.278g) 與 NaBH4 合成，攪拌速度 500 rpm。（來源：paper_A）\n"
    "[事實2] 反應在氮氣氛圍下進行 30 分鐘。（來源：paper_B）"
)

draft_good = (
    "根據[事實1]，NZVI 使用 FeSO4 (0.278g) 與 NaBH4 合成，攪拌速度 500 rpm。"
    "根據[事實2]，反應在氮氣氛圍下進行 30 分鐘。"
)

passed1, issues1 = verifier.verify(draft_good, fixed_knowledge_base)
print(f"\n結果：passed = {passed1}")
if passed1:
    print("✅ 測試2 通過：PASS 判斷正確")
else:
    print("⚠️  測試2 異常：預期 PASS 但得到 FAIL")
    print(f"原因：{issues1[:300]}")

# ══════════════════════════════════════════════════════════
# 測試 3：AnswerVerifier - 應該 FAIL
# ══════════════════════════════════════════════════════════
print("\n【測試 3】AnswerVerifier — 幻覺內容（預期 FAIL）")
print("-" * 60)

draft_bad = (
    "NZVI 合成溫度為 800°C，使用了鹽酸作為催化劑，根據[事實3]效果極佳。"
    "反應需要 24 小時才能完成，且必須在真空環境下操作。"
)

passed2, issues2 = verifier.verify(draft_bad, fixed_knowledge_base)
print(f"\n結果：passed = {passed2}")
if not passed2:
    print("✅ 測試3 通過：FAIL 判斷正確")
    print(f"發現的問題：\n{issues2[:400]}")
else:
    print("⚠️  測試3 異常：預期 FAIL 但得到 PASS（驗證器可能過於寬鬆）")

# ══════════════════════════════════════════════════════════
# 測試 4：verify_and_correct 完整流程
# ══════════════════════════════════════════════════════════
print("\n【測試 4】verify_and_correct — 完整修正流程")
print("-" * 60)

corrected = verifier.verify_and_correct(draft_bad, fixed_knowledge_base)
print("\n=== 修正後輸出 ===")
print(corrected)

if corrected != draft_bad:
    print("\n✅ 測試4：初稿有被修正")
else:
    print("\n⚠️  測試4：初稿未被修正（可能修正失敗或驗證器視為通過）")

# ══════════════════════════════════════════════════════════
# 總結
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  測試完成")
print("=" * 60)