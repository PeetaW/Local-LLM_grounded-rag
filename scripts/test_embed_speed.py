# scripts/test_embed_speed.py
# 測試 bge-m3 embedding 單次速度與並行行為
#
# 使用方式：
#   conda activate llm_env
#   cd E:\Projects\rag_project
#   python scripts/test_embed_speed.py

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import requests
import time
import concurrent.futures

URL = "http://localhost:11434/api/embeddings"
MODEL = "bge-m3"

TEXTS = [
    "What are the specific reagents, ratios and reaction conditions used in the synthesis of glycine-modified nZVI?",
    "What is the role of gelatin aerogel in the process, and what specific values are mentioned?",
    "What is the main research objective and novelty of this paper?",
    "What are the key findings regarding glycine-modified NZVI@gelatin aerogel for tetracycline degradation?",
]

# 問題文字的逐步縮減，找出哪個片段觸發 NaN
NAN_CANDIDATES = [
    "What are the specific reagents, ratios and reaction conditions used in the synthesis of glycine-modified nZVI?",
    "synthesis of glycine-modified nZVI",
    "glycine-modified nZVI",
    "nZVI",
    "glycine-modified",
    "reagents ratios reaction conditions synthesis",
    "What are the specific reagents",
]


def do_embed(text: str, label: str = "") -> dict:
    t0 = time.time()
    try:
        r = requests.post(URL, json={"model": MODEL, "prompt": text}, timeout=120)
        elapsed = time.time() - t0
        if r.status_code != 200:
            return {"label": label, "status": r.status_code, "elapsed": elapsed,
                    "error": r.text[:100]}
        emb = r.json().get("embedding", [])
        nan_count = sum(1 for x in emb if x != x)
        return {"label": label, "status": r.status_code, "elapsed": elapsed,
                "dims": len(emb), "nan": nan_count}
    except requests.exceptions.Timeout:
        return {"label": label, "status": "TIMEOUT", "elapsed": time.time() - t0}
    except Exception as e:
        return {"label": label, "status": "ERROR", "elapsed": time.time() - t0,
                "error": str(e)[:100]}


# ── 單次測試 ──────────────────────────────────────────
print("=" * 60)
print("【1】單次 embed 速度（連跑 3 次，確認是否穩定）")
print("=" * 60)
for i in range(3):
    result = do_embed(TEXTS[0], label=f"single-{i+1}")
    print(f"  [{i+1}] {result}")

print()

# ── 並行測試 ─────────────────────────────────────────
print("=" * 60)
print("【2】4 個並行 embed（模擬 Phase A 實際情況）")
print("=" * 60)
t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    futures = {ex.submit(do_embed, text, f"parallel-{i+1}"): i
               for i, text in enumerate(TEXTS)}
    for f in concurrent.futures.as_completed(futures):
        print(f"  {f.result()}")
print(f"  總耗時：{time.time() - t0:.2f}s")

print()

# ── 串行測試（基準對比）──────────────────────────────
print("=" * 60)
print("【3】4 個串行 embed（基準對比）")
print("=" * 60)
t0 = time.time()
for i, text in enumerate(TEXTS):
    result = do_embed(text, label=f"serial-{i+1}")
    print(f"  {result}")
print(f"  總耗時：{time.time() - t0:.2f}s")

print()

# ── NaN 觸發片段診斷 ─────────────────────────────────
print("=" * 60)
print("【4】逐步縮減診斷：哪個片段觸發 NaN？")
print("=" * 60)
for text in NAN_CANDIDATES:
    result = do_embed(text, label=text[:40])
    status = result["status"]
    marker = "❌ NaN/500" if status != 200 else "✅ OK"
    print(f"  {marker}  {text!r}")

print()

# ── 二分搜尋：找出最小觸發組合 ──────────────────────
print("=" * 60)
print("【5】二分搜尋：找出觸發 NaN 的最小組合")
print("=" * 60)

BAD = "What are the specific reagents, ratios and reaction conditions used in the synthesis of glycine-modified nZVI?"
words = BAD.split()
print(f"  完整句子：{len(words)} 個詞")

# 從完整句子逐步移除前半/後半，找到最小觸發組合
def is_nan(text):
    r = do_embed(text)
    return r["status"] != 200

# 先測試前半 / 後半
mid = len(words) // 2
first_half = " ".join(words[:mid])
second_half = " ".join(words[mid:])
print(f"  前半 [{first_half!r}]: {'❌' if is_nan(first_half) else '✅'}")
print(f"  後半 [{second_half!r}]: {'❌' if is_nan(second_half) else '✅'}")

# 逐步從完整句子縮減，找到最短觸發句
print()
print("  逐步從頭縮短（移除尾端）：")
for end in range(len(words), 0, -1):
    candidate = " ".join(words[:end])
    bad = is_nan(candidate)
    marker = "❌" if bad else "✅"
    print(f"  {marker} ({end} 詞) {candidate!r}")
    if not bad:
        print(f"  → 最短觸發句是 {end+1} 詞版本")
        break

print()

# ── prepare_query_text 端對端驗證 ────────────────────
print("=" * 60)
print("【6】_prepare_query_text 端對端驗證（含截短邏輯）")
print("=" * 60)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rag.query_engine import _prepare_query_text

result_text = _prepare_query_text(BAD)
final_status = do_embed(result_text)
marker = "✅ 截短後 OK" if final_status["status"] == 200 else "❌ 仍失敗"
print(f"  {marker}")
print(f"  原文:   {BAD!r}")
print(f"  最終用: {result_text!r}")

print()

# ── 深度診斷：找出真正觸發原因 ──────────────────────
print("=" * 60)
print("【6】深度診斷：是詞語本身還是位置/長度？")
print("=" * 60)

base = "What are the specific reagents, ratios and reaction conditions used in the synthesis of"
# 把第 15 個詞換成不同詞，看哪個會觸發
word15_candidates = [
    "glycine-modified",  # 原始（已知失敗）
    "glycine modified",  # 去連字號
    "glycine",           # 只留前半
    "modified",          # 只留後半
    "nanoparticles",     # 完全不同的詞
    "iron",              # 簡單詞
    "zero-valent",       # 另一個連字號詞
    "nano",              # 短詞
]
for w in word15_candidates:
    text = f"{base} {w}"
    result = do_embed(text)
    marker = "❌ NaN" if result["status"] != 200 else "✅ OK"
    print(f"  {marker}  第15詞={w!r}")

print()
print("【7】長度門檻確認：14詞 OK，那 15 詞任意詞都失敗？")
for extra in ["iron", "nano", "the", "a", "of"]:
    text = f"{base} {extra}"
    result = do_embed(text)
    marker = "❌ NaN" if result["status"] != 200 else "✅ OK"
    print(f"  {marker}  ({len(text.split())} 詞) 加上 {extra!r}")
