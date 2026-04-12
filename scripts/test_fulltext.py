"""
test_fulltext.py
================
目的：
    繞過 RAG retrieval，直接把論文全文塞進 prompt，
    測試模型本身能不能正確回答合成步驟問題。

使用方式：
    conda activate llm_env
    python test_fulltext.py

    可用 --paper 指定論文檔名（預設測試 1-s2.0-S1878029613002417-main.pdf）
    python test_fulltext.py --paper 你的論文.pdf

結果判讀：
    ✅ 模型答對 → 問題在 retrieval 層，跟模型無關
    ❌ 模型還是答錯 → 問題在模型本身或 PDF 解析
"""

import os
import sys
import time
import httpx
import fitz  # PyMuPDF

from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike

# ── LLM 設定（與主腳本完全相同）─────────────────────────
http_client = httpx.Client(timeout=httpx.Timeout(28800.0, connect=30.0))

Settings.llm = OpenAILike(
    model="deepseek-r1:32b",
    api_base="http://localhost:11434/v1",
    api_key="ollama",
    is_chat_model=True,
    timeout=28800.0,
    http_client=http_client,
    context_window=32768,
    system_prompt=(
        "你是一個學術論文分析助手。"
        "請只根據提供的論文內容回答問題，使用繁體中文。"
        "請務必使用繁體中文，絕對不可使用簡體中文。"
        "如果論文中沒有相關資訊，請直接說明「此論文未涉及此議題」，"
        "不要自行推測或補充論文以外的內容。"
        "回答時請盡量引用論文中的具體數據、步驟與條件。"
    ),
)

# ── PDF 讀取（與主腳本完全相同的解析邏輯）──────────────
def load_fulltext(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full_text = ""
    ref_section_started = False

    for page_num, page in enumerate(doc, 1):
        text = page.get_text("text")
        text = text.replace("\r\n", " ").replace("\r", " ")
        if not text.strip():
            continue

        lines = text.strip().split("\n")
        has_ref_header = any(
            line.strip() in ("References", "參考文獻", "REFERENCES")
            for line in lines
        )
        doi_count = text.lower().count("doi:")

        if has_ref_header and doi_count >= 3:
            ref_section_started = True
            print(f"  🔍 第{page_num}頁偵測到參考文獻區塊，後續略過")

        if ref_section_started:
            continue

        full_text += f"\n{text}\n"

    doc.close()
    return full_text


# ── 主程式 ───────────────────────────────────────────────
papers_dir = "papers"

# 預設測試這篇
target_pdf = "1-s2.0-S1878029613002417-main.pdf"
if "--paper" in sys.argv:
    idx = sys.argv.index("--paper")
    if idx + 1 < len(sys.argv):
        target_pdf = sys.argv[idx + 1]

pdf_path = os.path.join(papers_dir, target_pdf)
if not os.path.exists(pdf_path):
    print(f"❌ 找不到檔案：{pdf_path}")
    sys.exit(1)

print(f"\n{'='*65}")
print(f"  全文直接問答測試")
print(f"  論文：{target_pdf}")
print(f"{'='*65}\n")

# 讀取全文
print("📄 讀取論文全文中...")
full_text = load_fulltext(pdf_path)
char_count = len(full_text)
token_estimate = char_count // 3
print(f"  → 全文長度：{char_count:,} 字元（估計 {token_estimate:,} tokens）\n")

if token_estimate > 30000:
    print("⚠️  全文 token 數接近 context window 上限，可能被截斷")

# 測試問題
question = (
    "請仔細閱讀以下論文全文，詳細回答：\n"
    "1. ZVI（零價鐵奈米粒子）的完整合成步驟是什麼？\n"
    "2. 合成過程中使用了哪些試劑（reagents）？\n"
    "3. 每種試劑的用量是多少？（請列出具體數字，例如幾克、幾 mL、幾 mM）\n"
    "4. 合成過程中的操作條件是什麼？（攪拌速度、溫度、時間等）\n"
    "請直接引用論文原文中的數字，不要推測或補充。"
)

prompt = f"""以下是論文全文：

{full_text}

---

{question}
"""

print(f"{'='*65}")
print(f"  問題：")
print(f"  {question}")
print(f"{'='*65}\n")

# 串流輸出
print("🤖 模型回答中（串流輸出）：\n")
start = time.time()
full_response = ""

for chunk in Settings.llm.stream_complete(prompt):
    print(chunk.delta, end="", flush=True)
    full_response += chunk.delta

elapsed = time.time() - start
minutes = int(elapsed // 60)
seconds = int(elapsed % 60)

print(f"\n\n{'='*65}")
print(f"⏱ 耗時：{minutes}分{seconds}秒")
print(f"{'='*65}")

# 判讀提示
print("\n📋 結果判讀：")
keywords = ["0.27785", "0.46", "NaBH4", "FeSO4", "500 rpm", "ice-bath", "20 mM"]
found = [kw for kw in keywords if kw.lower() in full_response.lower()]
missing = [kw for kw in keywords if kw.lower() not in full_response.lower()]

if found:
    print(f"  ✅ 有提到的關鍵數字/詞：{', '.join(found)}")
if missing:
    print(f"  ❌ 沒有提到的關鍵數字/詞：{', '.join(missing)}")

if len(found) >= 5:
    print("\n  → 結論：模型本身能正確回答，問題在 RAG retrieval 層")
elif len(found) >= 2:
    print("\n  → 結論：模型部分正確，retrieval 和模型都有問題")
else:
    print("\n  → 結論：模型本身也有問題，需要進一步排查 PDF 解析")
