# rag/metadata_manager.py
# 自動生成並管理每篇論文的 metadata
# 新論文加入時自動觸發，不需要手動維護

import os
import json
import fitz  # PyMuPDF

import config as cfg


def load_metadata() -> dict:
    """載入現有的 metadata，若不存在則回傳空 dict"""
    if os.path.exists(cfg.METADATA_PATH):
        with open(cfg.METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_metadata(metadata: dict):
    """儲存 metadata 到 JSON 檔案"""
    with open(cfg.METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def needs_metadata(paper_name: str) -> bool:
    """檢查這篇論文是否還沒有 metadata"""
    metadata = load_metadata()
    return paper_name not in metadata


def extract_paper_preview(pdf_path: str, max_chars: int = 3000) -> str:
    """
    抽取論文前幾頁的文字作為預覽。
    只取前 3000 字元，足夠讓小模型判斷論文主題。
    """
    doc = fitz.open(pdf_path)
    preview = ""

    for page in doc:
        text = page.get_text("text").replace("\r\n", " ").replace("\r", " ")
        preview += text
        if len(preview) >= max_chars:
            break

    doc.close()
    return preview[:max_chars]


def generate_metadata_for_paper(pdf_path: str) -> dict:
    """
    用小模型自動生成這篇論文的 metadata。
    回傳格式：
    {
        "title": "論文標題",
        "keywords": ["關鍵字1", "關鍵字2", ...],
        "short_desc": "一句話描述論文主題（中文）",
        "main_topic": "論文主要研究對象"
    }
    """
    from rag.llm_client import planning_llm

    preview = extract_paper_preview(pdf_path)

    prompt = f"""以下是一篇學術論文的開頭內容：

{preview}

請根據以上內容，用 JSON 格式輸出這篇論文的基本資訊：
{{
  "title": "論文英文標題（從原文中找）",
  "keywords": ["最重要的5個英文關鍵字"],
  "short_desc": "一句話描述這篇論文的主題，使用繁體中文，20字以內",
  "main_topic": "這篇論文主要研究的材料或方法名稱，英文，例如：NZVI, pectin-nZVI, nZVI-Rectorite"
}}

只輸出 JSON，不要其他文字。"""

    try:
        response = planning_llm.complete(prompt)
        raw = response.text.strip()
        import re as _re
        raw = _re.sub(r'<think>.*?</think>', '', raw, flags=_re.DOTALL).strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as je:
            print(f"  ⚠️  metadata JSON 解析失敗（{je}），使用預設值")
            return {
                "title": os.path.basename(pdf_path),
                "keywords": [],
                "short_desc": "（無描述）",
                "main_topic": "",
            }

        # 確保必要欄位存在且型別正確
        result.setdefault("title", os.path.basename(pdf_path))
        if not isinstance(result.get("keywords"), list):
            result["keywords"] = []
        result.setdefault("short_desc", "（無描述）")
        result.setdefault("main_topic", "")

        return result

    except Exception as e:
        print(f"  ⚠️  metadata 生成失敗：{e}")
        return {
            "title": os.path.basename(pdf_path),
            "keywords": [],
            "short_desc": "（無描述）",
            "main_topic": "",
        }


def ensure_metadata(pdf_path: str) -> dict:
    """
    確保這篇論文有 metadata。
    若已有則直接回傳，若沒有則自動生成並儲存。
    """
    pdf_filename = os.path.basename(pdf_path)
    paper_name = pdf_filename.replace(".pdf", "")

    metadata = load_metadata()

    if paper_name not in metadata:
        print(f"  📝 自動生成 metadata：{paper_name[:50]}...")
        paper_meta = generate_metadata_for_paper(pdf_path)
        metadata[paper_name] = paper_meta
        save_metadata(metadata)
        print(f"  ✅ metadata 已儲存：{paper_meta['short_desc']}")
    
    return metadata[paper_name]