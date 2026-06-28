# rag/pdf_loader.py
# 負責 PDF 解析與 VL 圖片描述融合
# 回傳 LlamaIndex Document 列表供後續建立索引使用

import os
import sys
import json
import fitz  # PyMuPDF
from llama_index.core.schema import Document

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg

# Latin f-連字正規化：PDF 常把 fi/fl 等存成單一 Unicode 連字（U+FB00–FB04），
# 導致 BM25「fluoride」比對不到「ﬂuoride」、grounding entailment 也被 token 不符拉低。
# 只換這 5 個 f-連字（外科手術式），不用整段 NFKC，避免動到化學式的上下標/特殊符號。
_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}


def _normalize_ligatures(text: str) -> str:
    for lig, rep in _LIGATURES.items():
        text = text.replace(lig, rep)
    return text


def load_pdf_with_pymupdf(pdf_path, vl_output_dir=None):
    """
    用 PyMuPDF 解析 PDF，抽取文字內容。
    同時載入對應的 VL 圖片描述（若存在）。
    回傳多個 Document：1 個文字 + N 個圖片描述。
    """
    if vl_output_dir is None:
        vl_output_dir = cfg.VL_OUTPUT_DIR

    pdf_filename = os.path.basename(pdf_path)
    paper_name = pdf_filename.replace(".pdf", "")

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
            print(f"  🔍 第{page_num}頁偵測到參考文獻區塊，後續頁面略過")

        if ref_section_started:
            continue

        full_text += f"\n{text}"

    doc.close()

    # 若過濾後內容為空，退回完整文字（不過濾 reference）
    if not full_text.strip():
        print(f"  ⚠️  參考文獻過濾後內容為空，退回完整文字")
        doc = fitz.open(pdf_path)
        full_text = ""
        for page_num, page in enumerate(doc, 1):
            text = page.get_text("text")
            text = text.replace("\r\n", " ").replace("\r", " ")
            if text.strip():
                full_text += f"\n{text}"
        doc.close()

    full_text = _normalize_ligatures(full_text)

    documents = [Document(
        text=full_text,
        metadata={
            "file_name": pdf_filename,
            "file_path": pdf_path,
            "source_type": "pdf_text",
            "parser": "pymupdf",
        }
    )]

    # 載入 VL 圖片描述（若存在）
    vl_result_path = os.path.join(vl_output_dir, paper_name, "vl_test_result.json")

    if not os.path.exists(vl_result_path):
        print(f"  ℹ️  找不到VL描述：{paper_name}，僅使用PDF文字")
        return documents

    with open(vl_result_path, "r", encoding="utf-8") as f:
        vl_data = json.load(f)

    img_count = 0
    skipped_count = 0
    for img in vl_data.get("images", []):
        if not img.get("success", False):
            continue
        if img.get("needs_review", False):
            skipped_count += 1
            continue

        description = img.get("description", "").strip()
        if not description:
            continue

        img_text = (
            f"【圖片描述】\n"
            f"來源論文：{pdf_filename}\n"
            f"圖片檔名：{img['filename']}（第{img['page']}頁）\n\n"
            f"{description}"
        )

        documents.append(Document(
            text=img_text,
            metadata={
                "file_name": pdf_filename,
                "source_type": "image_description",
                "image_filename": img["filename"],
                "page": img["page"],
            }
        ))
        img_count += 1

    print(f"  ✅ 載入 {img_count} 張圖片描述", end="")
    if skipped_count > 0:
        print(f"（跳過 {skipped_count} 張 needs_review）", end="")
    print()

    return documents


if __name__ == "__main__":
    # 自檢：連字正規化（離線）。
    assert _normalize_ligatures("boroxines toward ﬂuoride and eﬃcient ﬁlms") == \
        "boroxines toward fluoride and efficient films"
    assert _normalize_ligatures("no ligatures here") == "no ligatures here"
    print("pdf_loader ligature self-check OK")
