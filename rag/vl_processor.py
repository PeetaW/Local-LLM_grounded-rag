# rag/vl_processor.py
# 負責自動偵測並執行 VL 圖片分析
# 若論文已有 VL JSON 則跳過，否則自動呼叫 VL 模型分析

import os
import json
import base64
import httpx
import fitz  # PyMuPDF

import config as cfg


def _vl_result_path(paper_name: str) -> str:
    return os.path.join(cfg.VL_OUTPUT_DIR, paper_name, "vl_test_result.json")


def needs_vl_analysis(paper_name: str) -> bool:
    """
    檢查這篇論文是否需要跑全量 VL 分析（JSON 不存在）。
    """
    return not os.path.exists(_vl_result_path(paper_name))


def get_failed_vl_images(paper_name: str) -> list:
    """
    回傳 vl_test_result.json 中所有 needs_review=True 的圖片 entry。
    """
    path = _vl_result_path(paper_name)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [img for img in data.get("images", []) if img.get("needs_review") and not img.get("skipped")]


def has_failed_vl_images(paper_name: str) -> bool:
    return bool(get_failed_vl_images(paper_name))


def extract_images_from_pdf(pdf_path: str, output_dir: str) -> list:
    """
    從 PDF 抽取所有圖片，儲存到 output_dir。
    回傳圖片資訊列表：[{"filename": ..., "page": ..., "path": ...}]
    """
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    images = []

    for page_num, page in enumerate(doc, 1):
        image_list = page.get_images(full=True)
        for img_idx, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            filename = f"page{page_num}_img{img_idx+1}.{ext}"
            img_path = os.path.join(output_dir, filename)

            with open(img_path, "wb") as f:
                f.write(image_bytes)

            images.append({
                "filename": filename,
                "page": page_num,
                "path": img_path,
            })

    doc.close()
    print(f"  📸 抽取 {len(images)} 張圖片")
    return images


def analyze_image_with_vl(image_path: str, paper_name: str) -> str:
    """
    把單張圖片送給 VL 模型分析，回傳描述文字。
    """
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    ext = os.path.splitext(image_path)[1].lstrip(".")
    media_type = f"image/{ext}" if ext != "jpg" else "image/jpeg"

    prompt = (
        "你是一個學術論文圖片分析助手。"
        "請詳細描述這張圖片的內容，包括：\n"
        "1. 圖片類型（圖表、顯微鏡照片、示意圖等）\n"
        "2. 主要內容與數據\n"
        "3. 座標軸標籤與單位（如有）\n"
        "4. 關鍵數值與趨勢\n"
        "5. 科學意義\n"
        "請用繁體中文回答。"
    )

    try:
        response = httpx.post(
            f"{cfg.OLLAMA_BASE_URL}/api/chat",
            json={
                "model": cfg.VL_MODEL,
                "messages": [{
                    "role": "user",
                    "content": prompt,
                    "images": [image_data],
                }],
                "stream": False,
            },
            timeout=600.0,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
    except Exception as e:
        print(f"  ⚠️  VL 分析失敗：{e}")
        return ""


def run_vl_analysis(pdf_path: str) -> bool:
    """
    對單篇 PDF 執行完整 VL 分析流程：
    1. 抽取圖片
    2. 逐張送 VL 模型分析
    3. 結果存成 JSON

    回傳 True 表示成功，False 表示失敗或無圖片。
    """
    if not cfg.VL_AUTO_RUN:
        return False

    pdf_filename = os.path.basename(pdf_path)
    paper_name = pdf_filename.replace(".pdf", "")
    vl_output_dir = os.path.join(cfg.VL_OUTPUT_DIR, paper_name)
    vl_result_path = os.path.join(vl_output_dir, "vl_test_result.json")

    print(f"\n  🔍 開始 VL 分析：{paper_name}")

    # 抽取圖片
    images = extract_images_from_pdf(pdf_path, vl_output_dir)
    if not images:
        print(f"  ℹ️  此 PDF 無圖片，跳過 VL 分析")
        # 建立空的結果檔，避免下次重複觸發
        with open(vl_result_path, "w", encoding="utf-8") as f:
            json.dump({"images": []}, f)
        return False

    # 逐張分析
    results = []
    for i, img_info in enumerate(images, 1):
        print(f"  🖼️  分析圖片 [{i}/{len(images)}]：{img_info['filename']}")
        description = analyze_image_with_vl(img_info["path"], paper_name)

        failed = not bool(description)
        if failed:
            print(f"  ❌ 圖片解析失敗，已標記待人工審查：{img_info['filename']}")
        results.append({
            "filename": img_info["filename"],
            "page": img_info["page"],
            "success": not failed,
            "needs_review": failed,
            "description": description,
        })

    # 儲存結果
    with open(vl_result_path, "w", encoding="utf-8") as f:
        json.dump({"images": results}, f, ensure_ascii=False, indent=2)

    failed = [r for r in results if r["needs_review"]]
    if failed:
        print(f"  ✅ VL 分析完成，共 {len(results)} 張圖片")
        print(f"  ⚠️  {len(failed)} 張圖片解析失敗，需人工審查：")
        for r in failed:
            print(f"       - {r['filename']} (第 {r['page']} 頁)")
        print(f"  💡 執行 python main.py --rerun-vl {paper_name} 可重新掃描失敗圖片")
    else:
        print(f"  ✅ VL 分析完成，共 {len(results)} 張圖片描述已儲存")
    return True


def backfill_needs_review(paper_name: str) -> int:
    """
    補丁函數：修正舊版 JSON 中 success=False 但 needs_review=False 的漏標問題。
    回傳補標的圖片數量。
    """
    path = _vl_result_path(paper_name)
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    images = data.get("images", [])
    fixed = 0
    for img in images:
        if not img.get("success") and not img.get("needs_review"):
            img["needs_review"] = True
            fixed += 1
    if fixed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"images": images}, f, ensure_ascii=False, indent=2)
    return fixed


def rerun_failed_vl(pdf_path: str) -> int:
    """
    只重新分析 needs_review=True 的失敗圖片，更新 JSON。
    回傳成功修復的圖片數量。
    若有任何圖片從失敗變成成功，呼叫端應重建該論文的 index。
    """
    pdf_filename = os.path.basename(pdf_path)
    paper_name = pdf_filename.replace(".pdf", "")
    vl_result_path = _vl_result_path(paper_name)

    if not os.path.exists(vl_result_path):
        print(f"  ⚠️  找不到 VL 結果檔，請先執行完整 VL 分析")
        return 0

    with open(vl_result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    failed = [img for img in images if img.get("needs_review")]
    if not failed:
        print(f"  ✅ {paper_name} 沒有待審查的圖片")
        return 0

    print(f"  🔄 重新分析 {len(failed)} 張失敗圖片...")
    vl_output_dir = os.path.join(cfg.VL_OUTPUT_DIR, paper_name)
    fixed = 0

    for img in images:
        if not img.get("needs_review"):
            continue
        img_path = os.path.join(vl_output_dir, img["filename"])
        if not os.path.exists(img_path):
            img["needs_review"] = False
            img["skipped"] = True
            print(f"  ⏭️  圖片已刪除，標記為手動略過：{img['filename']}")
            continue

        print(f"  🖼️  重新分析：{img['filename']}")
        description = analyze_image_with_vl(img_path, paper_name)
        if description:
            img["description"] = description
            img["success"] = True
            img["needs_review"] = False
            fixed += 1
            print(f"  ✅ 修復成功：{img['filename']}")
        else:
            print(f"  ❌ 仍然失敗：{img['filename']}")

    with open(vl_result_path, "w", encoding="utf-8") as f:
        json.dump({"images": images}, f, ensure_ascii=False, indent=2)

    remaining = sum(1 for img in images if img.get("needs_review"))
    print(f"  📊 修復結果：{fixed} 張成功，{remaining} 張仍需審查")
    return fixed