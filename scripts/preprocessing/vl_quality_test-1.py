"""
vl_quality_test.py
==================
目的：驗證 Qwen3-VL:32b 對論文圖片的理解品質。
流程：
  Phase 1：從所有PDF抽取圖片（已有圖片則跳過）
  Phase 2：逐一queue進VL模型分析（已分析成功則跳過）
"""

import os
import re
import json
import fitz  # PyMuPDF
import httpx
import base64
import time
from datetime import datetime

# ── 設定區 ─────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
VL_MODEL = "qwen3-vl:32b"
PAPERS_DIR = "papers"
OUTPUT_DIR = "vl_test_output"

MAX_IMAGES_PER_PAPER = 9999
MIN_WIDTH = 150
MIN_HEIGHT = 150
MAX_RETRY = 2 #驗證失敗最多跑幾次

# 細碎圖片偵測：單頁嵌入圖片數量 >= 此值時，改光柵化整頁
FRAGMENTED_PAGE_THRESHOLD = 8
# 向量圖偵測：無嵌入圖片但 drawing commands >= 此值時，改光柵化整頁
# 數據依據：表格邊框/裝飾線通常 <100，向量圖表通常 100-2000+
VECTOR_DRAWING_THRESHOLD = 100
RASTER_DPI = 400  # 光柵化解析度（DPI）

ANALYSIS_PROMPT = """你是一個專業的學術論文圖表分析助手，專精於材料科學與奈米技術領域。
請詳細描述這張來自ZVI（零價鐵奈米粒子）學術論文的圖片。

請依照以下格式回答：

1. 【圖片類型】
   說明這是什麼類型的圖（例如：XRD光譜圖、SEM電子顯微鏡照片、TEM照片、
   EDS元素分佈圖、數據表格、實驗流程示意圖、折線圖、柱狀圖等）

2. 【主要內容】
   詳細描述圖中呈現的資訊、數據趨勢或實驗結果

3. 【關鍵數值與標註】
   列出圖中所有可見的數字、單位、軸標籤、圖例、標尺等

4. 【科學意義】
   說明這張圖對ZVI合成方法、表面特性、或降解效能研究的意義

請使用繁體中文回答，內容盡量詳細完整。"""

# ── 工具函式 ────────────────────────────────────────────

def load_existing_results(result_file: str) -> dict:
    """
    讀取已存在的分析結果
    回傳：以filename為key的dict，方便快速查詢
    """
    if not os.path.exists(result_file):
        return {}
    try:
        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 只保留成功的結果，失敗的允許重跑
        return {
            img["filename"]: img
            for img in data.get("images", [])
            if img.get("success", False)
        }
    except Exception:
        return {}


def extract_all_images(papers_dir: str) -> list:
    """
    Phase 1：從所有PDF抽取圖片
    - 若該paper的output資料夾已有圖片 → 直接讀取，跳過PDF抽取
    - 若沒有圖片 → 從PDF重新抽取
    """
    all_images = []
    pdf_files = sorted([f for f in os.listdir(papers_dir) if f.endswith(".pdf")])

    print(f"\n{'='*65}")
    print(f"Phase 1：抽取所有圖片")
    print(f"{'='*65}")

    for pdf_idx, pdf_file in enumerate(pdf_files, 1):
        pdf_path = os.path.join(papers_dir, pdf_file)
        paper_name = pdf_file.replace(".pdf", "")
        paper_output_dir = os.path.join(OUTPUT_DIR, paper_name)
        os.makedirs(paper_output_dir, exist_ok=True)

        print(f"\n[{pdf_idx}/{len(pdf_files)}] {pdf_file}")

        # ── 檢查資料夾內是否已有圖片 ──────────────────────
        existing_imgs = [
            f for f in os.listdir(paper_output_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp"))
        ]

        if existing_imgs:
            # 已有圖片 → 直接讀取，跳過PDF抽取
            print(f"  ✅ 發現 {len(existing_imgs)} 張已存在的圖片，跳過抽取")
            for filename in sorted(existing_imgs):
                img_path = os.path.join(paper_output_dir, filename)
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
                ext = filename.rsplit(".", 1)[-1].lower()

                # 從檔名解析頁碼（格式：page{N}_img{M}.ext）
                try:
                    page_num = int(filename.split("page")[1].split("_")[0])
                except Exception:
                    page_num = 0

                all_images.append({
                    "pdf_file": pdf_file,
                    "paper_name": paper_name,
                    "paper_output_dir": paper_output_dir,
                    "page": page_num,
                    "ext": ext,
                    "filename": filename,
                    "save_path": img_path,
                    "bytes": img_bytes,
                    "width": 0,   # 已存圖片不重新讀尺寸
                    "height": 0,
                    "is_raster": "_raster." in filename,  # 識別光柵化整頁
                })
                print(f"  📂 載入：{filename}")
        else:
            # 沒有圖片 → 從PDF重新抽取
            print(f"  📄 資料夾無圖片，從PDF抽取中...")
            doc = fitz.open(pdf_path)
            paper_count = 0

            for page_num, page in enumerate(doc, 1):
                if paper_count >= MAX_IMAGES_PER_PAPER:
                    break

                image_list = page.get_images(full=True)

                # ── 細碎點陣圖偵測：改光柵化整頁 ─────────────────
                if len(image_list) >= FRAGMENTED_PAGE_THRESHOLD:
                    raster_reason = f"細碎嵌入圖片 x{len(image_list)}"
                # ── 向量圖偵測：無嵌入圖片但有大量 drawing commands ─
                elif len(image_list) == 0:
                    drawing_count = len(page.get_drawings())
                    if drawing_count >= VECTOR_DRAWING_THRESHOLD:
                        raster_reason = f"向量圖 ({drawing_count} drawing commands)"
                    else:
                        if drawing_count > 0:
                            print(f"  ⬜ 第{page_num}頁：無嵌入圖、drawings={drawing_count}（低於閾值{VECTOR_DRAWING_THRESHOLD}），跳過")
                        continue
                else:
                    raster_reason = None  # 走下方的正常個別抽取

                if raster_reason is not None:
                    print(f"  🔲 第{page_num}頁偵測到【{raster_reason}】，改用整頁光柵化（{RASTER_DPI} DPI）")
                    try:
                        mat = fitz.Matrix(RASTER_DPI / 72, RASTER_DPI / 72)
                        pix = page.get_pixmap(matrix=mat, alpha=False)
                        img_bytes = pix.tobytes("png")
                        filename = f"page{page_num}_raster.png"
                        save_path = os.path.join(paper_output_dir, filename)
                        with open(save_path, "wb") as f:
                            f.write(img_bytes)
                        all_images.append({
                            "pdf_file": pdf_file,
                            "paper_name": paper_name,
                            "paper_output_dir": paper_output_dir,
                            "page": page_num,
                            "ext": "png",
                            "width": pix.width,
                            "height": pix.height,
                            "filename": filename,
                            "save_path": save_path,
                            "bytes": img_bytes,
                            "is_raster": True,
                            "raster_reason": raster_reason,
                        })
                        paper_count += 1
                        print(f"  ✅ 光柵化：{filename} ({pix.width}x{pix.height}px)")
                    except Exception as e:
                        print(f"  ⚠️ 光柵化失敗：page{page_num} → {e}")
                    continue

                for img_index, img_info in enumerate(image_list):
                    if paper_count >= MAX_IMAGES_PER_PAPER:
                        break

                    xref = img_info[0]
                    try:
                        base_image = doc.extract_image(xref)
                        img_bytes = base_image["image"]
                        img_ext = base_image["ext"]
                        width = base_image["width"]
                        height = base_image["height"]

                        if width < MIN_WIDTH or height < MIN_HEIGHT:
                            print(f"  跳過小圖：page{page_num}_img{img_index+1} "
                                  f"({width}x{height}px)")
                            continue

                        filename = f"page{page_num}_img{img_index+1}.{img_ext}"
                        save_path = os.path.join(paper_output_dir, filename)

                        with open(save_path, "wb") as f:
                            f.write(img_bytes)

                        all_images.append({
                            "pdf_file": pdf_file,
                            "paper_name": paper_name,
                            "paper_output_dir": paper_output_dir,
                            "page": page_num,
                            "ext": img_ext,
                            "width": width,
                            "height": height,
                            "filename": filename,
                            "save_path": save_path,
                            "bytes": img_bytes,
                        })

                        paper_count += 1
                        print(f"  ✅ 抽取：{filename} ({width}x{height}px)")

                    except Exception as e:
                        print(f"  ⚠️ 抽取失敗：page{page_num}_img{img_index+1} → {e}")

            doc.close()
            print(f"  → 本篇抽取 {paper_count} 張")

    print(f"\n✅ Phase 1 完成！共 {len(all_images)} 張圖片準備送入分析")
    return all_images


def analyze_image_with_vl(img_bytes: bytes, img_ext: str) -> dict:
    """呼叫Qwen3-VL分析單張圖片"""
    ext_to_mime = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
    }
    mime_type = ext_to_mime.get(img_ext.lower(), "image/png")
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    payload = {
        "model": VL_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{img_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": ANALYSIS_PROMPT
                    }
                ]
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 16384, #加倍，防止對話輸出內容截斷
            "repeat_penalty": 1.15, #新增，防止運算到循環小數時陷入無限迴圈導致崩潰
        }
    }

    start_time = time.time()
    try:
        with httpx.Client(timeout=httpx.Timeout(14400.0, connect=30.0)) as client:
            response = client.post(
                f"{OLLAMA_BASE_URL}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            data = response.json()

        elapsed = time.time() - start_time
        content = data["choices"][0]["message"]["content"]
        return {
            "success": True,
            "description": content,
            "elapsed_seconds": round(elapsed, 1),
        }

    except httpx.TimeoutException:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "description": "",
            "error": "請求逾時（超過4小時）",
            "elapsed_seconds": round(elapsed, 1),
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "description": "",
            "error": str(e),
            "elapsed_seconds": round(elapsed, 1),
        }
    
def validate_description(description: str) -> dict:
    """
    自動驗證VL模型的輸出品質
    回傳：{ "valid": bool, "issues": list }
    """
    issues = []

    # 檢查1：重複迴圈（如 7777777 或 的的的的）
    if re.search(r'(.{1,3})\1{10,}', description):
        issues.append("REPETITION_LOOP")

    # 檢查2：句子截斷（結尾字元不正常）
    stripped = description.strip()
    if stripped and stripped[-1] not in ('。', ')', '）', '】', '.', '\n'):
        issues.append("TRUNCATED")

    # 檢查3：內容過短（正常輸出都在500字以上）
    if len(description) < 500:
        issues.append("TOO_SHORT")

    # 檢查4：四大項目是否齊全
    for section in ["圖片類型", "主要內容", "關鍵數值", "科學意義"]:
        if section not in description:
            issues.append(f"MISSING:{section}")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
    }

def analyze_all_images(all_images: list):
    """
    Phase 2：逐一queue進VL模型分析
    - 已成功分析的圖片 → 跳過
    - 未分析或失敗的 → 重新分析
    """
    total = len(all_images)
    success_count = 0
    skip_count = 0

    # 彙整每篇paper的結果（含已存在的）
    paper_results = {}

    # 預先載入所有已存在的結果
    for img_data in all_images:
        paper_name = img_data["paper_name"]
        if paper_name not in paper_results:
            result_file = os.path.join(
                img_data["paper_output_dir"], "vl_test_result.json"
            )
            paper_results[paper_name] = {
                "source_pdf": img_data["pdf_file"],
                "tested_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": VL_MODEL,
                "existing": load_existing_results(result_file),  # 已成功的結果
                "images": []
            }

    print(f"\n{'='*65}")
    print(f"Phase 2：開始分析（共 {total} 張圖片）")
    print(f"{'='*65}")

    for idx, img_data in enumerate(all_images, 1):
        pdf_file = img_data["pdf_file"]
        filename = img_data["filename"]
        paper_name = img_data["paper_name"]
        existing = paper_results[paper_name]["existing"]

        print(f"\n[{idx}/{total}] {pdf_file} → {filename}")

        # ── 斷點續跑：已成功分析則跳過 ────────────────────
        if filename in existing:
            skip_count += 1
            success_count += 1
            print(f"  ⏭️  已有分析結果，跳過")
            paper_results[paper_name]["images"].append(existing[filename])
            continue

        attempt = 0
        validation = {"valid": False, "issues": ["NOT_RUN"]}

        for attempt in range(1, MAX_RETRY + 2):  # 第1次正常跑 + 最多MAX_RETRY次重跑
            if attempt == 1:
                print(f"  分析中...（可能需要數分鐘）", end="", flush=True)
            else:
                print(f"  第{attempt - 1}次重跑中...（驗證未通過）", end="", flush=True)

            result = analyze_image_with_vl(
                img_bytes=img_data["bytes"],
                img_ext=img_data["ext"],
            )

            if not result["success"]:
                # API層面失敗，不重跑
                print(f" ❌ 失敗：{result.get('error', '未知錯誤')}")
                validation = {"valid": False, "issues": ["API_ERROR"]}
                break

            # 分析成功 → 驗證品質
            validation = validate_description(result["description"])
            result["validation"] = validation

            if validation["valid"]:
                # 驗證通過
                success_count += 1
                print(f" ✅ 完成！耗時 {result['elapsed_seconds']} 秒")
                preview = result["description"][:300]
                if len(result["description"]) > 300:
                    preview += "...(以下省略)"
                print(f"\n  📝 預覽：\n{preview}\n")
                break
            else:
                # 驗證失敗
                print(f" ⚠️ 驗證未通過：{validation['issues']}")
                if attempt <= MAX_RETRY:
                    print(f"  → 準備第{attempt}次重跑...")
                else:
                    # 重跑次數用盡
                    success_count += 1  # 仍算success（API有回應），但標記需檢查
                    print(f"  → 已重跑{MAX_RETRY}次，標記 needs_review")
                    preview = result["description"][:300]
                    if len(result["description"]) > 300:
                        preview += "...(以下省略)"
                    print(f"\n  📝 最後一次預覽：\n{preview}\n")

        img_result = {
            "filename": filename,
            "page": img_data["page"],
            "size": f"{img_data.get('width', '?')}x{img_data.get('height', '?')}",
            "is_raster": img_data.get("is_raster", False),
            "raster_reason": img_data.get("raster_reason", ""),
            "success": result["success"],
            "description": result.get("description", ""),
            "error": result.get("error", ""),
            "elapsed_seconds": result["elapsed_seconds"],
            "validation": validation,
            "retry_count": attempt - 1,
            "needs_review": not validation.get("valid", False),
        }
        paper_results[paper_name]["images"].append(img_result)

        # 每張分析完立即存檔（防止crash遺失）
        result_file = os.path.join(
            img_data["paper_output_dir"], "vl_test_result.json"
        )
        save_data = {
            "source_pdf": paper_results[paper_name]["source_pdf"],
            "tested_at": paper_results[paper_name]["tested_at"],
            "model": VL_MODEL,
            "images": paper_results[paper_name]["images"]
        }
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*65}")
    print(f"✅ Phase 2 完成！")
    print(f"   成功（含跳過）：{success_count}/{total}")
    print(f"   跳過（已有結果）：{skip_count}/{total}")
    print(f"   失敗：{total - success_count}/{total}")
    print(f"\n📁 結果存放在：{OUTPUT_DIR}/")
    print(f"{'='*65}")
    # 印出驗證摘要報告
    print_summary_report(paper_results)
    

def print_summary_report(paper_results: dict):
    """
    執行結束後印出驗證摘要報告
    """
    all_issues = []
    for paper_name, data in paper_results.items():
        for img in data["images"]:
            if img.get("needs_review"):
                all_issues.append((paper_name, img))

    print(f"\n{'='*65}")
    print(f"📋 驗證摘要報告")
    print(f"{'='*65}")

    if not all_issues:
        print("✅ 所有圖片驗證全部通過，無需人工處理！")
    else:
        print(f"⚠️  以下 {len(all_issues)} 張圖片驗證未通過，建議人工確認：\n")
        for i, (paper_name, img) in enumerate(all_issues, 1):
            print(f"  [{i}] {paper_name}")
            print(f"       檔案：{img['filename']}")
            print(f"       問題：{', '.join(img['validation']['issues'])}")
            print(f"       重跑次數：{img['retry_count']}")
            print()

    print(f"{'='*65}\n")
    
    
# ── 主程式 ──────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Qwen3-VL 品質驗證腳本（含斷點續跑）")
    print(f"  模型：{VL_MODEL}")
    print(f"  時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 確認模型是否在線
    print(f"\n🔍 確認 {VL_MODEL} 是否可用...")
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{OLLAMA_BASE_URL}/api/tags")
            models = [m["name"] for m in resp.json().get("models", [])]
            if not any(VL_MODEL in m for m in models):
                print(f"❌ 找不到模型 {VL_MODEL}")
                print(f"   請先執行：ollama pull {VL_MODEL}")
                print(f"   目前可用模型：{models}")
                return
            print(f"✅ 模型已就緒\n")
    except Exception as e:
        print(f"❌ 無法連線到Ollama：{e}")
        return

    # Phase 1：抽取所有圖片
    all_images = extract_all_images(PAPERS_DIR)

    if not all_images:
        print("❌ 沒有抽取到任何圖片，請確認papers資料夾內容")
        return

    # Phase 1完成後列出清單讓使用者確認
    print(f"\n📋 準備送入VL模型的圖片清單：")
    for i, img in enumerate(all_images, 1):
        print(f"  [{i:02d}] {img['pdf_file']} → {img['filename']}")

    print(f"\n⚠️  共 {len(all_images)} 張圖片，每張預計需要數分鐘")
    print(f"   確認要繼續嗎？(輸入 y 繼續，其他鍵離開)")
    confirm = input(">>> ").strip().lower()
    if confirm != "y":
        print("已取消，圖片已存在 vl_test_output/ 可手動檢查")
        return

    # Phase 2：逐一分析
    analyze_all_images(all_images)


if __name__ == "__main__":
    main()