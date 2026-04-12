"""
vl_quality_test.py
==================
目的：驗證 Qwen3-VL:32b 對論文圖片的理解品質。
流程：
  Phase 1：從所有PDF抽取圖片（已有圖片則跳過）
  Phase 2：逐一queue進VL模型分析（已分析成功則跳過）
"""

import os
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
            "num_ctx": 8192,
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

        print(f"  分析中...（可能需要數分鐘）", end="", flush=True)

        result = analyze_image_with_vl(
            img_bytes=img_data["bytes"],
            img_ext=img_data["ext"],
        )

        if result["success"]:
            success_count += 1
            print(f" ✅ 完成！耗時 {result['elapsed_seconds']} 秒")
            preview = result["description"][:300]
            if len(result["description"]) > 300:
                preview += "...(以下省略)"
            print(f"\n  📝 預覽：\n{preview}\n")
        else:
            print(f" ❌ 失敗：{result.get('error', '未知錯誤')}")

        img_result = {
            "filename": filename,
            "page": img_data["page"],
            "size": f"{img_data.get('width', '?')}x{img_data.get('height', '?')}",
            "success": result["success"],
            "description": result.get("description", ""),
            "error": result.get("error", ""),
            "elapsed_seconds": result["elapsed_seconds"]
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
    print(f"👉 品質確認後告訴我，我們繼續建立完整pipeline！")
    print(f"{'='*65}")


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