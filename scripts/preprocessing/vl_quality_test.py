"""
目的：在正式建立preprocessing pipeline之前，
      先驗證 Qwen3-VL:32b 對論文圖片的理解品質。

使用方式：
    1. 確認 Qwen3-VL:32b 已下載完成（ollama list確認）
    2. 確認 Ollama 正在執行中
    3. 執行：python vl_quality_test.py
    4. 觀察輸出，判斷品質是否符合需求

作者備註：
    - 這個腳本會從每篇PDF各抽取幾張圖片來測試
    - 不會修改任何現有索引或資料
    - 輸出結果會存到 vl_test_output/ 資料夾方便檢查
"""

import os
import json
import fitz  # PyMuPDF
import httpx
import base64
import time
from datetime import datetime
from pathlib import Path

# ── 設定區（根據你的環境修改）─────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"   # 直連Ollama（不經過WebUI）
VL_MODEL = "qwen3-vl:32b"                   # VL模型名稱
PAPERS_DIR = "papers"                        # PDF資料夾
OUTPUT_DIR = "vl_test_output"                # 測試輸出資料夾

# 每篇論文最多測試幾張圖片（品質驗證不需要全部跑）
MAX_IMAGES_PER_PAPER = 2

# 圖片最小尺寸過濾（太小的圖通常是logo或裝飾，不值得分析）
MIN_WIDTH = 150   # pixels
MIN_HEIGHT = 150  # pixels

# ── Prompt設計（這是核心，之後preprocess也會用這個）──────────
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
   說明這張圖對該文獻所做的研究的意義

請使用繁體中文回答，內容盡量詳細完整。"""

# ── 工具函式 ──────────────────────────────────────────────

def extract_images_from_pdf(pdf_path: str, max_images: int = MAX_IMAGES_PER_PAPER):
    """
    從PDF抽取圖片
    回傳：list of dict，每個dict包含圖片的bytes和頁碼資訊
    """
    doc = fitz.open(pdf_path)
    extracted = []

    for page_num, page in enumerate(doc, 1):
        if len(extracted) >= max_images:
            break

        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            if len(extracted) >= max_images:
                break

            xref = img_info[0]  # 圖片的xref編號

            try:
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                img_ext = base_image["ext"]
                width = base_image["width"]
                height = base_image["height"]

                # 過濾太小的圖（裝飾性圖片）
                if width < MIN_WIDTH or height < MIN_HEIGHT:
                    print(f"    跳過小圖：page{page_num}_img{img_index+1} "
                          f"({width}x{height}px)")
                    continue

                extracted.append({
                    "page": page_num,
                    "index": img_index + 1,
                    "ext": img_ext,
                    "width": width,
                    "height": height,
                    "bytes": img_bytes,
                    "filename": f"page{page_num}_img{img_index+1}.{img_ext}"
                })
                print(f"    抽取圖片：page{page_num}_img{img_index+1} "
                      f"({width}x{height}px, {img_ext})")

            except Exception as e:
                print(f"    ⚠️ 抽取失敗：page{page_num}_img{img_index+1} → {e}")
                continue

    doc.close()
    print(f"    共抽取 {len(extracted)} 張圖片")
    return extracted


def image_to_base64(img_bytes: bytes) -> str:
    """將圖片bytes轉成base64字串（給API用）"""
    return base64.b64encode(img_bytes).decode("utf-8")


def analyze_image_with_vl(img_bytes: bytes, img_ext: str, 
                           source_info: str = "") -> dict:
    """
    呼叫Qwen3-VL分析單張圖片
    回傳：dict包含回答內容和耗時
    """
    # 決定media type
    ext_to_mime = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
    }
    mime_type = ext_to_mime.get(img_ext.lower(), "image/png")

    # 轉base64
    img_b64 = image_to_base64(img_bytes)

    # 建構API請求（Ollama的multimodal格式）
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
            "temperature": 0.1,   # 低temperature：更穩定、減少幻覺
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
            "source_info": source_info
        }

    except httpx.TimeoutException:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "description": "",
            "error": "請求逾時（超過4小時）",
            "elapsed_seconds": round(elapsed, 1),
            "source_info": source_info
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "description": "",
            "error": str(e),
            "elapsed_seconds": round(elapsed, 1),
            "source_info": source_info
        }


def save_image_for_reference(img_bytes: bytes, output_path: str):
    """把圖片存到output資料夾，方便你對照VLM的描述"""
    with open(output_path, "wb") as f:
        f.write(img_bytes)


# ── 主程式 ────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Qwen3-VL 品質驗證腳本")
    print(f"  模型：{VL_MODEL}")
    print(f"  時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # 建立輸出資料夾
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
        print(f"   請確認Ollama正在執行中")
        return

    # 找PDF檔案
    pdf_files = sorted([f for f in os.listdir(PAPERS_DIR) if f.endswith(".pdf")])
    if not pdf_files:
        print(f"❌ 在 {PAPERS_DIR} 找不到任何PDF檔案")
        return

    print(f"📂 找到 {len(pdf_files)} 篇論文")
    print(f"   每篇最多測試 {MAX_IMAGES_PER_PAPER} 張圖片\n")

    # 彙整所有測試結果
    all_results = []
    total_images = 0
    total_success = 0

    for pdf_idx, pdf_file in enumerate(pdf_files, 1):
        pdf_path = os.path.join(PAPERS_DIR, pdf_file)
        paper_name = pdf_file.replace(".pdf", "")

        print(f"\n{'─' * 55}")
        print(f"[{pdf_idx}/{len(pdf_files)}] 處理：{pdf_file}")
        print(f"{'─' * 55}")

        # 為這篇論文建立輸出子資料夾
        paper_output_dir = os.path.join(OUTPUT_DIR, paper_name)
        os.makedirs(paper_output_dir, exist_ok=True)

        # 抽取圖片
        print(f"  📄 抽取圖片中...")
        images = extract_images_from_pdf(pdf_path)

        if not images:
            print(f"  ⚠️  此論文沒有抽取到符合尺寸的圖片，跳過")
            continue

        paper_results = []

        for img_data in images:
            total_images += 1
            filename = img_data["filename"]
            source_info = f"{pdf_file} - {filename}"

            print(f"\n  🖼️  分析：{filename} "
                  f"({img_data['width']}x{img_data['height']}px)")
            print(f"       送出給 {VL_MODEL} 分析中...", end="", flush=True)

            # 先把圖片存下來方便對照
            img_save_path = os.path.join(paper_output_dir, filename)
            save_image_for_reference(img_data["bytes"], img_save_path)

            # 呼叫VLM分析
            result = analyze_image_with_vl(
                img_bytes=img_data["bytes"],
                img_ext=img_data["ext"],
                source_info=source_info
            )

            if result["success"]:
                total_success += 1
                print(f" ✅ 完成！耗時 {result['elapsed_seconds']} 秒")
                # 印出描述（前300字預覽）
                preview = result["description"][:300]
                if len(result["description"]) > 300:
                    preview += "...(以下省略)"
                print(f"\n  📝 描述預覽：\n{preview}\n")
            else:
                print(f" ❌ 失敗：{result.get('error', '未知錯誤')}")

            paper_results.append({
                "filename": filename,
                "page": img_data["page"],
                "size": f"{img_data['width']}x{img_data['height']}",
                "success": result["success"],
                "description": result.get("description", ""),
                "error": result.get("error", ""),
                "elapsed_seconds": result["elapsed_seconds"]
            })

        # 儲存這篇論文的測試結果
        result_file = os.path.join(paper_output_dir, "vl_test_result.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump({
                "source_pdf": pdf_file,
                "tested_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": VL_MODEL,
                "images": paper_results
            }, f, ensure_ascii=False, indent=2)

        print(f"\n  💾 結果已儲存：{result_file}")
        all_results.extend(paper_results)

    # 最終統計
    print(f"\n{'=' * 65}")
    print(f"✅ 品質驗證完成！")
    print(f"   測試圖片總數：{total_images}")
    print(f"   成功分析：{total_success}")
    print(f"   失敗：{total_images - total_success}")
    print(f"\n📁 所有結果存放在：{OUTPUT_DIR}/")
    print(f"   請對照圖片檔案與json描述，評估品質是否符合需求")
    print(f"\n👉 品質確認後，告訴我結果，我們繼續建立完整pipeline！")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()