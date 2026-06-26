# rag/corpus_health.py
# 匯入健檢：抓語料三類污染——重複(精確+改名)、SI 當獨立論文、抽取壞(OCR/空)。零 gold 標籤。
# 兩入口：standalone 審計報告（main.py --health）+ ingestion 防呆（indexer 去重）。
# MVP：純計算 + 報告 + 去重跳過；不寫 metadata、不綁 SI、不自動刪檔、不自我查詢（皆 Phase 2，見 ingestion_health_spec.md）。
import os
import sys
import re
import hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # 讓 standalone 也能 import config
import fitz  # PyMuPDF
import config as cfg

# 健康門檻（ponytail: 啟發式上限，數值不夠用再調）
_MIN_TEXT_LEN = 500        # 正規化全文短於此 → 疑抽取失敗/掃描檔
_MAX_GARBAGE_RATIO = 0.05  # 亂碼/控制字元佔比高於此 → 疑 OCR 壞

# SI 偵測：大寫 SI 後綴（非字母邊界，避開 synthesis 的小寫 si；如 "2026SI" 命中）
_SI_RE = re.compile(r"(?<![A-Za-z])SI(?![A-Za-z])")
# supplement / supporting info（不分大小寫）
_SUPP_RE = re.compile(r"(?i)(support(?:ing)?[\s_\-]*info|supplement(?:ary|[\s_\-]*info)?)")


def _extract_text(pdf_path: str) -> str:
    """純文字抽取（fitz），不觸發 VL。"""
    try:
        doc = fitz.open(pdf_path)
        return "".join(page.get_text("text") for page in doc)
    except Exception:
        return ""


def _garbage_ratio(text: str) -> float:
    if not text:
        return 1.0
    bad = sum(1 for c in text if c == "�" or (ord(c) < 32 and c not in "\t\n\r"))
    return bad / len(text)


def _is_si(name: str) -> bool:
    return bool(_SI_RE.search(name) or _SUPP_RE.search(name))


def paper_health(pdf_file: str) -> dict:
    """單篇健檢指標。pdf_file = 檔名（含 .pdf）。"""
    pdf_path = os.path.join(cfg.PAPERS_DIR, pdf_file)
    raw = _extract_text(pdf_path)
    norm = re.sub(r"\s+", " ", raw.lower()).strip()
    name = pdf_file[:-4] if pdf_file.endswith(".pdf") else pdf_file
    return {
        "file": pdf_file,
        "name": name,
        "text_sha": hashlib.sha1(norm.encode("utf-8")).hexdigest() if norm else None,
        "text_len": len(norm),
        "garbage_ratio": round(_garbage_ratio(raw), 4),
        "is_si": _is_si(name),
    }


def scan_corpus() -> list:
    files = sorted(f for f in os.listdir(cfg.PAPERS_DIR) if f.endswith(".pdf"))
    return [paper_health(f) for f in files]


def _groups_by_sha(healths: list) -> dict:
    by_sha = {}
    for h in healths:
        if h["text_sha"]:
            by_sha.setdefault(h["text_sha"], []).append(h)
    return by_sha


def find_duplicate_groups(healths: list) -> list:
    """回傳 [[name,...], ...]，每組同 text_sha 的重複（>1）；排序首位為保留建議。"""
    return [sorted(h["name"] for h in g)
            for g in _groups_by_sha(healths).values() if len(g) > 1]


def _load_titles() -> dict:
    """name -> title（取自 papers_metadata.json）。讀不到回 {}。"""
    import json
    try:
        d = json.load(open(cfg.METADATA_PATH, encoding="utf-8"))
        return {k: (v.get("title", "") if isinstance(v, dict) else "") for k, v in d.items()}
    except Exception:
        return {}


def find_near_duplicate_groups(healths: list) -> list:
    """同 title 但 text_sha 不同 → 疑似近重複（改名/微差）。只報告供人工審查，不自動跳過
    （內容 sha 不同代表抽取文字有差，自動跳過有丟內容風險）。已被精確去重的組（同 sha）不重列。"""
    titles = _load_titles()
    by_title = {}
    for h in healths:
        t = re.sub(r"\s+", " ", (titles.get(h["name"], "") or "").lower()).strip()
        if t:
            by_title.setdefault(t, []).append(h)
    # 含 SI 成員的同 title 組＝主文+補充關係（歸 SI 區），非近重複，排除避免誤報。
    return [sorted(h["name"] for h in g)
            for g in by_title.values()
            if len(g) > 1 and len({h["text_sha"] for h in g}) > 1
            and not any(h["is_si"] for h in g)]


def duplicate_skip_set(pdf_files: list) -> dict:
    """給 ingestion 用：{冗餘檔名: 保留檔名}。同 sha 多份時，保留排序首位、其餘跳過。"""
    healths = [paper_health(f) for f in sorted(pdf_files)]
    skip = {}
    for g in _groups_by_sha(healths).values():
        if len(g) > 1:
            files = sorted(h["file"] for h in g)
            for f in files[1:]:
                skip[f] = files[0]
    return skip


def audit_report() -> str:
    healths = scan_corpus()
    out = [f"# 語料健檢報告（{len(healths)} 篇）", ""]

    dups = find_duplicate_groups(healths)
    out.append(f"## 重複（同正規化全文，自動跳過）：{len(dups)} 組")
    out += [f"  - 保留 `{g[0]}`；冗餘：{', '.join('`'+x+'`' for x in g[1:])}" for g in dups] or ["  （無）"]

    near = find_near_duplicate_groups(healths)
    out += ["", f"## 疑似近重複（同 title、內容微差，請人工審查）：{len(near)} 組"]
    out += [f"  - {', '.join('`'+x+'`' for x in g)}" for g in near] or ["  （無）"]

    sis = [h["name"] for h in healths if h["is_si"]]
    out += ["", f"## SI/補充資料當獨立論文：{len(sis)} 篇"]
    out += [f"  - `{n}`（檔名疑為 SI；建議綁回主文，Phase 2）" for n in sis] or ["  （無）"]

    broken = [h for h in healths
              if h["text_len"] < _MIN_TEXT_LEN or h["garbage_ratio"] > _MAX_GARBAGE_RATIO]
    out += ["", f"## 抽取健康警示：{len(broken)} 篇"]
    for h in broken:
        why = []
        if h["text_len"] < _MIN_TEXT_LEN:
            why.append(f"text_len={h['text_len']}<{_MIN_TEXT_LEN}")
        if h["garbage_ratio"] > _MAX_GARBAGE_RATIO:
            why.append(f"garbage={h['garbage_ratio']}>{_MAX_GARBAGE_RATIO}")
        out.append(f"  - `{h['name']}`：{', '.join(why)}")
    if not broken:
        out.append("  （無，全部抽取正常）")

    return "\n".join(out)


if __name__ == "__main__":
    # 自檢：離線測啟發式邏輯（不碰檔案）。
    assert _is_si("LAT1 ChemComm 2026SI")                 # 大寫 SI 後綴
    assert _is_si("Supplement_info_ ligand-enabled")      # supplement
    assert not _is_si("synthesis-and-biological-props")   # 小寫 si 不誤判
    assert not _is_si("s41467-024-45464-z (1)")           # S+數字不誤判
    assert _garbage_ratio("") == 1.0
    assert _garbage_ratio("clean text") == 0.0
    _fake = [{"name": "a", "file": "a.pdf", "text_sha": "x"},
             {"name": "b", "file": "b.pdf", "text_sha": "x"},
             {"name": "c", "file": "c.pdf", "text_sha": "y"}]
    assert find_duplicate_groups(_fake) == [["a", "b"]]
    assert duplicate_skip_set(["b.pdf"]) == {}  # 單檔不會誤判（需真檔，這裡只測無重複路徑）
    print("corpus_health.py self-check OK")
