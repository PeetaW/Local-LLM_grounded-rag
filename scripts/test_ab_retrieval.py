# scripts/test_ab_retrieval.py
# A/B 測試：比較有無 chunk summarization 時的 pipeline 檢索與答案品質。
#
# 使用方式：
#   # Step 1：用現有 index（condition A）跑並記錄
#   python scripts/test_ab_retrieval.py --run --label A
#
#   # Step 2：改 config + 刪 index + 重建後跑 B
#   #   config.py: CONTEXT_SUMMARY_ENABLED = False
#   #   刪除 index_storage/ 各論文子目錄
#   #   重啟一次讓 indexer 重建，再執行：
#   python scripts/test_ab_retrieval.py --run --label B
#
#   # Step 3：比較結果
#   python scripts/test_ab_retrieval.py --compare A B
#
# 結果存放位置：scripts/ab_results_A.json / ab_results_B.json

import sys
import os
import json
import time
import re
import io
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

RESULTS_DIR = os.path.dirname(__file__)


# ══════════════════════════════════════════════════════════════════
# 固定測試題組（維持不變，每次 A/B 都用相同題目）
#
# 題目類型說明：
#   direct_citation  → 答案高度仰賴單篇論文原文，直引 grounding 應 ≥ 0.8
#   multi_chunk      → 答案需跨多個 chunk 拼湊，grounding 中等
#   cross_paper      → 跨篇比較，grounding 預期偏低但不應矛盾
#   out_of_scope     → 資料庫沒有的主題，答案應誠實說明，grounding 低屬正常
# ══════════════════════════════════════════════════════════════════
TEST_QUESTIONS = [
    {
        "id": "Q1",
        "type": "direct_citation",
        "desc": "合成步驟直引（單篇，含具體數值）",
        "question": (
            "glycine 修飾 nZVI 的合成步驟中，具體使用了哪些試藥、"
            "比例與反應條件？gelatin aerogel 在其中扮演什麼角色？"
            "請盡量引用論文中的具體數值。"
        ),
    },
    {
        "id": "Q2",
        "type": "multi_chunk",
        "desc": "降解機制多段整合",
        "question": (
            "NZVI@G-GEL 降解四環素的機制是什麼？"
            "自由基與非自由基路徑各扮演什麼角色？"
            "請引用論文中的實驗數據支持你的說明。"
        ),
    },
    {
        "id": "Q3",
        "type": "cross_paper",
        "desc": "跨篇比較（multi-paper）",
        "question": (
            "目前資料庫的文獻中合成 nZVI 的方法有哪幾種？"
            "比較不同修飾策略對粒徑控制與反應活性的影響。"
        ),
    },
    {
        "id": "Q4",
        "type": "out_of_scope",
        "desc": "範圍外誠實性測試",
        "question": (
            "nZVI 在實際土壤修復工程中對重金屬（鉛、鎘）的現場去除效率如何？"
            "有沒有 field scale 實驗數據？"
        ),
    },
]


# ══════════════════════════════════════════════════════════════════
# 解析 grounding 分數
# ══════════════════════════════════════════════════════════════════

def _parse_grounding_scores(text: str) -> dict:
    """
    從 nli_report 文字中解析各段落 grounding 分數。
    回傳 dict，缺少的 key 為 None。
    """
    scores = {}

    # 直引依據率（section_scores 模式）
    m = re.search(r'直引依據率[：:]\s*([\d.]+)%', text)
    if m:
        scores["direct"] = round(float(m.group(1)) / 100, 3)

    # 整體論文依據率（fallback 模式）
    m = re.search(r'整體論文依據率[：:]\s*([\d.]+)%', text)
    if m:
        scores["overall"] = round(float(m.group(1)) / 100, 3)

    # 分段：論文直接依據 / 跨文獻推論 / 知識延伸推測
    for key, label in [
        ("direct",      "論文直接依據"),
        ("inference",   "跨文獻推論"),
        ("speculation", "知識延伸推測"),
    ]:
        m = re.search(rf'【{label}】[：:]\s*([\d.]+)%', text)
        if m:
            scores[key] = round(float(m.group(1)) / 100, 3)

    # 衝突句數
    m = re.search(r'偵測到\s*(\d+)\s*個陳述.*?矛盾', text)
    scores["conflict_count"] = int(m.group(1)) if m else 0

    return scores


def _parse_timing_from_status(status_msgs: list) -> dict:
    """從 on_status 訊息裡解析各 stage 耗時（ms）。"""
    timing = {}
    patterns = {
        "planning":       r'\[planning\].*elapsed_ms=(\d+)',
        "retrieval":      r'\[retrieval\].*elapsed_ms=(\d+)',
        "synthesis":      r'\[synthesis\].*elapsed_ms=(\d+)',
        "synthesis_llm":  r'\[synthesis-llm\].*elapsed_ms=(\d+)',
        "verification":   r'\[verification\].*elapsed_ms=(\d+)',
        "grounding":      r'\[grounding\].*elapsed_ms=(\d+)',
        "translation":    r'\[translation\].*elapsed_ms=(\d+)',
        "total":          r'\[pipeline\].*total_elapsed_ms=(\d+)',
    }
    combined = "\n".join(status_msgs)
    for key, pat in patterns.items():
        m = re.search(pat, combined)
        if m:
            timing[key] = int(m.group(1))
    return timing


def _count_embed_nan(stdout_capture: str) -> int:
    """從捕捉到的 stdout 裡計算 embed NaN 事件次數。"""
    return len(re.findall(r'\[embed-debug\].*NaN', stdout_capture))


def _count_embed_retry(stdout_capture: str) -> int:
    """計算 embed retry 次數。"""
    return len(re.findall(r'\[embed-debug\]', stdout_capture))


# ══════════════════════════════════════════════════════════════════
# 記錄一次完整跑
# ══════════════════════════════════════════════════════════════════

def run_and_record(label: str):
    import config as cfg

    print(f"\n{'='*65}")
    print(f"A/B 測試 — 記錄模式  label={label}")
    print(f"CONTEXT_SUMMARY_ENABLED = {cfg.CONTEXT_SUMMARY_ENABLED}")
    print(f"{'='*65}\n")

    # 延遲 import，讓 banner 先印出來
    from main import paper_engines
    from rag.query_pipeline import execute_structured_query

    results = {
        "label":           label,
        "timestamp":       datetime.now().isoformat(),
        "context_summary": cfg.CONTEXT_SUMMARY_ENABLED,
        "questions":       [],
    }

    for q in TEST_QUESTIONS:
        print(f"\n{'─'*65}")
        print(f"[{q['id']}] {q['desc']}")
        print(f"    {q['question'][:80]}...")
        print(f"{'─'*65}")

        status_msgs = []
        stdout_buf  = io.StringIO()

        # 攔截 stdout 以抓取 embed-debug 訊息
        _real_stdout = sys.stdout

        class _Tee:
            def write(self, data):
                _real_stdout.write(data)
                stdout_buf.write(data)
            def flush(self):
                _real_stdout.flush()
            def __getattr__(self, name):
                return getattr(_real_stdout, name)

        sys.stdout = _Tee()

        t0 = time.perf_counter()
        try:
            answer = execute_structured_query(
                question=q["question"],
                paper_engines=paper_engines,
                on_status=lambda msg: status_msgs.append(msg),
            )
            elapsed = time.perf_counter() - t0
            error   = None
        except Exception as e:
            elapsed = time.perf_counter() - t0
            answer  = ""
            error   = str(e)
            print(f"  ❌ 執行失敗：{e}")
        finally:
            sys.stdout = _real_stdout

        captured = stdout_buf.getvalue()
        scores   = _parse_grounding_scores(answer)
        timing   = _parse_timing_from_status(status_msgs)
        nan_cnt  = _count_embed_nan(captured)
        retry_cnt = _count_embed_retry(captured)

        q_result = {
            "id":             q["id"],
            "type":           q["type"],
            "desc":           q["desc"],
            "question":       q["question"],
            "answer_preview": answer[:300] if answer else "",
            "answer_full":    answer,
            "scores":         scores,
            "timing_ms":      timing,
            "elapsed_total_s": round(elapsed, 1),
            "embed_nan_count":   nan_cnt,
            "embed_retry_count": retry_cnt,
            "error":          error,
        }
        results["questions"].append(q_result)

        # 即時摘要
        direct = scores.get("direct", scores.get("overall", "N/A"))
        print(f"\n  ✓ 完成  直引 grounding={direct}  "
              f"NaN={nan_cnt}  耗時={elapsed:.0f}s")

    # 存檔
    out_path = os.path.join(RESULTS_DIR, f"ab_results_{label}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 結果已儲存：{out_path}")
    _print_summary(results)


# ══════════════════════════════════════════════════════════════════
# 單次結果摘要
# ══════════════════════════════════════════════════════════════════

def _print_summary(results: dict):
    label = results["label"]
    print(f"\n{'='*65}")
    print(f"摘要  label={label}  ({results['timestamp'][:16]})")
    print(f"context_summary={results['context_summary']}")
    print(f"{'─'*65}")
    print(f"  {'ID':<4}  {'類型':<16}  {'直引':>6}  {'推論':>6}  {'NaN':>4}  {'秒':>6}")
    print(f"  {'─'*4}  {'─'*16}  {'─'*6}  {'─'*6}  {'─'*4}  {'─'*6}")
    for q in results["questions"]:
        s      = q["scores"]
        direct = f"{s.get('direct', s.get('overall', 0)):.1%}" if (s.get('direct') or s.get('overall')) else "N/A"
        infer  = f"{s['inference']:.1%}" if s.get('inference') is not None else "N/A"
        nan    = q["embed_nan_count"]
        secs   = q["elapsed_total_s"]
        err    = " ❌" if q["error"] else ""
        print(f"  {q['id']:<4}  {q['desc'][:16]:<16}  {direct:>6}  {infer:>6}  {nan:>4}  {secs:>6.0f}{err}")
    print(f"{'='*65}\n")


# ══════════════════════════════════════════════════════════════════
# 比較兩次結果
# ══════════════════════════════════════════════════════════════════

def compare(label_a: str, label_b: str):
    path_a = os.path.join(RESULTS_DIR, f"ab_results_{label_a}.json")
    path_b = os.path.join(RESULTS_DIR, f"ab_results_{label_b}.json")

    for p, lbl in [(path_a, label_a), (path_b, label_b)]:
        if not os.path.exists(p):
            print(f"❌ 找不到結果檔案：{p}")
            print(f"   請先執行：python scripts/test_ab_retrieval.py --run --label {lbl}")
            sys.exit(1)

    with open(path_a, encoding="utf-8") as f:
        res_a = json.load(f)
    with open(path_b, encoding="utf-8") as f:
        res_b = json.load(f)

    qa_map = {q["id"]: q for q in res_a["questions"]}
    qb_map = {q["id"]: q for q in res_b["questions"]}

    print(f"\n{'='*75}")
    print(f"A/B 比較結果")
    print(f"  A = {label_a}  (context_summary={res_a['context_summary']})  {res_a['timestamp'][:16]}")
    print(f"  B = {label_b}  (context_summary={res_b['context_summary']})  {res_b['timestamp'][:16]}")
    print(f"{'='*75}")

    # ── 直引 grounding 比較 ──────────────────────────────────────
    print(f"\n{'─'*75}")
    print(f"  {'ID':<4}  {'描述':<18}  {'A 直引':>7}  {'B 直引':>7}  {'差異':>7}  {'A NaN':>6}  {'B NaN':>6}  {'A 秒':>6}  {'B 秒':>6}")
    print(f"  {'─'*4}  {'─'*18}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}")

    all_ids = sorted(set(list(qa_map.keys()) + list(qb_map.keys())))
    for qid in all_ids:
        qa = qa_map.get(qid)
        qb = qb_map.get(qid)

        def _score(q):
            if q is None:
                return None
            s = q["scores"]
            return s.get("direct", s.get("overall"))

        sa = _score(qa)
        sb = _score(qb)

        sa_str   = f"{sa:.1%}" if sa is not None else "N/A"
        sb_str   = f"{sb:.1%}" if sb is not None else "N/A"
        diff_str = "N/A"
        if sa is not None and sb is not None:
            diff = sb - sa
            sign = "+" if diff >= 0 else ""
            diff_str = f"{sign}{diff:.1%}"

        na_nan  = qa["embed_nan_count"] if qa else "N/A"
        nb_nan  = qb["embed_nan_count"] if qb else "N/A"
        na_sec  = qa["elapsed_total_s"] if qa else "N/A"
        nb_sec  = qb["elapsed_total_s"] if qb else "N/A"
        desc    = (qa or qb)["desc"][:18]

        print(f"  {qid:<4}  {desc:<18}  {sa_str:>7}  {sb_str:>7}  {diff_str:>7}  "
              f"{na_nan!s:>6}  {nb_nan!s:>6}  {na_sec!s:>6}  {nb_sec!s:>6}")

    # ── 整體統計 ─────────────────────────────────────────────────
    def _avg_score(results):
        scores = []
        for q in results["questions"]:
            s = q["scores"]
            v = s.get("direct", s.get("overall"))
            if v is not None:
                scores.append(v)
        return sum(scores) / len(scores) if scores else 0.0

    def _total_nan(results):
        return sum(q["embed_nan_count"] for q in results["questions"])

    def _avg_sec(results):
        secs = [q["elapsed_total_s"] for q in results["questions"] if q["elapsed_total_s"]]
        return sum(secs) / len(secs) if secs else 0.0

    avg_a = _avg_score(res_a)
    avg_b = _avg_score(res_b)
    diff  = avg_b - avg_a
    sign  = "+" if diff >= 0 else ""

    print(f"{'─'*75}")
    print(f"  {'平均直引 grounding':<24}  {avg_a:.1%}  →  {avg_b:.1%}   差異: {sign}{diff:.1%}")
    print(f"  {'embed NaN 總計':<24}  {_total_nan(res_a)}  →  {_total_nan(res_b)}")
    print(f"  {'平均耗時（秒）':<24}  {_avg_sec(res_a):.0f}  →  {_avg_sec(res_b):.0f}")
    print(f"{'='*75}")

    # ── 判讀建議 ─────────────────────────────────────────────────
    print(f"\n📌 判讀建議：")
    if diff > 0.05:
        print(f"  grounding 分數 B 高出 A {diff:.1%}，移除摘要對 NLI 計算有正面效果。")
        print(f"  ⚠️  grounding 上升可能部分來自 NLI premise 更乾淨，不代表答案品質一定更好。")
        print(f"  建議：人工對比幾篇 answer_full 確認內容品質後再決定是否移除摘要。")
    elif diff < -0.05:
        print(f"  grounding 分數 B 低於 A {abs(diff):.1%}，移除摘要可能使 retrieval 品質下降。")
        print(f"  建議：保留 CONTEXT_SUMMARY_ENABLED=True，或進一步分析哪些題目退步。")
    else:
        print(f"  grounding 分數差異在 ±5% 以內，兩者品質相近。")
        print(f"  可優先看 NaN 次數：若 B 的 NaN 明顯較少，傾向移除摘要。")

    nan_diff = _total_nan(res_b) - _total_nan(res_a)
    if nan_diff < 0:
        print(f"  ✅ embed NaN 次數 B 減少 {abs(nan_diff)} 次，符合移除中文摘要的預期效果。")
    elif nan_diff > 0:
        print(f"  ⚠️  embed NaN 次數 B 增加 {nan_diff} 次，可能有其他因素影響。")

    # ── 答案全文另存 ────────────────────────────────────────────
    out_path = os.path.join(RESULTS_DIR, f"ab_compare_{label_a}_vs_{label_b}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"A/B 答案全文比較  {label_a} vs {label_b}\n")
        f.write(f"生成時間：{datetime.now().isoformat()}\n")
        f.write("=" * 75 + "\n\n")
        for qid in all_ids:
            qa = qa_map.get(qid)
            qb = qb_map.get(qid)
            q_obj = (qa or qb)
            f.write(f"[{qid}] {q_obj['desc']}\n")
            f.write(f"問題：{q_obj['question']}\n\n")
            f.write(f"── {label_a} 答案 ──\n")
            f.write((qa["answer_full"] if qa else "（無資料）") + "\n\n")
            f.write(f"── {label_b} 答案 ──\n")
            f.write((qb["answer_full"] if qb else "（無資料）") + "\n\n")
            f.write("─" * 75 + "\n\n")

    print(f"\n📄 答案全文已存至：{out_path}")
    print(f"   請人工閱讀確認答案品質差異。\n")


# ══════════════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="A/B 測試：比較有無 chunk summarization 的 pipeline 品質"
    )
    sub = parser.add_subparsers(dest="cmd")

    run_p = sub.add_parser("--run", help="執行測試並記錄結果")
    run_p.add_argument("--label", required=True, help="結果標籤（例如 A 或 B）")

    cmp_p = sub.add_parser("--compare", help="比較兩次結果")
    cmp_p.add_argument("label_a", help="第一個標籤（例如 A）")
    cmp_p.add_argument("label_b", help="第二個標籤（例如 B）")

    # 相容 argparse 的 --run / --compare 前綴寫法
    args, _ = parser.parse_known_args()

    if "--run" in sys.argv:
        idx = sys.argv.index("--run")
        label_idx = sys.argv.index("--label") if "--label" in sys.argv else None
        if label_idx is None or label_idx + 1 >= len(sys.argv):
            print("❌ 請指定 --label，例如：--run --label A")
            sys.exit(1)
        label = sys.argv[label_idx + 1]
        run_and_record(label)

    elif "--compare" in sys.argv:
        idx = sys.argv.index("--compare")
        if idx + 2 >= len(sys.argv):
            print("❌ 請提供兩個標籤，例如：--compare A B")
            sys.exit(1)
        compare(sys.argv[idx + 1], sys.argv[idx + 2])

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
