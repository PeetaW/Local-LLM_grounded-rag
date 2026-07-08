# eval/run_eval.py
# Tier 0 評估骨架：把標準題組跑過完整 pipeline，計算量化指標。
# 這是「量尺」——之後任何 config / 檢索 / prompt 的改動，都用它跑回歸來證明有沒有效。
#
# 用法：
#   python eval/run_eval.py --run --label baseline      # 跑題組，存 results/eval_baseline.json
#   python eval/run_eval.py --run --label rerank24      # 改 config 後再跑一次
#   python eval/run_eval.py --compare baseline rerank24 # 比較兩次彙總指標
#
# 兩種模式（依 eval_set.json 是否填了 gold 欄位自動切換）：
#   Mode 1（gold 留空）  ：只報 grounding 分數、延遲、論文/子問題數 → 不用標答案即可用
#   Mode 2（填了 gold）  ：加報「論文選擇命中率」「檢索覆蓋率」→ 真正對照人標真相
#
# 注意：
#   - 本腳本直接呼叫 execute_structured_query，不寫入 ChromaDB 記憶（不污染 episodic）。
#   - import main 會載入/重建所有索引，首次啟動需數分鐘。

import sys
import os
import json
import time
import datetime
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import metrics  # 同目錄

EVAL_DIR    = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
EVAL_SET    = os.path.join(EVAL_DIR, "eval_set.json")


class _Tee:
    """同時把輸出寫到原本的終端機與一個 log 檔（供深度 debug / 給 Claude 看完整過程）。"""
    def __init__(self, stream, file_obj):
        self._stream = stream
        self._file = file_obj

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def __getattr__(self, attr):
        return getattr(self._stream, attr)


def _load_questions() -> list:
    with open(EVAL_SET, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [q for q in data.get("questions", []) if q.get("question")]


def _probe_selection(question: str, all_names: list):
    """
    獨立量測「粗篩 + 論文選擇」層是否保留了 gold 論文。
    複製 pipeline 的選擇邏輯（query_pipeline.py Stage 1 之前）。
    回傳 (selected_papers, detected_paper)。
    """
    import config as cfg
    from rag.query_planning import (
        detect_target_paper, _keyword_prefilter, select_relevant_papers,
    )

    detected = detect_target_paper(question, all_names)
    if cfg.REVIEW_MODE:
        return all_names, detected
    if detected:
        # 命中特定論文時，pipeline 會把子問題全部鎖定到該篇
        return [detected], detected
    prefiltered = _keyword_prefilter(question, all_names)
    selected = select_relevant_papers(question, prefiltered)
    return selected, detected


def _probe_retrieval(question: str, gold_papers: list, paper_engines: dict) -> list:
    """
    從 gold 論文的引擎檢索，回傳檢索到的 chunk 原文（供 span recall 用）。
    只在有 gold_spans 時呼叫。
    """
    from rag.query_embedding_guard import prepare_query_text

    texts = []
    try:
        qt = prepare_query_text(question)
    except Exception:
        return texts

    for name in gold_papers:
        engine = paper_engines.get(name)
        retr = getattr(engine, "retriever", None) if engine else None
        if retr is None:
            continue
        try:
            for nws in retr.retrieve(qt):
                texts.append(nws.node.get_content())
        except Exception:
            continue
    return texts


def _fmt(v, pct=False, suffix=""):
    """格式化指標值；None → N/A，負數（如 grounding 解析失敗的 -1）→ —。"""
    if v is None:
        return "N/A"
    if isinstance(v, (int, float)) and v < 0:
        return "—"
    if pct and isinstance(v, (int, float)):
        return f"{v:.1%}"
    return f"{v}{suffix}"


def _ms_to_s(ms):
    return "N/A" if ms is None else f"{ms / 1000:.1f}s"


def _q_status(row: dict) -> str:
    """為每題決定狀態 emoji，方便在報告裡一眼掃出有問題的題目。"""
    ans = row.get("answer", "")
    if isinstance(ans, str) and ans.startswith("[PIPELINE ERROR]"):
        return "❌"
    sel = row.get("paper_selection_recall")
    ret = row.get("retrieval_span_recall")
    gs  = row.get("grounding_score")
    bad = (
        (ret is not None and ret < 0.3)
        or (isinstance(gs, (int, float)) and 0 <= gs < 0.3)
    )
    if bad:
        return "❌"
    warn = (
        (sel is not None and sel < 1.0)
        or (ret is not None and ret < 0.7)
        or (isinstance(gs, (int, float)) and 0 <= gs < 0.8)
    )
    return "⚠️" if warn else "✅"


def _correctness_candidate(answer: str, artifacts: dict, reference: str = "") -> tuple[str, str]:
    candidate = (artifacts or {}).get("answer_for_judge")
    if candidate and not any("\u4e00" <= ch <= "\u9fff" for ch in reference or ""):
        return candidate, "answer_for_judge"
    return answer, "answer"


def _write_markdown_report(out: dict, path: str):
    """產生人類好掃的 Markdown 報告（與 JSON 並存）。"""
    s = out.get("summary", {})
    L = []
    L.append(f"# Eval Report — `{out.get('label')}`")
    L.append("")
    mode = "Mode 2（對照 gold 真相）" if out.get("mode") == "gold" else "Mode 1（自評／延遲）"
    L.append(f"- 模式：{mode}")
    L.append(f"- 產生時間：{datetime.datetime.now():%Y-%m-%d %H:%M}")
    L.append(f"- 題數：{s.get('n_questions')}")
    L.append("")

    L.append("## 彙總")
    L.append("")
    L.append("| 指標 | 值 |")
    L.append("|------|-----|")
    L.append(f"| 平均正確性（LLM-judge） | {_fmt(s.get('avg_correctness'))} |")
    L.append(f"| 平均 grounding 分數 | {_fmt(s.get('avg_grounding_score'))} |")
    L.append(f"| 平均論文選擇命中率 | {_fmt(s.get('avg_paper_sel_recall'), pct=True)} |")
    L.append(f"| 平均檢索覆蓋率 | {_fmt(s.get('avg_retrieval_recall'), pct=True)} |")
    L.append(f"| 平均總延遲 | {_ms_to_s(s.get('avg_total_ms'))} |")
    L.append(f"| 平均 planning 延遲 | {_ms_to_s(s.get('avg_planning_ms'))} |")
    L.append(f"| 平均 retrieval 延遲 | {_ms_to_s(s.get('avg_retrieval_ms'))} |")
    L.append(f"| └ Phase A embed/vector/BM25 | {_ms_to_s(s.get('avg_retrieval_phase_a_ms'))} |")
    L.append(f"| └ Phase B 子答案生成 | {_ms_to_s(s.get('avg_retrieval_phase_b_ms'))} |")
    L.append(f"| 平均 grounding 延遲 | {_ms_to_s(s.get('avg_grounding_ms'))} |")
    L.append(f"| └ 其中 NLI | {_ms_to_s(s.get('avg_grounding_nli_ms'))} |")
    L.append(f"| └ 其中 gemma4 | {_ms_to_s(s.get('avg_grounding_llm_ms'))} |")
    L.append("")

    L.append("## 逐題速覽")
    L.append("")
    L.append("| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |")
    L.append("|---|----|------|---------|---------|-----------|------|------|")
    for r in out.get("rows", []):
        lat = r.get("latency") or {}
        iss = r.get("issues") or {}
        L.append(
            f"| {_q_status(r)} | {r.get('id')} | {r.get('type','')} | "
            f"{_fmt(r.get('paper_selection_recall'), pct=True)} | "
            f"{_fmt(r.get('retrieval_span_recall'), pct=True)} | "
            f"{_fmt(r.get('grounding_score'))} | "
            f"{_ms_to_s(lat.get('total'))} | "
            f"C{iss.get('conflicts', 0)}/U{iss.get('unsupported', 0)} |"
        )
    L.append("")

    L.append("## 逐題細節")
    L.append("")
    for r in out.get("rows", []):
        L.append(f"### {_q_status(r)} {r.get('id')} · {r.get('type','')}")
        L.append("")
        L.append(f"**問題**：{r.get('question')}")
        L.append("")
        L.append(f"- detected_paper：`{r.get('detected_paper')}`")
        L.append(f"- 選出論文：{r.get('selected_papers')}")
        L.append(f"- gold_papers：{r.get('gold_papers')}")
        L.append(f"- correctness candidate：`{r.get('correctness_candidate_source', 'answer')}`")
        L.append(
            f"- 論文選擇命中率：{_fmt(r.get('paper_selection_recall'), pct=True)}　"
            f"檢索覆蓋率：{_fmt(r.get('retrieval_span_recall'), pct=True)}　"
            f"grounding：{_fmt(r.get('grounding_score'))}"
        )
        L.append(f"- 延遲：{_ms_to_s((r.get('latency') or {}).get('total'))}　問題標記：{r.get('issues')}")
        L.append("")
        ans = (r.get("answer") or "").strip()
        if len(ans) > 800:
            ans = ans[:800] + " …（完整內容見 JSON）"
        L.append("**答案預覽**：")
        L.append("")
        L.append("> " + ans.replace("\n", "\n> "))
        L.append("")
        L.append("---")
        L.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def _print_row(row: dict):
    sel = row["paper_selection_recall"]
    ret = row["retrieval_span_recall"]
    lat = row["latency"]
    print(f"  論文選擇命中率: {sel if sel is not None else 'N/A（無 gold_papers）'}"
          f"   detected={row['detected_paper']}")
    print(f"  選出論文      : {row['selected_papers']}")
    print(f"  檢索覆蓋率    : {ret if ret is not None else 'N/A（無 gold_spans）'}")
    cr = row.get("correctness")
    print(f"  正確性(judge) : {cr if cr is not None else 'N/A'}")
    print(f"  judge candidate: {row.get('correctness_candidate_source', 'answer')}")
    print(f"  grounding     : {row['grounding_score']}")
    def _s(k):
        v = lat.get(k)
        return f"{v/1000:.0f}s" if isinstance(v, (int, float)) else "n/a"
    print(f"  延遲      : plan={_s('planning')} retr={_s('retrieval')} synth={_s('synthesis')} "
          f"gen={_s('synthesis-llm')} verify={_s('verification')} grnd={_s('grounding')} "
          f"trans={_s('translation')} | total={_s('total')}")
    print(f"  問題標記      : {row['issues']}")


def run(label: str, limit: int = None, retrieval_only: bool = False, ids: str = None):
    # ── 把完整 console 輸出（索引載入、逐句 NLI、萬一的 traceback）同步寫進 log 檔 ──
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_path = os.path.join(RESULTS_DIR, f"eval_{label}.log")
    _log_file = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.stdout, _log_file)
    sys.stderr = _Tee(sys.stderr, _log_file)

    from main import paper_engines
    from rag.query_pipeline import execute_structured_query

    questions = _load_questions()
    if ids:
        want = {x.strip().upper() for x in ids.split(",") if x.strip()}
        questions = [q for q in questions if q.get("id", "").upper() in want]
        print(f"⚠️  --ids {sorted(want)}：只跑指定 {len(questions)} 題（挑代表題快速迭代）")
    if limit:
        questions = questions[:limit]
        print(f"⚠️  --limit {limit}：只跑前 {len(questions)} 題（快速測試/重現用）")
    all_names = list(paper_engines.keys())

    has_gold = any(q.get("gold_papers") for q in questions)
    print(f"\n{'#'*70}")
    print(f"# 評估模式：{'Mode 2（對照 gold 真相）' if has_gold else 'Mode 1（僅自評/延遲）'}")
    print(f"# 題數：{len(questions)}　論文庫：{len(all_names)} 篇　label={label}")
    print(f"{'#'*70}")

    rows = []
    for i, q in enumerate(questions, 1):
        qid   = q.get("id", f"Q{i}")
        qtext = q["question"]
        gold_papers = q.get("gold_papers", [])
        gold_spans  = q.get("gold_spans", [])

        print(f"\n{'='*70}\n[{i}/{len(questions)}] {qid}  ({q.get('type','?')})\n{qtext}\n{'='*70}")

        # 1) 選擇層探測（獨立量測，不影響主 pipeline）
        selected, detected = _probe_selection(qtext, all_names)
        sel_recall = metrics.paper_selection_recall(selected, gold_papers)

        # 2) 檢索層探測（只在有 gold_spans 時做）
        ret_recall = None
        if gold_spans and gold_papers:
            ret_texts = _probe_retrieval(qtext, gold_papers, paper_engines)
            ret_recall = metrics.retrieval_span_recall(ret_texts, gold_spans)

        # 3) 跑完整 pipeline（--retrieval-only 時跳過，只看選擇/檢索）
        status_lines = []
        artifacts = {}
        if retrieval_only:
            answer, wall_s = "", 0.0
        else:
            t0 = time.time()
            try:
                answer = execute_structured_query(
                    qtext, paper_engines, "", on_status=status_lines.append,
                    on_artifact=artifacts.__setitem__,
                )
            except Exception as e:
                answer = f"[PIPELINE ERROR] {e}"
            wall_s = round(time.time() - t0, 1)

        # 4) 正確性 LLM-judge（有 reference_answer 且非 retrieval-only 時）
        correctness, correctness_detail = None, None
        reference = q.get("reference_answer", "")
        answer_for_judge, candidate_source = _correctness_candidate(answer, artifacts, reference)
        if reference and not retrieval_only and answer:
            from judge import judge_correctness  # 同目錄
            j = judge_correctness(qtext, answer_for_judge, reference)
            correctness = j["score"]
            correctness_detail = {"raw": j["raw"], "reason": j["reason"]}
            print(f"  ⚖️  [Judge] correctness={correctness}（{j['raw']}/5, {candidate_source}）：{j['reason'][:80]}")

        row = {
            "id": qid,
            "type": q.get("type"),
            "question": qtext,
            "detected_paper": detected,
            "selected_papers": selected,
            "gold_papers": gold_papers,
            "paper_selection_recall": sel_recall,
            "retrieval_span_recall": ret_recall,
            "grounding_score": metrics.parse_grounding_score(answer),
            "correctness": correctness,
            "correctness_detail": correctness_detail,
            "correctness_candidate_source": candidate_source,
            "answer_for_judge": answer_for_judge,
            "knowledge_base": artifacts.get("knowledge_base"),
            "counts": metrics.parse_counts(status_lines),
            "latency": metrics.parse_stage_latencies(status_lines),
            "wall_seconds": wall_s,
            "issues": metrics.count_issues(answer),
            "answerability": next((ln.split("[answerability]", 1)[1].strip()
                                   for ln in status_lines if "[answerability]" in ln), None),
            "answer": answer,
        }
        rows.append(row)
        _print_row(row)

    summary = metrics.summarize(rows)
    out = {"label": label, "mode": "gold" if has_gold else "self", "summary": summary, "rows": rows}
    path = os.path.join(RESULTS_DIR, f"eval_{label}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    md_path = os.path.join(RESULTS_DIR, f"eval_{label}.md")
    _write_markdown_report(out, md_path)

    print(f"\n{'#'*70}\n# 彙總（label={label}）\n{'#'*70}")
    for k, v in summary.items():
        print(f"  {k:24s}: {v}")
    print(f"\nJSON 結果（給 Claude 細看）：{path}")
    print(f"Markdown 報告（給你快速掃）：{md_path}")
    print(f"完整 console log：{log_path}")

    # 還原 stdout/stderr 並關閉 log 檔
    sys.stdout = sys.stdout._stream
    sys.stderr = sys.stderr._stream
    _log_file.close()


def compare(label_a: str, label_b: str):
    def _load(lbl):
        with open(os.path.join(RESULTS_DIR, f"eval_{lbl}.json"), "r", encoding="utf-8") as f:
            return json.load(f)

    a, b = _load(label_a), _load(label_b)
    print(f"\n{'='*70}\n比較 {label_a} vs {label_b}\n{'='*70}")
    print(f"  {'指標':22s} {label_a:>14s} {label_b:>14s}")
    for k in sorted(set(a["summary"]) | set(b["summary"])):
        print(f"  {k:22s} {str(a['summary'].get(k)):>14s} {str(b['summary'].get(k)):>14s}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Tier 0 RAG 評估骨架")
    ap.add_argument("--run", action="store_true", help="跑題組")
    ap.add_argument("--label", default="baseline", help="這次結果的標籤（檔名用）")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 題（快速測試/重現用）")
    ap.add_argument("--retrieval-only", action="store_true", help="只測選擇/檢索覆蓋率，跳過完整 pipeline（幾分鐘）")
    ap.add_argument("--ids", default=None, help="只跑指定題號，逗號分隔，如 Q05,Q06,Q08（挑代表題快速迭代）")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="比較兩個 label 的彙總")
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
    elif args.run:
        run(args.label, args.limit, args.retrieval_only, args.ids)
    else:
        ap.print_help()
