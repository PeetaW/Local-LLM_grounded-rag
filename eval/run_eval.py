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
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import metrics  # 同目錄

EVAL_DIR    = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
EVAL_SET    = os.path.join(EVAL_DIR, "eval_set.json")


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


def _print_row(row: dict):
    sel = row["paper_selection_recall"]
    ret = row["retrieval_span_recall"]
    lat = row["latency"]
    print(f"  論文選擇命中率: {sel if sel is not None else 'N/A（無 gold_papers）'}"
          f"   detected={row['detected_paper']}")
    print(f"  選出論文      : {row['selected_papers']}")
    print(f"  檢索覆蓋率    : {ret if ret is not None else 'N/A（無 gold_spans）'}")
    print(f"  grounding     : {row['grounding_score']}")
    print(f"  延遲(ms)      : plan={lat.get('planning')} retr={lat.get('retrieval')} "
          f"grnd={lat.get('grounding')} total={lat.get('total')}")
    print(f"  問題標記      : {row['issues']}")


def run(label: str):
    from main import paper_engines
    from rag.query_pipeline import execute_structured_query

    questions = _load_questions()
    all_names = list(paper_engines.keys())
    os.makedirs(RESULTS_DIR, exist_ok=True)

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

        # 3) 跑完整 pipeline，擷取 status 訊息
        status_lines = []
        t0 = time.time()
        try:
            answer = execute_structured_query(
                qtext, paper_engines, "", on_status=status_lines.append,
            )
        except Exception as e:
            answer = f"[PIPELINE ERROR] {e}"
        wall_s = round(time.time() - t0, 1)

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
            "counts": metrics.parse_counts(status_lines),
            "latency": metrics.parse_stage_latencies(status_lines),
            "wall_seconds": wall_s,
            "issues": metrics.count_issues(answer),
            "answer": answer,
        }
        rows.append(row)
        _print_row(row)

    summary = metrics.summarize(rows)
    out = {"label": label, "mode": "gold" if has_gold else "self", "summary": summary, "rows": rows}
    path = os.path.join(RESULTS_DIR, f"eval_{label}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n{'#'*70}\n# 彙總（label={label}）\n{'#'*70}")
    for k, v in summary.items():
        print(f"  {k:24s}: {v}")
    print(f"\n結果已存：{path}")


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
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="比較兩個 label 的彙總")
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
    elif args.run:
        run(args.label)
    else:
        ap.print_help()
