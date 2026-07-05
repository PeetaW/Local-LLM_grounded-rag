# rag/query_retrieval.py
# Pipeline Stage 2: subquery task building and parallel retrieval execution.
# Phase A (parallel): embed guard + vector search (bge-m3 only).
# Phase B (serial):   LLM answer generation (gemma4 loaded once).

import concurrent.futures
import time

import config as cfg
from rag.query_embedding_guard import prepare_query_text


_NO_RESULT_PATTERNS = [
    "此論文未涉及",
    "empty response",
    "no information",
    "找不到",
    "未找到",
    "沒有相關",
    "無相關",
    "i don't have",
    "i cannot find",
    "not mentioned",
    "the context does not",
    "no relevant",
    "does not contain",
    "not found",
    "no context",
    "沒有找到",
    "查詢失敗",
]


def is_empty_result(text: str) -> bool:
    """Returns True if the RAG result contains no substantive content."""
    text_lower = text.lower().strip()
    if len(text_lower) < 30:
        return True
    # 只讓 no-result 片語對「短」答案生效：長答案有實質內容時，
    # 不該只因內含一句「這篇沒提到 X，但討論了 Y」（跨論文題常見）就被誤判為空。
    if len(text_lower) < 200 and any(pat in text_lower for pat in _NO_RESULT_PATTERNS):
        return True
    return False


def extract_paper_name(ans: str, fallback: str) -> str:
    """Extract the first 【paper name】 label from a sub-answer string."""
    import re
    m = re.search(r'【(.+?)】', ans)
    return m.group(1) if m else fallback


def _retrieve_nodes(engine, query_text: str):
    """
    Phase A-2: vector retrieval only — no LLM call.
    Returns retrieved nodes, or None if the engine exposes no standalone retriever.
    Thread-safe (read-only index).
    """
    retriever = engine.retriever if hasattr(engine, 'retriever') else None
    if retriever is not None:
        return retriever.retrieve(query_text)
    return None


def _rerank_nodes(engine, nodes, query_text: str):
    """
    Phase A-3: 若 RERANK_ENABLED，用 engine 既有的 cross-encoder reranker
    把檢索候選精選到 RERANKER_TOP_N（reranker 已內建 top_n）。
    重用 engine 既有的 reranker，不另外載入模型。失敗則退回原始 nodes。
    """
    if not cfg.RERANK_ENABLED or not nodes:
        return nodes
    postprocessors = getattr(engine, "_node_postprocessors", None) or []
    if not postprocessors:
        return nodes
    from llama_index.core import QueryBundle
    try:
        return postprocessors[0].postprocess_nodes(
            nodes, query_bundle=QueryBundle(query_text)
        )
    except Exception as e:
        print(f"  ⚠️  [rerank] 失敗，用原始檢索結果：{e}")
        return nodes


def _generate_from_nodes(engine, nodes, query_text: str) -> str:
    """
    Phase B: LLM answer generation (gemma4).
    Nodes are already retrieved; called serially to avoid model-switching overhead.
    """
    if nodes is None:
        return str(engine.query(query_text))
    from llama_index.core.response_synthesizers import get_response_synthesizer
    from llama_index.core import QueryBundle
    synthesizer = get_response_synthesizer()
    response = synthesizer.synthesize(query=QueryBundle(query_text), nodes=nodes)
    return str(response)


def build_subquery_tasks(sub_questions: list, paper_engines_to_use: dict, paper_engines: dict):
    """
    Flatten sub_questions × papers into a task list.
    Returns:
      valid_tasks — list of (idx, label, engine, sub_q) for the thread pool
      prefilled   — dict of {idx: (label, result_str)} for engines not found
    """
    valid_tasks = []
    prefilled = {}
    idx = 0

    for sq in sub_questions:
        paper = sq.get("paper", "ALL")
        sub_q = sq.get("sub_q", "")

        if paper == "ALL":
            for name, engine in paper_engines_to_use.items():
                valid_tasks.append((idx, f"【{name}】", engine, sub_q))
                idx += 1
        else:
            engine = paper_engines_to_use.get(paper)
            if engine is None:
                matched = next((k for k in paper_engines_to_use if paper in k), None)
                if matched is None:
                    matched = next((k for k in paper_engines if paper in k), None)
                    engine = paper_engines.get(matched) if matched else None
                else:
                    engine = paper_engines_to_use.get(matched)

            if engine:
                valid_tasks.append((idx, f"【{paper}】", engine, sub_q))
            else:
                prefilled[idx] = (f"【{paper}】", f"【{paper}】找不到對應論文")
            idx += 1

    return valid_tasks, prefilled


def run_subqueries_parallel(valid_tasks: list, prefilled: dict, on_status=None) -> list:
    """
    Two-phase execution to minimise Ollama model-switching overhead:
      Phase A (parallel): embed guard + vector retrieval (bge-m3 stays loaded)
      Phase B (serial):   LLM generation per task (gemma4 loaded once)

    Returns list of (label, result_str) in original sub-question order.
    """
    def _status(msg):
        if on_status:
            on_status(msg)
        else:
            print(msg)

    results = dict(prefilled)

    # ── Phase A: parallel embed + retrieval ─────────────────────────
    phase_a_start = time.perf_counter()

    def _retrieve_one(task):
        task_idx, label, engine, sub_q = task
        try:
            query_text = prepare_query_text(sub_q)
            raw_nodes = _retrieve_nodes(engine, query_text)
            nodes = _rerank_nodes(engine, raw_nodes, query_text)
            if cfg.RERANK_ENABLED and raw_nodes is not None:
                print(f"  🔎 [Phase A] {label} 檢索 {len(raw_nodes)} → rerank {len(nodes) if nodes else 0} 個 node")
            else:
                n = "n/a" if nodes is None else len(nodes)
                print(f"  🔎 [Phase A] {label} 檢索到 {n} 個 node")
            return task_idx, label, engine, query_text, nodes
        except Exception as e:
            print(f"  ⚠️  [Phase A] {label} 檢索失敗：{e}")
            return task_idx, label, engine, sub_q, None

    retrieved = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.SUBQUERY_MAX_WORKERS) as ex:
        futures = [ex.submit(_retrieve_one, t) for t in valid_tasks]
        for f in concurrent.futures.as_completed(futures):
            task_idx, label, engine, query_text, nodes = f.result()
            retrieved[task_idx] = (label, engine, query_text, nodes)
    phase_a_ms = int((time.perf_counter() - phase_a_start) * 1000)

    # ── Phase B: serial LLM generation (gemma4 loaded once) ─────────
    phase_b_start = time.perf_counter()
    for task_idx in sorted(retrieved.keys()):
        label, engine, query_text, nodes = retrieved[task_idx]
        try:
            result = _generate_from_nodes(engine, nodes, query_text)
            # P1 診斷：區分「檢索回空」與「有 node 但生成回空」
            tag = "（空/無內容）" if is_empty_result(result) else ""
            print(f"  ✍️  [Phase B] {label} 生成 {len(result)} 字元 {tag}")
            results[task_idx] = (label, result)
        except Exception as e:
            print(f"  ❌ [Phase B] {label} 生成例外：{e}")
            results[task_idx] = (label, f"{label}生成失敗：{e}")
    phase_b_ms = int((time.perf_counter() - phase_b_start) * 1000)
    _status(
        f"[retrieval-timing] tasks={len(valid_tasks)} prefilled={len(prefilled)} "
        f"phase_a_ms={phase_a_ms} phase_b_ms={phase_b_ms}"
    )

    return [(label, result) for _, (label, result) in sorted(results.items())]
