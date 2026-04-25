# rag/query_retrieval.py
# Pipeline Stage 2: subquery task building and parallel retrieval execution.
# Phase A (parallel): embed guard + vector search (bge-m3 only).
# Phase B (serial):   LLM answer generation (gemma4 loaded once).

import concurrent.futures

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
    return any(pat in text_lower for pat in _NO_RESULT_PATTERNS)


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


def run_subqueries_parallel(valid_tasks: list, prefilled: dict) -> list:
    """
    Two-phase execution to minimise Ollama model-switching overhead:
      Phase A (parallel): embed guard + vector retrieval (bge-m3 stays loaded)
      Phase B (serial):   LLM generation per task (gemma4 loaded once)

    Returns list of (label, result_str) in original sub-question order.
    """
    results = dict(prefilled)

    # ── Phase A: parallel embed + retrieval ─────────────────────────
    def _retrieve_one(task):
        task_idx, label, engine, sub_q = task
        try:
            query_text = prepare_query_text(sub_q)
            nodes = _retrieve_nodes(engine, query_text)
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

    # ── Phase B: serial LLM generation (gemma4 loaded once) ─────────
    for task_idx in sorted(retrieved.keys()):
        label, engine, query_text, nodes = retrieved[task_idx]
        try:
            result = _generate_from_nodes(engine, nodes, query_text)
            results[task_idx] = (label, result)
        except Exception as e:
            results[task_idx] = (label, f"{label}生成失敗：{e}")

    return [(label, result) for _, (label, result) in sorted(results.items())]
