# rag/query_retrieval.py
# Pipeline Stage 2: subquery task building and parallel retrieval execution.
# Phase A (parallel): embed guard + vector search (bge-m3 only).
# Phase B (serial):   LLM answer generation (gemma4 loaded once).

import concurrent.futures
import re
import time

import config as cfg
from rag.comparison_json_validator import exact_isotope_terms
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


def _source_note(label: str) -> str:
    name = label.strip("【】")
    try:
        from rag.metadata_manager import load_metadata
        meta = load_metadata().get(name, {})
    except Exception:
        meta = {}
    title = str(meta.get("title", "") or "")
    desc = str(meta.get("short_desc", "") or "")
    role = "review/comparison source" if "review" in title.lower() or "綜述" in desc else "paper source"
    return f"Source metadata (not paper evidence): role_hint={role}; title={title or name}; desc={desc}"


def _is_comparison_query(query_text: str) -> bool:
    text = (query_text or "").lower()
    return any(term in text for term in (
        "compare", "comparison", "different", "routes", "approaches",
        "scalability", "cost-effectiveness", "isotopic", "safety",
        "比較", "差異", "路線", "可擴展", "成本", "同位素", "安全",
    ))


def _strip_context_summary(text: str) -> str:
    text = " ".join(text.split())
    if text.startswith("[摘要："):
        _, marker, text = text.partition("]")
        if marker:
            return text.lstrip()
    return text


_QUERY_STOPWORDS = {
    "according", "across", "also", "and", "are", "does", "for", "from", "give",
    "how", "into", "main", "paper", "papers", "reported", "study", "that", "the",
    "their", "these", "they", "this", "under", "used", "using", "what", "which",
    "with",
}
_VALUE_QUERY_MARKERS = ("value", "values", "data", "parameter", "condition", "dose", "yield")
_VALUE_EVIDENCE_MARKERS = ("ic50", "km", "vmax", "mol", "mm", "nm", "μm", "°c", "%", "ph")
_MECHANISM_QUERY_MARKERS = ("mechanism", "bind", "binding", "inhibit", "role", "how")
_MECHANISM_EVIDENCE_MARKERS = (
    "bind", "bond", "interact", "inhibit", "occup", "convert", "exchange", "cross-link",
    "uptake", "efflux", "collapse", "reform", "leads to", "results in",
)
_COMPARISON_DIMENSION_TERMS = (
    "high cost", "cost-effectiveness", "cost effectiveness", "isotopically enriched",
    "10b", "scalability", "safety", "toxicity", "contamination", "risk", "cost",
    "multiple routes",
)


def _query_terms(query_text: str) -> set[str]:
    return {
        term for term in re.findall(r"[a-z0-9][a-z0-9+.-]*", (query_text or "").lower())
        if len(term) > 2 and term not in _QUERY_STOPWORDS
    }


def _query_window_score(text: str, query_text: str) -> int:
    lower = (text or "").lower()
    query_lower = (query_text or "").lower()
    score = 3 * sum(term in lower for term in _query_terms(query_text))
    if any(marker in query_lower for marker in _VALUE_QUERY_MARKERS):
        score += 4 * bool(re.search(r"\d", lower))
        score += 2 * sum(marker in lower for marker in _VALUE_EVIDENCE_MARKERS)
    if any(marker in query_lower for marker in _MECHANISM_QUERY_MARKERS):
        score += 2 * sum(marker in lower for marker in _MECHANISM_EVIDENCE_MARKERS)
    score -= min(8, lower.count("doi.org") + lower.count(" et al.") + len(re.findall(r"\(20\d{2}\)", lower)))
    return score


def _sentence_window(text: str, position: int, limit: int) -> str:
    start = max(0, position - limit // 3)
    sentence_start = max(text.rfind(marker, max(0, start - 240), start) for marker in (". ", "? ", "! "))
    if sentence_start >= 0:
        start = sentence_start + 2
    end = min(len(text), start + limit)
    sentence_ends = [
        pos for marker in (". ", "? ", "! ")
        if (pos := text.find(marker, end, min(len(text), end + 240))) >= 0
    ]
    if sentence_ends:
        end = min(sentence_ends) + 1
    return text[start:end].strip()


def _query_aware_window(text: str, query_text: str, limit: int) -> str:
    text = _strip_context_summary(text)
    if len(text) <= limit:
        return text
    lower = text.lower()
    query_lower = (query_text or "").lower()
    anchors = list(_query_terms(query_text))
    if any(marker in query_lower for marker in _VALUE_QUERY_MARKERS):
        anchors.extend(_VALUE_EVIDENCE_MARKERS)
    if any(marker in query_lower for marker in _MECHANISM_QUERY_MARKERS):
        anchors.extend(_MECHANISM_EVIDENCE_MARKERS)
    positions = {
        match.start()
        for anchor in anchors
        for match in re.finditer(re.escape(anchor), lower)
    }
    if not positions:
        return text[:limit]
    windows = [_sentence_window(text, position, limit) for position in sorted(positions)]
    return max(windows, key=lambda window: (_query_window_score(window, query_text), len(window)))


def _clip_evidence_snippet(text: str, query_text: str, limit: int = 900) -> str:
    is_comparison = _is_comparison_query(query_text)
    text = _strip_context_summary(text) if is_comparison else " ".join(text.split())
    if len(text) <= limit:
        return text
    query_aware = getattr(cfg, "STAGE2_QUERY_AWARE_EVIDENCE_ENABLED", False)
    if query_aware and not (
        is_comparison and any(term in (query_text or "").lower() for term in _COMPARISON_DIMENSION_TERMS)
    ):
        return _query_aware_window(text, query_text, limit)
    if not is_comparison:
        return text[:limit]

    lower = text.lower()
    positions = [lower.find(term) for term in _COMPARISON_DIMENSION_TERMS if lower.find(term) >= 0]
    if not positions:
        return text[:limit]
    pos = min(positions)
    start = max(0, pos - limit // 3)
    sentence_start = text.rfind(". ", max(0, start - 200), start)
    if sentence_start >= 0:
        start = sentence_start + 2
    end = min(len(text), start + limit)
    # ponytail: finish only a nearby sentence; raise the 200-char tail if eval still clips key terms.
    anchor = max(position for position in positions if position < end)
    sentence_end = text.rfind(". ", anchor, end)
    if sentence_end < 0:
        sentence_end = text.find(". ", end, min(len(text), end + 200))
    if sentence_end >= 0:
        end = sentence_end + 1
    return text[start:end].rstrip()


def _nodes_to_evidence_block(nodes, query_text: str, label: str = "") -> str:
    if nodes is None:
        return "No standalone retriever output was available."
    if not nodes:
        return "No relevant retrieved evidence snippets."

    lines = [
        f"Sub-question: {query_text}",
        _source_note(label),
        "Use only snippets below as citable paper evidence; metadata/guidance lines are not paper evidence.",
        "Retrieved evidence snippets:",
    ]
    # ponytail: fixed tiny evidence pack; tune count/length only if eval shows quality loss.
    snippet_count = (
        getattr(cfg, "COMPARISON_EVIDENCE_SNIPPETS_PER_TASK", 4)
        if _is_comparison_query(query_text)
        else getattr(cfg, "STAGE2_EVIDENCE_SNIPPETS_PER_TASK", 2)
    )
    query_aware = getattr(cfg, "STAGE2_QUERY_AWARE_EVIDENCE_ENABLED", False)
    if query_aware and not _is_comparison_query(query_text):
        snippet_count = max(4, snippet_count)

    selected_nodes = nodes[:snippet_count]
    if (
        _is_comparison_query(query_text)
        and getattr(cfg, "COMPARISON_JSON_DIRECT_RENDER_ENABLED", False)
        and (
            not query_aware
            or any(term in (query_text or "").lower() for term in _COMPARISON_DIMENSION_TERMS)
        )
    ):
        def _node(nws):
            return getattr(nws, "node", nws)

        text_nodes = [
            nws for nws in nodes
            if getattr(_node(nws), "metadata", {}).get("source_type") != "image_description"
        ]
        candidates = text_nodes if len(text_nodes) >= snippet_count else list(nodes)
        selected_nodes = candidates[:min(2, snippet_count)]
        dimension_terms = (
            ("isotopically enriched", "isotopic enrichment", "10b"),
            ("scalability", "scale-up", "gram-scale", "few steps", "reaction steps", "workup"),
            ("cost-effectiveness", "cost effectiveness", "high cost", "major cost", "cost"),
            ("safety", "safe", "toxicity", "risk"),
        )
        query_lower = query_text.lower()
        prefer_isotope_cost_witness = (
            any(term in query_lower for term in ("isotop", "enrich", "10b", "同位素", "富集"))
            and any(term in query_lower for term in ("cost", "成本"))
        )

        def _coverage(nws):
            node = _node(nws)
            text = node.get_content() if hasattr(node, "get_content") else str(node)
            lower = _strip_context_summary(text).lower()
            score = sum(any(term in lower for term in group) for group in dimension_terms)
            if (
                prefer_isotope_cost_witness
                and exact_isotope_terms(lower, require_context=False)
                and any(term in lower for term in dimension_terms[2])
            ):
                score += len(dimension_terms)
            return score

        remaining = candidates[len(selected_nodes):]
        remaining = sorted(enumerate(remaining), key=lambda pair: (-_coverage(pair[1]), pair[0]))
        selected_nodes += [nws for _, nws in remaining[:snippet_count - len(selected_nodes)]]
    elif query_aware:
        def _content(nws):
            node = getattr(nws, "node", nws)
            return node.get_content() if hasattr(node, "get_content") else str(node)

        selected_nodes = [
            nws for _, nws in sorted(
                enumerate(nodes),
                key=lambda pair: (
                    -_query_window_score(
                        _query_aware_window(_content(pair[1]), query_text, 1400),
                        query_text,
                    ),
                    pair[0],
                ),
            )[:snippet_count]
        ]

    for i, nws in enumerate(selected_nodes, 1):
        node = getattr(nws, "node", nws)
        text = node.get_content() if hasattr(node, "get_content") else str(node)
        limit = 1400 if query_aware and not (
            _is_comparison_query(query_text)
            and any(term in (query_text or "").lower() for term in _COMPARISON_DIMENSION_TERMS)
        ) else 900
        text = _clip_evidence_snippet(text, query_text, limit=limit)
        lines.append(f"[Snippet {i}] {text}")
    return "\n".join(lines)


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

    specific_targets = set()
    for sq in sub_questions:
        paper = sq.get("paper", "ALL")
        if paper == "ALL":
            continue
        matched = paper if paper in paper_engines_to_use else next(
            (name for name in paper_engines_to_use if paper in name), None
        )
        if matched:
            specific_targets.add(matched)

    for sq in sub_questions:
        paper = sq.get("paper", "ALL")
        sub_q = sq.get("sub_q", "")

        if paper == "ALL":
            for name, engine in paper_engines_to_use.items():
                if name in specific_targets:
                    continue
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
            if cfg.STAGE2_LLM_SUBANSWERS_ENABLED:
                result = _generate_from_nodes(engine, nodes, query_text)
                tag = "（空/無內容）" if is_empty_result(result) else ""
                print(f"  ✍️  [Phase B] {label} 生成 {len(result)} 字元 {tag}")
            else:
                result = _nodes_to_evidence_block(nodes, query_text, label)
                print(f"  🧾 [Phase B] {label} evidence block {len(result)} 字元")
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
