"""
Rebuild only the Q07 source paper index after the PyMuPDF ligature fix.

Run from the repo root inside llm_env:
    python scripts\rebuild_q07_ligfix.py
    python eval\run_eval.py --run --label q07_ligfix --ids Q07
"""

from pathlib import Path
import json
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import config as cfg
from rag.llm_client import init_llm_and_embedding
from rag.pdf_loader import load_pdf_with_pymupdf
from rag.vl_processor import needs_vl_analysis, run_vl_analysis
from rag.metadata_manager import ensure_metadata
from rag.chunk_summarizer import add_summaries_to_nodes
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter


PAPER_FILE = "41467_2024_Article_45464.pdf"
PAPER_NAME = PAPER_FILE[:-4]


def write_index_config() -> None:
    path = ROOT / cfg.INDEX_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(cfg.INDEX_BUILD_CONFIG, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"updated {path.relative_to(ROOT)}")


def rebuild_paper() -> None:
    pdf_path = ROOT / cfg.PAPERS_DIR / PAPER_FILE
    index_dir = ROOT / cfg.INDEX_BASE_DIR / PAPER_NAME
    cache_path = index_dir / "chunk_summaries.json"
    cached_summaries = cache_path.read_text(encoding="utf-8") if cache_path.exists() else None

    if needs_vl_analysis(PAPER_NAME):
        print(f"VL missing; running VL analysis for {PAPER_NAME}")
        run_vl_analysis(str(pdf_path))

    ensure_metadata(str(pdf_path))

    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    if cached_summaries:
        cache_path.write_text(cached_summaries, encoding="utf-8")
        print("preserved chunk_summaries.json cache")

    docs = load_pdf_with_pymupdf(str(pdf_path))
    splitter = SentenceSplitter(
        chunk_size=cfg.CHUNK_SIZE,
        chunk_overlap=cfg.CHUNK_OVERLAP,
    )
    nodes = splitter.get_nodes_from_documents(docs)
    nodes = add_summaries_to_nodes(nodes, PAPER_NAME)
    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=str(index_dir))
    print(f"rebuilt {index_dir.relative_to(ROOT)}")


def check_docstore() -> int:
    docstore = ROOT / cfg.INDEX_BASE_DIR / PAPER_NAME / "docstore.json"
    text = docstore.read_text(encoding="utf-8")
    fluoride = text.lower().count("fluoride")
    lig_fl = text.count("ﬂ")
    print(f"docstore check: fluoride={fluoride}, ligature_fl={lig_fl}")
    if fluoride <= 0 or lig_fl != 0:
        print("WARNING: ligature fix did not look fully applied.")
        return 1
    print("OK: ligature fix is visible in the rebuilt docstore.")
    print("next: python eval\\run_eval.py --run --label q07_ligfix --ids Q07")
    return 0


def main() -> int:
    write_index_config()
    init_llm_and_embedding()
    rebuild_paper()
    return check_docstore()


if __name__ == "__main__":
    raise SystemExit(main())
