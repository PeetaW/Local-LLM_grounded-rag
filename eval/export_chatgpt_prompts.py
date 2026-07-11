import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL_SET = ROOT / "eval" / "eval_set.json"
PAPERS_DIR = ROOT / "projects" / "boron_bnct" / "papers"


SYSTEM_PROMPT = """You are answering ONLY from the uploaded PDFs.
If the uploaded PDFs do not contain enough evidence, say so explicitly.
Do not use web knowledge.
For every factual claim, cite the PDF file name and the supporting passage or section.
Separate your answer into:
1. Direct evidence
2. Cross-paper inference
3. Speculation or external knowledge
Answer in Traditional Chinese.
"""


def load_questions(ids):
    data = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    wanted = {x.strip().upper() for x in ids.split(",") if x.strip()}
    return [q for q in data["questions"] if q["id"].upper() in wanted]


def paper_path(name):
    path = PAPERS_DIR / f"{name}.pdf"
    return path if path.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="Q06,Q08,Q11,Q12")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    questions = load_questions(args.ids)
    ids_label = "_".join(q["id"].lower() for q in questions)
    out = Path(args.out) if args.out else ROOT / "eval" / f"chatgpt_prompts_{ids_label}.md"
    answers = ROOT / "eval" / f"chatgpt_answers_{ids_label}.jsonl"

    all_papers = []
    for q in questions:
        for name in q.get("gold_papers", []):
            if name not in all_papers:
                all_papers.append(name)

    lines = [
        "# ChatGPT Comparison Prompts",
        "",
        "## Upload These PDFs",
        "",
    ]
    if all_papers:
        for name in all_papers:
            path = paper_path(name)
            lines.append(f"- {name}.pdf" + (f" -> `{path}`" if path else " -> MISSING"))
    else:
        lines.append("- No gold PDFs for this subset. Use the full project corpus if testing abstention.")

    lines.extend(["", "## Shared Instruction", "", "```text", SYSTEM_PROMPT.strip(), "```", ""])

    for q in questions:
        lines.extend([
            f"## {q['id']} - {q['type']}",
            "",
            "```text",
            SYSTEM_PROMPT.strip(),
            "",
            f"Question: {q['question']}",
            "```",
            "",
        ])

    out.write_text("\n".join(lines), encoding="utf-8")

    with answers.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps({
                "id": q["id"],
                "question": q["question"],
                "answer": "",
            }, ensure_ascii=False) + "\n")

    print(out)
    print(answers)


if __name__ == "__main__":
    main()
