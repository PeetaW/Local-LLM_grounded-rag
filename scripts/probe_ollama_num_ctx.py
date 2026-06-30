import json
from pathlib import Path
import sys
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config as cfg

_ROWS = []


def _emit(row):
    _ROWS.append(row)
    print(json.dumps(row, ensure_ascii=False))


def _post_json(url, payload):
    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=600)
        data = resp.json() if resp.content else {}
        status = resp.status_code
    except Exception as e:
        data = {"error": f"{type(e).__name__}: {e}"}
        status = "EXCEPTION"
    return status, data, round(time.perf_counter() - t0, 1)


def probe_generate(model, prompt, ctx_values):
    url = f"{cfg.OLLAMA_BASE_URL}/api/generate"
    for ctx in ctx_values:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
                "num_ctx": ctx,
                "num_predict": 8,
                "thinking": False,
            },
        }
        status, data, elapsed = _post_json(url, payload)
        _emit({
            "route": "/api/generate",
            "model": model,
            "requested_num_ctx": ctx,
            "status": status,
            "elapsed_s": elapsed,
            "prompt_eval_count": data.get("prompt_eval_count"),
            "eval_count": data.get("eval_count"),
            "done_reason": data.get("done_reason"),
            "response": (data.get("response") or "")[:80],
            "error": data.get("error"),
        })


def probe_v1_chat(model, prompt, ctx_values):
    url = f"{cfg.OLLAMA_BASE_URL}/v1/chat/completions"
    for ctx in ctx_values:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0,
            "max_tokens": 8,
            # This is the candidate path for LlamaIndex OpenAILike(additional_kwargs=...).
            "options": {"num_ctx": ctx, "thinking": False},
        }
        status, data, elapsed = _post_json(url, payload)
        usage = data.get("usage") or {}
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        _emit({
            "route": "/v1/chat/completions",
            "model": model,
            "requested_num_ctx": ctx,
            "status": status,
            "elapsed_s": elapsed,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "finish_reason": choice.get("finish_reason"),
            "response": (message.get("content") or "")[:80],
            "error": data.get("error"),
        })


def main():
    ctx_values = [512, 1024, 2048, 4096]
    if len(sys.argv) > 1:
        ctx_values = [int(x) for x in sys.argv[1].split(",") if x.strip()]
    model = sys.argv[2] if len(sys.argv) > 2 else cfg.SYNTHESIS_MODEL

    prompt = (
        "BEGIN_MARKER\n"
        + ("alpha beta gamma delta " * 1400)
        + "\nEND_MARKER\nReply with exactly: OK"
    )
    probe_generate(model, prompt, ctx_values)
    probe_v1_chat(model, prompt, ctx_values)

    out_dir = Path("eval") / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace(":", "_").replace("/", "_").replace("\\", "_")
    label = f"ollama_num_ctx_probe_{safe_model}_{int(time.time())}"
    jsonl_path = out_dir / f"{label}.jsonl"
    md_path = out_dir / f"{label}.md"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in _ROWS:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        f"# Ollama num_ctx Probe: `{model}`",
        "",
        "| route | requested_num_ctx | status | prompt tokens/eval | output tokens | finish | elapsed_s | response/error |",
        "|---|---:|---|---:|---:|---|---:|---|",
    ]
    for row in _ROWS:
        prompt_count = row.get("prompt_eval_count", row.get("prompt_tokens"))
        output_count = row.get("eval_count", row.get("completion_tokens"))
        finish = row.get("done_reason", row.get("finish_reason"))
        msg = row.get("error") or row.get("response") or ""
        msg = str(msg).replace("|", "\\|").replace("\n", " ")[:120]
        lines.append(
            f"| {row['route']} | {row['requested_num_ctx']} | {row['status']} | "
            f"{prompt_count} | {output_count} | {finish} | {row['elapsed_s']} | {msg} |"
        )
    lines.extend([
        "",
        "## How To Read",
        "",
        "- `/api/generate`: compare `prompt_eval_count` across requested ctx values.",
        "- `/v1/chat/completions`: compare `prompt_tokens`, `finish_reason`, and errors across requested ctx values.",
        "- If counts or failures change with requested ctx, the route is likely honoring the ctx option.",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {jsonl_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
