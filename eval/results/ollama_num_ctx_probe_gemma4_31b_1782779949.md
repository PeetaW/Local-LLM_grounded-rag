# Ollama num_ctx Probe: `gemma4:31b`

| route | requested_num_ctx | status | prompt tokens/eval | output tokens | finish | elapsed_s | response/error |
|---|---:|---|---:|---:|---|---:|---|
| /api/generate | 512 | 200 | 2047 | 1 | length | 63.0 | OK |
| /api/generate | 1024 | 200 | 2047 | 1 | length | 3.1 | OK |
| /api/generate | 2048 | 200 | 2047 | 1 | length | 3.2 | OK |
| /api/generate | 4096 | 200 | 4095 | 1 | length | 25.6 | OK |
| /v1/chat/completions | 512 | 200 | 5633 | 8 | length | 30.2 |  |
| /v1/chat/completions | 1024 | 200 | 5633 | 8 | length | 3.2 |  |
| /v1/chat/completions | 2048 | 200 | 5633 | 8 | length | 3.1 |  |
| /v1/chat/completions | 4096 | 200 | 5633 | 8 | length | 3.1 |  |

## How To Read

- `/api/generate`: compare `prompt_eval_count` across requested ctx values.
- `/v1/chat/completions`: compare `prompt_tokens`, `finish_reason`, and errors across requested ctx values.
- If counts or failures change with requested ctx, the route is likely honoring the ctx option.