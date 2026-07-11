# ChatGPT Comparison - Q06/Q08/Q11/Q12

- Date: 2026-07-08
- ChatGPT surface: chatgpt.com, uploaded 3 PDFs
- Uploaded PDFs:
  - `1-s2.0-S0378517325007926-main.pdf`
  - `CMDC-20-e202500059.pdf`
  - `bbb0683.pdf`
- Prompt mode: one combined prompt, no web knowledge/search, Traditional Chinese output
- Raw capture: `eval/chatgpt_raw_q06_q08_q11_q12.txt`

## Summary

This small test is favorable to ChatGPT because only the gold/source PDFs were uploaded, not the full noisy corpus. Even so, it is useful: ChatGPT answered the two hard in-corpus questions well and handled both abstention-style questions correctly.

| ID | Type | Our baseline_v5 | ChatGPT manual score | Result |
|---|---|---:|---:|---|
| Q06 | multi_chunk | correctness 1.0, grounding 0.722 | correctness ~1.0 | Tie on correctness; ChatGPT answer was compact and strong |
| Q08 | cross_paper | correctness 0.5, grounding 0.727 | correctness ~1.0 | ChatGPT wins this run |
| Q11 | out_of_scope | correctness 1.0 | correctness ~1.0 | Tie; both abstain correctly |
| Q12 | false_premise | correctness 1.0 | correctness ~1.0 | Tie; both reject oral premise |

## Notes

- Q06: ChatGPT covered PVA-BPA boronate ester complexation, LAT1-mediated endocytosis, higher tumor boron accumulation, reduced efflux/retention, survival improvement, and dose data.
- Q08: ChatGPT gave a stronger synthesis-route comparison than our baseline_v5 answer, especially on isotope economics, B2pin2 waste, Pd/NaIO4 concerns, and the Nakao/Kirihata hybrid route.
- Q11: ChatGPT correctly refused to fabricate phase III glioblastoma overall-survival data and distinguished mouse thoracic tumor survival from clinical results.
- Q12: ChatGPT correctly identified the false premise: these PDFs support infusion/intravenous administration, not oral BPA administration, and report no oral bioavailability values.

## Interpretation

For answer quality on a small, clean, gold-PDF set, ChatGPT is already very competitive and likely stronger on cross-paper synthesis. The project should not claim to beat ChatGPT at general academic answering.

The project's defensible value remains the audit layer: fixed local corpus, repeatable evals, retrieval coverage, NLI grounding, answerability gate, and raw evidence traceability. A fair next test should run each question independently with the full corpus or gold+decoy PDFs, then score ChatGPT with the same grounding/eval harness.
