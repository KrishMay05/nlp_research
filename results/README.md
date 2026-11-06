# Recorded results

These artifacts were generated on November 6, 2026. They distinguish completed
checks from the live experiment that remains blocked.

- `offline/report.json`: all 151 snippets; exact accuracy, argument coverage,
  paired bootstrap intervals, and exact McNemar comparisons.
- `offline/predictions.jsonl`: a prediction for every snippet and cached method.
- `purdue-models.json`: available model IDs and descriptions; account identifiers removed.
- `purdue-selection.json`: the hosted LLM's resource-efficient model recommendation.
  Pricing was unavailable; this does not prove the model is cheapest.
- `purdue-smoke.json`: successful authenticated API connection and quote extraction,
  unsuccessful probability checks, diagnostics, and 11 passing local tests.
- `audit.jsonl`: reproducibly sampled 40-item audit worksheet. Human fields are
  blank; this is NOT a completed human evaluation or gold annotation set.

## Completed offline baseline

| Method | Correct | Total | Accuracy |
| --- | ---: | ---: | ---: |
| Majority A | 83 | 151 | 54.97% |
| Cached XOR | 82 | 151 | 54.30% |
| Unfiltered grounded | 69 | 151 | 45.70% |

## Incomplete live experiment

Gemma3 12B produced a valid quote with the expected schema and speaker attribution
on one argument. Gemma3 12B and Qwen3 4B returned no token probabilities. GPT-OSS
120B exposed probabilities for a control token, requiring an answer-token adapter.
The full quote-filtered factorial and zero-shot evaluation have NOT been run.
No measured improvement from live rescoring is claimed.

## Provenance and reproduction

Upstream: https://github.com/kandicew/social-reasoning

Commit: `fed11b69765b4a398ce4a985adedfb65ec3098ab`.

Source cache SHA-256:
`e4be3c7c6a3543a77ad07eddde44bfa42f6e58fa6a4bf47df53ea9e894f1e318`.

The source dataset and dependency checkout are fetched by `scripts/setup.sh`
and remain local under ignored `data/`. Audit transcripts and the smoke quote
are excerpts from that public release, not newly generated conversations.
Credentials and API response caches are excluded. To rerun offline evaluation,
follow the root README. Live output may vary by hosted model version.
