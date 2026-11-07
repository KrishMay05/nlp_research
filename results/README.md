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

## Subsequent full-experiment attempt

`model-qualification.json` records that Gemma4 26B-A4B returned usable Yes/No
probabilities and quote JSON, so it was selected on API functionality rather
than test accuracy. `docs/experiment-plan.md` fixes all comparisons before the
live evaluation. The newer implementation has 18 passing local tests.

`experiment/` contains resumable live exports, which are **partial unless
`status.json` says `complete` and all 151 IDs appear in the final evaluation**.
Do not treat a prefix of annotations as a full experiment. The raw API cache is
kept locally and excluded from Git. The human audit remains pending even after
a successful automated scoring/evaluation run.

The prompt-off factorial control is now rescored by the same current model,
separate from the published cached XOR. This avoids a model/prompt confound;
see the plan amendment. The controlled budget is 14,647 calls before retries.
