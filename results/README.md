# Recorded results

The [quote-filter experiment report](experiment/quote_evaluation/README.md)
contains the completed results on all 151 snippets and 7,248 argument texts.
Both filtered methods scored 64/151 (42.38%), compared with cached XOR 82/151
(54.30%) and majority A 83/151 (54.97%). Quote filtering did not help in this run.

| Artifact | Contents |
| --- | --- |
| [Final metrics](experiment/quote_evaluation/report.json) | Six methods, quote diagnostics, coverage, paired bootstrap intervals, exact McNemar tests |
| [Final predictions](experiment/quote_evaluation/predictions.jsonl) | Every method on every snippet |
| [Final validation](experiment/quote_evaluation/validation.json) | Exact coverage, independent recount, cache provenance, pending human audit |
| [Quote annotations](experiment/quote_annotations.jsonl) | All 7,248 model quote judgments |
| [Zero-shot predictions](experiment/zero_shot.jsonl) | All 151 predictions and probabilities |
| [Run status](experiment/status.json) | Completed reduced scope; original factorial remains partial |
| [Human audit](experiment/audit.jsonl) | Seeded 40-item worksheet with blank human judgments |
| [Model qualification](model-qualification.json) | Gemma4 JSON and token-logprob qualification |
| [Cache reuse check](experiment/cache_reuse_check.json) | Offline inventory at the user-directed scope amendment |
| [Partial factorial annotations](experiment/annotations.jsonl) | Preserved 3,332-item export; not complete-dataset factorial results |
| [Attempt history](experiment/attempt_history.json) | Earlier run failures, interruptions, and scope change |
| [Earlier baseline evaluation](experiment/baseline_evaluation/README.md) | Completed cached and zero-shot baseline report |
| [Offline metrics](offline/report.json) | Original cached-score baselines |

`purdue-models.json`, `purdue-selection.json`, and `purdue-smoke.json` are earlier
qualification artifacts, including unsuccessful Gemma3/Qwen3/GPT-OSS probability
checks. They are superseded by successful Gemma4 qualification. The original
`audit.jsonl` is an earlier unfilled worksheet. Historical reports retain their
original test counts; the current suite has 22 passing tests.

The original corrected-prompt factorial was deferred at the user's request to
reduce calls. New rebuttal probabilities are ignored in the completed fixed-score
comparison. All successful cache entries remain local for potential resumption;
cache identity includes endpoint/model/payload. No Nous credit was spent.

Source: https://github.com/kandicew/social-reasoning at
`fed11b69765b4a398ce4a985adedfb65ec3098ab`. Cache SHA-256:
`e4be3c7c6a3543a77ad07eddde44bfa42f6e58fa6a4bf47df53ea9e894f1e318`.
Setup fetches the source dependency into ignored `data/`. Credentials, raw API
caches, and the user-uploaded proposal PDF remain excluded from GitHub.
