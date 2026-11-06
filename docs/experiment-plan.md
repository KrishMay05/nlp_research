# Experiment plan fixed before live evaluation

Date: November 6, 2026. Source: the supplied proposal, *Quote-Grounded
Arguments for Conversational Power*. No accuracy target is promised. The 151
released institutional-role labels are the evaluation set; there is no training
or development split. Do not optimize prompts, thresholds, weights, tie rules,
or model selection against these labels.

## Model qualification

Select by working API features and inference resources, not test accuracy.
Gemma3 12B and Qwen3 4B did not expose usable logprobs through Purdue. GPT-OSS
120B exposed control-token probabilities requiring a specialized adapter.
Gemma4 26B-A4B successfully returned quote JSON and both Yes and No probabilities
on a synthetic question and a released rebuttal. Use `gemma4:26b-a4b` throughout
all live comparisons. Its model identifier indicates 4B activated parameters;
Purdue's response does not expose prices or prove that this model is cheapest.
Measure usage and latency where available. No paid external API is used.

## Fixed methods

1. Majority A, with all 151 snippets included.
2. Cached XOR reproduction: released rules and 20/18/1 weights, CBC binary MAP.
3. Same-model, temperature-zero zero-shot baseline on the raw transcript.
4. Complete factorial: quote filter on/off × corrected rebuttal prompt on/off ×
   XOR/grounded aggregation. All arms reuse a single annotation pool.

The quote prompt requires a verbatim span (or NONE), support judgment and
correct-speaker attribution. Deterministic matching lowercases and collapses
whitespace within the cited speaker's turn. Removed arguments never remove a
snippet. The corrected rebuttal prompt explicitly names both the rebuttal and
its paired reason. Scores normalize first-token Yes/No probabilities, not model
self-reported confidence. Missing probability mass candidates are errors, not
invented values.

Dung's unweighted attacks do not depend on corrected rebuttal probabilities.
Consequently the corrected-prompt grounded arms are identical for a fixed quote
filter, as required by the proposal. Counts of IN reasons decide the speaker,
then summed grounding probabilities over retained reasons, then A.

The baseline full-graph XOR matches 82/151. Quote filtering removes failed
variables and any paired reason/rebuttal XOR touching them; other constraints
remain. With zero surviving reasons, its power OROUT factors and power XOR
cannot be jointly satisfied. In that case use the proposal's final tie rule A,
and record this explicit empty-evidence fallback. Do not report an infeasible
solver result as a valid optimum.

## Execution and reporting

Run one snippet through every live scoring step first. On failure, preserve all
successful content-addressed requests and diagnose before spending the full
budget. Then resume over all 151 snippets with one sequential scorer, paced at
3.2 seconds/request and bounded backoff. About 11,023 calls are required, not the
proposal's estimated 3,600: 7,248 quote checks, 3,624 rebuttal scores, 151 zero-shot
predictions. At 20 requests/minute the rate-limit floor is over nine hours.

Require complete ID coverage before headline metrics. Report exact accuracy,
retained-argument coverage, paired bootstrap intervals (10,000 resamples, seed
2026), and exact two-sided McNemar tests against cached XOR, majority-A and
zero-shot. All factorial comparisons are descriptive; do not select the best arm
and imply a prespecified significant improvement. Report all arms and ties.

Export the existing seeded 40-item human audit with model annotations. Human
judgments must be supplied by a person and remain distinct from automated
checks. Do not claim the human audit is complete until those fields are filled.

The API may truncate structured JSON. For an explicitly length-truncated quote
response, retry with up to two doubled output budgets (500 → 1,000 → 2,000).
Never turn malformed or truncated output into a failed-quote judgment. A repeated
failure blocks scoring rather than removing the corresponding argument/snippet.
Successful requests are cached, so restarting the runner resumes prior work.
