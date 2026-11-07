# Completed quote-filter experiment

Completed November 7, 2026 (America/Indianapolis). All **151 snippets**, **7,248
argument texts** (3,624 reasons and 3,624 rebuttals), and **151 zero-shot
predictions** are included. No snippet was removed. This completes the user's
reduced quote-filter scope, not the original corrected-prompt factorial.

| Method | Correct / 151 | Accuracy | Argument retention |
| --- | ---: | ---: | ---: |
| Majority A | 83 | 54.97% | — |
| Cached XOR | 82 | 54.30% | 100% |
| Unfiltered grounded | 69 | 45.70% | 100% |
| Gemma4 zero-shot | 66 | 43.71% | — |
| Quote-filtered XOR, cached scores fixed | 64 | 42.38% | 33.84% |
| Quote-filtered grounded, cached scores fixed | 64 | 42.38% | 33.84% |

Quote filtering did **not** improve classification accuracy. Filtered XOR lost
18 correct predictions relative to cached XOR; filtered grounded lost five
relative to unfiltered grounded. The two filtered methods have equal accuracy
but disagree on 12 snippet predictions.

| Paired contrast | Accuracy change (percentage points) | Bootstrap 95% interval | Exact McNemar p |
| --- | ---: | ---: | ---: |
| Filtered XOR − cached XOR | −11.92 | [−20.53, −3.31] | 0.01135 |
| Filtered grounded − unfiltered grounded | −3.31 | [−12.58, 5.96] | 0.58313 |
| Filtered XOR − zero-shot | −1.32 | [−10.60, 7.95] | 0.89192 |
| Filtered grounded − zero-shot | −1.32 | [−11.26, 8.61] | 0.89385 |

Intervals use 10,000 paired bootstrap resamples with seed 2026; McNemar is exact,
two-sided, and unadjusted for multiple comparisons. Different procedures can
disagree at a threshold. These small-test-set contrasts are descriptive; there
is no demonstrated improvement and no claim of confirmatory significance after
multiple comparisons.

## Evidence and graph behavior

| Diagnostic | All arguments | Reasons | Rebuttals |
| --- | ---: | ---: | ---: |
| Total | 7,248 | 3,624 | 3,624 |
| Normalized span matches anywhere | 3,151 | 1,692 | 1,459 |
| Span matches within cited speaker turn | 2,964 | 1,589 | 1,375 |
| Model support judgment true | 2,958 | 1,666 | 1,292 |
| Model attribution judgment true | 4,707 | 2,613 | 2,094 |
| Passes all retention requirements | 2,453 | 1,393 | 1,060 |

Overall retention is 33.84%; reason retention is 38.44% and rebuttal retention
29.25%. Only 85/3,624 source reasons (2.35%) are labelled IN after filtering.
Filtered grounded decisions use IN-reason counts in 30 snippets and cached
reason-grounding sums in 121. Unfiltered grounded uses the sum tie-break in all
151 snippets because its unattacked rebuttals make every reason OUT. No filtered
XOR example needed the declared no-surviving-reasons fallback. These diagnostics
explain decision paths, not human evidence correctness or a causal explanation
of the accuracy loss. The seeded 40-item human audit remains unfilled.

## Model, cache, and scope

Purdue GenAI Studio served `gemma4:26b-a4b` at temperature zero. Model selection
used API functionality and resource requirements, not label accuracy. Monetary
prices were unavailable; the model is not proven to be globally cheapest.
All 7,248 quote exports exactly match successful cache entries for the requested
Gemma4 model, Purdue endpoint, and unchanged payloads. Served-model metadata
records Gemma4 for 7,227 responses; 21 early responses predate that metadata.
No successful response was requeried or selected according to test accuracy.

At the user's runtime/cost amendment, 3,333 quote checks and all zero-shot calls
were cached. The remaining 3,915 quote calls were completed. Skipping the 1,960
remaining requests for each rebuttal prompt saved 3,920 prospective requests
before retries. Both scoring prompts' 1,664 completed calls and the 3,332-item
partial factorial export remain available, but are **not a complete prompt
ablation**. The quote-plus-zero-shot design requires 7,399 unique requests;
older rescoring/qualification cache entries are additional sunk work, not a
billing ledger. No Nous endpoint was used and no Nous credit was spent.

The filtered/unfiltered comparisons hold every released cached weight fixed.
The completed study therefore isolates this implementation's quote filter at
fixed weights; it cannot answer whether corrected rebuttal scoring would help.
The 151 released institutional-role labels, small sample, conservative span
rule, model semantic judgments, and pending human audit limit generalization.
No prompts, thresholds, weights, or models were changed after seeing these
argument-arm accuracies. See the [methods and amendments](../../../docs/experiment-plan.md).

## Files and reproduction

- [Metrics and all paired comparisons](report.json).
- [Every snippet prediction](predictions.jsonl).
- [Artifact validation](validation.json).
- [Complete quote annotations](../quote_annotations.jsonl).
- [Complete zero-shot predictions](../zero_shot.jsonl).
- [Human audit worksheet](../audit.jsonl).

After `bash scripts/setup.sh` and activating the environment:

```bash
quote-power evaluate --quote-only \
  --annotations results/experiment/quote_annotations.jsonl \
  --zero-shot results/experiment/zero_shot.jsonl \
  --output results/reproduced-quotes
```

This evaluation makes no API calls. The implementation's 22 tests passed.
Validation independently recounted accuracy, checked exact unique ID coverage,
validated finite probabilities and threshold consistency, compared every quote
with its exact cache entry without network calls, verified preservation of the
partial factorial quotes, and confirmed blank human judgment fields.
