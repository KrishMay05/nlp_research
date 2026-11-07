# Completed baseline evaluation

All 151 released snippets were evaluated. The quote-filtered factorial is still running; these are complete baseline results, not final treatment results.

| Method | Correct | Accuracy |
| --- | ---: | ---: |
| Majority A | 83/151 | 54.97% |
| Cached XOR reproduction | 82/151 | 54.30% |
| Unfiltered cached grounded | 69/151 | 45.70% |
| Gemma4 zero-shot | 66/151 | 43.71% |

## Same-model zero-shot baseline

Model: `gemma4:26b-a4b`, Purdue GenAI Studio, temperature 0.

The binary question is: “Does speaker A hold more conversational power than B? Answer only Yes or No.” Scores use first-token P(Yes)/(P(Yes)+P(No)); values at least 0.5 predict A. All 151 IDs and finite probability values were checked.

Against majority A, the accuracy difference is -11.26 percentage points; paired bootstrap 95% interval [-24.50, 1.99] points; exact two-sided McNemar p=0.1145. The point estimate does not improve on majority A.

The model and argument-method settings remain fixed. No prompt or model is selected from these test-label results. The main factorial will use the same full dataset and the same cached model judgments.

The report includes every baseline comparison, including the distinct paired-bootstrap and exact-test results. They should be read together; a borderline bootstrap interval and exact p-value can differ because these are different procedures on discrete paired outcomes.

See `report.json` for all intervals/tests and `predictions.jsonl` for every prediction. No human audit has been completed.
