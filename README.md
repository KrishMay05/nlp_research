# Quote-Grounded Arguments for Conversational Power

Research implementation of the supplied Patel, Kim, and Alladin proposal.
The **quote-filter experiment is complete** on all 151 released snippets and
7,248 argument texts. Quote filtering did not improve accuracy in this run.
The corrected-rebuttal prompt ablation remains partial after the user's
cost/runtime amendment; it is not included in the completed comparison.

| Method | Correct / 151 | Accuracy |
| --- | ---: | ---: |
| Majority A | 83 | 54.97% |
| Cached XOR reproduction | 82 | 54.30% |
| Grounded, unfiltered cached arguments | 69 | 45.70% |
| Gemma4 zero-shot | 66 | 43.71% |
| XOR with quote filtering, fixed cached scores | 64 | 42.38% |
| Grounded with quote filtering, fixed cached scores | 64 | 42.38% |

Filtered XOR is 11.92 percentage points below cached XOR (paired bootstrap
95% interval −20.53 to −3.31 points; exact McNemar p=0.0114, unadjusted).
Filtering retained 2,453/7,248 arguments (33.84%). These are descriptive results
on a small released test set; no prompts or weights were tuned on its labels.

- [Full report](results/experiment/quote_evaluation/README.md), [metrics](results/experiment/quote_evaluation/report.json), and [every prediction](results/experiment/quote_evaluation/predictions.jsonl).
- [All quote annotations](results/experiment/quote_annotations.jsonl), [validation](results/experiment/quote_evaluation/validation.json), and [run status](results/experiment/status.json).
- [Methods and amendments](docs/experiment-plan.md), [source code](src/quote_power), [tests](tests), and [results index](results/README.md).
- [40-item human audit worksheet](results/experiment/audit.jsonl): human judgments remain pending.

## Reproduce locally

Python 3.11+ on Linux; tested with Python 3.12 and pinned PuLP/CBC. No GPU,
training split, database, AD3 installation, or web server is required.

```bash
bash scripts/setup.sh
source .venv/bin/activate
quote-power evaluate
quote-power evaluate --quote-only \
  --annotations results/experiment/quote_annotations.jsonl \
  --zero-shot results/experiment/zero_shot.jsonl \
  --output results/reproduced-quotes
python -m pytest -q --disable-warnings
```

Setup installs `requirements.lock`, fetches and verifies the pinned upstream
cache, imports all snippets, runs tests, and evaluates cached baselines. Set
`QUOTE_POWER_VENV` to reuse a different environment. The 22 tests cover graph
semantics, filtering, statistics, API recovery, cache isolation, and reduced-scope
execution. PuLP emits deprecation warnings; its dependency range excludes 4.0.

## Live scoring and cache reuse

Purdue's base URL is `https://genai.rcac.purdue.edu/api`. Supply `LLM_API_KEY`
securely in environment settings, scoped to `genai.rcac.purdue.edu`, and set
`LLM_BASE_URL` to that base URL. Never commit credentials.

```bash
python scripts/run_experiment.py --quote-only
```

This resumes the completed quote-only design from ignored
`results/experiment-cache/`, reusing exact successful requests. It writes a
separate quote export and evaluation, preserving the partial factorial outputs.
A fresh run requires 7,248 quote checks plus 151 zero-shot requests. Purdue calls
are paced to one per 3.2 seconds, with bounded transient/format retries and an
exclusive runner lock. Setup itself makes no live model calls.

`gemma4:26b-a4b` was selected by JSON and token-probability compatibility, not
accuracy. Its ID advertises 4B activated parameters. Purdue did not expose prices,
so this is a verified resource-efficient choice, not proof of the cheapest model.
The API was authenticated successfully. Historical qualification reports document
why Gemma3, Qwen3 and GPT-OSS were unsuitable for this scorer's token probabilities.

Cache keys include endpoint, request kind, model, prompt, and parameters. A
provider/model switch cannot treat Purdue responses as another model's outputs.
Nous access has a saved draft requirement but was not active in the completed run;
no Nous credit was spent. Do not mix providers across snippets in a headline arm.

## Methods and limitations

The converter verifies a pinned hash and CSV alignment using a restricted pickle
reader. XOR reproduces the released 98-variable factor rules with weights 20/18/1
and rounded exponentiated cached potentials, including the apparent repeated
contradiction pair. CBC solves binary MAP; this differs from AD3's LP relaxation.
Quote filtering drops failed argument variables and paired XOR constraints that
touch them. A declared A fallback handles zero surviving reasons; none occurred.

The quote model returns a verbatim span, evidence speaker, support judgment, and
attribution judgment. Keep an argument only when both judgments are true and
normalized text matches within the cited speaker's turn. String matching and
model judgments are not human validation. Grounded aggregation uses unweighted
attacks and an IN-reason count, then cached reason grounding sums, then A.
The fixed-score comparison holds all released weights constant with filtering
off/on; partial current-model rebuttal scores are ignored in both arms.

Labels describe institutional roles in academic meetings; they do not establish
universal conversational power. The 151 examples have no independent development
split. The human audit is unfilled. All contrasts and uncertainty are reported;
no accuracy improvement is claimed, and the deferred prompt ablation is untested.

Upstream: https://github.com/kandicew/social-reasoning, pinned at
`fed11b69765b4a398ce4a985adedfb65ec3098ab`. Cache SHA-256:
`e4be3c7c6a3543a77ad07eddde44bfa42f6e58fa6a4bf47df53ea9e894f1e318`.
Source data and dependency checkout remain under ignored `data/`. API caches,
credentials, and the user-uploaded proposal PDF are excluded from GitHub.
