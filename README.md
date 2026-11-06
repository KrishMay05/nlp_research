# Quote-Grounded Arguments for Conversational Power

Research implementation of the supplied Patel, Kim, and Alladin proposal.
The offline experiment is runnable now. Live quote checks, corrected rebuttal
scores, and zero-shot predictions require an explicitly selected model and API
credentials; no model-generated results are bundled or fabricated.

## Quick start

After cloning this repository, change into its checkout directory. The setup
script also works outside `/workspace`; its default environment is `.venv`.
Set `QUOTE_POWER_VENV` to reuse an existing environment.

Python 3.11+ on Linux; tested on Python 3.12 with bundled PuLP/CBC.

```bash
cd /workspace/nlp_research
bash scripts/setup.sh
source .venv/bin/activate
quote-power evaluate
quote-power audit
```

`setup.sh` installs the pinned dependency set, verifies the upstream version,
converts its cached data, runs the tests, and evaluates all 151 snippets. It is
safe to rerun. Existing upstream checkouts at another revision are left intact
and reported. The script uses the environment's configured proxy and TLS trust.
No web server, database, GPU, training split, or AD3 installation is needed.

## Implemented workflow

- Convert upstream cached objects to JSONL with a restricted pickle reader,
  a pinned SHA-256, CSV alignment checks, stable argument IDs, and provenance.
- Reproduce the released 98-variable factor graph in PuLP/CBC: XOR, OR, OROUT,
  weights 20/18/1, exponentiated cached log-potentials rounded to two decimals.
  Every original rule is retained, including the apparent repeated pair in the
  first aspect's contradiction rules. This is a binary MAP model, not AD3's LP
  relaxation. The Python solver also exposes `relaxation=True` for diagnostics.
- Compute Dung grounded labels by a fixed point. Each surviving rebuttal attacks
  its paired reason; all opposing reasons attack each other. Select the speaker
  with more IN reasons, then the sum of grounding probabilities across retained
  reasons, then A. No arguments passing the filter still yields A.
- Check normalized verbatim spans within the cited speaker's turn, with explicit
  model judgments of support and correct attribution. String matching alone
  cannot establish semantic attribution. Failed arguments drop out, never snippets.
- Run the complete 2×2×2 factorial from one shared annotation pool, with complete
  ID coverage enforced. Quote filtering fixes removed graph variables to zero
  and drops a reason/rebuttal XOR if either endpoint is removed. Other factors
  remain. These filtering semantics are an explicit implementation choice.
- Report accuracy, argument coverage, paired bootstrap 95% intervals (10,000
  resamples, seed 2026), and exact two-sided McNemar tests against majority-A,
  cached XOR, and zero-shot when available. Export every prediction by snippet.
- Sample 40 arguments with paired reasons and blank human judgment fields.
  Keep this audit separate from development decisions; it is not a gold dataset.

## Verified offline results

| Method | Correct | Accuracy |
| --- | ---: | ---: |
| Majority A | 83/151 | 54.97% |
| Cached XOR reproduction | 82/151 | 54.30% |
| Grounded, all cached arguments retained | 69/151 | 45.70% |

The XOR reproduction matches the published rounded 54.3%. These are descriptive
results on the released test set, not evidence of a significant improvement.
Unfiltered Dung has all rebuttals unattacked and IN, so all reasons are OUT;
its predictions are decided by the declared grounding tie rule.
`results/offline/report.json` contains the comparisons and uncertainty;
`predictions.jsonl` has full coverage. Generated source data and API caches are ignored by Git. Verified reports, per-snippet predictions, the unfilled audit sample, and API smoke-test outcomes are committed under `results/`.

## Live model scoring (optional, incurs API charges)

Supply `LLM_API_KEY` securely in environment settings; `OPENAI_API_KEY`, if already
available, is also recognized. Do not commit credentials. The default endpoint is
`https://api.openai.com/v1`; a compatible endpoint may be selected with `--base-url`.
The selected model must support temperature 0, JSON output, and top token logprobs.
Model compatibility and live API access have not been validated in this environment.

```bash
# Select a real model ID supported by your account. Start with one snippet.
quote-power score --model YOUR_MODEL_ID --limit 1
# Resume using the same content-addressed cache, then score the full release.
quote-power score --model YOUR_MODEL_ID
quote-power evaluate --factorial \
  --annotations results/scored/annotations.jsonl \
  --zero-shot results/scored/zero_shot.jsonl --output results/factorial
quote-power audit --annotations results/scored/annotations.jsonl
```

Scores are conditional P(Yes)/(P(Yes)+P(No)) from the first token's logprobs.
Both tokens must be present in the returned top candidates or scoring fails
explicitly. Model, prompt, temperature, and endpoint enter cache keys. Every
successful call is saved immediately; rerunning reconstructs partial exports.
A limited smoke run cannot satisfy complete-dataset evaluation.
Use one model/endpoint configuration for all paired arms. Audit JSONL contains
blank human fields, the returned quote, full transcript, and paired reason.

## Proposal details resolved explicitly

The release has **3,624 reasons and 3,624 rebuttals**, for **7,248 argument texts**.
Quote-checking all texts requires 7,248 calls, plus 3,624 corrected rebuttal calls
and 151 zero-shot calls: up to **11,023 calls**, before retries. This exceeds the
proposal's approximate 3,600-call budget. No live calls run during setup.

Under the proposal's unweighted grounded semantics, a corrected rebuttal score
does not change whether a surviving rebuttal attacks. The two corrected-prompt
settings therefore give identical grounded predictions for a fixed quote filter.
Introducing a score threshold would be a new experimental choice; this project
does not silently introduce one. The corrected score replaces the rebuttal's
cached grounding/rebuttal log-potential in XOR, keeping its validity potential.

Upstream `rebuttal_a` means **against A's reason**, not spoken by A. The converter
preserves the released pairing rather than deriving speaker identity from that
field name. Quote support and speaker attribution still need the proposed human
audit. The similarity/contradiction factors are reproduced from upstream rather
than reinterpreted or repaired.

## Data provenance and limitations

Source: https://github.com/kandicew/social-reasoning

Pinned commit: `fed11b69765b4a398ce4a985adedfb65ec3098ab`.
The importer verifies the expected pickle hash before loading and permits only
specific data classes and NumPy scalar types. Do not use it on arbitrary pickles.
Upstream has no explicit license file at this revision; its data/code are fetched
as an ignored research dependency and are not redistributed as project source.
Check applicable permissions before publishing data or derivative artifacts.

Tests cover graph cycles/chains, quote attribution, filtering, tie rules, corrected
XOR scores, statistics, prompts, API cache behavior, and all 151 cached predictions.
Run `python -m pytest -q` after activating the environment. PuLP 3.3 emits
migration warnings about its planned 4.0 API; the dependency range excludes 4.0.

## Purdue GenAI Studio

Purdue's documented OpenAI-compatible base URL is
`https://genai.rcac.purdue.edu/api` (without `/v1`). In GenAI Studio, create a
personal key under **Settings > Account > API Keys**, then store it securely in
this cloud environment as `LLM_API_KEY`, allowed only for
`genai.rcac.purdue.edu`. This project uses `LLM_API_KEY` for the key that Purdue's
examples call `GENAI_API_KEY`. Set `LLM_BASE_URL` to the base URL above or use:

```bash
quote-power score --base-url https://genai.rcac.purdue.edu/api \
  --model MODEL_ID_FROM_YOUR_ACCOUNT --limit 1
```

List account-accessible models through the authenticated `GET /api/models`
endpoint before selecting an ID. Purdue's examples mention `gpt-oss:120b`, but
availability and suitability for token scoring must be checked with your account.
The scorer paces network calls to at most one per 3.2 seconds and backs off on
rate limits, transient failures, and JSON `null` responses. Run only one scoring
process at a time; other API usage on your account shares the 20 requests/minute
limit. At 11,023 calls, pacing alone requires roughly 9.8 hours, plus inference
latency. Interrupted requests resume through the existing cache.

The documentation does **not** establish token-logprob support; its example
responses have `logprobs: null`. Test the Yes/No probability request before a full
quote run. If the chosen backend does not expose probabilities, the proposed
corrected-score experiment is blocked; do not substitute invented probabilities.
For structured output, use a vLLM-backed model supporting `response_format` with
this scorer. Ollama's documented `format` parameter needs a separate adapter.
No Purdue model/API call has yet been validated with a real token.

## Live experiment execution plan

The fixed [experiment plan](docs/experiment-plan.md) records the proposal-based
comparisons and model-selection rules. Model qualification found that
`gemma4:26b-a4b` provides valid quote JSON and usable Yes/No log probabilities on
representative prompts. This supersedes the earlier smoke tests of Gemma3,
Qwen3 and GPT-OSS; those reports remain as historical evidence. No claim is made
that the full live experiment has finished or improved accuracy.

Quote responses now use Purdue's documented strict JSON schema to avoid missing
or mistyped fields. The XOR arm explicitly falls back to A if no reasons survive,
recording `empty_evidence_fallbacks` in evaluation reports. Successful new API
cache entries include server usage and elapsed time when exposed. Cached calls
remain local, while final exports are intended for version control.
