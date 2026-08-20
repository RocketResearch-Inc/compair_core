# Baseline generation qualification examination v1

This non-production eval surface freezes the final 120-case synthetic
baseline-generation qualification examination. It was authored and audited
without loading, downloading, contacting, or invoking a model or generation
provider.

The source checkout was the clean remote `feature/baseline-v1` commit
`e83eb2ffd7adf0f8718b6b2d388c0f3af60e4326`, which contains Phase 2B2M.4.6.
The artifacts are under `tests/evals`, are not Python packages, and are pruned
from both wheel and sdist by `MANIFEST.in`.

## Frozen files

- `baseline-generation-qualification-examination.v1.json`: the 120 ordered
  cases and a base64 lossless copy of the original 16-case anchor payload.
- `baseline-generation-qualification-examination.v1.sha256`: direct-byte
  fixture digest.
- `semantic-audit.v1.json`: non-provider-visible, inference-free review ledger
  proving each negative is consistent and each positive has exactly one
  objective material contradiction.
- `semantic-audit.v1.sha256`: direct-byte audit digest.
- `validator.py`: strict structural, balance, anchor, hash, privacy, and
  semantic-audit validator.

Expected direct-byte SHA-256 values:

- examination: `2f1d8d204de06173fbfbe7fabf00aeb5771ef9869c09cbd959b2e7b4789d5863`
- semantic audit: `6c6778ed3e007caaa4f7d76c2efa6b37863938dcb5f43170e8230369b2eb1167`
- decoded 16-case anchor: `886ce0e93ac0749ade3bb109e736e3ffc0a08d0893c23fc5a83430bb0b700f2a`

## Evaluation tooling

`runner.py` is an evaluation-only production-adapter qualification runner. It
is not part of the frozen examination and stores no source, evidence, feedback
text, endpoint, or raw provider output. Like the frozen assets, it remains
under `tests/evals` and is excluded from distributions.

Hash the examination and audit files exactly as stored. Do not parse,
normalize, canonicalize, or reserialize them first. Both are strict UTF-8 with
LF line endings, no BOM, and exactly one final LF. The anchor digest is over
the 6,366 bytes obtained by strict base64 decoding of
`anchor_fixture.payload_base64`; it likewise has no JSON canonicalization step.

Per-case hashes are SHA-256 over UTF-8 JSON for only these fields:
`ordinal`, `case_id`, `surface`, `expected_outcome`, `source_text`, and
`evidence_renderer_input`. Keys are sorted lexicographically, non-ASCII is
emitted directly, separators are comma and colon without spaces, and non-finite
numbers are forbidden. The exact executable procedure is
`validator.canonical_case_bytes`.

## Examination shape

There are 12 surfaces with 10 cases per surface. Every surface contains five
`no_findings` and five `findings` cases, for 60 of each outcome. Provider-visible
input consists only of `source_text` and the ordered
`evidence_renderer_input`. Expected outcomes, hashes, anchor metadata, and the
semantic audit are evaluator-only and must never be included in a provider
request.

The 16 original anchors remain independently recoverable byte-for-byte and are
also identifiable in the examination by `anchor_case_id`. Their source,
ordered evidence, and outcome are checked against the decoded raw payload.

The examination is bound to the unchanged production output contract:

- schema version: `baseline-generation-output.v2`
- schema SHA-256: `fc5a85d5d38c18775afe0966987ea74e7e9ac072148822c1be60a199e32cca27`
- specification SHA-256: `e670731777b253f9d5e3984405c2d99871ba26f637a17e6221cc82d97bc8beb1`

It does not select, contact, attest, or qualify a model by itself. A later
execution phase must use Core's unchanged production prompt, parser, provider,
attestation, and supported runtime profile. Provider responses and examination
results must be stored outside this source tree.

## Qualification runner

The runner uses `OllamaGenerationConfig`,
`OllamaBaselineGenerationProvider._prepare_chat`,
`OllamaBaselineGenerationProvider.generate`, and
`BaselineGenerationService._parse_output` directly. This keeps qualification
on the production request, attestation, context-budget, serialization, and
structured-result paths without adding a second prompt or parser.

Configure the exact qualified provider through the normal Core environment and
choose a result directory outside the checkout. Run one case by stable ID or
one-based ordinal:

```console
python3 tests/evals/baseline_generation_qualification_v1/runner.py \
  --output-dir /external/eval-runs/qwen3-case-1 \
  --case 1
```

Run the complete ordered examination:

```console
python3 tests/evals/baseline_generation_qualification_v1/runner.py \
  --output-dir /external/eval-runs/qwen3-full
```

After an interruption, repeat the identical configuration and selection with
`--resume`. Before contacting the provider, resume validates the frozen
fixture and audit, run/configuration/selection fingerprints, every completed
case's identity and token bounds, and the ordered result hash chain. A changed
endpoint, seed, timeout, request/response limit, qualified profile, selection,
fixture, or result record is a mismatch and cannot resume.

The result directory contains:

- `run.json`: frozen input hashes, privacy-safe configuration fingerprint,
  qualified provider identity, and non-inference attestation latency;
- `cases.jsonl`: one atomically checkpointed record per completed case with
  case identity/hash, input-token and request-byte counts, expected and actual
  outcome, finding count, match result, provider identity fingerprint, and
  generation latency;
- `summary.json`: completion, outcome, token, latency, and result-file totals.

The runner never persists source/evidence strings, feedback strings, raw model
output, or the provider endpoint. Result checkpoints are replaced atomically
and fsynced after structured-output validation. Because the qualified Ollama
adapter has no channel-side idempotency contract, a process termination after
inference but before the atomic checkpoint can repeat that case on resume.

## Validation

Run:

```console
python3 tests/evals/baseline_generation_qualification_v1/validator.py
python3 -m pytest -q tests/test_baseline_generation_qualification_eval.py
python3 -m pytest -q tests/test_baseline_generation_qualification_runner.py
```

The validator and tests print only privacy-safe summaries and use no model or
network. Running `runner.py` is the explicit inference action; do not invoke it
until the exact model/runtime qualification execution is authorized.
