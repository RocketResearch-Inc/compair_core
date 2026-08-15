# Retrieval baseline plan (Phase 0)

## Scope and recorded baseline

This document records Phase 0 discovery only.  No production retrieval, API, task,
storage, configuration, or test code has been changed.

| Item | Recorded value |
| --- | --- |
| Core repository | `sources/compair_core` |
| Working branch | `feature/baseline-v1` |
| Current commit | `9c36b04e38b372532c1143c3f526a2e1a749cb4e` (`Release v0.10.4`) |
| `origin/main` commit | `9c36b04e38b372532c1143c3f526a2e1a749cb4e` |
| Package version | `compair-core 0.10.4` |
| Initial status | `?? AGENTS.md`, `?? CODEX_TASK.md`, and `?? EVAL_POLICY.md`; these pre-existing task instructions are intentionally preserved. |
| Adjacent CLI inspected | `sources/compair-cli`, clean at `df50c66d8f2ffb106f58d75c7f70a08b6a78326a` |

The documented source setup is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The required baseline commands are:

```bash
python -m compileall -q compair_core tests
python -m pytest -q
```

This environment has `python3`, not `python`.  The equivalent commands were run
from an isolated `/private/tmp/compair-core-phase0-venv` after installing the
documented development dependencies there:

```bash
/private/tmp/compair-core-phase0-venv/bin/python -m compileall -q compair_core tests
/private/tmp/compair-core-phase0-venv/bin/python -m pytest -q
```

Compilation passed.  Pytest completed with **105 passed, 3 failed** in 10.65s.
The failures are present before retrieval changes:

1. `tests/test_api_load_documents.py::test_load_documents_executes_only_paginated_query`
   fails during SQLAlchemy mapper configuration because a test-installed
   `sqlalchemy.orm` stub lacks `foreign`.
2. `tests/test_reference_reranker.py::ReferenceRerankerTests::test_load_model_resolves_latest_manifest_for_xgboost`
   requires scikit-learn, which is not listed in the development extra.
3. `tests/test_tokenization_special_text.py::test_embedding_token_helpers_treat_special_token_markers_as_source_text`
   encounters a test-installed `requests` module without `get` when tiktoken
   attempts to resolve `cl100k_base`.

The suite also emits existing Pydantic v2 and SQLAlchemy mapped-dataclass
deprecation warnings.  These are baseline findings, not Phase 0 fixes.

## Current Core architecture

### End-to-end data flow

```text
CLI repository sync (adjacent repository, not modified in this phase)
  └─ builds snapshot or accumulated change text
     └─ POST /process_doc: doc_id, doc_text(_b64), feedback and scope options
        └─ api._dispatch_process_document_task(...)
           └─ tasks.process_document_task(...)
              └─ main.process_document(...)
                 ├─ chunk document content and persist Chunk + embedding
                 ├─ choose changed/new source chunks for feedback
                 └─ main.process_text(...) per selected source chunk
                    ├─ retrieve and select legacy reference Chunks
                    ├─ persist Reference rows
                    └─ feedback.get_feedback(...)
                       └─ render at most four reference snippets into a model prompt
```

The core API's ingestion path is `POST /process_doc` in `compair_core/api.py`.
It accepts `doc_id`, `doc_text` or `doc_text_b64`, `generate_feedback`,
`chunk_mode`, `reanalyze_existing`, optional `reference_doc_ids`, `skip_index`,
and a focus manifest.  It does **not** accept a raw Git diff, changed repository
identity, a change-set identifier, a repository/file manifest, retrieval engine,
or generation-bypass option.  It returns only a task ID or the `skip_index`
acknowledgement.

The core task currently writes the supplied `doc_text` into `Document.content`
and calls `process_document`.  Although the API stages a payload and passes
`snapshot_payload_key`, the Core task implementation does not dereference that
key.  A Phase 2 nonempty-payload sentinel test is therefore mandatory before
using Redis or external payload staging for retrieval work.

`process_document` in `compair_core/compair/main.py` sanitizes the document,
compares current and historical chunk text, calculates focus/change context,
creates embeddings for new chunks, and calls `process_text`.  This is a
per-chunk feedback workflow; it is not a single whole-change-set query.

### Persistent model and indexing

The persistent data model has:

* `Document`: title, full document content, document type, publication state,
  topic tags, and group membership.
* `Chunk`: a chunk hash, chunk content, document/note ownership, and an
  embedding.  The only relevant index is document/type/note/hash.
* `Reference`: a selected source chunk to selected reference chunk relation.

Embeddings are stored either as JSON arrays or pgvector values according to
`COMPAIR_VECTOR_BACKEND`.  The configured dimension is process-global from
`COMPAIR_EMBEDDING_DIM` (default 384 in Core).  `Embedder` can call local or
OpenAI providers, but a failed local/OpenAI request falls back to deterministic
SHA-derived hash vectors.  Vectors have no stored provider/model/revision,
dimension fingerprint, tokenizer version, namespace, or index version.  The
existing cosine helper returns `None` for missing, zero, or wrong-dimension
vectors, but it does not reject non-finite values.

There is no durable file-level corpus table, repository identity, canonical
relative file path, content hash at file granularity, token-frequency state, or
atomic lexical-index version.  Code-repository snapshots may contain `### File:`
headers, but Core stores/retrieves their chunks rather than a complete stable
file corpus.  `Document` titles and snapshot headers are insufficient to claim
the frozen comparator's repository/path scope without an explicit mapping.

### Legacy retrieval and selection

The executable legacy selection block is in `process_text`, after the source
chunk is persisted and before `Reference` rows are written.  It:

1. Determines a scope from the source document's groups (or an explicitly
   allowed same document), publication state, and optional document-ID filter.
2. Applies `_filter_reference_candidates`, including chunk type, same source
   chunk/file/document, and metadata/header filters.
3. Fetches dense candidates from pgvector or scores all JSON-vector chunks by
   cosine similarity.  Exact score ties are not consistently broken by a stable
   repository/path key.
4. For code-focused documents, creates ephemeral in-memory SQLite FTS5 rows and
   also produces lexical, anchor, and counterpart candidate lanes.  Query
   variants include primary, full, anchor, and counterpart text.
5. Interleaves the dense and optional lane candidates and invokes
   `_rerank_reference_chunks`.
6. Limits the result to the legacy code-review default of four references with
   a maximum of two chunks per source, plus diversity penalties.  It can use a
   configured reranker and deterministic reference adjudicator.

The optional legacy hybrid branch is not the frozen comparator.  It uses vector,
FTS, lexical, anchor, and counterpart ranks; defaults to `RRF k=40`; has unequal
rank weights and many additional heuristic signals; and is followed by
reranking/adjudication/diversity selection.  The FTS index is rebuilt in memory
per query and SQLite `bm25()` scoring is not frozen BM25.

The selected `Chunk` objects are saved as `Reference` rows.  `get_feedback`
renders local references using the first four snippets.  That generation path
is downstream of selection and is deliberately outside baseline fidelity.

### API, capability, task, and rendering boundaries

`compair_core/compair/schema.py` has document, reference, feedback, and
whole-bundle `review_now` models, but no retrieval request/result schema.
`/capabilities` currently reports broad inputs/models/features only; it does
not declare an engine, embedding provenance, index state, retrieval health, or
degraded state.  The Core task and API route have no structured retrieval-only
path.  Therefore an external caller cannot distinguish “no selected evidence”,
“missing index”, “embedding fallback”, and “no finding” in a machine-readable
retrieval contract.

The adjacent CLI confirms the boundary.  It sends snapshot/diff text to
`/process_doc` as `doc_text_b64`, but does not pass the exact
`git diff HEAD^ HEAD --no-ext-diff` required by the comparator as a named
retrieval query.  Its `sync --json` currently prints progress to stdout before
the final JSON and exits nonzero only through generic command errors.  CLI
changes are intentionally deferred until the Core retrieval schema is stable.

## Frozen comparator versus current hybrid selector

| Dimension | Frozen `baseline_v1` | Current Core legacy hybrid |
| --- | --- | --- |
| Query | Exact raw `git diff HEAD^ HEAD --no-ext-diff` | Current source snapshot chunk or compact focus/change context, plus variants |
| Candidate repositories | All sibling repositories, changed repo excluded before scoring | Documents in shared group; same document can be allowed; no repository-level exclusion invariant |
| Candidate unit | Complete UTF-8 file | Persisted snapshot/document chunk |
| Discovery/filtering | Stable repo/path walk; exclude named directories, non-UTF-8, and files over 200,000 bytes | SQL document/chunk scope plus header, document, same-file, and publication filters; no comparable file size/encoding enumeration |
| Ranking text | `Repository file: <repo/path>\n\n<first 12,000 chars>` | Snapshot chunk text and separately engineered lane/query text |
| Lexical ranking | Frozen tokenizer/stopwords and whole-corpus BM25 (`k1=1.5`, `b=0.75`) | Ephemeral FTS5 plus custom token overlap/path/artifact/anchor heuristics |
| Dense ranking | FastEmbed `BAAI/bge-small-en-v1.5`; raw vector dot product | Configurable Core embeddings, cosine similarity, and current fallback-to-hash behavior |
| Fusion | Equal BM25/dense reciprocal-rank fusion, `k=60` | Five possible lanes, `k=40`, unequal weights, plus heuristic blends |
| Tie break | Complete `repo/path` string at every final ranking | Several score-only sorts preserve database/interleave order on ties |
| Retrieval cut | Six ranked whole files | Candidate/merge/trim limits; normally four references selected |
| Final evidence | Deduplicate content, then fill four unique items within 16,000 characters | Source/path caps and diversity selection before generation; existing downstream filtering can leave fewer slots without refill |
| Failure semantics | Reference implementation does not expose a Core contract | Local embedding failure silently substitutes hash vectors; no engine/index provenance contract |

Two small implementation details are material for parity.  The vendored source
labels its dense lane “cosine” in prose, but the actual implementation computes
`vectors @ query`; baseline fidelity must reproduce that dot product, without
adding Core normalization unless a golden test proves the representations make
it equivalent.  Also, the comparator's common normalizer applies content-hash
deduplication and a four-item/16,000-character cap *after* retrieval.  The Core
legacy selector's source caps/diversity policy are therefore a different
selection algorithm, not a baseline optimization.

The archived evaluation reports make this distinction operationally important.
On the open 60-pair/120-case corpus, the canonical retrieval-complete counts are
36/60 finding, 34/60 compatible, 70/120 cases, and 32/60 pairs for fresh hybrid,
versus 27/60, 24/60, 51/120, and 20/60 for the tested Core workflow.  These are
open-development/exploratory reference values, not a new confirmation target.
The fairness audit also found that Core could select changed-repository evidence
before normalization removed it, leaving two or three delivered sibling chunks
with no refill.  `baseline_v1` must exclude changed-repository files before
scoring and refill after every post-ranking filter or dedupe removal.

## Proposed retrieval-engine seam

### Interface and ownership

Introduce a small retrieval package, independent of feedback generation:

```text
compair_core/compair/retrieval/
  __init__.py
  types.py       # versioned request/result, candidates, evidence, states
  factory.py     # explicit legacy/baseline_v1 registration and validation
  legacy.py      # adapter around the current selector
  baseline.py    # pure frozen BM25+BGE+RRF implementation
```

The public internal contract should be conceptually:

```python
class RetrievalEngine(Protocol):
    name: str

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult: ...

    def capability(self) -> RetrievalCapability: ...
```

`RetrievalRequest` must carry a schema version, caller-supplied request ID,
query kind and text, changed repository identity, candidate scope, group/corpus
identity, and engine configuration.  `baseline_v1` requests additionally need
an explicit raw diff and a complete, stable file-level sibling corpus.  A
`RetrievalResult` must include its schema version, request/correlation ID,
status (`ok`, `insufficient`, `degraded`, or `error`), engine/config fingerprint,
corpus/index version, embedding provider/model/revision/dimension fingerprint,
candidate provenance and every lane's score/rank, selected file/span/content
hash, timings, budget/refill counters, and an explicit error/fallback object.

The minimal integration seam is the selection section of `process_text`: after
the source chunk exists and before `Reference` rows are saved.  Extract that
section behind one `retrieve_reference_evidence(request)` facade.  The `legacy`
adapter calls a narrowly extracted private bridge containing the present query,
candidate discovery, interleaving, `_rerank_reference_chunks`, and ordering
logic.  Its adapter must return the exact legacy `Chunk` sequence unchanged.
Existing `process_text` remains responsible for persisting `Reference` rows and
calling generation.  This makes the selection boundary explicit without moving
or rewriting all legacy scoring code.

`baseline_v1` must not pretend that a source chunk is an exact comparator query.
Until a request provides `query_kind=raw_git_diff_v1`, changed repository ID,
and a complete sibling file corpus, it returns `insufficient` with a precise
reason (for example `raw_diff_absent` or `file_corpus_incomplete`).  It must
never call `legacy` as an implicit fallback.  The default factory selection is
always `legacy`; an unknown engine or invalid baseline configuration is a
configuration error.

### File and corpus model needed after pure fidelity

The existing document/chunk tables cannot establish parity by themselves.  The
first persistent integration needs a versioned file-level corpus record or a
faithful, explicitly versioned reconstruction from source snapshots.  Each
candidate must retain at least repository ID/name, relative path, full decoded
content, byte size, UTF-8 eligibility, content hash, and corpus/index version.
The collection must be one coherent version for a request.  The model/index
fingerprint binds embedding provider, model, revision, dimension, tokenizer,
and schema version; changing any of them creates a new namespace or requires a
rebuild.

For fidelity, exact BM25 is computed from the stable file corpus (or durable
term-frequency/document-length state implementing the same formula).  FTS5 and
PostgreSQL text rank remain separately named experimental lanes until golden
rank parity exists.  File-level persistence and later chunk-level span selection
are separate concerns: `baseline_v1` ranks whole files, then renders the first
12,000 file characters and applies the common evidence budget.

## Migration, compatibility, and failure behavior

* `legacy` stays the default.  Its current request path, API fields, environment
  variables, `Reference` persistence, and generation behavior stay intact.
* The first retrieval-only response is a new versioned API contract.  Do not
  rename legacy fields or make legacy endpoints return a different shape.
* Phase 2 adds `retrieval_query`/`change_set` fields and a retrieval-only API
  rather than inferring raw-diff parity from compact per-chunk text.  Ordinary
  non-Git clients may use a separately named deterministic
  `core_change_set_v1` query, but that profile is not `baseline_v1`.
* A missing query, empty/incomplete corpus, or no eligible candidates is
  `insufficient`, never a successful “safe” result.  Filtered candidates are
  classified as `candidate_absent`, `candidate_present_not_selected`,
  `filtered`, or `render_truncated` as applicable.
* Missing/stale/mixed/wrong-dimension/non-finite embeddings are fail-closed
  `error` results for `baseline_v1`.  Hash vectors may remain an explicitly
  diagnostic legacy/provider behavior, but never satisfy baseline capability or
  comparable evaluation requirements.
* Provider timeout and index failure remain explicit; degraded execution is
  permitted only under an explicitly named configuration and is excluded from
  comparable benchmark runs.  There is no silent dense-to-lexical or
  baseline-to-legacy fallback.
* Candidate exclusions, symlink/path-traversal protection, deduplication, and
  changed-repository removal happen before final selection.  If an item is
  removed, select the next eligible ranked item until four items/the character
  budget/corpus exhaustion, and record underfill/refill counts.
* **Deliberate comparator delta:** production `baseline_v1` rejects every
  symlink candidate, including links whose resolved target remains inside the
  repository.  The vendored comparator follows file symlinks; Core does not,
  because an escaping link would expand the declared sibling-repository corpus.
* Index/file writes and corpus-statistics changes are atomic.  Readers observe
  one version; deletes and renames invalidate the old candidate in the same
  transition.  Restart persistence and concurrent read/update are integration
  gates, not assumptions.

## Exact test plan

All new unit fixtures will be authored toy corpora.  They will not contain
benchmark case IDs, paths, decisive strings, ground-truth spans, or evaluator
imports.  Production modules will never import the read-only benchmark trees.

### Phase 1: pure parity tests

1. Tokenizer and stopword golden cases cover complete tokens, split forms,
   numeric tokens, punctuation trimming, and case normalization.  Assert the
   exact frozen token list.
2. A toy whole-file corpus verifies document frequency, average length, query
   term frequency, BM25 `k1=1.5`, `b=0.75`, numeric BM25 scores, and stable
   repository/path ordering.
3. A fixture embedding provider returns fixed vectors.  Assert the unnormalized
   dot-product scores, dense ranks, deterministic score ties, and rejection of
   missing/wrong-dimension/NaN/infinite vectors.
4. Assert exact equal-weight RRF scores with `k=60`, ranks beginning at one, and
   complete-path tie resolution.
5. Exercise candidate enumeration with excluded directory components,
   oversized files, non-UTF-8 bytes, empty corpus, changed-repository removal,
   symlink escape prevention, and duplicate content.
6. Verify retrieve-six/normalize-four behavior: content-hash dedupe, filtering,
   refill from the fused ordering, a 16,000-character cap, correct underfill
   accounting, and render truncation attribution.
7. Run a fixed request twice and compare all deterministic response bytes after
   excluding generated request IDs/timestamps.  Compare selected order, BM25,
   RRF, normalized evidence, and dense ordering with a frozen locally authored
   golden package; dense numeric tolerance must be documented and cannot change
   ordering.
8. Test factory/default selection: `legacy` is default, `baseline_v1` is
   explicit, unknown engines and invalid configurations fail, and baseline
   preconditions produce `insufficient` rather than an implicit legacy result.

### Phase 2 integration tests

1. API/task propagation preserves one full `raw_git_diff_v1` change-set query,
   changed repo, candidate scope, request ID, and engine flag through the Core
   retrieval-only API without invoking generation.
2. Legacy contract snapshots show that the `legacy` engine returns the same
   selected references and keeps current `process_text` behavior.
3. Initial index, incremental edit, deletion, rename, restart persistence,
   empty and partially indexed group, atomic concurrent reader/update, and
   SQLite/PostgreSQL behavior where supported are tested.
4. A nonempty staged-payload sentinel demonstrates the task indexes the actual
   submitted document body before Redis/payload staging is enabled.
5. Engine capabilities report actual index/provider/model/revision/dimension
   state; failures include embedding timeout, missing/stale/mixed model,
   wrong dimension, and non-finite vectors.
6. API schema tests assert all result provenance, timing, status, candidate
   stages, selected spans, and errors, plus unchanged legacy fields.

### Evaluation protocol and artifacts

Do not run or inspect `ground_truth.json` to tune production behavior.  After
the fidelity/integration gates, the evaluator alone may receive it.  The
existing open-development reconstruction commands are:

```bash
python compair-statistical-drift/harness/run_retrieval_study.py \
  --output runs/candidate-001/retrieval \
  --fixture-specs compair-statistical-drift/fixture_specs.json \
  --ground-truth compair-statistical-drift/ground_truth.json \
  --fixture-module compair-statistical-drift/harness/fixture_builder.py \
  --runner compair-hard-drift/harness/run_experiment.py \
  --project-root compair-hard-drift \
  --cli bin/compair \
  --source-cli sources/compair-cli \
  --source-core sources/compair_core \
  --embedding-cache experiment/models \
  --runtime-root /tmp/compair-candidate-001

python compair-statistical-drift/harness/aggregate_shards.py \
  --shards-root runs/candidate-001/retrieval/shards \
  --output runs/candidate-001/retrieval/combined \
  --fixture-specs compair-statistical-drift/fixture_specs.json \
  --ground-truth compair-statistical-drift/ground_truth.json \
  --blind-seed paired-drift-confirmatory-v1 \
  --score-script compair-hard-drift/harness/score_retrieval.py
```

Phase 3 should wrap these as one stable command that writes a new append-only
run directory containing `run.json`, `metrics.json`, `cases.jsonl`,
`deltas.json`, and `summary.md`.  Record source commits/dirty state, input and
scorer hashes, engine/index/model configuration, candidate recall at declared
depths, selected-set recall, refill/budget use, timing, and failures.  Results
on this corpus are open-development/exploratory only; promotion requires a new
sealed cohort.

## Step-by-step implementation sequence

1. Add pure, dependency-injected baseline types, frozen tokenization, file
   enumeration/filtering, BM25, dense dot-product ranking, RRF, evidence
   normalization, and result serialization.  No database/API/task changes.
2. Add the parity/factory tests above and prove the pure engine's golden output.
3. Add the engine factory and a thin `legacy` adapter; extract only the
   `process_text` selection block into an invocation boundary.  Keep legacy
   default and verify its selected sequence is unchanged.
4. Define the retrieval-only schema and `raw_git_diff_v1`/`core_change_set_v1`
   request kinds.  Add API/task propagation and capability reporting, but do
   not change CLI yet.
5. Add the versioned file-level corpus/index state and atomic lifecycle.  Use
   exact portable BM25 over that corpus; retain FTS/pgtext as experiments.
6. Wire `baseline_v1` through the real Core retrieval-only endpoint.  Enforce
   fail-closed embedding/index contracts and prove all lifecycle tests.
7. Add one append-only open-development evaluation command and produce a
   case-level parity report before any optimization.  Keep `baseline_v1`
   frozen thereafter.
8. Only after those gates, add separately named `ensemble_v1` experiments one
   ablatable hypothesis at a time.  CLI integration follows the stable Core
   schema, not the reverse.

## Proposed Phase 1 file changes

| File | Phase 1 change |
| --- | --- |
| `compair_core/compair/retrieval/__init__.py` | Export the internal retrieval contract without importing benchmark code. |
| `compair_core/compair/retrieval/types.py` | Versioned request/result, candidate/evidence/provenance, budget, capability, and status types. |
| `compair_core/compair/retrieval/baseline.py` | Pure `baseline_v1`: frozen tokenization, candidate handling, BM25, injected BGE dense lane, RRF-60, deterministic selection/rendering. |
| `compair_core/compair/retrieval/factory.py` | Validated `legacy`/`baseline_v1` selection; default `legacy`; no fallback on invalid/missing baseline capability. |
| `compair_core/compair/retrieval/legacy.py` | Thin adapter/bridge contract for current legacy selection; do not reimplement its algorithm. |
| `pyproject.toml` | Add a pinned/recorded FastEmbed retrieval dependency (or dedicated optional extra) for the actual BGE provider; pure unit tests continue to use an injected fixture embedder. |
| `compair_core/compair/main.py` | Minimal extraction of the existing `process_text` reference-selection block behind the legacy bridge only after legacy snapshot tests exist. |
| `tests/test_retrieval_baseline.py` | Pure comparator parity, filter/dedupe/refill, budget, dense failure, and deterministic-order tests using new toy corpora. |
| `tests/test_retrieval_factory.py` | Engine registration/default/invalid-engine and precondition behavior. |
| `tests/test_main_retrieval.py` | Narrow regression coverage proving the legacy adapter preserves existing output/order. |

The following are intentionally Phase 2 or later, not Phase 1: database models
and migrations for file/index state; API/schema/task/settings/capabilities
changes; an endpoint; CLI flags/JSON/exit semantics; and benchmark runner code.

## Risks and decisions required before promotion

1. **No parity query or corpus exists in Core today.**  Raw diff and stable
   sibling-file identity must be explicit input/index state.  Inferring them
   from a chunk would create an unlabelled approximate profile.
2. **Embedding behavior is presently unsafe for comparable runs.**  Hash fallback,
   absent provenance, and lack of finite-vector validation must be contained by
   baseline capability checks before baseline results are compared.
3. **Persistent schema work is non-trivial.**  Existing chunks are not files;
   migration must retain legacy indexes while introducing coherent file-corpus
   versions and a rebuild boundary.
4. **Legacy extraction can change ordering inadvertently.**  It needs snapshot
   regression tests first, explicit stable ordering only in the new engine, and
   no opportunistic cleanup in the same change.
5. **The open benchmark cannot validate superiority.**  It is suitable for
   parity/regression and exploratory diagnostics only.  A separately sealed
   organic/executable holdout is required before promotion.
6. **No evidence is not compatibility.**  Retrieval `insufficient`, provider
   error, and evaluation underfill must remain separate from feedback/generation
   policy results and should not enable blocking enforcement.
