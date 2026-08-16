# Durable baseline evidence to Reference bridge

Status: Option A was approved. Phase 2B2F.1 freezes and implements the schema
and migration under `baseline-reference-bridge.v1`. Bridge writes, reads,
retrieval integration, API serialization, and generation remain disabled and
are Phase 2B2G or later work. The frozen names and lifecycle in
`docs/design/baseline-evidence-schema.md` supersede earlier provisional names
below where they differ.

## Current-state findings

Core's generation path accepts an ordered `list[Chunk]`. `process_text` inserts
one `Reference` per selected legacy chunk, commits those rows, and then passes
the same in-memory chunk list to `get_feedback`. That preserves order within
the current call, but order is not durable: `Reference` has no ordinal, the
`Chunk.references` relationship has no `order_by`, and `/load_references` uses
an unordered `.all()` query.

`Reference.reference_chunk_id` is consumed only by the Core API serializers and
the ORM relationship. `_reference_content` prefers the referenced chunk and
falls back to a referenced document or note; `_reference_file_path` infers a
path from snapshot text inside a chunk. `/load_references` joins
`reference_chunk` but filters out every row without a referenced `Document`.
The document-feedback serializer also iterates the unordered relationship.
Existing response schemas expose legacy IDs, content, and inferred path, but no
rank or retrieval provenance.

The persistent baseline is deliberately different:

- `RetrievalResult.evidence` contains already selected, ordered evidence and
  query hash/length/origin, never raw query text.
- `RetrievalEvidence.document_id` currently means
  `RetrievalBaselineIndexDocument.index_document_id`, not a Core `Document` ID.
- Candidate content is the first 12,000 source characters. Normalization may
  further clip that selected content to the shared 16,000-character budget.
- `RetrievalEvidence.content_hash` hashes the normalized selected content. The
  whole-file hash remains in `RetrievalCorpusFile.content_hash` and
  `RetrievalBaselineIndexDocument.source_content_hash`.
- The immutable whole UTF-8 file, repository identity, path, byte size, and
  optional source document/snapshot IDs remain in `RetrievalCorpusFile`.
- Corpus generations and index artifacts use cascading foreign keys. Activation
  marks the prior generation and index stale but does not currently garbage
  collect them. The publication pointer uses `SET NULL` when its index is
  deleted.

Core deletion is mixed. ORM relationships and PostgreSQL foreign keys specify
cascades, while several API paths perform explicit bulk deletes. SQLite does
not currently enable foreign-key enforcement globally, and some bulk document
deletes bypass the explicit `_delete_document_records` helper. A bridge cannot
assume database cascades alone provide lifecycle parity.

## Design options

### Option A: dedicated immutable baseline evidence

Create a baseline retrieval-run record, a content-addressed immutable file
artifact, and an ordered selected-evidence record. Add a nullable baseline
evidence target to `Reference`. Baseline rows use `reference_type =
"baseline_file"` and leave all legacy target IDs null.

Advantages:

- Whole-file provenance, selected content, scores, and rank have natural typed
  storage instead of being encoded into chunk text.
- Evidence is absent from `chunk`, has no legacy embedding, and therefore
  cannot enter vector, FTS, lexical, anchor, counterpart, or reranker lanes.
- Historical evidence can remain stable after corpus re-ingestion while corpus
  and index artifacts follow their own retention policy.
- Generation can consume a purpose-built immutable evidence view without
  pretending an index document is an authored Core document.

Costs are an additive schema and explicit baseline branches in future
Reference serializers and generation adapters. Those changes are visible and
testable, which is preferable to relying on implicit `Chunk` behavior.

### Option B: controlled materialization as a stable Core chunk

Materialize each selected file as a `Chunk` with `document_id = NULL`,
`note_id = NULL`, `chunk_type = "baseline_file"`, and no embedding. Put the
selected content in `Chunk.content`, store provenance in a companion table, and
continue using `Reference.reference_chunk_id`.

This is viable only with a unique content/provenance key, explicit cleanup, a
database check prohibiting a document/note/embedding, and regression tests that
every legacy candidate query requires `chunk_type = "document"` and a real
document join. Current primary legacy retrieval does those checks, but other
chunk loaders are less strict and future queries could accidentally omit them.
Materialization also cannot represent both the complete file and the exact
budget-selected content without another table, and a chunk without a document
does not fit current serialization or authorization assumptions.

### Recommendation

Use Option A. A dedicated evidence type makes accidental legacy retrieval
structurally impossible and preserves baseline's file-level semantics. Option B
saves one consumer adapter today at the cost of permanent ambiguity in the
central legacy candidate table.

## Recommended schema

Use an explicit schema version `baseline-reference-bridge.v1`. IDs are
`VARCHAR(36)`, hashes/fingerprints are lowercase 64-character SHA-256 values,
content is UTF-8 `TEXT`, and scores are PostgreSQL `DOUBLE PRECISION` / SQLite
`REAL`. Dense scores are the exact float32 values promoted losslessly for
storage; BM25 and RRF retain their internal full precision. No output rounding
occurs in these tables.

### `baseline_retrieval_run`

One row represents one successfully persisted `RetrievalResult`:

- `run_id` primary key and `idempotency_key` unique;
- `source_chunk_id` and `source_document_id`, both indexed, with source deletion
  cascading to the run;
- `request_id`, `result_schema_version`, `engine`, and `engine_version`;
- `query_kind`, `query_sha256`, `query_length`, and `query_origin`; there is no
  raw-query column;
- `corpus_scope_key`, `corpus_id`, `generation_id`, `generation_version`, and
  `corpus_manifest_hash`;
- `index_id`, `index_version`, `index_schema_version`, and `index_fingerprint`;
- `config_fingerprint`, embedding provider/model/revision/dimension/fingerprint;
- candidate, retrieved, filtered, duplicate, refill, selected, and evidence
  character counts plus `underfilled`;
- `authorization_scope_hash` and canonical `authorization_group_ids_json` as a
  creation-time audit snapshot;
- generation state and lease fields. Phase 2B2I supersedes the provisional four
  states with the durable `baseline-generation-state.v1` lifecycle documented
  in `docs/design/baseline-generation-lifecycle.md`.

Only an `ok` result can create a run. The idempotency key is SHA-256 over a
versioned canonical tuple containing source chunk, request ID, engine/version,
query hash, corpus generation, index fingerprint, and config fingerprint.

### `baseline_evidence_artifact`

One immutable, content-addressed snapshot represents a selected indexed file
and is reusable by runs against the same publication:

- `artifact_id` primary key and `artifact_key` unique;
- repository ID/name and normalized relative path;
- source corpus/file/generation IDs and generation version;
- index ID, index document ID, index fingerprint, and indexed-document hash;
- optional source Core document ID and source snapshot ID;
- complete source UTF-8 content, whole-file content hash, byte size, and
  character count;
- created timestamp.

`artifact_key` covers the bridge schema version, generation, index fingerprint,
index document ID, repository/path, and whole-file hash. Corpus/index IDs are
immutable provenance values rather than retention foreign keys. The bridge
verifies the live foreign rows before insertion, then the self-contained
artifact may outlive later corpus/index garbage collection without losing its
identity or content.

### `baseline_reference_evidence`

One immutable row represents one selected item exactly as delivered by the
result:

- `evidence_id` primary key;
- `run_id` foreign key with `ON DELETE CASCADE`;
- `artifact_id` foreign key with `ON DELETE RESTRICT`;
- one-based `ordinal` and `fused_rank`;
- exact `selected_content`, selected-content hash, and selected character count;
- `ranking_truncated`, derived by comparing the complete artifact with the
  frozen 12,000-character unit, and `budget_truncated`, copied from the result's
  normalization boundary;
- BM25 score/rank, dense score/rank, and RRF score;
- created timestamp.

Required unique constraints are `(run_id, ordinal)`, `(run_id, artifact_id)`,
and `(run_id, selected_content_hash)`. Checks require ordinals/ranks to be
positive, selected content to be nonempty, no more than four rows per run, and
finite scores. The ordinal here is the single authoritative Reference and
generation order; it is never duplicated into a second mutable rank column.

### Existing-table additions

Add nullable `Reference.baseline_evidence_id`, unique when non-null, referencing
`baseline_reference_evidence` with `ON DELETE CASCADE`. A check requires a
baseline target to have `reference_type = "baseline_file"` and null
`reference_chunk_id`, `reference_document_id`, and `reference_note_id`. Existing
legacy rows remain valid and unchanged when `baseline_evidence_id` is null.

Add nullable `Feedback.baseline_run_id` and `Feedback.finding_ordinal`. A partial
unique index on `(baseline_run_id, finding_ordinal)` prevents duplicate baseline
feedback persistence while leaving all legacy feedback unchanged. The existing
`model` and source-chunk linkage remain authoritative.

Every baseline Reference read must join its evidence and order by
`baseline_reference_evidence.ordinal`. Legacy queries and serialization retain
their current contract. Baseline serializers must read content/path from the
dedicated evidence and must not populate or synthesize `reference_chunk_id`.

## Migration plan

This must be a named, explicit, reversible migration, not another best-effort
startup `ALTER` or automatic rebuild.

1. Create the three new tables and indexes.
2. Add nullable bridge columns to `reference` and `feedback`; no legacy
   backfill is required.
3. PostgreSQL: add foreign keys and checks as `NOT VALID`, deploy read-compatible
   code, validate constraints, then enable writes. Use a partial unique index
   for baseline feedback.
4. SQLite: add nullable columns and partial unique indexes additively. Because
   global SQLite foreign keys are currently not guaranteed, add bridge-specific
   validation/cascade triggers or perform a separately approved, backup-and-
   verify table-copy migration. Do not silently rebuild `reference` at startup.
5. Record `baseline-reference-bridge.v1` in a durable migration-version table
   and make baseline writes fail closed if the version or constraints are
   absent.

Rollback before writes may remove the new nullable columns/tables using the
dialect-specific down migration. After any baseline write, application rollback
must first disable baseline writes and leave the additive schema/data in place;
old code ignores nullable columns. A destructive schema rollback requires an
export, row-count/hash verification, deletion in Reference/Feedback/evidence/
artifact/run order, and an explicit SQLite backup/copy verification. It must
never rewrite or renumber legacy References.

## Transaction and idempotency contract

The bridge accepts only a complete result with `status == ok`, no error, no
fallback engine, nonempty evidence, and complete corpus/index/embedding/query
provenance. `insufficient` or `error` returns before opening a write transaction
and creates no run, artifact, evidence, Reference, Feedback, generation claim,
or task result containing evidence.

For an `ok` result:

1. Begin one short write transaction. Lock the `RetrievalCorpus` and publication
   rows with `FOR UPDATE` on PostgreSQL; use `BEGIN IMMEDIATE` on SQLite.
2. Re-resolve the source chunk/document and authorize the caller. Verify that
   the corpus generation is still active, the same compatible index is still
   published, every fingerprint equals the result, and the changed repository
   remains excluded.
3. Resolve every result `document_id` as an index-document ID in that exact
   index. Load its supported corpus file, recompute the whole-file and selected
   hashes, verify repository/path/content, ranks, finite scores, budgets, and
   contiguous result order. A missing, superseded, deleted, or changed row
   aborts the transaction as stale.
4. Insert the run by idempotency key. On conflict, compare the complete immutable
   manifest. If it matches and has the expected ordered References, return the
   existing run; any mismatch is an integrity error, never an overwrite.
5. Upsert immutable artifacts by artifact key, then insert evidence and
   References by enumerating `RetrievalResult.evidence` from ordinal one.
6. Commit once. Any injected failure rolls back the run, artifacts created by
   this transaction, evidence, and References together.

Activation and publication use the same corpus row lock, so they cannot pass
the stale check concurrently. Retries cannot duplicate artifacts, evidence, or
References because all three levels have stable unique keys.

Generation is outside this database transaction: no database transaction may
remain open during an LLM call. After commit, a worker claims the run's
generation lease and reloads its evidence ordered by ordinal. A crash after a
provider call may repeat that external call unless the provider supports an
idempotency key, but Feedback persistence remains idempotent through
`(baseline_run_id, finding_ordinal)`.

## Generation contract

Define a read-only `GenerationEvidence` value with ordinal, repository/path,
exact selected content, selected/full hashes, and whole-file provenance. Load
it only from the persisted run with `ORDER BY ordinal`.

Generation receives exactly that sequence. There is no reranking, filtering,
deduplication, refill, fallback, or content clipping after `RetrievalResult`.
Provider prompt renderers may add a fixed provenance label such as
`Repository file: <repository/path>` outside the selected content, but they
must preserve the selected content bytes and order. The existing generic
`_local_references` path strips whitespace and applies its own limits, so the
baseline bridge must bypass it or prove an identity-preserving baseline mode.
Because baseline already returns at most four items and 16,000 characters, a
provider that cannot accept the rendered prompt fails generation explicitly;
it must not silently drop or truncate evidence.

The complete file remains in the artifact for inspection and provenance. Only
the already selected content is sent to generation. This preserves the fact
that ranking and selection operated on a whole-file corpus unit without
expanding generation beyond the frozen evidence budget.

## Lifecycle, authorization, and observability

- Normal re-ingestion marks old generations/indexes stale but does not mutate
  historical artifacts or References. New References can originate only from
  the current active compatible publication.
- Corpus/index garbage collection may remove stale source artifacts after no
  bridge transaction is using them; self-contained evidence snapshots remain.
- A run is retained while its source chunk, References, or Feedback are
  retained. Deleting a source chunk/document must explicitly delete its runs,
  evidence, and References on SQLite as well as relying on PostgreSQL cascades.
  Unreferenced content-addressed artifacts are then garbage-collected.
- An explicit repository/user privacy purge finds artifacts by repository ID,
  removes affected selections/References and derived Feedback according to the
  product retention policy, and retains at most aggregate non-content audit
  counts. Ordinary rename/delete in a newer corpus does not rewrite history.
- Creation requires an authoritative mapping from `corpus_scope_key` to the
  source document's allowed group/repository scope. Store the sorted group-ID
  hash/snapshot for audit. Reads authorize through the source document's current
  owner/group/publication rules; evidence IDs never grant access by themselves.
- `/load_references` currently lacks a user authorization dependency and cannot
  expose baseline evidence until that is corrected without changing legacy
  response compatibility.
- Safe traces may contain run/request IDs, engine/version, status/error code,
  counts, ordinals, corpus/index IDs and hashes, embedding fingerprint, query
  hash/length/origin, and optionally hashed repository/path identifiers. Never
  log raw query, selected/full content, vectors, credentials, endpoint URLs, or
  unredacted sensitive paths.

## Narrow implementation sequence and tests

1. Add a versioned migration runner and the additive schema. Test fresh and
   upgraded SQLite/PostgreSQL databases, existing legacy rows, constraints,
   downgrade-without-writes, and data-preserving application rollback.
2. Add a pure result validator and publication resolver. Test ambiguous
   `document_id`, full-file versus selected hashes, 12k/16k truncation flags,
   finite full-precision scores, changed-repository exclusion, and stale races.
3. Add the transactional bridge repository. Test every non-`ok` status produces
   zero rows; exact ordered persistence; injected rollback at every insert;
   same-result retries; conflicting idempotency keys; and concurrent retries on
   both databases.
4. Add baseline Reference read/serialization support. Snapshot legacy payloads
   unchanged and assert baseline payload order is ordinal, not insertion/UUID
   order. Test authorization before content loading.
5. Add the structured generation adapter. Spy all provider paths and assert the
   identical evidence order/content, no reranking/fallback/truncation, explicit
   prompt-size failure, generation retry leases, and idempotent Feedback rows.
6. Add deletion, re-ingestion, GC, and privacy-purge services. Test source
   document deletion, sibling rename/delete, stale corpus GC, shared artifacts,
   SQLite orphan audits, and PostgreSQL cascades.
7. Enable writes only for explicit `baseline_v1` after schema capability,
   corpus/index readiness, authorization, and generation-contract tests pass.

## Risks

- The current result's `document_id` name is ambiguous and its content hash is
  not the whole-file hash. Treating either as a Core document/full-file value
  would corrupt provenance.
- Current SQLite deletion and foreign-key behavior can leave orphans unless all
  deletion paths use the bridge lifecycle service.
- Current Reference serializers are unordered and document-centric.
- Current generation helpers strip and label content. Reusing them without a
  baseline-specific contract would violate the post-selection boundary.
- Full-file snapshots increase sensitive-data retention; content addressing,
  authorization, explicit purge, and GC are required before rollout.

## Phase 2B2F prerequisites

Before implementation begins, explicitly approve and freeze:

1. Option A and `baseline-reference-bridge.v1` field/constraint names.
2. The meaning of `RetrievalEvidence.document_id` as index-document ID, or a
   typed replacement carrying index-document and corpus-file IDs separately.
3. The authoritative `corpus_scope_key` to group/repository authorization map.
4. A versioned SQLite/PostgreSQL migration mechanism and SQLite integrity
   strategy; startup best-effort DDL is insufficient.
5. The exact `GenerationEvidence` renderer and provider prompt-size failure
   policy.
6. Product retention/privacy-purge periods for whole-file artifacts and derived
   Feedback.
7. A deletion-path audit that routes every source document/chunk deletion
   through explicit bridge cleanup.

No embedding, index, or ranking blocker remains. The bridge should not be
implemented until these prerequisites are decided and tested as contracts.
