# Baseline evidence persistence contract

Phase 2B2G introduces one typed write boundary and does not connect it to the
API, task, CLI, retrieval, notification, or generation paths.

## Command and receipt

`BaselineEvidencePersistenceCommand` carries the authoritative `group_id`,
`source_chunk_id`, `source_document_id`, an opaque caller-generated
`idempotency_key`, and one `RetrievalResult`. The result must be
`retrieval-result.v2`, `ok`, and produced by
`baseline_v1.persistent.v1`. It must contain one to four evidence items in
ascending fused-rank order. The service returns the durable run, selected
evidence, and Reference identifiers plus an explicit replay flag.

The canonical group corpus scope is `group:{group_id}`. Query text is not an
input to this service. Only the result's explicit query SHA-256, character
length, and origin are persisted. The idempotency key must not equal the query
hash.

## Frozen materialization contracts

The renderer version is `baseline-evidence-renderer.v1` and renders exactly:

```text
Repository file: {repository_name}/{normalized_relative_path}

{selected_content}
```

The renderer output and its SHA-256 are stored without another truncation.
`selected_content_hash` hashes the exact budgeted selected content, while
`whole_file_content_hash` hashes the immutable complete corpus file.

An artifact key is SHA-256 over canonical sorted compact JSON containing:

- bridge schema version;
- corpus generation ID;
- publication fingerprint;
- index document ID;
- repository ID and name;
- normalized relative path; and
- whole-file content hash.

The database uniqueness boundary remains `(group_id, artifact_key)`, so an
artifact is never shared across authorization scopes.

Before writing, the service reconstructs and verifies every published index
document and every returned candidate against the active trusted corpus. It
also recomputes deterministic lane order, full-precision RRF, top-six
selection, content deduplication, filtering, refill counters, four-item limit,
and the 16,000-character budget. Results created with a non-frozen external
filter are intentionally not persistable.

## Transaction and locking

SQLite starts `BEGIN IMMEDIATE`, serializing evidence writers before
validation. PostgreSQL locks the current authorization rows, corpus,
generation, ingestion provenance, publication, build, and index-state rows in
that order. Corpus activation and index publication already serialize through
the corpus row, so the validated snapshot cannot become stale before commit.

All validation, idempotency checks, and these write stages share one
transaction:

1. `baseline_retrieval_run`;
2. reused or new `baseline_evidence_artifact` rows;
3. `baseline_selected_evidence` rows in explicit ordinal order; and
4. `Reference` rows with `reference_type=baseline_file`, a required source
   chunk, and only `baseline_selected_evidence_id` as their target.

Any exception rolls the transaction back. A same-group, same-key replay first
revalidates current authorization and publication, then compares all persisted
intent, artifacts, selections, renderer output, and References before returning
the prior identifiers. A different intent fails with `idempotency_conflict`.

No `Feedback` is created and generation remains pending and uninvoked.

