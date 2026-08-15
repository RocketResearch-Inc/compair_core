# Frozen immutable baseline-evidence schema

Status: implemented by Phase 2B2F.1 and corrected by Phase 2B2F.2 as schema
and migrations only. No bridge
writer, reader, serializer, retrieval change, generation adapter, API change,
or baseline finding enablement is included.

This document freezes the approved dedicated-evidence option from
`baseline-reference-bridge.md`. It is authoritative where earlier provisional
names differ.

## Migration and versions

The forward registry contains, in order:

1. `0000_core_schema_baseline`, the existing-schema recognition marker;
2. `0001_baseline_evidence_bridge_v1`, the additive bridge migration; and
3. `0002_baseline_evidence_retention_v1`, the source-lifecycle retention
   correction.

The durable contract constants are:

- bridge schema: `baseline-reference-bridge.v1`;
- provenance schema: `baseline-evidence-provenance.v1`; and
- renderer: `baseline-evidence-renderer.v1`.

The migration creates missing objects, validates the final schema, and records
its checksum and `applied` state in the same transaction. A failure rolls back
the whole pending batch, records a sanitized `failed` state separately, and
fails startup. Migration `0002` uses a reviewed SQLite copy-and-swap solely for
the four FK/nullability corrections; it is transactional, preserves all rows
and explicit schema objects, and must pass `foreign_key_check` before commit.

## Tables

### `baseline_retrieval_run`

A row is the immutable provenance envelope for one future successfully
persisted `baseline_v1` result. It belongs to exactly one `group_id`; nullable
`source_document_id` is used consistently and no ambiguous `document_id`
column exists. It stores:

- caller intent: opaque `idempotency_key`, unique with `group_id`, plus request
  and result schema identity;
- query provenance: kind, SHA-256, length, and `explicit` origin, never raw
  retrieval-query text;
- corpus generation and manifest identity;
- index identity, version, publication fingerprint and publication time;
- engine/config and embedding provider/model/revision/dimension fingerprints;
- versioned authorization-scope fingerprint;
- retrieval and evidence counts; and
- future generation lease/state metadata.

Checks accept only `status = ok`, an explicit nonempty query provenance, one to
four selected items, positive evidence size, fixed schema versions, valid hash
lengths, positive embedding dimension, and valid generation states. The
caller-provided key must be nonempty and cannot equal the query hash; query hash
alone is never the idempotency identity.

Delete rules are `group -> run CASCADE`, while source chunk and source document
provenance use nullable `ON DELETE SET NULL`. Removing normal source objects
does not erase an auditable run.

### `baseline_evidence_artifact`

A row is one immutable whole-file audit snapshot. It belongs to exactly one
group. `(group_id, artifact_key)` is unique, so content-address reuse never
crosses an authorization boundary. It preserves repository ID/name, normalized
relative path, complete content, whole-file content hash, byte/character size,
corpus file/generation/manifest provenance, index publication/document/hash
provenance, and optional Core source document/snapshot provenance.

`source_document_id` is nullable and uses `ON DELETE SET NULL`. Corpus and index
identifiers are copied provenance, not foreign keys. A later re-ingestion,
rename, delete, or stale-index collection therefore cannot erase evidence that
backs an auditable Reference. Group deletion cascades the artifact.

### `baseline_selected_evidence`

A row is the exact evidence item selected for one run. Composite foreign keys
bind both its run and artifact to the same `group_id`. It stores:

- positive one-based `ordinal` and fused rank;
- exact selected content, hash, character count, and truncation flags;
- full-precision BM25, dense, and RRF scores and ranks; and
- renderer version, exact stored renderer output, output hash, and character
  count.

`(run_id, ordinal)`, `(run_id, artifact_id)`, and
`(run_id, selected_content_hash)` are unique. Ordinal is constrained to 1..4.
Selection has its own `group_id -> group CASCADE`. Its relationships to run and
artifact are `ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED`: neither a run
nor artifact can be used as an undeclared historical purge while selected.
One transaction may delete the owning group; the direct group cascade removes
the selection and satisfies both deferred restrictions before commit.

Durable order is always `ORDER BY ordinal`; insertion, UUID, and relationship
iteration order have no semantic meaning.

## Existing-table compatibility

`reference.baseline_selected_evidence_id` is nullable, unique when non-null,
and references `baseline_selected_evidence` with `ON DELETE CASCADE`. Every new
Reference must satisfy exactly one branch:

```text
legacy:   baseline_selected_evidence_id IS NULL
          AND reference_chunk_id IS NOT NULL

baseline: baseline_selected_evidence_id IS NOT NULL
          AND reference_chunk_id IS NULL
          AND reference_document_id IS NULL
          AND reference_note_id IS NULL
          AND reference_type = 'baseline_file'
```

The migration neither rewrites nor backfills legacy rows. Existing normal
Reference persistence continues using the legacy branch.

`feedback.baseline_retrieval_run_id` and
`feedback.baseline_finding_ordinal` are nullable as a pair. When populated, the
pair references the selected evidence `(run_id, ordinal)` with `ON DELETE
CASCADE`; the ordinal is positive and the pair is unique per run. This makes an
explicit selected-evidence retention purge remove its Feedback without making
ordinary run deletion a purge mechanism. Existing Feedback remains unchanged.

`reference.source_chunk_id` and `feedback.source_chunk_id` are nullable and use
`ON DELETE SET NULL`. A before-delete trigger on `chunk` first deletes only
legacy rows (baseline target fields are null); baseline References and Feedback
survive with their source pointers cleared. Core's explicit document/note bulk
deletion helpers apply the same legacy-only predicate before deleting chunks.

PostgreSQL uses named `NOT VALID` constraints followed by explicit validation
for the new nullable foreign keys and Feedback pair. The Reference target check
deliberately remains `NOT VALID`: PostgreSQL still enforces it for every new or
updated row, while pre-`reference_chunk_id` document-only audit rows remain
untouched. SQLite achieves the same new-write policy with insert/update
validation triggers. Every SQLite connection enables `PRAGMA foreign_keys=ON`;
post-migration validation reads `PRAGMA foreign_key_list` directly and verifies
all bridge triggers.

## Lifecycle matrix

| Operation | Run | Selection | Baseline Reference | Baseline Feedback | Artifact |
| --- | --- | --- | --- | --- | --- |
| Delete run while selected | restricted | retained | retained | retained | retained |
| Delete selected artifact directly | retained | retained | retained | retained | restricted |
| Delete source document/chunk | retained; pointers null | retained | retained; source chunk null | retained; source chunk null | retained; source document null |
| Re-ingest/delete corpus or index | retained | retained | retained | retained | retained |
| Delete group | cascade | cascade | cascade | cascade | cascade |

No normal corpus lifecycle operation deletes bridge rows. Product retention,
privacy purge, and unselected-artifact garbage collection require explicit
future services and authorization checks. Legacy References and Feedback remain
source-chunk-owned and are deleted when their source chunk is deleted.

### Explicit retention purge

An explicit retention purge is the sole non-group path allowed to destroy
historical baseline evidence. No purge endpoint or writer exists in this
phase. A future service must require an authenticated, group-authorized data
retention/privacy administrator; enforce any legal hold and minimum-retention
policy; require a reason and external approval/ticket; and emit a durable audit
event outside the rows being erased containing actor, group, reason, approval,
timestamp, affected run/artifact/reference/feedback counts, and a hash of the
purge manifest. The operation must be one transaction in this order:

1. lock and delete selected evidence in the approved run set (cascading only
   its baseline References and Feedback);
2. delete now-unselected runs;
3. delete only approved, now-unselected artifacts; and
4. verify the audited counts before commit.

Direct run/artifact deletion while selected is restricted, so it cannot
silently impersonate this workflow.

## Manual downgrade and failure recovery

The migration system is forward-only. Deploying an older binary leaves these
additive tables and nullable columns in place; that is the normal application
rollback. There is no automatic down migration.

For a failed, uncommitted migration:

1. keep the sanitized failed registry row for diagnosis;
2. stop writers and verify a database backup;
3. verify that transactional DDL left no partial bridge objects;
4. correct the reviewed migration/environment; and
5. remove only that `failed` registry row before retrying the same immutable
   definition, or publish a new corrective migration.

Never rewrite an `applied` row or checksum.

For an explicitly approved destructive downgrade after bridge data exists:

1. disable bridge writes and stop all application writers;
2. take and restore-test a backup;
3. export bridge rows and record counts/content hashes;
4. delete baseline Feedback and References, then selections, runs, and
   artifacts;
5. verify every legacy Reference row and order-relevant field against the
   export;
6. PostgreSQL: drop bridge constraints/indexes/nullable columns/tables in one
   reviewed transaction;
7. SQLite: use a reviewed copy-and-swap for columns that cannot be dropped
   safely, run `foreign_key_check` and integrity checks, and retain the original
   database until verification completes; and
8. record the operational downgrade outside the forward registry.

Schema downgrade is a data-retention decision, not an automatic code rollback.

## Phase 2B2G persistence-bridge prerequisites

The next narrow phase must define and test, before enabling any finding:

1. a typed write command and pure validator mapping each persistent
   `RetrievalResult` index-document ID to its corpus file and complete artifact;
2. an authoritative source chunk/document to single-group authorization
   resolver and exact authorization-scope fingerprint input;
3. canonical group-scoped artifact-key construction and caller idempotency-key
   issuance/retry semantics;
4. the exact renderer used for stored output, including repository/path
   escaping and prompt-size failure;
5. one short transaction that revalidates/locks the active corpus and compatible
   publication, then writes run, artifact, selections, and References in result
   ordinal order;
6. zero-write behavior for every non-`ok` result and stale publication;
7. same/conflicting/concurrent retry and injected rollback tests on SQLite and
   PostgreSQL; and
8. proof that no raw query text, evidence content, credentials, or endpoints
   enter logs, task status, or traces.

Reference reads/serialization, generation, API/CLI behavior, and baseline
finding enablement remain separate later phases.
