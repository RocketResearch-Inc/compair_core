# Baseline compatible-index continuation

Phase 2B2L.1C adds durable orchestration around the existing
`BaselineIndexBuilder` and `BaselineIndexLifecycle.publish`. It does not define
a second index format and does not enable baseline run submission.

## Submission and authority

The internal continuation service accepts the frozen
`baseline-control-plane.v1` `index_build_submit` message. It names an explicit
group, one successfully ingested immutable corpus generation, sealed snapshot
and control-manifest hashes, frozen tokenizer/index versions, and the complete
pinned embedding identity. The caller must be the submitter on the successful
ingestion continuation and remain an active group member.

The frozen schema requires `operations.index_build` to be `unavailable`.
Consequently, `POST /baseline/control/v1/index-builds` returns the safe
`capability_unavailable` error before constructing the continuation service or
creating a job. Manual/internal invocation remains possible for controlled
validation but is not an authenticated control-plane capability. Public
submission requires a new protocol version/hash that can truthfully advertise
`index_build: safe` and its dispatch policy.

An exact `(group, corpus generation, index intent hash)` replay returns the
existing job. Reusing a `(group, operation, idempotency key)` for a different
intent is a conflict. Both replay paths repeat authorization and corpus
eligibility checks.

Repository authority comes only from the approved registrations captured by
the sealed snapshot. Request names, revisions, and paths grant no authority.
Submission, claim, completed-result replay, and publication all recheck the
submitter, group, source document, every registration, snapshot hashes,
ingestion provenance, and active corpus generation.

## State and lease lifecycle

```text
queued ------------------------------> running -----------------> succeeded
  |                                      |   ^
  |                                      |   | expired lease reclaim
  |                                      v   |
  +----------------------------> retryable_failed
  |                                      |
  +----------------------------> terminal_failed
  +----------------------------> cancelled
```

Only an internal worker with service identity
`compair-core-compatible-index` may claim a job. Claims use opaque bounded
leases; expired leases are reclaimable and attempt counts are durable.
Completion, cancellation, and failure transitions require the current
unexpired token. Tokens, idempotency keys, stack traces, provider errors,
repository paths, and content never appear in status.

Revoked authorization, a deleted source, stale/inactive generation, missing
file hashes, or incompatible provenance is terminal for the immutable job.
Transient adapter and staging/publication failures are retryable.

## Builder and publication boundary

The worker attests `baseline-embedding-http.v1` before invoking the existing
builder. The submitted and configured identity must exactly match provider
`baseline_http_v1`, model `BAAI/bge-small-en-v1.5`, its pinned immutable
revision, dimension 384, dtype `float32`, and its fingerprint. Existing finite
float32 and dimension checks remain authoritative; there is no hash fallback.

The builder stages its existing whole-file lexical and dense artifacts against
the immutable generation. Its injected publication callback runs in the same
builder transaction and:

1. locks the index job and, on PostgreSQL, continuation/generation rows;
2. verifies the unexpired lease;
3. repeats user, group, source, repository, active-generation, and complete
   ingestion-provenance checks;
4. verifies the staged build's corpus manifest, counts, versions, retrieval
   configuration, and embedding identity;
5. invokes existing `BaselineIndexLifecycle.publish` to validate and move the
   compatible publication pointer;
6. records only safe result hashes/counts; and
7. marks the control job `succeeded` and clears its lease.

These steps commit together. Any exception after pointer mutation but before
the outer builder transaction commits rolls back the pointer, safe result, and
success transition. The previous compatible publication therefore remains
active on every pre-commit failure.

Only after `BaselineIndexBuilder.build` returns has the outer transaction
committed. A response or process failure after that point cannot demote the job:
the failure recorder requires the old running state, current lease token, and
unexpired lease. Recovery observes both the compatible publication and durable
succeeded job, or neither. Status rejects inconsistent partial state instead of
exposing its publication ID.

## Safe state and capabilities

The extension stores safe corpus/generation/snapshot IDs, versions, manifest
and provenance fingerprints, pinned embedding/index identity, document/token
counts, result manifest hashes, timestamps, and publication ID. Authenticated
job status exposes only its frozen safe subset, including corpus-generation and
successful-publication IDs.

The frozen `baseline-control-plane.v1` capability bytes remain unchanged:
corpus ingestion, index build, and baseline run report `unavailable`. Endpoint
admission and capability construction share one server-side status, so an
unavailable operation cannot enqueue a job. A later protocol revision may
advertise readiness after worker deployment and dispatch policy are health
checked.

## Phase 2B2L.1D prerequisites

- Define a protected authenticated `baseline_run_submit` contract without
  changing frozen v1 bytes, or publish a new protocol version and hash.
- Select only a succeeded index job whose compatible publication remains
  current for the active trusted group corpus.
- Reauthorize caller, group, source, every repository registration, corpus, and
  index fingerprints at submit, claim, and retrieval/persistence commit.
- Create a random parent run identity and replay-safe per-source intent keys;
  never derive either from query or source text.
- Apply protected retrieval-query transport and retain only hash, length, and
  origin in status and traces.
- Invoke the existing persistent retriever and ordered evidence persistence
  path without legacy fallback; every non-OK result remains zero-write.
- Keep generation and notifications behind their later-phase state machines.
- Add SQLite and PostgreSQL authorization, revocation, stale-publication,
  retry/concurrency, privacy, and legacy-regression coverage.
