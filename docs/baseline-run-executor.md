# Internal document-level baseline run executor

Phase 2B2L.1D.1C.1 introduces the private worker contract
`baseline-run-worker.v1` for service identity
`compair-core-baseline-runner`. Its only dispatch value is an opaque
`baseline_control_run_job.job_id`. There is no public claim, lease, completion,
or cancellation API and no task adapter in this phase.

## State and lease contract

The worker claims `queued` or `retryable_failed` jobs, or reclaims `running`
jobs whose lease has expired. A claim atomically sets `running`, an opaque lease
token and expiry, attempt count, the fixed worker identity, and the immutable
first-start timestamp. Renewal and every durable outcome require the current,
unexpired token. A stale worker therefore cannot persist evidence after lease
loss.

```text
queued -------------------------------> cancelled
  |                                        ^
  v                                        |
running --infrastructure failure--> retryable_failed
  |   ^                                  |
  |   +------- claim/reclaim ------------+
  +--> references_persisted
  +--> insufficient
  +--> terminal_failed
  +--> blocked
  +--> cancelled
```

`references_persisted`, `insufficient`, `terminal_failed`, `blocked`, and
`cancelled` are terminal for the retrieval executor. `references_persisted` is
the precise handoff to the separately leased coordinated generation service.

## Authorization and protected-input lifetime

Claim and each durable effect revalidate the submitting user, group, source
document, changed-repository registration/approval, active trusted corpus,
succeeded index continuation, compatible publication, and frozen tokenizer,
engine, embedding, index, snapshot-manifest, and corpus-file-manifest
identities. The two manifest hashes are intentionally distinct and both are
bound: the v2 submission carries the sealed snapshot manifest, while the
persistent retriever returns the indexed corpus-file manifest.

The worker authenticates and decrypts the existing AES-256-GCM payload only
after claim. The stored AAD binds the job, scope, source, publication, protocol,
and query provenance. Decryption also checks UTF-8 byte count, character count,
SHA-256, and the parent-secret fingerprint. Authentication or integrity failure
blocks the job and deletes the encrypted payload. Plaintext exists only in
worker-local values while one execution is active; Python memory zeroization is
not guaranteed.

Status, errors, result fingerprints, and object representations exclude the raw
query, ciphertext, nonce, key ID, parent secret, repository paths, evidence
content, and caller idempotency key.

## Retrieval and persistence boundary

The invocation path is:

```text
opaque job ID
  -> lease claim and authorization
  -> authenticated payload decryption
  -> one document-level RetrievalRequest containing the complete Git diff
  -> PersistentBaselineV1Retriever.retrieve
  -> BaselineEvidencePersistenceService.persist
  -> references_persisted
```

The executor does not call `process_document`, `process_text`, a legacy
selector, generation, Feedback, or notifications. The persistent retriever
continues to own full-corpus BM25, pinned BGE scoring, equal RRF-60, the top-six
boundary, deterministic ordering, and the job-wide four-item/16,000-character
budget.

For an OK result, evidence rows, ordered `baseline_file` References, the
one-to-one control-job/run link, safe result fingerprint/counts, state
transition, lease clearing, and payload deletion commit in the existing single
evidence-persistence transaction. The document-level idempotency identity is
HMAC-SHA-256 over the frozen run intent using the decrypted random parent
secret. It is not derived from query or source text.

An insufficient result atomically records `insufficient`, deletes the payload,
and creates no evidence effects. Retryable infrastructure failures retain an
unexpired payload. Blocked, terminal, and cancelled outcomes erase it. Internal
cleanup skips a valid active lease, expires abandoned payloads, and removes any
payload left beside an already durable `references_persisted` job.

## Crash recovery

| Crash point | Recovery |
|---|---|
| Before claim/retrieval | A later worker claims the immutable job. |
| During embedding or before evidence commit | The lease expires or a retryable failure is recorded; retry may recompute against the same publication. |
| Evidence transaction failure | Evidence, References, job link/state, and payload deletion all roll back; payload remains for retry. |
| Commit succeeds but response is lost | Replay loads the linked run and returns the exact selected-evidence and Reference IDs without invoking retrieval. |
| Durable run with residual payload | Replay or cleanup deletes the residual payload. |
| Stale worker after lease loss | Its effect write fails the token/expiry predicate; the current worker may reclaim. |
| Contradictory durable link/count/state | Recovery fails closed as `job_state_incompatible`. |

The Phase 2B2L.1D.1D.1 manual operator composes this executor with the existing
Phase 2B2L.1D.1C.2 generation service. It calls the provider only after
`references_persisted`, consumes exact stored ordinal renderer bytes, and
finishes at `feedback_persisted` (including zero-finding success). Positive
finding runs create a suppressed internal digest; zero-finding runs create no
outbox row. External delivery remains disabled and the operator takes only the
opaque job ID.
