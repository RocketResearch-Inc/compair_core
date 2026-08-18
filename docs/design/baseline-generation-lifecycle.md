# Baseline generation lifecycle

Phase 2B2I adds `baseline-generation-input.v1` and the forward migration
`0003_baseline_generation_state_v1`. It applies only to persisted
`baseline_v1` retrieval runs. The legacy retrieval, generation, Feedback, and
notification path is unchanged.

## Input contract

Generation starts from a persisted `baseline_retrieval_run`. The adapter joins
its `baseline_selected_evidence` and `baseline_evidence_artifact` rows and
always executes `ORDER BY baseline_selected_evidence.ordinal ASC`. Each typed
input item carries the ordinal, repository identity/name, relative path,
selected/full-content hashes, corpus generation, index publication identity,
and exact stored `renderer_output`.

The adapter verifies the stored content, byte/character counts, hashes,
renderer version, renderer reconstruction, contiguous ordinals, and run-level
corpus/index identity. It neither creates `Chunk` objects nor reranks, filters,
deduplicates, clips, or otherwise rewrites the evidence. The provider receives
each renderer output as one unchanged string in that exact order. A provider
whose context limit cannot accept the input must fail explicitly; Core does not
truncate it.

The input fingerprint is a length-delimited SHA-256 digest over the exact
source text, ordered renderer bytes, evidence identities/hashes, group/source
identity, and persisted query/corpus/index fingerprints. Raw inputs are not
written to generation state or logs.

Source authorization is scope-aware. `legacy_chunk` inputs require the retained
authoritative chunk. `control_document` inputs require a null chunk and use the
authoritative document, group membership, and unique control-job relationship;
the adapter never substitutes an arbitrary document chunk.

## Structured output contract

Baseline providers receive the frozen `baseline-generation-output.v2` version
and schema SHA-256 plus `maximum_findings=min(4, reference_count)`. Their
transport envelope must return model content as a string containing exactly one
JSON object. Core rejects plaintext, blank/`NONE`, malformed JSON, duplicate
keys, non-finite values, unknown properties, blank findings, outcome/count
contradictions, and excessive findings. There is no compatibility fallback to
the former plaintext separator convention.

`no_findings` with an empty array is a successful resolved generation outcome:
the retrieval run succeeds, the control job reaches `feedback_persisted`, and
zero Feedback/outbox rows are created. A positive `findings` array is persisted
verbatim in array order as ordinal 1–4 and retains the existing internal outbox
scheduling behavior. Neither outcome invokes legacy `get_feedback`.

The supported native provider uses adapter contract
`baseline-generation-ollama-http.v1` and Ollama's nonstreaming `/api/chat`
directly. It sends the exact packaged output schema, reattests the configured
model tag's immutable digest before every evidence-bearing request, disables
thinking/streaming/tools, and uses deterministic bounded decoding. It neither
pulls a model nor falls back to the generic HTTP provider or legacy generation.
The separate generic strict HTTP adapter remains an explicit configuration.

The existing durable provider/model/version fields are sufficient: the native
version binds adapter contract, Ollama runtime, immutable digest, and output
specification hash; the existing schema and provider-fingerprint fields bind
the output schema and complete provider identity. No schema migration is
needed.

## State machine

The durable states on `baseline_retrieval_run` are:

| Current state | Event | Next state |
| --- | --- | --- |
| `pending` | lease acquired after authorization/provenance validation | `running` |
| `retryable_failed` | retry lease acquired | `running` |
| `running` | unexpired competing claim | remains `running`; caller gets `generation_lease_active` |
| `running` | expired lease is reclaimed | `running` with a new token and incremented attempt |
| `running` | provider or transient database failure | `retryable_failed` |
| `running` | malformed/unsupported provider output | `terminal_failed` |
| `running` | authorization, source, group, corpus, or publication invalid | `blocked` |
| `running` | Feedback rows and completion commit together | `succeeded` |
| `succeeded` | retry | remains `succeeded`; prior receipt is returned |

Lease acquisition uses `BEGIN IMMEDIATE` on SQLite and a row lock on
PostgreSQL. The lease stores a random token, expiry, start/update timestamps,
attempt count, provider/model/version, and the exact input fingerprint.
Authorization and current corpus/publication fingerprints are checked both
before the external call and again inside the Feedback commit transaction.
Only sanitized error codes and an error-class/code hash are durable; exception
messages and provider payloads are not.

Feedback insertion and the transition to `succeeded` are one transaction.
`(baseline_retrieval_run_id, baseline_finding_ordinal)` remains the durable
uniqueness boundary, so retries cannot duplicate findings. Each baseline
Feedback row stores provider/model/version plus input/output fingerprints.
No Notification row or event is created in this phase.

For document-level control jobs, migration
`0012_baseline_control_generation_v1` coordinates this authoritative
retrieval-run lifecycle with the existing control-job lease. The lock order is
always control job, then linked retrieval run. The same opaque token and expiry
guard both rows. Provider invocation remains outside the final transaction;
Feedback insertion, retrieval-run success, optional positive-finding outbox,
and control-job `feedback_persisted` completion commit atomically. The
encrypted query payload is required to be absent and is checked only by row
existence; generation never decrypts or reconstructs it.

## External-call duplication boundary

Core cannot make an external model call exactly once across a process crash.
A worker can receive a response and crash before the Feedback transaction;
after lease expiry, another worker must call the provider again. Core supplies
a stable, content-free provider idempotency key derived from the run and input
fingerprint. A provider that durably honors that key can collapse the repeated
call. Without provider-side idempotency, at-least-once external invocation is
unavoidable, while Core's Feedback writes remain idempotent.

## Operations

`0003_baseline_generation_state_v1` is forward-only. PostgreSQL adds nullable
columns and replaces the old generation-state check transactionally. SQLite
uses the migration runner's transactional copy/swap while preserving the Phase
F retention foreign keys, indexes, and triggers. Operational rollback means
disabling baseline generation and retaining the additive columns/data; a
destructive downgrade requires the reviewed backup/export procedure already
defined for the baseline evidence bridge.
