# Baseline preview API

`baseline-preview.v1` is an authenticated, read-only view of one completed
document-level baseline control job. The unreleased draft formerly used GET
and interpreted `run_id` as `baseline_retrieval_run.run_id`. That ambiguous
draft is replaced in place: the authoritative identity is now
`baseline_control_run_job.job_id`.

## Request

```text
POST /baseline/preview/v1
Content-Type: application/json
```

```json
{
  "schema_version": "baseline-preview.v1",
  "request_id": "00000000-0000-4000-8000-000000000001",
  "group_id": "00000000-0000-4000-8000-000000000002",
  "job_id": "00000000-0000-4000-8000-000000000003"
}
```

Exactly one of `job_id` or `digest_id` is required. A digest is an alternate
authorized lookup for the same control job. Query-string IDs and the obsolete
GET endpoint are unsupported. Requests use the control-plane's strict UTF-8
JSON parser, 4,096-byte limit, duplicate-key and non-finite-number rejection,
actual-peer HTTPS/trusted-proxy policy, and explicit loopback exception.

## Response

A response is available only for an authorized terminal `feedback_persisted`
job. It contains the control job, its one linked persisted retrieval run,
job-wide evidence and Reference counts, source scope, ordered Feedback, an
optional digest, and safe query/corpus/index/embedding/generation provenance.

For a successful no-findings generation, `feedback=[]`, `feedback_count=0`,
`notification_outbox_count=0`, and `digest=null`. Retrieval provenance and the
positive evidence/Reference counts remain present. No placeholder Feedback is
created.

For positive findings, Feedback is loaded strictly by durable finding ordinal.
The one authorized in-app digest must contain the same ordered Feedback IDs.
Reading a suppressed digest does not change its state, perform delivery, or
create an outbox or `NotificationEvent` row.

The `source.chunk_id` field is null for `control_document`; the response shape
also preserves a non-null chunk ID for any historically retained
`legacy_chunk` representation. New document-level control jobs do not
manufacture source chunks.

## Authorization and privacy

Every read revalidates the exact submitter/caller, explicit group membership,
source-document access, active changed-repository registration and approval,
the linked retrieval-run relationship, evidence-to-Reference manifest,
generation and Feedback fingerprints, and, when present, digest recipient and
immutable finding manifest. Missing, deleted, unauthorized, cross-group, or
contradictory records all return the same generic 404 response.

The response excludes raw retrieval query, source and evidence text, renderer
output, repository paths, encryption metadata, idempotency material, lease
tokens, prompts, provider bodies, credentials, and internal endpoints. Request
and response bodies must not be logged; error records contain only sanitized
codes and the safely parsed request ID.

The frozen contract is in [baseline-preview.v1.md](../protocol/baseline-preview.v1.md)
with its JSON Schema and fixtures in the same protocol tree. Core and CLI keep
byte-identical copies.
