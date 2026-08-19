# Baseline control-document corpus identity

Status: implemented internal contract, version
`baseline-control-document-corpus-scope.v1`.

## Identity

Each document-level control-plane corpus is owned by exactly this tuple:

1. `group_id`;
2. `changed_repository_registration_id`;
3. `source_document_id`.

The scope key is the lowercase SHA-256 digest of the RFC 8785/JCS encoding of
this exact object:

```json
{
  "changed_repository_registration_id": "<opaque registration ID>",
  "group_id": "<opaque group ID>",
  "scope_contract_version": "baseline-control-document-corpus-scope.v1",
  "source_document_id": "<opaque document ID>"
}
```

The stored form is
`baseline-control-document-corpus-scope.v1:sha256:<64 lowercase hex>` and is
bounded by the existing 256-character corpus-scope column. The hash input has
named fields and a version, so it does not depend on delimiter concatenation.
Repository names, local paths, revisions, file content, diffs, and query text
are not inputs.

The source document is the stable control identity. A later immutable snapshot
for the same tuple creates a new generation in the same corpus. A different
source document or changed-repository registration in the same group creates a
different corpus, active-generation pointer, and compatible-index publication.

## Resolution and authorization

The sealed-snapshot continuation derives the identity only after revalidating
the authoritative staging record and registered changed repository. It passes
the selected scope to the trusted ingestion service. Downstream index submit,
index claim/publication, run submit, retrieval, evidence persistence,
generation, capability readiness, and preview independently check the stored
corpus against the same tuple.

Control-document consumers resolve a corpus by its durable ID and then compare
its stored scope, changed-repository registration, and source document. They do
not select an arbitrary corpus for a group. A cross-wired publication or run
fails before evidence effects. The internal collision/corruption reason is
`control_document_corpus_scope_conflict`; public status continues to use the
existing sanitized, schema-compatible error projection.

The legacy chunk processing path is separate. It continues to use its existing
`group:<group_id>` request scope and behavior.

### Audited flow

| Boundary | Previous control-document behavior | Source-specific behavior | Legacy chunk behavior |
|---|---|---|---|
| Sealed continuation | Constructed `group:<group_id>` | Selects exact legacy match or constructs the versioned key | Not used |
| Corpus lifecycle | Generic get/create by supplied scope | Unchanged; receives the authoritative key | Unchanged |
| Active generation | One group-wide corpus pointer | Pointer belongs to the exact corpus ID | Group-only pointer unchanged |
| Index submit/claim/publish/replay | Compared the corpus with a group-only key | Reconstructs the sealed tuple and checks corpus provenance before every boundary | Not used |
| Run readiness | Searched only the group-only key | Considers only publications whose stored tuple matches the requested group | Legacy processing readiness unchanged |
| Run admission | Required the group-only key | Reauthorizes group, registration, document, corpus, generation, and publication together | Not used |
| Retrieval request | Executor manufactured a group-only key | Executor passes the authorized corpus's stored key | `main.py` still supplies the group-only key |
| Persistent retriever | Read by request scope | Unchanged; now receives the exact control scope | Unchanged |
| Evidence persistence/replay | Resolved the group-only corpus | Locks the result corpus ID and checks its exact tuple | Resolves the group-only corpus as before |
| Generation revalidation | Resolved the group-only corpus | Locks the persisted run corpus and checks the control job tuple | Existing chunk and group checks unchanged |
| Status and preview | Relied on durable IDs without a scope comparison | Status retains durable IDs; preview also compares job, run, and stored corpus identity | Existing preview compatibility retained |

Generic corpus ingestion/index primitives remain scope-parameterized and were
not given control-plane authorization semantics. Notification behavior was not
changed.

## Legacy group-only transition

No schema migration or destructive backfill is required.

- If an existing `group:<group_id>` corpus stores the exact changed-repository
  registration and source-document IDs, control ingestion reuses it. Completed
  continuation, index, and run jobs retain and replay their existing durable
  corpus/generation/publication/run identities.
- If that row does not match both stored identities, it is never mutated or
  reassigned. The incoming source uses its new source-specific scope.
- A stored source-specific scope with different provenance is treated as
  corruption or a cryptographic collision and fails closed.
- New control-document corpora always use the versioned source-specific key.

This lazy transition preserves auditable history and permits multiple
independently active source-document corpora in one group without changing the
existing corpus schema.

## Lifecycle matrix

| Operation | Same tuple | Different source or changed registration |
|---|---|---|
| Snapshot ingestion | New immutable generation; same corpus | Separate corpus |
| Activation | Updates only that corpus pointer | Independent pointer |
| Index publication | Replaces only that corpus publication | Independent publication |
| Run admission | Exact corpus/publication required | Cross-wire rejected |
| Retrieval/persistence/generation/preview | Exact stored tuple revalidated | No fallback or reassignment |

Group deletion retains its existing privacy cascade behavior. Normal
re-ingestion does not rewrite old generations, evidence, References, or
Feedback; it only advances the matching corpus active pointer.

## Out-of-scope follow-up

The existing run-result `replayed=false` observability discrepancy remains a
separate follow-up. This corpus-identity change does not alter replay
projection semantics.
