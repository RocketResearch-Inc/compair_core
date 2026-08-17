# Baseline evidence source scope

Migration `0010_baseline_document_source_scope_v1` adds the versioned
`baseline-source-scope.v1` contract to immutable baseline retrieval runs.

`legacy_chunk` preserves the existing Core processing path. A new run requires
the authoritative source document and source chunk at creation. Normal source
deletion may later set either retained provenance pointer to `NULL`; the stored
scope never changes.

`control_document` is reserved for the document-level v2 control-plane path. A
new run requires the authoritative source document, prohibits a source chunk,
and is atomically attached to exactly one leased
`baseline_control_run_job`. The existing nullable
`baseline_control_run_job.persisted_run_id` is the sole relationship direction:
the unique, deferred composite foreign key targets
`baseline_retrieval_run(run_id, group_id)`. No reverse link or synthetic chunk
is created.

The persistence transaction inserts one run, one ordered set of one to four
selected evidence rows, the same number of `baseline_file` References with a
null source chunk, and changes the control job from `running` to
`references_persisted`. It clears the lease and records the one durable run ID
in that same transaction. A failed transaction leaves both the evidence and
job link absent. A replay reauthorizes current document/group/publication state
and returns the same run, evidence, and Reference IDs.

Creation-time triggers enforce the source shape, immutable scope, 16,000
character evidence budget, group-consistent relationship, and attachment
transition. They intentionally allow an existing source document or chunk
pointer to become null through the audited retention policy. Group deletion
cascades the complete scope. Corpus/index deletion remains disconnected from
the copied immutable evidence provenance.

Generation and the notification outbox authorize `control_document` runs from
the current source document, group membership, and control-job relationship;
they never select an arbitrary chunk. Baseline Feedback and Reference source
chunk pointers remain null. Existing legacy-chunk consumers retain their chunk
checks.

The revised unreleased `baseline-preview.v1` is anchored on the authoritative
control job and represents `control_document` with `source.chunk_id=null`.
Its response shape retains a non-null chunk for a historical `legacy_chunk`
representation, but new control-plane persistence continues to require
`control_document` and never creates a synthetic chunk.

Operational rollback is forward-only: disable document-scoped execution and
retain the discriminator and relationship. A destructive downgrade requires a
reviewed backup/export, verification that no `control_document` rows or linked
jobs remain, removal of the triggers/FK/unique constraint, and backend-specific
table reconstruction on SQLite. It is not automatic.
