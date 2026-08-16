# Baseline notification outbox

Phase 2B2J adds a baseline-only, migration-owned outbox. It does not use or
modify Core's legacy `NotificationEvent` behavior and it has no email, Slack,
webhook, or other external delivery adapter.

## Scheduling contract

A baseline generation transaction first writes ordered `Feedback`, transitions
its `baseline_retrieval_run` to `succeeded`, and inserts one `in_app` digest for
the generation caller. Those operations commit atomically. The caller must
still be a member of the run's explicit group and must retain access to the
source document at scheduling time. Database triggers reject entries for runs
that are not durably `succeeded`.

Migration does not backfill historical succeeded runs: those rows do not
durably identify an authorized recipient. New entries are created only by a
generation success transaction that carries the explicit caller identity.

The outbox payload is `baseline-notification-digest.v1`. It contains only the
ordered `feedback_id` and ordinal pairs plus their count. It never contains
the retrieval query, evidence or source content, generated finding text,
credentials, or provider response metadata. One unique row is allowed for a
`(run_id, recipient_user_id, channel)` tuple, and a stable SHA-256 digest key is
available as a channel-side idempotency key.

`COMPAIR_BASELINE_NOTIFICATIONS_ENABLED` defaults to `false`. A successful run
still records its sole digest in the terminal `suppressed` state, which makes
the default-off policy auditable without creating a deliverable item. Setting
the flag to `true` permits only the injected internal `in_app` dispatcher.
Invalid boolean values fail configuration rather than enabling delivery.

## State and delivery contract

The durable states are `pending`, `running`, `delivered`, `retryable_failed`,
`terminal_failed`, `suppressed`, and `cancelled`. A worker transactionally
leases a pending/retryable/expired-running row. SQLite serializes claims with
`BEGIN IMMEDIATE`; PostgreSQL uses row locking with `SKIP LOCKED`. Membership,
source access, the succeeded run state, and the exact ordered Feedback
identifier manifest are validated when claimed and immediately before the
sink call.

Revoked authorization becomes `suppressed`. Deleted recipient or source
provenance becomes `cancelled`; group deletion follows Core's stronger privacy
rule and cascades the row. Malformed or stale manifests fail terminally. Sink
failures are either retryable or terminal according to the injected sink's
typed error.

Delivery is explicitly **at least once**. A crash after the sink accepts a
digest but before Core records `delivered` can repeat the call. The dispatcher
always provides the stable digest key, but exactly-once effects require the
future in-app channel implementation to enforce that key transactionally.

## Authenticated consumption

The read adapter returns ordered finding identifiers only after rechecking the
recipient, group membership, source access, succeeded run state, and manifest.
An authenticated API may later use those identifiers to load Feedback in
ordinal order. Phase 2B2J does not add that API or change CLI behavior.
