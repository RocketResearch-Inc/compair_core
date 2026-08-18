# Core schema migration foundation

## Scope

Phase 2B2F.0 added the migration registry and runner. Phase 2B2F.1 uses it for
the additive immutable baseline-evidence schema, and Phase 2B2F.2 adds the
forward retention correction. Later baseline-only migrations add the leased
generation state and the notification digest outbox without changing legacy
retrieval or notification behavior.

The previous startup path uses SQLAlchemy `create_all(checkfirst=True)` and a
set of `_ensure_*` helpers. That approach can create a missing table, but it
cannot identify which schema revisions have run, detect edited migration
definitions, serialize concurrent upgrades, or distinguish a complete upgrade
from a partially failed one. Some existing helpers also log and continue after
an error. It is therefore retained only as the compatibility bootstrap for
schemas that predate the registry. New schema changes must use the versioned
runner.

## Registry and upgrade path

`core_schema_migration` contains one row per migration:

- immutable migration identifier and SHA-256 definition checksum;
- runner version;
- `applied` or `failed` state;
- start/finish timestamps; and
- a stable, sanitized error code, never an arbitrary exception or request
  value.

For an existing database created by the current Core startup path, startup now
does the following:

1. Run the existing create-if-missing and compatibility ensure steps.
2. Create `core_schema_migration` if absent under a bootstrap lock.
3. Acquire the migration lock and inspect the bridge-relevant Core and
   persistent-retrieval tables and columns.
4. If the schema is recognized, commit the checksummed
   `0000_core_schema_baseline` marker. No application DDL is performed by this
   marker.
5. On later starts, verify the marker checksum and schema invariants before
   accepting it as applied.

An empty database follows the same route: the existing metadata creates the
current schema first, then the registry records the baseline. A partially
recognized database fails startup with a stable missing-table or missing-column
code. It is not silently declared current.

All pending migrations in one application release run as a single transaction.
An upgrade and its `applied` row commit together. If an upgrade or its validator
fails, the entire pending batch rolls back; a separate transaction records only
the failed migration identifier/checksum and sanitized failure code. The raised
`SchemaMigrationError` is not caught by startup, so workers and API processes do
not run against a partially upgraded schema. A recorded failure blocks retry
until an operator diagnoses the cause and explicitly clears or supersedes the
failed state according to a reviewed recovery procedure.

Editing an applied migration changes its checksum and fails closed. Corrections
must be new, ordered migrations.

## Implemented additive bridge migration

`0001_baseline_evidence_bridge_v1` performs these additive operations:

1. Create immutable evidence/run tables and their indexes.
2. Add any nullable relationship column with a `NULL` default, so existing
   legacy `reference` rows remain valid.
3. Add and validate the foreign key using the backend-specific form below.
4. Create the lookup/order and partial unique indexes.
5. Validate tables, columns, constraints, and indexes before recording the
   migration as applied.

It creates `baseline_retrieval_run`, `baseline_evidence_artifact`, and
`baseline_selected_evidence`; adds nullable bridge target fields to `reference`
and `feedback`; and does not rebuild, rewrite, or backfill legacy rows. The
exact contract and manual downgrade procedure are in
`docs/design/baseline-evidence-schema.md`.

## Implemented retention migration

`0002_baseline_evidence_retention_v1` corrects source-deletion semantics without
rewriting evidence content or adding bridge behavior. Source chunk/document
foreign keys become nullable `SET NULL`; baseline selected evidence directly
group-cascades but restricts run/artifact deletion; baseline Feedback targets
the selected `(run_id, ordinal)`; and a scoped chunk trigger preserves legacy
source-owned deletion while retaining baseline Reference and Feedback rows.
Corpus/index identifiers remain copied values with no lifecycle foreign keys.

## Implemented notification outbox migration

`0004_baseline_notification_outbox_v1` creates the group-scoped
`baseline_notification_outbox`. It stores only an ordered manifest of Feedback
identifiers/ordinals and privacy-safe hashes/state metadata. Composite run/group
ownership cascades with the baseline run, group deletion follows Core's privacy
cascade, and recipient deletion sets the recipient pointer to `NULL` so a
dispatcher can record cancellation. SQLite and PostgreSQL triggers reject rows
for non-succeeded runs and prevent mutation of digest identity/payload fields.
The full state and delivery contract is in
`docs/design/baseline-notification-outbox.md`.

### SQLite

- `BEGIN IMMEDIATE` serializes bootstrap and migration writers. SQLite DDL is
  included in the transaction, so failed DDL rolls back with the pending
  batch.
- A nullable foreign key can be introduced without rebuilding the owner table
  by using `ALTER TABLE ... ADD COLUMN ... REFERENCES ...` with the implicit
  `NULL` default, after creating the referenced table. The index is a separate
  `CREATE INDEX` statement.
- SQLite cannot generally add an independently named table constraint with
  `ALTER TABLE ... ADD CONSTRAINT`. The reference clause must therefore be part
  of `ADD COLUMN`. Core now establishes and tests per-connection
  `PRAGMA foreign_keys=ON` before any bridge constraint or cascade is used.
- Migration `0002` is the reviewed copy-and-swap needed to change existing FK
  actions and nullability. The locked migration connection disables FK actions
  before `BEGIN IMMEDIATE`, copies all columns and explicit indexes/triggers,
  swaps the four affected tables transactionally, then requires a clean
  `PRAGMA foreign_key_check` before commit. It restores and verifies
  `PRAGMA foreign_keys=ON` before returning the connection to the pool. An
  injected failure proves the original tables/data survive and reviewed retry
  succeeds.

### PostgreSQL

- A transaction-scoped advisory lock serializes first-time registry creation;
  an exclusive registry-table lock serializes migration batches.
- PostgreSQL transactional DDL rolls back with the batch.
- Add the nullable column first, then add a named foreign key as `NOT VALID` and
  validate it explicitly. Existing rows are unaffected because the new column
  is null. The Reference XOR check intentionally remains `NOT VALID`, which
  enforces new/updated rows without rejecting historical document-only rows
  created before `reference_chunk_id`. Index creation uses ordinary
  transactional `CREATE INDEX` while the migration lock is held. `CREATE INDEX
  CONCURRENTLY` is intentionally outside this transaction model and would
  require a separately designed migration state machine.
- Migration `0002` uses named `ALTER TABLE` operations: drop only the audited
  old FKs, drop source-column `NOT NULL`, install and validate the replacement
  `SET NULL`/restrictive/composite FKs, and create the scoped chunk trigger.

### Document-scoped baseline evidence

`0010_baseline_document_source_scope_v1` adds the explicit
`baseline-source-scope.v1` discriminator and deterministically labels every
existing retrieval run `legacy_chunk`. It enforces the one-way, one-to-one,
group-scoped `baseline_control_run_job.persisted_run_id` relationship for new
`control_document` runs. SQLite reconstructs only the control-run job table to
add its composite foreign key and unique constraint, then installs creation
guards. PostgreSQL uses additive columns/constraints and equivalent triggers.
Source deletion can still null retained provenance; the source scope and
control relationship remain durable.

### Document-level baseline executor metadata

`0011_baseline_run_executor_v1` adds the fixed internal worker identity,
immutable first-start timestamp, and safe retrieval-result fingerprint to the
existing control run job. Lease token, expiry, attempt count, sanitized reason,
and update/finish timestamps already existed and are reused. SQLite installs
insert/update guards; PostgreSQL adds named checks and an immutable-metadata
trigger. Existing queued audit jobs remain valid with null executor metadata
until first claim. The migration contains no evidence table, run endpoint, or
dispatch behavior.

### Document-level baseline generation coordination

`0012_baseline_control_generation_v1` adds only generation-coordination
metadata to `baseline_control_run_job`: a generation attempt counter, first
start/completion timestamps, provider/model/version and provider-idempotency
attestation, the frozen `baseline-generation-output.v2` version/schema hash,
and exact input/output fingerprints. The existing control-job lease columns
remain the control-side lease authority and receive the same opaque token as
the linked `baseline_retrieval_run` generation lease. Workers lock the control
job before the retrieval run on every claim, revalidation, failure, and success
transaction.

Existing jobs upgrade with `generation_attempt_count=0` and all new metadata
null. SQLite insert/update guards and PostgreSQL named checks reject partial
attempt metadata and contradictory `feedback_persisted` rows. A terminal
`feedback_persisted` row requires an invoked generation, a durable output
fingerprint and completion time, `feedback_count <= reference_count`, and—when
the count is zero—no notification outbox effect. Completed generation metadata
and counts are immutable. The migration adds no public endpoint, retrieval
execution, payload access, preview, or capability enablement.

### Database worker heartbeat

`0013_baseline_database_worker_v1` adds only
`baseline_database_worker_instance`, the privacy-safe operational heartbeat
registry for the optional database-backed baseline worker. Rows contain an
opaque instance UUID, fixed worker contract, supported job-type booleans,
start/heartbeat timestamps, draining state, and bounded concurrency/active
counts. They contain no hostname, path, endpoint, credential, lease token, or
job data. SQLite enforces one recent non-draining worker; PostgreSQL permits
multiple compatible instances. Stale rows are operational metadata and may be
deleted after the configured heartbeat TTL.

`0014_baseline_worker_runtime_attestation_v1` additively creates
`baseline_database_worker_attestation` without changing migration `0013`.
Each row is a one-to-one, cascading extension of a worker instance and stores
the exact `baseline-runtime-config.v1`, embedding identity, and generation
identity SHA-256 fingerprints. The migration validates its contract check,
foreign key, and runtime lookup index on SQLite and PostgreSQL. Failed DDL or
validation rolls back with the migration batch; it stores no endpoint, DSN,
path, secret, protected payload, or job identity.

## Failure recovery and rollback

This is a forward-only runner. “Rollback” has two distinct meanings:

- Before commit, SQLite and PostgreSQL roll back the entire pending migration
  batch automatically. The previous applied schema remains active.
- After commit, application rollback may continue using additive nullable
  tables/columns, but schema downgrade is not assumed safe. Operators should
  stop writers, take and verify a database backup/snapshot, deploy the compatible
  application revision, and use a reviewed backend-specific down procedure only
  if the added schema must actually be removed. Data-bearing evidence tables
  must never be dropped merely because an older binary is deployed.

For a recorded failure, operators should retain the error state for diagnosis,
repair the environmental or schema cause, verify a backup, then either remove
that one `failed` row for an idempotent retry or publish a new corrective
migration. An `applied` row must never be manually rewritten to bypass a
checksum mismatch.

## Phase 2B2G prerequisites

Before implementing bridge writes:

- define the pure result/provenance/authorization validator;
- freeze group-scoped artifact-key and caller idempotency-key semantics;
- revalidate active corpus/index state inside the write transaction;
- persist run/artifact/selection/Reference atomically in explicit ordinal order;
- prove non-`ok`, stale, retry, concurrent, and injected-failure behavior on
  SQLite and PostgreSQL; and
- keep Reference reads, generation, API/CLI changes, and finding enablement out
  of that persistence-only step.
