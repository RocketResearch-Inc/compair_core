# Baseline database worker

Phase 2B2L.1E.0 adds an optional separate Core process that polls the existing
database job records. It introduces no broker and no fourth job state machine.
The sealed-snapshot continuation, compatible-index service, and baseline run
operator remain the only components allowed to claim and execute their jobs.

## Deployment lifecycle

The default remains:

```text
COMPAIR_BASELINE_WORKER_MODE=manual
```

To enable automatic database dispatch, give the API and worker the same Core
database and existing baseline provider/keyring settings, then set:

```text
COMPAIR_BASELINE_WORKER_MODE=database
```

Start a one-job process for operational testing:

```text
compair-core-worker --once
```

Run the durable poller in normal service supervision:

```text
compair-core-worker --poll
```

The command line accepts only one of those lifecycle flags. Provider secrets,
database settings, and model configuration remain environment/secret-manager
configuration. The worker should start after migrations and stop accepting
traffic before the API is switched back to manual mode. Send SIGTERM or SIGINT
for graceful draining; the worker stops selecting new work, finishes the
current service call, records a draining heartbeat, and exits. A hard process
kill relies on the existing per-job lease expiry and reclaim rules.

## Heartbeat and readiness

Migration `0013_baseline_database_worker_v1` owns
`baseline_database_worker_instance`. Its rows contain only an opaque instance
ID, `baseline-database-worker.v1`, supported job-type flags, safe timestamps,
draining state, and bounded capacity counts. They deliberately contain no host,
path, endpoint, environment value, credential, lease, or job payload.

Forward migration `0014_baseline_worker_runtime_attestation_v1` adds the
one-to-one `baseline_database_worker_attestation` row. It records only the
`baseline-runtime-config.v1` version and exact runtime, embedding-identity, and
generation-identity SHA-256 fingerprints. Instance deletion cascades this
operational row. No environment value, URL, DSN, path, key, payload, or job
identity is stored.

The defaults are a five-second heartbeat and 30-second health TTL. SQLite
permits one recent active worker/concurrency slot. PostgreSQL permits multiple
workers; readiness sums their advertised one-process slots. Automatic run or
index submission is ready only when the application runtime is ready, migration
and database validators pass, a recent non-draining worker supports the job and
cleanup types, and pending work is below the capacity threshold. Missing or
full automatic dispatch is `not_ready/worker_unavailable`; there is no fallback
to manual. The API computes its own canonical runtime fingerprint and requires
an exact recent heartbeat match. A healthy but differently configured worker
is counted as mismatched, does not make readiness safe, and re-attests before
selecting any job. Existing exact run submission replays remain read-only and
never extend protected payload expiry. See
[baseline runtime operations](baseline-runtime-operations.md) for the canonical
field and doctor contracts.

## Scheduling and backpressure

Each poll opens a short transaction, observes at most the oldest eligible item
in each of the ingestion, index, and run lanes, and closes the transaction
before calling a service. PostgreSQL selection uses `FOR UPDATE SKIP LOCKED`;
the service lease remains authoritative after the observation lock closes.
Normally the globally oldest lane head wins, with stable job-type and opaque-ID
ties. A run whose encrypted payload expires within 120 seconds is urgent, but
at most three urgent runs are selected consecutively while ingestion/index work
exists. This bounds expiry risk without permanently starving the other lanes.

The default admission threshold is eight pending jobs per healthy concurrency
slot (`COMPAIR_BASELINE_WORKER_MAX_PENDING_PER_SLOT=8`). It is a bounded
capacity rule, not a latency promise. New protected runs are rejected before
write when full. Polling or exact replay does not extend their 15-minute default
payload lifetime. Polling starts at two seconds and internal-failure backoff is
exponential up to 30 seconds. Service attempt counts remain authoritative; at
the configured default maximum of five retryable attempts, the worker uses the
existing terminal state and sanitized `worker_unavailable` reason.

## Crash and recovery

| Boundary | Durable result | Recovery |
|---|---|---|
| API before job commit | no job | caller may resubmit |
| API after job commit, before worker observation | queued job | any healthy worker observes it |
| Worker after observation, before service claim | unchanged eligible job | later poll observes it again |
| Worker after service claim, before service commit | running lease only | existing lease expiry/reclaim |
| Service transaction commits, process dies before response | committed durable boundary | replay returns committed identity/effects |
| Generation provider call succeeds, process dies before commit | provider may have been called twice | existing generation lease/idempotency boundary; Feedback is not duplicated |
| Worker forced while active | no artificial success transition | lease expiry/reclaim |

Cleanup runs only in this process and calls existing staging-expiry and
protected-payload cleanup routines; ordinary scheduling provides lease reclaim
and recoverable post-commit continuation. Stale worker heartbeats are deleted
after the TTL. Cleanup failures and service failures are logged only with
opaque IDs and frozen safe reason codes.

Worker logs may contain job type/state, attempts, safe counts, elapsed time,
and already-permitted fingerprints. They do not contain worker/job UUIDs and
must never contain raw
queries, encrypted fields, source paths/content, evidence, prompts/responses,
Feedback, credentials, idempotency material, nonces, keys, or lease tokens.
