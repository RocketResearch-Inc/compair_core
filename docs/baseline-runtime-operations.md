# Baseline runtime operations

`baseline-runtime-config.v1` is the privacy-safe compatibility contract shared
by the installed Core API and database worker. It is RFC 8785-canonicalized and
SHA-256 hashed. A worker heartbeat is eligible for automatic dispatch only
when its exact runtime fingerprint matches the API's current fingerprint.

## What the fingerprint means

The canonical input includes fields that can change security or observable
baseline behavior:

- Core, runtime-contract, baseline-engine, tokenizer, whole-file document,
  index, and float32 vector format versions;
- exact BM25 parameters, RRF constant, candidate boundary, evidence item and
  character budgets, token pattern, and stopwords;
- frozen control-plane and generation-output hashes;
- embedding contract/provider/model/revision/dimension/dtype, plus a normalized
  endpoint identity hash and its transport classification;
- generation adapter/provider/model/digest/output contract, deterministic
  generation limits, plus an endpoint identity hash and transport class;
- worker contract, complete supported-job set, mode, heartbeat/poll/cleanup,
  capacity, retry, and backoff settings;
- baseline run and notification enablement, protected-query lifetime, active
  key ID, and a private keyring identity fingerprint;
- database backend, credential-free database identity hash, trusted-proxy
  allowlist hash, loopback policy, and relevant request/provider limits.

The output never contains keys, passwords, tokens, DSNs, endpoint URLs,
filesystem or cache paths, host/user names, query/source/evidence/finding text,
prompts, or model responses. Database and endpoint identity hashes are computed
from normalized credential-free identities. Keyring identity hashes contain a
hash of each decoded key, not key bytes; changing a key changes the runtime
fingerprint, while changing only the active key is also represented explicitly.

API and worker must inherit the same effective environment and database. A
different runtime fingerprint is operational drift: that worker neither makes
automatic dispatch ready nor reaches job selection/claim. Existing per-job
authorization, corpus, publication, provider, and fingerprint checks remain
authoritative after this process-level gate.

## Supported installed commands

After installing `compair-core` (and the `baseline-embedding` extra for the BGE
service), start each Core-owned process from the same environment:

```sh
compair-core-embedding-service --host 127.0.0.1 --port 9010
compair-core-generation verify
compair-core-api --host 127.0.0.1 --port 8000
compair-core-worker --poll
```

The API and embedding service bind loopback by default. A non-loopback API bind
requires `--allow-non-loopback`; production deployments must put it behind a
TLS reverse proxy, list only actual proxy peers in
`COMPAIR_BASELINE_CONTROL_PLANE_TRUSTED_PROXY_ALLOWLIST`, and redact query
strings and bodies from proxy access/error logs. The API does not trust
forwarding headers by default and its installed launcher disables Uvicorn
access logs. It does not make remote plaintext transport safe.

Core never starts Ollama and never downloads a model automatically. Start the
attested Ollama runtime separately and use `compair-core-generation verify`
before starting baseline work. BGE model acquisition remains the explicit
`compair-core-models fetch baseline-v1` action; service startup is offline and
refuses a missing or unverified cache.

SIGINT and SIGTERM initiate Uvicorn graceful shutdown and worker draining. The
worker stops selecting work, completes its active service boundary, writes a
draining heartbeat, and exits; durable job leases cover hard-process failure.

## Doctor contract

The read-only operational diagnostic is:

```sh
compair-core doctor
compair-core doctor --json
compair-core doctor --require-baseline
compair-core doctor --probe-generation
```

The default command performs ordinary database/provider health reads but does
not apply migrations, modify durable state, run generation inference, download
models, or start services. `--probe-generation` alone sends the existing
private-data-free structured-output probe to Ollama.

`baseline-doctor-result.v1` is one JSON object containing overall
`ready|degraded|not_ready`, the runtime fingerprint, safe component statuses
and reason codes, permitted versions/fingerprints/counts, UTC timestamp,
whether generation was probed, and stable operator action codes. JSON mode
writes exactly one value to stdout and nothing diagnostic to stderr. Text mode
writes the same safe facts for an operator.

The exit contract is:

| Code | Meaning |
|---:|---|
| 0 | requested readiness is satisfied |
| 1 | degraded, with complete baseline readiness not explicitly required |
| 2 | invalid configuration or command usage |
| 3 | database unavailable, migration missing/pending/mismatched |
| 4 | embedding unavailable or incompatible under `--require-baseline` |
| 5 | generation unavailable or incompatible under `--require-baseline` |
| 6 | automatic worker unavailable, stale, draining, full, or mismatched under `--require-baseline` |
| 7 | sanitized internal diagnostic failure |

Checks cover configuration safety, database identity/connectivity and migration
state, protected-query keyring and referenced-key availability, expired
payloads, run enablement, exact BGE cache/service identity, exact Ollama
runtime/model/digest and static schema support, optional generation probing,
worker heartbeat/fingerprint/job support/capacity, manual versus automatic
readiness, notification default-off state, retained model staging, safe job
state counts, and threshold-only disk sufficiency. Core intentionally does not
read the CLI repository-binding installation secret.

Stable action codes never embed commands or private values. The usual recovery
order is: correct configuration, restore database connectivity, apply pending
migrations through a normal API/worker startup, restore the query keyring,
verify/start BGE and Ollama, then start a matching database worker. Add a new
query encryption key before selecting it active; do not remove an old key while
doctor reports protected payload references to it. No destructive cleanup or
key-rotation command is provided in this phase.
