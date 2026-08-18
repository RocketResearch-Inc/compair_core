# Baseline run protected-query storage

Phase 2B2L.1D.1D.1 exposes the frozen authenticated POST submission and status
messages behind `COMPAIR_BASELINE_RUNS_ENABLED`. The setting defaults to
`false`. Disabled deployments advertise `baseline_run=unavailable`, reject
both run endpoints before any job/payload write, and create no dispatch work.

When explicitly enabled, submission is admitted only after read-only runtime
checks confirm migration `0012_baseline_control_generation_v1`, its tables and
constraints, database connectivity, the AES-GCM keyring, the internal executor
and cleanup/recovery methods, a live attested baseline embedding identity, a
current authorized compatible publication for the requested group, and a
configured generation provider compatible with the strict
`baseline-generation-output.v2` parser. With the default
`COMPAIR_BASELINE_WORKER_MODE=manual`, the truthful ready operation is:

```json
{
  "submission": "safe",
  "endpoint": "authenticated_post",
  "dispatch": "manual",
  "readiness": "ready",
  "reason_code": null
}
```

Enabled deployments with a missing prerequisite advertise `not_ready` plus a
frozen safe reason and reject new run submission before writes. An authorized
exact replay remains read-only and does not refresh its payload lifetime.

`COMPAIR_BASELINE_WORKER_MODE=database` opts into the separately deployed
database worker. Only a recent compatible heartbeat, validated migration,
required job-type support, complete runtime readiness, and bounded queue
capacity change dispatch to `automatic` and readiness to `ready`. A missing or
full worker advertises `automatic/not_ready/worker_unavailable`; it never falls
back to manual dispatch. See `docs/baseline-database-worker.md`.

## Authenticated encryption and dependency

Core uses `cryptography==49.0.0` and its `AESGCM` recipe with a 256-bit key and
a fresh 96-bit cryptographically random nonce for every new payload. The
maintained recipe supplies authenticated encryption and avoids maintaining a
local cryptographic primitive. Core performs no implicit normalization of the
raw UTF-8 query before encryption.

The ciphertext additional authenticated data is RFC 8785 canonical JSON with
these exact fields:

- `aad_version`;
- `payload_schema_version`;
- `job_id`;
- `group_id`;
- `submitted_by_user_id`;
- `source_document_id`;
- `changed_repository_registration_id`;
- `corpus_generation_id`;
- `index_publication_id`;
- `protocol_version` and `protocol_sha256`;
- `query_sha256`, `query_byte_length`, and `query_origin`.

Decryption verifies the authentication tag, UTF-8 query hash, character and
byte lengths, and parent-processing-secret fingerprint. Authentication failure
blocks the safe job and deletes the envelope. It never returns cryptographic
diagnostics or plaintext.

## External keyring

`COMPAIR_BASELINE_RUN_ENCRYPTION_KEYRING` is a secret-managed JSON value. For
first-time local POSIX setup, generate the exact production shape without
printing it:

```sh
compair-core config init
CONFIG_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/compair-core/baseline.env"
set -a
. "$CONFIG_FILE"
set +a
```

The resulting value has this structure (the command never emits the secret
fields shown here):

```json
{
  "version": "baseline-run-keyring.v1",
  "active_key_id": "opaque-rotation-id",
  "keys": [
    {
      "key_id": "opaque-rotation-id",
      "key_base64": "<base64-encoded 32-byte key>"
    }
  ]
}
```

The local file is `0600`, no-overwrite, and symlink-rejecting. Load the same
fragment into API, worker, and doctor, or supply an equivalent value through a
deployment secret manager. Never commit it or pass it as a command-line
argument. Key IDs are opaque and may appear only in the safe local initializer
result and runtime attestation; secret keys are never returned through
capabilities, status, errors, logs, reprs, task metadata, or initializer output.
Missing, malformed, duplicate, short, or unknown keys fail closed.

Rotation is an add-before-remove procedure:

1. Generate a new independent 32-byte key in the secret manager.
2. Retain every key referenced by an unexpired payload, append the new key, and
   make only the new key ID active.
3. Restart Core and verify protected envelopes authenticate. Exact submission
   replays retain their original nonce and ciphertext; only newly accepted jobs
   use the active key.
4. Run expiry/terminal cleanup. Remove an inactive key only after no protected
   payload row references it.

Losing a retained key makes its payload unrecoverable and fail-closed. Keys are
never stored in the database.

## Lifetime and erasure

`COMPAIR_BASELINE_RUN_PAYLOAD_TTL_SECONDS` defaults to 900 seconds and is
bounded to 60–3600 seconds. Cleanup skips an unexpired active lease. Once the
lease expires, an expired pending payload is deleted and its safe audit job is
marked internally `blocked` with `payload_expired`. Because the frozen v2 safe
reason enum does not contain that new code, the frozen status projection uses
`worker_unavailable`; the internal audit reason remains exact.

Terminal, cancelled, successfully consumed, corrupt, and expired jobs must not
retain a protected payload. Cleanup is internal and idempotent; no public
endpoint exposes cleanup or lease control.

The trusted manual callable is
`compair_core.compair.retrieval.run_operator.process_baseline_run_job(job_id)`.
Its only argument is the opaque job UUID. It composes the existing document
retrieval/Reference executor and coordinated generation/Feedback service. It
accepts no query, ciphertext, key, source path, evidence, prompt, or provider
body. There is no Celery dispatch, import-time thread, or public claim API.

The submission endpoint accepts no URL query string and enforces the frozen
8,100,000-byte request limit and 8,000,000-byte raw UTF-8 query limit before
processing. Status reauthorizes the original submitter, group, source,
repository approval, and publication and returns only frozen safe fields.
Neither endpoint returns or logs raw queries, ciphertext, nonces, key IDs,
parent secrets, caller idempotency values, evidence/renderer bytes, provider
prompts/responses, Feedback text, or lease data.
