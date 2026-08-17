# Baseline local self-host runbook (blocked draft)

This is the shortest currently identifiable local sequence for
`baseline_v1`. It is an audit artifact, not a supported release runbook.
Unsupported or manual boundaries are called out inline. Do not use this draft
to claim release readiness or benchmark parity.

Read the readiness audit first:
[baseline-local-self-host-readiness.md](baseline-local-self-host-readiness.md).

## Why this runbook cannot complete from a clean clone

At the audited commits:

- the Core clean clone lacks the control-plane worker and most current workflow
  modules; and
- the CLI clean clone returns `unknown command "baseline"`.

The remaining commands in this document describe the current post-Phase source
trees. Those files must first be reviewed, committed, packaged, and released.
Do not copy untracked modules or temporary validation helpers into a clean
checkout.

The sequence then stops at repository registration. The registration contract
requires an approved immutable authority and provider-stable repository UID.
There is no approved rule for deriving those values for a local-only Git
repository. A local path, remote display string, friendly name, or commit is
not authorization. Obtain a reviewed policy before proceeding.

## 1. Obtain and install the sources

Supported legacy source-install commands are:

```sh
git clone https://github.com/RocketResearch-Inc/compair_core.git
git clone https://github.com/RocketResearch-Inc/compair-cli.git

cd compair_core
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev,postgres]'

cd ../compair-cli
go build -o ./compair .
```

**Unsupported today:** at the audited commits, the resulting `./compair` has no
baseline command and the Core install has no `compair-core-worker` entry point.
The remainder requires the reviewed post-Phase source tree.

Check the expected help before doing anything else:

```sh
./compair baseline --help
./compair baseline scan --help
./compair baseline upload --help
./compair baseline index --help
./compair baseline run --help
./compair baseline preview --help
compair-core-worker --help
```

## 2. Choose the database

### SQLite

Core startup creates and migrates the database automatically. Use an explicit,
private location shared by API and worker:

```sh
export COMPAIR_DB_DIR="$PWD/.local-state/core"
export COMPAIR_DB_NAME="compair.sqlite"
```

Do not run SQL or create baseline tables manually.

### PostgreSQL

Set a SQLAlchemy URL for an already administered database:

```sh
export COMPAIR_DATABASE_URL='postgresql+psycopg2://<user>:<password>@127.0.0.1:<port>/<database>'
```

**Unsupported today:** neither repository contains a supported baseline-ready
PostgreSQL launcher. The CLI development compose file is not suitable: its
worker sleeps indefinitely and its model service uses hash embeddings and
heuristic generation. Provisioning PostgreSQL itself is therefore an operator
step outside this runbook. Normal Core startup must be the only schema writer.

## 3. Configure the control plane and worker

API and worker must receive the identical environment. The current
`.env.example` is incomplete; the values below are assembled from focused
documents and are not yet a supported single configuration artifact.

```sh
export COMPAIR_EDITION=core
export COMPAIR_REQUIRE_AUTHENTICATION=false
export COMPAIR_BASELINE_CONTROL_PLANE_ALLOW_INSECURE_LOOPBACK=true
export COMPAIR_BASELINE_WORKER_MODE=database
export COMPAIR_BASELINE_RUNS_ENABLED=true
export COMPAIR_BASELINE_RUN_PAYLOAD_TTL_SECONDS=900

export COMPAIR_BASELINE_EMBEDDING_PROVIDER=http
export COMPAIR_BASELINE_EMBEDDING_ENDPOINT=http://127.0.0.1:9010
export COMPAIR_BASELINE_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
export COMPAIR_BASELINE_EMBEDDING_REVISION=52398278842ec682c6f32300af41344b1c0b0bb2
export COMPAIR_BASELINE_EMBEDDING_DIMENSION=384
export COMPAIR_BASELINE_EMBEDDING_TIMEOUT_SECONDS=10
export COMPAIR_BASELINE_EMBEDDING_BATCH_SIZE=32
export COMPAIR_BASELINE_EMBEDDING_ALLOW_INSECURE_LOOPBACK=true
```

Generate an independent 32-byte AES key with a trusted secret-management
workflow and provide this JSON as a secret, not a committed file or command-line
argument:

```json
{
  "version": "baseline-run-keyring.v1",
  "active_key_id": "<opaque-rotation-id>",
  "keys": [
    {
      "key_id": "<opaque-rotation-id>",
      "key_base64": "<base64-encoded-32-byte-key>"
    }
  ]
}
```

Set the secret as `COMPAIR_BASELINE_RUN_ENCRYPTION_KEYRING` in both API and
worker service managers. See
[baseline-run-query-protection.md](baseline-run-query-protection.md) for the
add-before-remove rotation rule.

**Manual gap:** no supported command proves that an inactive key has no
remaining protected payload references. Do not remove an old key based only on
elapsed wall time.

## 4. Start BGE

The only committed service is the operator validation helper documented in
[baseline-embedding-service.md](baseline-embedding-service.md). It requires the
exact model snapshot to exist already:

```text
model: BAAI/bge-small-en-v1.5
FastEmbed artifact: qdrant/bge-small-en-v1.5-onnx-Q
revision: 52398278842ec682c6f32300af41344b1c0b0bb2
dimension: 384
```

With that snapshot already present, the documented source-checkout sequence is:

```sh
python3.11 -m venv /path/to/baseline-embedding-venv
/path/to/baseline-embedding-venv/bin/pip install \
  -r scripts/requirements-baseline-embedding-live.txt

export COMPAIR_BASELINE_EMBEDDING_SNAPSHOT_DIR='/absolute/path/to/models--qdrant--bge-small-en-v1.5-onnx-Q/snapshots/52398278842ec682c6f32300af41344b1c0b0bb2'
export COMPAIR_BASELINE_EMBEDDING_THREADS=8
export HF_HUB_OFFLINE=1

/path/to/baseline-embedding-venv/bin/uvicorn \
  scripts.live_baseline_embedding_service:app \
  --host 127.0.0.1 --port 9010 --workers 1 --no-access-log
```

In another terminal, from the Core source checkout:

```sh
python scripts/smoke_baseline_embedding.py \
  --endpoint http://127.0.0.1:9010 \
  --revision 52398278842ec682c6f32300af41344b1c0b0bb2
```

**Manual gap:** there is no supported model download/prefetch command, package
entry point, or service image. The helper is not included in the Python wheel.
Do not allow it to resolve or download a model at service startup.

## 5. Start strict local generation

The Core baseline adapter requires a provider that accepts
`baseline-generation-input.v1` and returns:

```json
{
  "content": "{\"schema_version\":\"baseline-generation-output.v2\",\"outcome\":\"findings\",\"findings\":[{\"feedback\":\"...\"}]}"
}
```

The inner object may instead use `outcome: "no_findings"` with an empty
`findings` array. Plain text, blank output, `NONE`, and malformed JSON are
invalid.

**STOP — missing supported service.** Native Ollama does not accept/return this
envelope. The bundled local model also does not satisfy it. The temporary
translation proxy used for development validation is not committed and must not
be copied into this workflow. No safe command can be supplied for this step.

After a production adapter exists, its expected Core settings are:

```sh
export COMPAIR_GENERATION_PROVIDER=http
export COMPAIR_GENERATION_ENDPOINT=http://127.0.0.1:<adapter-port>/<adapter-route>
export COMPAIR_BASELINE_GENERATION_MODEL='<pinned-model-name>'
export COMPAIR_BASELINE_GENERATION_MODEL_VERSION='<immutable-model-or-runtime-revision>'
export COMPAIR_BASELINE_GENERATION_TIMEOUT=30
```

Those settings currently share the legacy HTTP provider selector, while the
baseline request/response shape differs from the legacy contract. A supported
adapter and documentation must make that distinction explicit.

## 6. Start API and worker

Conditional on a compatible generation provider and identical environments:

```sh
uvicorn compair_core.server.app:create_app \
  --factory --host 127.0.0.1 --port 8000
```

Then start the durable worker from the same Core checkout/environment:

```sh
compair-core-worker --poll
```

Graceful SIGINT/SIGTERM stops new claims, drains the current call, updates the
heartbeat, and exits. A hard stop relies on job lease expiry/reclaim.

**Diagnostic gap:** there is no `compair baseline doctor`. API capabilities
are checked internally by index/run commands, but several not-ready causes are
reported only as `capability_unavailable` or `worker_unavailable`. Do not
proceed if capability preflight is not `safe` and `ready` for the requested
operation.

## 7. Authenticate, select a group, and create the source document

Point the CLI profile at the loopback API using the existing profile/Core
configuration convention, then:

```sh
compair login
compair group create
compair group ls
compair group show <group-name-or-id>
compair group use <group-name-or-id>
```

From the changed repository checkout:

```sh
compair track --group <group-id>
compair group files <group-id>
compair docs list --all-groups --own-only --json
```

Record the exact group ID and authoritative repository-document ID returned by
these public commands. Do not infer either from a name or path.

## 8. Register the changed and sibling repositories

The current Core API shape is documented in
[baseline-control-plane-continuation.md](baseline-control-plane-continuation.md):

```text
POST /baseline/control/admin/v1/repositories/register
POST /baseline/control/admin/v1/repositories/state
```

An authenticated current group administrator must create each registration.
The changed registration binds `source_document_id`; sibling registrations use
`null`. The returned opaque `registration_id` becomes the scanner's
`repository_id`.

**STOP — blocked security decision and missing CLI.** There is no CLI command
or list endpoint, and no approved local-only `authority` / `repository_uid`
derivation. Do not use local paths, display names, remotes, or revisions as a
substitute. The remaining sequence is conditional and cannot be claimed as a
clean-clone run.

## 9. Create and scan the immutable plan (conditional)

After approved registration IDs exist, create a strict
`baseline-scanner-inputs.v1` JSON file containing:

- the explicit group ID;
- changed repository local path, registration ID, display name, immutable head
  revision, and authoritative source-document ID;
- one or more sibling local paths, registration IDs, display names, and
  immutable revisions;
- distinct immutable base/head revisions; and
- `dry_run: true` and `json: true`.

Resolve revisions from immutable Git objects, not a dirty working tree:

```sh
git -C /path/to/changed rev-parse <base-ref>^{commit}
git -C /path/to/changed rev-parse <head-ref>^{commit}
git -C /path/to/sibling rev-parse <snapshot-ref>^{commit}
```

Run the scanner:

```sh
compair baseline scan \
  --group <group-id> \
  --plan /path/to/baseline-scanner-inputs.v1.json \
  --dry-run --json > /path/to/scan-report.json
```

The report is safe metadata. The scanner re-reads immutable Git objects and
does not write source/diff bytes to its report.

**Usability gap:** no supported command generates the input plan from listed
registrations/documents, so the JSON must currently be authored by hand.

## 10. Upload and wait for ingestion (conditional)

```sh
compair baseline upload \
  --group <group-id> \
  --plan /path/to/baseline-scanner-inputs.v1.json \
  --allow-loopback-http \
  --wait --timeout 20m --json \
  > /path/to/baseline-snapshot-upload-result.json
```

If interrupted, repeat the identical command with `--resume`. Do not edit the
plan or immutable revisions. The worker reconstructs and validates the sealed
snapshot before atomically activating a complete corpus generation.

## 11. Build and wait for the compatible index (conditional)

```sh
compair baseline index \
  --group <group-id> \
  --upload-result /path/to/baseline-snapshot-upload-result.json \
  --allow-loopback-http \
  --wait --timeout 30m --json \
  > /path/to/baseline-index-result.json
```

If interrupted, repeat with `--resume`. The submitted intent comes from the
server's attested pinned embedding/config identity; there are no model flags.

## 12. Submit one document-level run (conditional)

```sh
compair baseline run \
  --group <group-id> \
  --plan /path/to/baseline-scanner-inputs.v1.json \
  --index-result /path/to/baseline-index-result.json \
  --allow-loopback-http \
  --wait --timeout 30m --json \
  > /path/to/baseline-run-result.json
```

The raw diff is recomputed from the immutable plan and sent only in the
protected POST body. It is never accepted as a command-line argument. One job
runs retrieval exactly once for the complete document-level change set and may
persist at most four ordered References total. `feedback_persisted` is success
with zero or more findings; `insufficient` is a zero-effect terminal outcome.

If interrupted, repeat with `--resume` before the protected payload expires.
Replay does not extend the original payload lifetime.

## 13. Preview ordered findings (conditional)

Read the run job ID from the safe run-result JSON, then:

```sh
compair baseline preview \
  --group <group-id> \
  --job-id <run-job-id>
```

Preview is read-only. It does not count as notification delivery. Findings are
returned by durable ordinal; zero-finding success returns an empty array.

## 14. Retained state and safe shutdown

Stop in this order:

1. stop accepting new baseline submissions;
2. send SIGTERM/SIGINT to `compair-core-worker` and let it drain;
3. stop the API;
4. stop the strict generation adapter;
5. stop the BGE service; and
6. stop PostgreSQL, if used, without deleting its volume.

Retained artifacts include:

- SQLite file or PostgreSQL database;
- sealed snapshot content, active corpus content, selected evidence/renderer
  output, References, and Feedback;
- model snapshot/cache and provider runtime cache;
- safe CLI configuration and HMAC-protected baseline resume state under
  `~/.compair/state`; and
- service logs, whose reverse-proxy/request-body policy is outside Core.

The raw query is retained only as short-lived AES-GCM ciphertext while a queued
run needs it. Successful, terminal, cancelled, corrupt, and expired processing
erases the payload. Query hash/length/origin remain as safe provenance.

There is no general non-group purge for sealed snapshots or historical
evidence. Do not advertise a privacy deletion workflow beyond the implemented
group cascade until authorized/audited retention tooling exists.

## 15. Restart and resume

Restart BGE and the strict generation adapter with the identical immutable
identities, then API and worker with the same database and keyring. Wait for a
fresh worker heartbeat before submission. Re-run interrupted CLI operations
with their original inputs plus `--resume`.

Expected recovery boundaries are documented in
[baseline-database-worker.md](baseline-database-worker.md). A provider call may
occur more than once if the process crashes after the external call but before
the database commit; durable Feedback and Reference uniqueness prevents row
duplication. This is not exactly-once model execution.

**Cleanup gap:** no public command lists or prunes obsolete upload/index/run
resume state after a protocol rollover. Do not delete
`baseline-upload-install-secret.v1`; doing so invalidates all retained replay
identities. An operator cleanup command and runbook are prerequisites for a
supported restart story.

## Release gate

Replace every **STOP**, **unsupported**, **manual gap**, and **usability gap**
above with a committed command or an explicit reviewed operator contract. Then
run the clean-clone acceptance plan in
[baseline-local-self-host-readiness.md](baseline-local-self-host-readiness.md)
on macOS arm64 and Linux amd64, with both SQLite and PostgreSQL, before
publishing this as an end-user runbook.
