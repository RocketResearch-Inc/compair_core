# Baseline local self-host runbook (orchestration draft)

This is the shortest currently identifiable local sequence for
`baseline_v1`. It is an audit artifact, not a supported release runbook.
Unsupported or manual boundaries are called out inline. Do not use this draft
to claim release readiness or benchmark parity.

Read the readiness audit first:
[baseline-local-self-host-readiness.md](baseline-local-self-host-readiness.md).

## Checkpoint and why this runbook still cannot complete

This draft is pinned to:

- Core `feature/baseline-v1` at
  `4a31a47c79a6768319433e4835edb2688d21daae`; and
- CLI `main` at `94031136df4702d1613f0bd62467098d01b4e909`.

Clean clones at those checkpoints contain the baseline API, migrations, worker,
CLI scan/upload/index/run/preview commands, and protocol artifacts. The CLI
checkpoint temporarily depends on the Core feature checkpoint's API and
protocol behavior. This pair is a development checkpoint, not a releasable
cross-branch combination; do not publish or automate it.

Repository registration, scan-plan creation, pinned BGE, and native Ollama
generation are supported after Phase 2B2M.3B. The sequence remains a
development workflow until combined service orchestration and clean-machine
acceptance are complete. Local paths, remote display strings, friendly names,
commits, and filesystem metadata are never repository authorization.

## 1. Obtain and install the checkpoint sources

Clone the paired source-of-truth branches. The checkpoint SHAs above establish
the minimum committed baseline implementation; the branch heads must also
contain the reviewed Phase 2B2M.1 closeout before using this draft.

```sh
git clone --branch feature/baseline-v1 --single-branch \
  https://github.com/RocketResearch-Inc/compair_core.git
git clone --branch main --single-branch \
  https://github.com/RocketResearch-Inc/compair-cli.git

cd compair_core
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install '.[dev,postgres]'

cd ../compair-cli
go build -o ./compair .
```

Check the expected help before doing anything else:

```sh
./compair baseline --help
./compair baseline repository --help
./compair baseline plan create --help
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

API and worker must receive the identical environment. `.env.example` now
enumerates every baseline setting with safe/default-off placeholders. The
values below illustrate the explicit local-development opt-ins needed after
the missing providers and registration policy are supplied; copying them does
not make the stack supported.

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

Create the first local keyring with the installed command. It generates one
independent 32-byte AES-GCM key and opaque key ID, validates the exact
production `baseline-run-keyring.v1` contract, and writes only the fixed shell
assignment:

```sh
compair-core config init
CONFIG_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/compair-core/baseline.env"
set -a
. "$CONFIG_FILE"
set +a
```

When `XDG_CONFIG_HOME` is set, the default is
`$XDG_CONFIG_HOME/compair-core/baseline.env`; otherwise macOS and Linux use
`~/.config/compair-core/baseline.env`. Use `compair-core config init --output
<absolute-path>` to select another destination. The command creates private
parents, publishes a `0600` file without overwrite, and never prints the key.
Load the same secrets fragment into API, worker, and doctor processes. Do not
commit it or pass its contents as an argument.

The initializer is POSIX-only in this checkpoint. It rejects symlinked paths,
unsafe parents, and filesystems that cannot provide the required exclusive
same-directory publication rather than weakening atomicity. Windows operators
must use a deployment secret manager with equivalent ACL and no-overwrite
guarantees; Core fails closed with `platform_security_unsupported`.

See [baseline-run-query-protection.md](baseline-run-query-protection.md) for
the add-before-remove rotation rule.

**Remaining rotation gap:** no supported command proves that an inactive key has no
remaining protected payload references. Do not remove an old key based only on
elapsed wall time.

## 4. Start BGE

Install the supported Python 3.11+ service extra. The wheel includes the
verified manifest and entry points, never the model weights:

```sh
python3.11 -m venv /path/to/baseline-embedding-venv
/path/to/baseline-embedding-venv/bin/pip install \
  "compair-core[baseline-embedding]"
```

The only permitted model identity is:

```text
model: BAAI/bge-small-en-v1.5
FastEmbed artifact: qdrant/bge-small-en-v1.5-onnx-Q
revision: 52398278842ec682c6f32300af41344b1c0b0bb2
dimension: 384
```

Fetch is the sole network-enabled action. It stages each frozen artifact,
checks its exact size and SHA-256, and atomically publishes only the complete
five-file snapshot:

```sh
/path/to/baseline-embedding-venv/bin/compair-core-models fetch baseline-v1
/path/to/baseline-embedding-venv/bin/compair-core-models verify baseline-v1
/path/to/baseline-embedding-venv/bin/compair-core-embedding-service \
  --host 127.0.0.1 --port 9010
```

The default private cache is `~/.cache/compair-core/models`; set
`COMPAIR_BASELINE_MODEL_CACHE` identically for fetch, verify, and service to
choose another absolute cache root. Serving is offline-only and rejects a
missing, partial, corrupt, symlinked, or unexpected artifact set. Interrupted
staging is never eligible and can be removed explicitly with:

```sh
compair-core-models clean baseline-v1 --incomplete
```

From a source checkout, the older fixed-probe smoke helper remains available:

```sh
python scripts/smoke_baseline_embedding.py \
  --endpoint http://127.0.0.1:9010 \
  --revision 52398278842ec682c6f32300af41344b1c0b0bb2
```

The smoke client, live retrieval validator, old validation adapter, and frozen
validation requirements under `scripts/` remain source-only validation tools.
Production instructions use the installed commands above. Top-level
`protocol/` and `docs/` remain repository-only specification/operator material;
the wheel carries the machine-readable model manifest inside the package.

## 5. Start strict local generation

Install Ollama separately and ensure the pinned model is already present. Core
never pulls it. Configure the native provider with its exact immutable digest:

```sh
export COMPAIR_BASELINE_GENERATION_PROVIDER=ollama
export COMPAIR_BASELINE_GENERATION_ENDPOINT=http://127.0.0.1:11434
export COMPAIR_BASELINE_GENERATION_MODEL=qwen3:14b
export COMPAIR_BASELINE_GENERATION_MODEL_DIGEST=sha256:bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8
export COMPAIR_BASELINE_GENERATION_TIMEOUT_SECONDS=300
export COMPAIR_BASELINE_GENERATION_ALLOW_LOOPBACK_HTTP=true
```

This is the supported CPU-only timeout profile. It preserves the qualified
32,768-token context and 1,024-token output limit. Plan for 24 GiB total memory
minimum (32 GiB preferred), approximately 15 GB measured 32K inference
allocation, 25 GB free storage after installation, and 40 GB free during model
acquisition or upgrades. Accelerated deployments may retain the 60-second
default only when their measured provider bound fits it. For routine CPU jobs,
configure the Ollama service with a keep-alive such as
`OLLAMA_KEEP_ALIVE=30m`; Core does not control model acquisition or silently
substitute another model.

Verify runtime, model, and digest without inference, then opt into one benign
strict-schema probe:

```sh
compair-core-generation verify
compair-core-generation verify --probe
```

Both commands emit exactly one safe JSON value. Native Core calls `/api/chat`
directly with the exact packaged `baseline-generation-output.v2` schema; no
translation proxy, legacy generation, automatic pull, or provider fallback is
used. See [baseline-ollama-generation.md](baseline-ollama-generation.md).

## 6. Start API and worker

With the verified provider and identical API/worker environments:

```sh
compair-core-api --host 127.0.0.1 --port 8000
```

Then start the durable worker from the same Core checkout/environment:

```sh
compair-core-worker --poll
```

Graceful SIGINT/SIGTERM stops new claims, drains the current call, updates the
heartbeat, and exits. A hard stop relies on job lease expiry/reclaim.
The worker derives the generation lease from the configured provider timeout.
At the 300-second CPU bound the lease is 360 seconds, reserving 60 seconds for
strict response validation and the atomic Feedback commit.

Before starting the worker, inspect the common environment and database:

```sh
compair-core doctor --json
```

Automatic dispatch must be not ready until a matching worker heartbeat exists.
After starting the worker, require complete baseline readiness:

```sh
compair-core doctor --require-baseline
```

Add `--probe-generation` only when explicitly choosing the synthetic Ollama
schema inference. Doctor never prints endpoints, DSNs, paths, secrets, job IDs,
or private content. Do not proceed if either doctor or capability preflight is
not ready for the requested operation. See
[baseline-runtime-operations.md](baseline-runtime-operations.md).

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

`compair track` prints `doc_id=<uuid>`. The same ID can be rediscovered with:

```sh
compair --group <group-id> docs list --own-only --all-pages --json
```

Record the exact group ID and authoritative repository-document ID returned by
these public commands. Do not infer either from a name or path. `track` is the
existing Core/CLI document workflow; baseline does not create a second document
concept.

## 8. Register the changed and sibling repositories

Run these as a current administrator of the explicit group. For a loopback HTTP
Core, the exception must be explicit on every control command:

```sh
compair baseline repository register \
  --group <group-id> \
  --path /path/to/changed \
  --source-document-id <document-id> \
  --name changed \
  --allow-loopback-http --json

compair baseline repository register \
  --group <group-id> \
  --path /path/to/sibling \
  --name sibling \
  --allow-loopback-http --json

compair baseline repository list \
  --group <group-id> --allow-loopback-http --json

compair baseline repository inspect \
  --group <group-id> \
  --registration-id <registration-id> \
  --allow-loopback-http --json
```

Registration generates a random local UID once and sends only the identity
descriptor, explicit group, and optional source document. It never sends the
path or remote. The server returns its opaque registration ID and the CLI
creates the initial protected local binding.

To bind a moved/recloned or additional working copy, do not register implicitly:

```sh
compair baseline repository bind \
  --group <group-id> \
  --registration-id <registration-id> \
  --path /new/local/working-copy \
  --allow-loopback-http --json
```

The binding state is under
`~/.compair/state/baseline-repositories/`, uses mode 0700/0600 on POSIX, atomic
replacement, symlink rejection, and HMAC integrity through
`~/.compair/state/baseline-upload-install-secret.v1`. Do not edit or copy a
binding file independently of its installation secret.

## 9. Create and scan the immutable plan (conditional)

Create the exact existing `baseline-scanner-inputs.v1` plan. Base and head may
be local refs; the command resolves them to immutable commit IDs and pins each
sibling's current `HEAD`. It performs authorization and Git metadata checks but
does not scan or upload:

```sh
compair baseline plan create \
  --group <group-id> \
  --changed /path/to/changed \
  --base <base-ref-or-commit> \
  --head <head-ref-or-commit> \
  --sibling /path/to/sibling \
  --output /private/path/baseline-scanner-inputs.v1.json \
  --allow-loopback-http --json
```

Repeat `--sibling` for additional repositories. Existing output is never
silently replaced; add `--overwrite` only after reviewing the target.

Run the scanner:

```sh
compair baseline scan \
  --group <group-id> \
  --plan /path/to/baseline-scanner-inputs.v1.json \
  --dry-run > /path/to/scan-report.json
```

The report is safe metadata. The scanner re-reads immutable Git objects and
does not write source/diff bytes to its report.

Disable/reactivate testing is explicit and preserves audit history:

```sh
compair baseline repository state \
  --group <group-id> --registration-id <sibling-registration-id> \
  --active=false --allow-loopback-http --json

# plan create and subsequent submission now fail closed

compair baseline repository state \
  --group <group-id> --registration-id <sibling-registration-id> \
  --active=true --allow-loopback-http --json
```

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
