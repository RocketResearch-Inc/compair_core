# Baseline local self-host readiness audit

Status: **blocked; not release-ready**  
Audit date: 2026-08-17  
Core committed revision: `16b05d162c3bfae63a76aba127930dc606e76019`  
CLI committed revision: `df50c66d8f2ffb106f58d75c7f70a08b6a78326a`

This is a Phase 2B2M.0 audit of the `baseline_v1` local workflow:

```text
scan -> upload -> ingestion -> index -> run -> preview
```

It changes no runtime behavior. “Clean clone” below means the files recorded by
the two revisions above, not uncommitted files present in a developer working
tree. The audit also inspected the current post-Phase working trees to identify
what would become available after those changes are reviewed and committed.

This audit does not claim benchmark parity. The earlier live validation proved
that individual components can interoperate when manually provisioned; it did
not prove that a new operator can reproduce the workflow from supported release
artifacts.

## Verdict

A technically capable user cannot currently complete the workflow from clean
clones under the stated constraints. The first decisive failure is release
integrity:

- the committed CLI has no `compair baseline` command;
- the committed Core has no control-plane worker entry point and does not
  contain most of the current control-plane/run implementation; and
- the current post-Phase source tree can build a Python wheel containing the
  worker and runtime modules, but the loopback BGE helper, protocol artifacts,
  and operator documents are not installed by that wheel.

Even after the current working trees are committed, the public workflow stops
at repository approval. Core has a documented, authenticated group-admin POST
API for registration, but the CLI has no register/list/state commands and no
supported way to obtain the registration IDs for a scan plan. More importantly,
the approved descriptor requires an authority and a provider-stable repository
UID, while no committed security policy defines those values for a local-only
Git repository. A checkout path, remote display name, or revision is explicitly
not authorization. This audit therefore does not invent a mapping.

The remaining release blockers are a supported strict local generation
service, a reproducible local service stack, and operational discovery/
diagnostics. The temporary Ollama translation proxy used during validation is
not committed or supported, and Core cannot call native Ollama directly through
the current strict baseline adapter.

## Classification

| Classification | Meaning |
| --- | --- |
| ready | A committed public command or documented service path is complete and usable. |
| usable but undocumented | The committed implementation is usable, but a user would have to inspect code or infer configuration. |
| requires manual/internal action | The implementation exists, but completing the step requires an internal callable, direct API construction, database inspection, or an unbundled helper. |
| missing | No supported implementation or operator path exists. |
| blocked | A security/product decision is required before a safe path can be documented. |

## End-to-end prerequisite audit

| Step | Clean-clone state | Post-Phase working-tree state | Finding |
| --- | --- | --- | --- |
| Install Core | **ready** for the committed legacy package, assuming dependency download access | **ready** as an editable source install; current wheel includes the Python runtime modules | An offline isolated `pip install .` could not download build dependencies during this audit. `pip wheel --no-build-isolation --no-deps` succeeded. |
| Install CLI | **ready** for legacy CLI | **ready** from source | The committed binary builds but `compair baseline --help` is an unknown command. Current working-tree binary exposes scan/upload/index/run/preview. |
| Initialize SQLite | **ready** | **ready** | Import/startup creates the SQLite database and runs the forward migration registry. No separate command exists to inspect migration readiness. |
| Initialize PostgreSQL | **requires manual/internal action** | **requires manual/internal action** | A PostgreSQL URL is supported and tests have exercised PostgreSQL, but there is no baseline-ready supported compose profile. The CLI development compose uses `postgres:15-alpine`, has a placeholder worker, and is not the baseline stack. |
| Start Core API | **ready** for committed behavior | **ready** from source | `uvicorn compair_core.server.app:create_app --factory` is documented. The published container/`compair core up` path is not pinned to the audited working tree. |
| Create/authenticate user | **ready** | **ready** | Single-user mode auto-provisions a user/session; account mode has CLI signup/login. Baseline endpoints still require an authenticated identity. |
| Create/select group | **ready** | **ready** | `compair group create`, `group ls`, `group show`, and `group use` expose the group ID. |
| Create authoritative source document | **ready** | **ready** | `compair track` creates the repository document; `compair group files` and `compair docs list --json` expose document IDs. |
| Register repositories | **missing** from CLI; API absent in clean Core | **blocked** for local-only identity; otherwise **requires manual/internal action** | The admin POST API exists only in the current tree. There is no CLI register/list/state surface. Local authority/UID policy is unspecified. |
| Configure AES-GCM keyring | **missing** | **usable but undocumented** in the main setup surface | The keyring format and add-before-remove procedure are documented in [baseline-run-query-protection.md](baseline-run-query-protection.md), but `.env.example` omits it and there is no safe inspect/drain command for deciding when an old key can be removed. |
| Start pinned BGE | Helper is committed, but not packaged | **requires manual/internal action** | [baseline-embedding-service.md](baseline-embedding-service.md) documents an operator-only validation adapter and a pre-downloaded snapshot. There is no supported downloader, launcher, image, or compose service. |
| Configure strict local generation | **missing** | **missing** | Native Ollama is incompatible with the current request/response envelope. The validation proxy is temporary and uncommitted. The bundled local model is a legacy hash/heuristic service and is not permitted as a baseline fallback. |
| Start `compair-core-worker` | **missing** | **ready from the source tree/wheel**, but deployment is manual | Current help is `compair-core-worker (--once | --poll)`. The CLI compose worker remains `sleep infinity`. |
| Verify capabilities | **missing** | **requires manual/internal action** | Baseline commands preflight internally, but no standalone baseline doctor/capability command exists. Safe reason codes collapse several causes into `capability_unavailable` or `worker_unavailable`. |
| Create scan plan | **missing** | **requires manual/internal action**, then **blocked** by registration policy | The scanner is pure and deterministic, but the user must hand-author JSON containing opaque registration and source-document IDs. |
| Scan | **missing** | **ready**, conditional on a valid plan | `compair baseline scan --dry-run --json` is local-only and rejects mutable/nonconforming input. |
| Upload and ingest | **missing** | **ready**, conditional on registration, services, and worker | Upload is resumable and the database worker can execute the existing continuation. |
| Build index | **missing** | **ready**, conditional on live pinned BGE and worker | The CLI submits the existing compatible-index continuation and can wait/status it. |
| Run | **missing** | **ready**, conditional on the strict provider, keyring, worker, and exact publication | Document-level retrieval is fail-closed and has no legacy/hash fallback. |
| Preview | **missing** | **ready**, conditional on a completed job | Preview returns findings in durable ordinal order and accepts an explicit group plus job/digest ID. |
| Shutdown/restart | **missing as one stack** | **usable but undocumented** | Individual processes drain/recover by leases. No supported launcher orders startup/shutdown or verifies a common configuration. |

## Identifier discovery

No identifier in this workflow should be guessed.

| Identity | Supported discovery | Assessment |
| --- | --- | --- |
| user/session | `compair login`, `compair whoami`; single-user Core session | ready |
| group ID | `compair group ls` or `compair group show` | ready |
| source-document ID | `compair track`, then `compair group files` or `compair docs list --json` | ready |
| repository registration ID | returned by the admin create API only; there is no list command/endpoint | missing discovery path |
| corpus/generation ID | successful `baseline upload --wait` JSON | ready once upload is possible |
| ingestion continuation/job ID | upload result/status JSON | ready once upload is possible |
| index job/publication ID | `baseline index` result/status JSON | ready once index submission is possible |
| run job and persisted retrieval-run IDs | `baseline run` result/status JSON | ready once run submission is possible |
| notification digest ID | authenticated status/preview data when a digest exists | not required for job-ID preview |

The scanner input field named `repository_id` is the opaque registration ID,
not a Git remote, directory name, or friendly repository name. The current CLI
help and scanner-input documentation do not make the end-to-end acquisition of
that value possible.

## Explicit investigation results

### Repository authorization and provisioning

The current Core tree implements:

```text
POST /baseline/control/admin/v1/repositories/register
POST /baseline/control/admin/v1/repositories/state
```

Authorization is based on durable group membership plus `administrator` /
`admin_to_group` relationships. This is a real group-admin boundary, not a
request-supplied role. Ordinary members cannot provision or reactivate a
registration.

The API contract is documented in
[baseline-control-plane-continuation.md](baseline-control-plane-continuation.md),
but it is not a complete self-host operator experience:

- no CLI command submits the admin request;
- no list/read endpoint discovers existing registrations;
- the normal CLI login does not expose a token for a separate documented curl
  workflow; and
- no policy defines `authority` and `repository_uid` for local-only repositories.

The final item is a security decision. Persisting a checkout path, friendly
name, or request revision as authority would violate the frozen trust model.

### Generation and Ollama

`ReviewerBaselineGenerationProvider` sends local/HTTP providers:

```json
{
  "contract_version": "baseline-generation-input.v1",
  "document": "...",
  "references": ["..."],
  "output_contract": {},
  "idempotency_key": "..."
}
```

It requires an HTTP response whose `content` string is strict
`baseline-generation-output.v2` JSON. Native Ollama endpoints use a different
request and response shape; the bundled local model returns `output`, `text`,
and `feedback`, and may return plain text or `NONE`. Neither satisfies this
contract. Core therefore cannot point directly at Ollama today.

The translation proxy used in live validation was outside the repositories and
is not a supported artifact. A production-capable loopback adapter—or direct
provider implementation with schema enforcement—remains missing. The baseline
path must not use the bundled hash embedding or heuristic generation fallback.

### FastEmbed/BGE

The committed `scripts/live_baseline_embedding_service.py` is contract-correct
for the frozen local validation snapshot:

- model: `BAAI/bge-small-en-v1.5`;
- FastEmbed artifact: `qdrant/bge-small-en-v1.5-onnx-Q`;
- immutable revision: `52398278842ec682c6f32300af41344b1c0b0bb2`;
- dimension: 384 float32 values;
- bind: operator-supplied loopback only; and
- model/config/tokenizer files: SHA-256 checked before service startup.

It is deliberately described as an operator-only validation helper. It refuses
downloads and requires an already populated absolute snapshot directory. The
requirements snapshot is pinned, but the helper and requirements file are not
inside the Core wheel. There is no model acquisition command with size/consent,
service supervision, health wait, Docker image, or macOS/Linux release test.
It is suitable as the implementation seed for a supported adapter, not as the
normal local startup path.

### Configuration consistency

`.env.example` contains only two baseline-related settings: legacy/default
retrieval selection and notifications disabled. It omits all settings required
for a real baseline run:

- control-plane loopback/proxy policy;
- baseline run enablement;
- database-worker mode, polling, heartbeat, capacity, cleanup, and retry values;
- baseline embedding provider, endpoint, pinned revision/dimension, limits, and
  loopback policy;
- AES-GCM keyring and payload lifetime; and
- strict baseline generation model/version/timeout values.

The settings are distributed across focused documents, while the general
README describes the legacy HTTP generation shape (`length_instruction` and a
`feedback` response), not the baseline shape (`output_contract` and a `content`
response). A user following only the main setup material will configure an
incompatible endpoint.

API and worker processes independently construct settings from their own
environments. A database heartbeat attests worker contract/job-type flags,
capacity, and timestamps, but not a hash of database URL, embedding identity,
generation identity, keyring availability, or relevant runtime configuration.
Pinned job intent catches some mismatches later and fails closed; capabilities
can still say a worker is present while its configuration differs from the API.
That is safe against fallback but operationally silent drift.

### Capability truthfulness

Capabilities correctly keep operations unavailable/default-off and enforce
pre-write rejection. They do not explain every not-ready condition precisely:

- index schema/database problems collapse to `capability_unavailable`;
- run schema, keyring, cleanup/executor, and several database checks collapse to
  `capability_unavailable`;
- local/HTTP generation configuration failures collapse to
  `worker_unavailable`; and
- automatic worker absence, queue pressure, heartbeat expiry, or worker config
  mismatch share `worker_unavailable`.

Embedding unavailable versus identity mismatch is distinguishable. The safe
reason vocabulary is intentionally non-reflective, but an authenticated
operator needs a separate diagnostic view that identifies the failed
prerequisite without exposing endpoints or secrets.

### Resume, restart, and cleanup

Upload, index, and run clients persist HMAC-protected safe resume state under
`~/.compair/state`. Successful operations remove their operation record after
writing final JSON; retryable interruptions retain it. Raw diff, source text,
credentials, idempotency keys, and provider bodies are excluded.

There is no supported inspect/prune command for stale state after a protocol
rollover. Upload documentation says an orphaned server staging record must be
resolved before starting over, but no public operator action performs that
resolution. Deleting the shared installation secret is unsafe because it makes
all retained operations unverifiable. Per-operation manual file deletion is
possible but not a documented end-to-end recovery contract.

Worker restart is lease-safe. A graceful stop drains the current process; a
hard stop relies on existing lease expiry and reclaim. A protected run query
expires after the configured 60–3600 second lifetime (900 seconds by default),
so an extended outage can intentionally end in a blocked run. API and worker
must restart with the same database, keyring, and provider identities.

### Encryption-key rotation

The add-before-remove algorithm is sound and documented. It is not fully
operable without internal/database inspection: no supported command reports
whether any unexpired payload still references an inactive key ID, and cleanup
has no public one-shot operator command. Rotation is therefore **usable but
undocumented/manual**, not ready for a clean-clone runbook.

### Platform and packaging

The CLI documents different installation paths for macOS and Linux. Core's
source venv and uvicorn commands are portable POSIX commands; Windows is not in
this audit. The BGE validation environment is pinned to Python 3.11 but has no
documented macOS-arm64 versus Linux-amd64 compatibility matrix. The exact
snapshot ran previously on this macOS host; a clean Linux run was not executed.

Package checks found:

- the committed CLI builds successfully, but contains no baseline command;
- the current working-tree CLI builds with baseline commands because Go embeds
  their source at compile time;
- a clean committed Core wheel contains the earlier retrieval modules only and
  no `compair-core-worker` entry point;
- a current working-tree Core wheel contains `compair_core.worker`, the new
  control-plane modules, and the console entry point; and
- neither Core wheel contains top-level `scripts/`, `protocol/`, or `docs/`
  files. In particular, the BGE service and frozen requirements are not
  installed with `pip install compair-core`.

The CLI development compose stack is not a solution: its worker command is a
placeholder sleep, its model uses deterministic hash embeddings and heuristic
text, and it does not launch the baseline BGE or strict generation services.

## Sensitive retained data

The statement “raw query absent from retained database” needs a precise
boundary. A queued run stores an AES-256-GCM ciphertext, nonce, and opaque key
ID until successful consumption, terminal cleanup, or expiry. It does not
store query plaintext. After retrieval, only query hash/length/origin remain.

Other source material is intentionally durable:

| Location | Sensitive material retained | Lifecycle |
| --- | --- | --- |
| sealed snapshot content parts | canonical JSON containing uploaded sibling file content | immutable audit record; no general purge exists |
| active corpus files | complete supported UTF-8 sibling file content | generation/corpus lifecycle; normal re-ingestion does not erase audit evidence |
| baseline evidence artifacts/selections | complete selected file content and exact renderer output | retained for auditable References; group deletion cascades |
| Feedback | generated finding text | retained in ordinal order; group/evidence retention rules apply |
| run payload | encrypted raw query plus nonce/key ID | short-lived; erased at durable boundaries or cleanup |
| CLI resume state | safe IDs, hashes, counts, timestamps, HMAC | removed on success; retained after retryable interruption |
| normal status/capabilities/outbox | no raw query, evidence, or finding text | safe metadata only |

The design permits group deletion as the strong privacy cascade. There is no
implemented non-group retention/privacy purge for sealed snapshots or baseline
evidence. Access logs must disable request-body capture for control/run routes;
the application cannot enforce reverse-proxy logging policy. An acceptance test
must scan API, worker, model-adapter, and proxy logs for sentinels.

## Minimum usability surface

The smallest useful additions are inspectable primitives, not one opaque
orchestrator.

| Proposed addition | Recommendation | Why |
| --- | --- | --- |
| `compair baseline doctor` | add | One authenticated read should report protocol pins and each prerequisite separately, including API/worker configuration identity agreement. It must not print endpoints or secrets. |
| repository register/list/state CLI | add after local identity policy is approved | This closes the first authorization/ID-discovery gap. Restrict mutation to group admins and make descriptor provenance explicit. |
| local initialization/bootstrap | keep narrow | Add a command that creates/prints safe group/document references and writes an example plan, but do not let it invent repository authority or hide approval. |
| production BGE adapter | promote and package | Reuse the existing protocol/helper; add model acquisition consent, hash verification, health wait, launcher/image, and platform tests. Do not create another protocol. |
| production Ollama adapter | add or implement direct native support | Translate to/from the frozen strict contracts, bind to loopback, redact bodies, and keep schema parsing in Core. Do not use the temporary validation proxy. |
| complete example environment | add | API and worker must consume the same reviewed settings; examples must distinguish legacy and baseline provider contracts. |
| service launcher/Compose profile | add | Launch API, worker, PostgreSQL (optional), BGE, and strict generation with health ordering and persistent volumes. SQLite should remain a supported smaller profile. |
| one end-to-end tutorial | add last | Generate it from the acceptance path so every command is continuously checked. |

`compair doctor` and the proposed baseline doctor should be consolidated where
possible: baseline checks can be a versioned section/subcommand rather than a
second unrelated diagnostic framework. Likewise, promote the existing BGE
adapter instead of adding a second embedding protocol. Remove the placeholder
sleep worker and hash-model service from any profile advertised for
`baseline_v1`; they may remain clearly labeled legacy/demo services.

## Highest-priority gaps

1. **Release integrity:** commit/review the current Core and CLI workflow, make
   package/container contents reproducible, and test a checkout containing only
   tracked files.
2. **Repository authorization and discovery:** approve a local repository
   identity policy, then add admin register/list/state commands and safe ID
   discovery/plan creation.
3. **Supported local providers:** package the pinned BGE service and replace the
   temporary Ollama proxy with a maintained strict adapter or direct support.
4. **One truthful service configuration:** ship a complete environment/compose
   profile, attest API/worker config agreement, and make readiness diagnostics
   actionable.
5. **Operations and retention:** add stale-resume inspection/cleanup, safe
   key-rotation inspection, and an authorized/audited retention purge policy.

## Minimum implementation phases

### M1 — release and package integrity

Likely files:

- Core `pyproject.toml`, package/build manifests, container build files, and CI;
- current Core control-plane/worker/protocol modules and tests;
- current CLI baseline commands, `internal/baseline`, protocol copies, Go module
  files, release configuration, and CI.

Required tests:

- fresh `git clone`/`git ls-files` build with a clean status;
- Core sdist/wheel install and console-entry smoke;
- CLI release binary help and protocol-byte parity;
- assert required runtime/provider assets are present in the selected delivery
  artifact; and
- assert no untracked helper is referenced.

### M2 — authorization bootstrap and discovery

Blocked prerequisite: approve the stable local repository identity authority/
UID policy.

Likely files:

- Core registration list/read contract and authenticated API, if approved;
- CLI `baseline repository register|list|state`, plan-init primitives, and docs;
- safe capability/doctor projections; and
- authorization and audit tests.

Required tests:

- admin/member/cross-group/revoked behavior on SQLite and PostgreSQL;
- local and hosted descriptor normalization/collision cases;
- no path/name/revision establishing authority;
- registration/source/group IDs flow into a plan without guessing; and
- no credentials or descriptors in unsafe logs.

### M3 — production local providers and configuration

Likely files:

- promoted BGE service package/image/launcher and pinned model manifest;
- a maintained Ollama adapter or native provider module;
- `.env.example`, provider docs, health/capability checks, and compose/service
  definitions; and
- API/worker configuration-fingerprint heartbeat fields through a new reviewed
  migration if durable attestation is chosen.

Required tests:

- exact BGE identity and float32 order on macOS arm64 and Linux amd64;
- model download consent, cache hash, offline restart, and corrupt-cache failure;
- positive and zero-finding strict generation output against real Ollama;
- malformed/timeout/unavailable provider behavior and idempotency boundary;
- API/worker configuration mismatch is visible and pre-write fail-closed; and
- no legacy/hash fallback.

### M4 — operations, acceptance, and tutorial

Likely files:

- baseline doctor, resume-state inspect/prune, keyring reference inspection, and
  authorized retention tooling;
- SQLite/PostgreSQL launcher profiles and shutdown/backup documentation; and
- one generated/checked self-host tutorial plus clean-machine CI harness.

Required tests are the acceptance plan below, plus interrupted startup,
shutdown, replay, key rotation, retention authorization/audit, and proxy-log
redaction.

## Clean-clone acceptance plan

This plan is intentionally not executable yet. It defines the release gate.
The harness must run from an empty temporary root and may use only tracked
source plus built release artifacts.

1. Create empty Core source, CLI source, database, CLI home, model cache, log,
   and output directories. Assert no previous Compair processes, containers,
   state, or validation helpers are reachable.
2. Build/install Core sdist and wheel and the CLI release binary from clean
   clones. Assert `git status --porcelain` is empty before and after builds.
3. Run the complete matrix once with SQLite and once with a disposable real
   PostgreSQL service. Apply migrations only through normal Core startup.
4. Use a documented model-acquisition command that reports the exact BGE
   artifact, immutable revision, expected bytes, destination, and hashes before
   downloading. Confirm consent; verify the cached snapshot; start only on
   loopback.
5. Start the committed strict local generation adapter backed by a pinned local
   model. Verify its health contract and both `findings` and `no_findings`
   schema outputs. Record model/runtime revision and hardware, not prompts.
6. Start API and database worker from the same generated, permission-restricted
   environment. Assert baseline doctor reports exact protocol, migrations,
   transport, worker dispatch, embedding identity, generation identity, and
   keyring readiness without endpoints/secrets.
7. Authenticate; create/select a group; create the source document; register
   changed and sibling repositories through public commands; list them; and
   generate the scanner plan from returned IDs. No manual database or JSON-ID
   fabrication is allowed.
8. Run scan, upload with ingestion wait, index with wait, and run with wait.
   Assert automatic database-worker dispatch and one document-level retrieval:
   one retrieval run, one ordered evidence set, 1–4 total References, and the
   job-wide four-item/16,000-character budget.
9. Prove exact pinned BGE, full BM25 plus dense equal RRF-60, top-six selection,
   and explicit `baseline_v1` fingerprints. Assert no hash embedding, legacy
   selector, reranking, or fallback path is called.
10. Run one positive-finding case and one valid zero-finding case. Preview each
    by job ID and assert ordinal order, zero placeholders, and no external
    notification delivery.
11. Place unique sentinels in the raw diff and evidence. Scan CLI state, retained
    database fields intended not to carry raw query, API/worker/adapter/proxy
    logs, status, capabilities, and errors. The query sentinel may appear only
    in the short-lived encrypted payload as ciphertext—not as plaintext. Report
    separately the intentional durable corpus/evidence content retention.
12. Interrupt each lane after submission and during a lease; restart API,
    providers, and worker; use `--resume`; assert stable job/evidence/Reference/
    Feedback identities and no duplicates. Repeat after a provider call before
    commit to demonstrate the documented at-least-once external-call boundary.
13. Rotate the query key by add-before-remove using only supported commands.
    Prove old pending payload recovery, new-key use for new jobs, cleanup, and a
    safe “old key no longer referenced” check before removal.
14. Exercise stale resume-state inspection/prune after a synthetic obsolete
    protocol identity without deleting the installation secret or inspecting
    the database manually.
15. Gracefully stop worker, API, generation, embedding, and database; record the
    exact retained databases, volumes, model cache, CLI state, and logs. Restart
    from those retained artifacts and verify preview plus exact replay.
16. Run the same documented command transcript on macOS arm64 and Linux amd64.
    Validate every command in the published tutorial against `--help` output.

The harness must capture exit status and wall time for every process, archive
only safe logs/results, and fail if any command is absent from the public
documentation.

## Verification performed in this audit

- cloned both repositories locally from their recorded commits into an empty
  temporary directory;
- built the committed CLI from source: exit 0;
- inspected committed CLI help: `compair baseline` returned unknown command,
  exit 1;
- built the current working-tree CLI and inspected help for baseline,
  scan/upload/index/run/preview, group/document, login, doctor, and Core;
- built clean and current Core wheels without dependencies/build isolation and
  inspected their file and entry-point manifests;
- inspected `compair-core-worker --help` from the current source with an
  isolated temporary SQLite database: exit 0;
- traced API routes, authorization relationships, migration startup, provider
  envelopes, capabilities, worker heartbeat, resume stores, and deletion/
  retention schemas; and
- compared `.env.example`, general setup documentation, focused baseline
  documentation, and actual setting reads.

Not executed:

- network-dependent clean `pip install .` (the sandbox could not reach the
  package index for isolated build dependencies);
- a fresh model download;
- a new live Ollama run, because no committed compatible adapter exists;
- Linux/amd64 validation; and
- an end-to-end clean-clone workflow, because the clean clones do not contain
  the workflow and the local repository identity decision is blocked.

Previous validation of the current development tree recorded two unrelated
Core failures:

```text
tests/test_api_load_documents.py::test_load_documents_executes_only_paginated_query
tests/test_reference_reranker.py::ReferenceRerankerTests::test_load_model_resolves_latest_manifest_for_xgboost
```

Neither is on the baseline control-plane path. They do not explain the
clean-clone blockers, but a release gate should still either fix or explicitly
quarantine them. The normal CLI suite passes. The full CLI race run has an
existing global-state race in `TestCollectNotificationGateResultSkipsDropEvents`
and `TestParseWaitTimeoutSeconds`; targeted baseline race tests pass. The race
does not touch baseline scan/upload/index/run state, but it weakens the full
release signal and should be fixed before calling the CLI release clean.

