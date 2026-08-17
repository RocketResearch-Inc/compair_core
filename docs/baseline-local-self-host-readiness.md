# Baseline local self-host readiness audit

Status: **blocked; not release-ready**
Audit date: 2026-08-17
Core checkpoint (`feature/baseline-v1`): `4a31a47c79a6768319433e4835edb2688d21daae`
CLI checkpoint (`main`): `94031136df4702d1613f0bd62467098d01b4e909`

This is the Phase 2B2M.0 audit updated through the Phase 2B2M.2 local
repository-provisioning closeout for the `baseline_v1` workflow:

```text
scan -> upload -> ingestion -> index -> run -> preview
```

“Clean clone” below means the files recorded by the two revisions above, not
uncommitted files, local excludes, editable installs, caches, or validation
directories. Phase 2B2M.1 confirmed that both checkpoint clones now contain the
baseline implementation. The CLI `main` checkpoint temporarily depends on the
Core `feature/baseline-v1` checkpoint's API and protocol behavior. Do not
publish or automate a mixed-branch release until that compatibility constraint
is removed.

The M2 repository-provisioning implementation described here is currently a
tested working-tree change set, not part of those recorded revisions. Its
“ready” classifications become clean-clone claims only after the Core and CLI
changes are committed, pushed, and rerun from those new revisions.

This audit does not claim benchmark parity. The earlier live validation proved
that individual components can interoperate when manually provisioned; it did
not prove that a new operator can reproduce the workflow from supported release
artifacts.

## Verdict

A technically capable user still cannot complete the entire model-backed
workflow from clean clones under the stated constraints, but repository
provisioning and scan-plan creation are now supported. The CLI exposes
repository register/list/inspect/state/bind, plan creation, and the existing
scan/upload/index/run/preview commands. Core exposes authenticated discovery in
addition to its existing group-admin registration and state services.

The local authority policy is now explicit: the existing
`repository-identity.v1` envelope uses
`authority=compair-local-repository.v1` and a cryptographically random UID
generated once by the CLI. Core's opaque group-scoped registration ID remains
the authority used by manifests. Paths, names, remotes, revisions, root commits,
and the local Git sanity fingerprint do not authorize anything.

The remaining release blockers are a supported strict local generation service, a supported BGE
service package/launcher, a reproducible local service stack, and operational
discovery/diagnostics. The temporary Ollama translation proxy used during
validation is not committed or supported, and Core cannot call native Ollama
directly through the current strict baseline adapter.

## Classification

| Classification | Meaning |
| --- | --- |
| ready | A committed public command or documented service path is complete and usable. |
| usable but undocumented | The committed implementation is usable, but a user would have to inspect code or infer configuration. |
| requires manual/internal action | The implementation exists, but completing the step requires an internal callable, direct API construction, database inspection, or an unbundled helper. |
| missing | No supported implementation or operator path exists. |
| blocked | A security/product decision is required before a safe path can be documented. |

## End-to-end prerequisite audit

| Step | Current implementation state | Finding |
| --- | --- | --- |
| Install Core | **ready** | A clean Python 3.11 install with declared `dev,postgres` extras builds wheel/sdist; a fresh wheel install exposes `compair-core-worker`. |
| Install CLI | **ready** | A clean Go build exposes baseline scan/upload/index/run/preview. It is temporarily paired to the Core feature checkpoint above. |
| Initialize SQLite | **ready** | Import/startup creates the SQLite database and runs migrations `0000` through `0013`. No separate command inspects migration readiness. |
| Initialize PostgreSQL | **requires manual/internal action** | PostgreSQL is supported and tested, but there is no supported baseline-ready compose profile. |
| Start Core API | **ready from source or wheel** | `uvicorn compair_core.server.app:create_app --factory` uses only packaged runtime modules. |
| Create/authenticate user | **ready** | Single-user mode auto-provisions a user/session; account mode has CLI signup/login. Baseline endpoints still require an authenticated identity. |
| Create/select group | **ready** | `compair group create`, `group ls`, `group show`, and `group use` expose the group ID. |
| Create authoritative source document | **ready** | `compair track`, `group files`, and `docs list --json` expose document IDs. |
| Register repositories | **ready** | `baseline repository register/list/inspect/state/bind` uses authenticated Core services and protected local bindings. |
| Configure AES-GCM keyring | **documented but manual** | `.env.example` now shows a deliberately nonfunctional structure. There is still no safe inspect/drain command for rotation. |
| Start pinned BGE | **validation-only helper** | The source-only helper requires a pre-downloaded snapshot; there is no supported downloader, package entry point, image, or supervisor. |
| Configure strict local generation | **missing** | Native Ollama and the bundled legacy service do not implement the strict baseline envelope; the temporary proxy is not committed. |
| Start `compair-core-worker` | **ready**, deployment manual | `compair-core-worker (--once | --poll)` is installed by the wheel. The CLI development compose worker remains unsuitable. |
| Verify capabilities | **requires manual/internal action** | Commands preflight, but no standalone baseline doctor explains all not-ready causes. |
| Create scan plan | **ready**, conditional on active registrations | `baseline plan create` resolves protected bindings, pins Git revisions, reauthorizes registrations, and writes the exact scanner-input contract. |
| Scan | **ready**, conditional | `compair baseline scan --dry-run` emits JSON, is local-only, and fails closed. |
| Upload and ingest | **ready**, conditional | Upload is resumable and the database worker can execute the continuation. |
| Build index | **ready**, conditional | Requires a live, pinned BGE adapter and worker. |
| Run | **ready**, conditional | Requires strict generation, keyring, worker, authorization, and the exact publication; there is no fallback. |
| Preview | **ready**, conditional | Findings are returned in durable ordinal order. |
| Shutdown/restart | **usable but undocumented as one stack** | Individual leases recover; no supported launcher verifies a common configuration. |

## Identifier discovery

No identifier in this workflow should be guessed.

| Identity | Supported discovery | Assessment |
| --- | --- | --- |
| user/session | `compair login`, `compair whoami`; single-user Core session | ready |
| group ID | `compair group ls` or `compair group show` | ready |
| source-document ID | `compair track`, then `compair group files` or `compair docs list --json` | ready |
| repository registration ID | `compair baseline repository register/list/inspect` | ready |
| corpus/generation ID | successful `baseline upload --wait` JSON | ready once upload is possible |
| ingestion continuation/job ID | upload result/status JSON | ready once upload is possible |
| index job/publication ID | `baseline index` result/status JSON | ready once index submission is possible |
| run job and persisted retrieval-run IDs | `baseline run` result/status JSON | ready once run submission is possible |
| notification digest ID | authenticated status/preview data when a digest exists | not required for job-ID preview |

The scanner input field named `repository_id` is the opaque registration ID,
not a Git remote, directory name, or friendly repository name. `baseline plan
create` supplies it only after resolving and revalidating an HMAC-protected
local binding.

## Explicit investigation results

### Repository authorization and provisioning

The current Core tree implements:

```text
POST /baseline/control/admin/v1/repositories/register
POST /baseline/control/admin/v1/repositories/state
POST /baseline/control/admin/v1/repositories/list
POST /baseline/control/v1/repositories/inspect
```

Authorization is based on durable group membership plus `administrator` /
`admin_to_group` relationships. This is a real group-admin boundary, not a
request-supplied role. Ordinary members cannot provision or reactivate a
registration.

The discovery contract is frozen in
`protocol/baseline-repository-discovery.v1.*`. List is group-admin-only;
inspect is available to a current authorized group member because plan and run
submission must resolve an already approved registration. Responses contain
the immutable descriptor and hash, opaque IDs, state, nullable source document,
and timestamps. They contain no local path, remote URL, credentials, audit-user
identity, private idempotency material, content, diff, or query.

The CLI stores bindings under
`~/.compair/state/baseline-repositories/<binding-id>.json`. Each record is
versioned, HMAC-authenticated with the existing installation secret, written
atomically with private permissions, and contains the group, registration,
random UID, descriptor hash, canonical local path, path hash, and a
non-authoritative Git sanity hash. Moving or recloning requires the explicit
`repository bind` operation. Disabling in Core immediately blocks plan and
subsequent control-plane submission while preserving registration and local
audit state.

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

`.env.example` now enumerates the baseline retrieval default, explicit-query
transport exception, control-plane loopback/proxy policy, run gate, worker
timing/capacity/retry values, a deliberately nonfunctional AES-GCM keyring
shape, payload lifetime, pinned embedding identity and limits, strict generation
identity/timeout, notification default-off behavior, and a commented disposable
PostgreSQL test URL. All opt-in or insecure-local switches remain false, and no
working credential is present.

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

Checkpoint package checks found:

- the clean CLI builds with baseline scan/upload/index/run/preview commands;
- the clean Core wheel contains API startup, migrations `0000` through `0013`,
  control-plane v1/v2, database worker, corpus ingestion, index publication,
  run execution, evidence persistence, generation coordination, notification
  outbox, preview, and the `compair-core-worker` entry point;
- runtime code does not open top-level `protocol/`, `docs/`, or `scripts/` paths;
  protocol identities and schemas used at runtime are frozen in package modules,
  while repository tests enforce byte/hash identity against Core and CLI copies;
- `protocol/` and `docs/` therefore remain repository-only specification and
  operator material; and
- `scripts/` remains source-only validation tooling. In particular,
  `live_baseline_embedding_service.py` and its frozen requirements are not a
  supported installed service and are intentionally absent from the wheel.

No supported command depends on a source-checkout-relative path. If the BGE
helper is promoted to a supported self-host service later, it must gain a
packaged entry point or separate versioned distribution rather than silently
relying on `scripts/`.

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

1. **Supported local providers:** package the pinned BGE service and replace the
   temporary Ollama proxy with a maintained strict adapter or direct support.
2. **One truthful service configuration:** ship a complete service/compose
   profile, attest API/worker config agreement, and make readiness diagnostics
   actionable.
3. **Operations and retention:** add stale-resume inspection/cleanup, safe
   key-rotation inspection, and an authorized/audited retention purge policy.

## Minimum implementation phases

### M1 — release and package integrity (checkpoint complete)

The committed baseline checkpoint plus the Phase 2B2M.1 closeout change set has
clean-clone Core wheel/sdist builds, wheel-install worker smoke, CLI command
help, protocol byte/hash checks, package manifest inspection, and full-suite
dependency/test-isolation corrections. This is not a release: CLI `main` still
depends on Core `feature/baseline-v1`, no tag was created, and no publish
workflow was invoked.

### M2 — authorization bootstrap and discovery (implementation complete; commit pending)

The local authority is frozen as `compair-local-repository.v1`: an
administrator approves a random CLI-generated UID, while Core's opaque,
group-scoped registration remains authoritative. Authenticated register,
list, inspect, state, and explicit bind commands now resolve durable IDs into
the existing scanner-input contract. Protected HMAC state detects moved,
recloned, corrupt, and symlinked bindings; paths and remotes never establish or
reach Core as authorization claims. SQLite, PostgreSQL, race, scanner
compatibility, disable/reactivate, and no-disclosure tests cover the workflow.

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

- cloned the remote Core feature branch and CLI main branch into a new temporary
  directory and confirmed the starting checkpoint SHAs above;
- installed Core on Python 3.11 with declared `dev,postgres` extras, built wheel
  and sdist, installed the wheel non-editably, imported API/worker modules, and
  enumerated migrations `0000` through `0013`;
- ran installed `compair-core-worker --help`: exit 0;
- built the clean CLI and ran baseline plus scan/upload/index/run/preview help:
  every command exited 0;
- ran 44 focused Core protocol tests plus the CLI protocol/artifact identity
  tests: pass;
- applied the Phase 2B2M.1 packaging/test-isolation changes in the paired source
  worktrees and ran the final Core suite: 562 passed, 42 skipped, no failures;
- ran CLI `go test ./...`, `go vet ./...`, `go build`, and
  `go test -race ./...`: all passed;
- inspected wheel/sdist member lists and extracted content for credentials,
  databases, local paths, model/cache/state/log/report/environment artifacts,
  bytecode, and test caches; none were present; and
- confirmed every package Python runtime module is present in the wheel and
  that the sdist contains the same runtime source without repository-only tests,
  protocol copies, docs, or validation scripts.

The initial clean Core suite reproduced exactly two failures. The paginated
document test failed because two collection-time unit-test loaders replaced
global SQLAlchemy modules without restoring them; both loaders now restore the
real modules in `finally`. The XGBoost manifest test failed because the declared
development extra installed XGBoost but not its scikit-learn wrapper dependency;
`scikit-learn>=1.4` is now explicit in that extra. The CLI race was two parallel
tests mutating shared notification-gate globals; those two tests now run
serially. These are test/development packaging corrections, not production
retrieval or legacy behavior changes.

Not executed:

- a fresh model download;
- a new live Ollama run, because no committed compatible adapter exists;
- Linux/amd64 validation; and
- an end-to-end baseline workflow, because local repository authority and
  supported model-service decisions remain blocked.

No commit, tag, push, release, or publish action was performed. The only Core
GitHub workflow at this checkpoint is a review workflow, not a package-release
workflow. The CLI release behavior was not invoked; this audit created only
local temporary build outputs.
