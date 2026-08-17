# Baseline control-plane v2 freeze

This development freeze revises the unreleased v2 artifacts in place before
public run activation. It does not add a run route, worker, migration,
retrieval execution, generation behavior, notification behavior, CLI command,
or legacy-path change.

The immediately preceding unreleased specification draft had SHA-256
`c9486b3deb1a494781513109df17d8e8df1281fbc9687960ace711485b50d174`.
It has no runtime compatibility or downgrade path; an exact request carrying
that hash is rejected as a protocol mismatch. Git history is the archive for
earlier drafts.

The authoritative shared artifacts are:

| Artifact | Raw file SHA-256 |
| --- | --- |
| `protocol/baseline-control-plane.v2.md` | `b278abe007779f05e92509db068f555701c03cba5cf236151e8df231a9b44091` |
| `protocol/baseline-control-plane.v2.schema.json` | `10170faf5cecab1861a0e3c831080cbe1073f437b4c668b55c39dd3be9ca631a` |
| `protocol/fixtures/baseline-control-plane.v2.valid.json` | `d06ea3ab7194c2ef58eea9af555835ed0f1d29eb8a431fb8d5c68976d2b76003` |
| `protocol/fixtures/baseline-control-plane.v2.invalid.json` | `64f06b80f17cc4804f72f8bfd599139dc1ab7e681c9f8d37c244f55612894e3a` |

The RFC 8785 hashes of the parsed valid and invalid fixture values are
`8b43d80e15a84f2bafdfa143a0ddbaa7a9912b63f28586b93a0a7c988f1c8d34`
and `de66f15097d0346d0f66191f91f79e79fac29cedf6bddd7186b0ad847d92f731`,
respectively. Tests pin both the raw file bytes and canonical fixture values.

The separate structured generation-output freeze is:

| Artifact | Raw file SHA-256 |
| --- | --- |
| `protocol/baseline-generation-output.v2.md` | `1dccd3a11ec659a5e8705f9b8acf333a64a21f056265fcd7c96e9c6ac197bb20` |
| `protocol/baseline-generation-output.v2.schema.json` | `39f8e8eaf5e5a219e806d34f46af887d69268a88d5f1d06d45e6c56465e250ed` |
| `protocol/fixtures/baseline-generation-output.v2.valid.json` | `b9781155870350dd8b72619e562ea8da6997125229f2064a39947e71a494b488` |
| `protocol/fixtures/baseline-generation-output.v2.invalid.json` | `489164e6b5f1596134ce0a4e0092dcdc65a80d0fd173870beafa01fe73ea108f` |

The RFC 8785 hashes of its parsed valid and invalid fixture values are
`b428181d7fecbb4c2f6bfca00e120ec3347182fc4ef9c43a4ec50066e9d71336`
and `24126307ddf2257f8cf16f2b9d30a6ed740688653fc7f80c5bf11b2b5a214ed3`.
This contract accepts only structured `no_findings` or ordered `findings`
outcomes; plain text, blank output, and `NONE` are invalid. Production parsing
and generation execution remain unchanged in this phase.

V1 remains independently frozen:

| V1 artifact | SHA-256 |
| --- | --- |
| Specification | `3b45287a54d04cea751e9cc3209c5f0783192de13062e682eadcae40af322650` |
| Schema | `4ea2bbd09c6362b0510cf6cc43dc16f0ec3458fda2525a2409a59d299e801200` |
| Valid messages | `bd89803abcdeac97a57bf0c22b9460cf61be8e0b186b58db8fc0c5cfd3dd60c4` |
| Scanner fixture | `e483e017270aff1997aafce4225e4b4787e643084ffe716dfe36acb40c03c553` |

## Existing-contract mapping

V2 reuses the established `raw_git_diff_v1`, `baseline-index.v1`,
`baseline_v1_frozen_tokenizer.v1`, `baseline-embedding-http.v1`, explicit
group authorization, persistent publication identity, and safe Phase 2A query
provenance. The public state names map to existing retrieval/persistence and
generation boundaries, while the document-level orchestration remains a
future executor responsibility:

| Durable boundary or outcome | V2 public state |
| --- | --- |
| retrieval queued/executing | `queued` / `running` |
| `BaselineProcessingStatus.REFERENCES_PERSISTED` | `references_persisted` |
| `BaselineProcessingStatus.FEEDBACK_PERSISTED` | `feedback_persisted` |
| `RetrievalStatus.INSUFFICIENT` | `insufficient` |
| generation retryable failure | `retryable_failed` |
| generation terminal failure or unrecoverable processing error | `terminal_failed` |
| generation/source/authorization blocked | `blocked` |
| accepted job cancelled before further work | `cancelled` |

The public status renderer must distinguish retrieval failure from a later
generation failure. Only a non-OK retrieval is required to have zero evidence,
Reference, Feedback, generation, and notification effects; a later failure may
accurately report already committed ordered References.

`feedback_persisted` means generation completed successfully and its Feedback
outcome was durably resolved. It permits zero through four ordered Feedback
rows. Zero findings still requires a successful generation lifecycle, one
persisted retrieval run, positive equal evidence and Reference counts, and no
notification outbox row. It never creates placeholder, empty, synthetic, or
`NONE` Feedback.

## Frozen document-level run unit

One v2 run is one complete change-set query for one authoritative source
document. Retrieval runs once, independently of Core's existing per-chunk
`process_document` fan-out. It may create one retrieval run, one ordered
evidence set, 1–4 References total, one generation lifecycle, and ordered
Feedback. The shared evidence budget is four items and 16,000 selected-content
characters. The schema has one `persisted_run_id`, rejects child-run and
per-chunk outcome fields, and advertises the job-wide character cap as a
capability limit without exposing evidence content.

The server derives one document-level persistence identity from its random
parent secret and the immutable intent. New control-plane persistence must be
document-authoritative and must neither require nor manufacture a source chunk.
Existing legacy and per-chunk workflows are outside this contract and remain
unchanged.

## Capability truth table

| Submission | Endpoint | Dispatch | Readiness | Required behavior |
| --- | --- | --- | --- | --- |
| `unavailable` | `unavailable` | any declared safe enum | any declared safe enum | Pre-write HTTP 503 `capability_unavailable`. |
| `safe` | `authenticated_post` | `manual` | `ready` | Authenticated submission exists; an operator starts dispatch. |
| `safe` | `authenticated_post` | `automatic` | `ready` | Authenticated submission and automatic worker dispatch both exist. |
| `safe` | `authenticated_post` | any | `not_ready` or `unavailable` | Endpoint exists but rejects before writes until readiness is restored. |

The schema forbids `safe` with an unavailable endpoint. The canonical fixture
continues to advertise `baseline_run` as unavailable because no v2 run endpoint
or executor is enabled.

## Phase 2B2L.1D.1B.1 schema-alignment prerequisites

Before a document-level executor can write evidence, a forward migration must:

1. align SQLAlchemy metadata with the already nullable retention form of
   `baseline_retrieval_run.source_chunk_id`, `reference.source_chunk_id`, and
   `feedback.source_chunk_id` while preserving the required chunk target for
   every new legacy row;
2. require `source_document_id` for new control-plane retrieval runs without
   rewriting historical per-chunk runs;
3. add a group-consistent, one-to-one durable relationship from one control job
   to its optional `baseline_retrieval_run`, with no child-run table or array;
4. enforce the job-wide selected count of 1–4 and evidence-character count of
   1–16,000 at the document-run persistence boundary;
5. let document-level baseline References and Feedback retain their existing
   selected-evidence/run targets with a null source chunk, while leaving legacy
   Reference/Feedback constraints and deletion behavior unchanged; and
6. validate copied legacy databases plus SQLite/PostgreSQL upgrade, rollback,
   retention, authorization, uniqueness, and restart behavior.

## Executor activation prerequisites

Production implementation requires, in order:

1. migration-owned, group-scoped index and run submission jobs with private
   intent hashes, caller idempotency keys, safe public IDs, leases, and forward
   migration/recovery tests;
2. authenticated POST handlers that reuse the strict control-plane parser,
   actual-peer transport policy, group/source/repository authorization, exact
   version/hash negotiation, pre-buffer limits, and pre-write capability gate;
3. server-side creation of the random parent processing secret and derivation
   of one document-level persistence identity from the immutable run intent,
   with neither value serialized publicly;
4. an index dispatcher that calls the existing index-continuation service and
   a document-level run dispatcher that calls the existing persistent retriever
   exactly once, then document-scoped transactional evidence persistence and
   baseline generation without legacy fallback or `process_document` fan-out;
5. safe status projection implementing the frozen state/effect invariants and
   excluding request text, evidence, findings, keys, leases, endpoints, and
   credentials;
6. SQLite/PostgreSQL authorization, replay/conflict, concurrency, crash,
   retry, zero-write, redaction, and capability/endpoint truth tests; and
7. a separately gated CLI client that first negotiates exact v2 capabilities,
   transports raw diff through protected stdin/file input rather than command
   arguments, and maps the frozen exit classifications without changing legacy
   or preview commands.
