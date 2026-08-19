# Native Ollama baseline generation

Core supports native, fail-closed `baseline_v1` generation through Ollama's
nonstreaming `POST /api/chat`. This provider is independent of legacy
generation and the separate generic strict HTTP provider. There is no provider
fallback and no translation proxy.

## Supported contract

The adapter contract is `baseline-generation-ollama-http.v1`. Every request
uses the packaged JSON Schema whose raw SHA-256 is
`fc5a85d5d38c18775afe0966987ea74e7e9ac072148822c1be60a199e32cca27`.
The specification SHA-256 is
`e670731777b253f9d5e3984405c2d99871ba26f637a17e6221cc82d97bc8beb1`.
Core neither projects nor simplifies this schema.

Before evidence is sent, Core calls `GET /api/version` and `GET /api/tags` and
requires:

- Ollama 0.32.13 or newer;
- the exact configured model name; and
- the exact configured `sha256:` digest.

Core never calls a pull endpoint. The same attestation is repeated immediately
before each evidence-bearing request, so a changed mutable tag fails closed.
The request sets `stream=false`, `think=false`, no tools, temperature zero, a
fixed seed, and bounded `num_ctx`/`num_predict`. Ordered stored renderer output
is passed without reranking, clipping, or normalization. Findings remain
bounded by both the four-item schema and the persisted Reference count.

## Recommended qualified profile

The supported recommended local profile is:

- model: `qwen3:14b`;
- quantization: Q4_K_M;
- immutable Ollama manifest digest:
  `sha256:bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8`;
- validated Ollama runtime: 0.32.14;
- context: 32,768 tokens;
- output limit: 1,024 tokens; and
- adapter: `baseline-generation-ollama-http.v1`.

The provider qualification produced 32/32 expected outcomes across two
independent cold cycles. Peak Ollama RSS was 14.88 GB, maximum observed latency
was 87.76 seconds, and a separate cold probe took 119.47 seconds. This is a
provider qualification result, not the final 120-case examination.

Only `message.content` is passed to Core's existing strict output validator.
Plain text, `NONE`, Markdown fences, blank or malformed JSON, duplicate keys,
non-finite numbers, extra properties, blank findings, and excessive findings
are terminal failures. Provider/network timeouts and transient HTTP statuses
are retryable. Core never claims exactly-once model invocation because Ollama
does not support a channel-side idempotency key; durable Feedback remains
idempotent across retries.

## Configuration

```sh
export COMPAIR_BASELINE_GENERATION_PROVIDER=ollama
export COMPAIR_BASELINE_GENERATION_ENDPOINT=http://127.0.0.1:11434
export COMPAIR_BASELINE_GENERATION_MODEL=qwen3:14b
export COMPAIR_BASELINE_GENERATION_MODEL_DIGEST=sha256:bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8
export COMPAIR_BASELINE_GENERATION_TIMEOUT_SECONDS=300
export COMPAIR_BASELINE_GENERATION_ALLOW_LOOPBACK_HTTP=true
```

The 300-second value is the supported CPU-only timeout profile. The settings
default remains 60 seconds for compatibility with accelerated deployments.
Core derives a 360-second internal generation/control-job lease from the CPU
timeout, preserving a 60-second response-validation and atomic-commit margin.
Configurations cannot use a provider timeout above 300 seconds, and an
internally supplied lease shorter than the derived bound is rejected.

For CPU deployments, configure model residency on the Ollama service, for
example `OLLAMA_KEEP_ALIVE=30m`, to avoid a cold model load on routine jobs.
Choose a shorter duration when reclaiming resident memory is more important.
Core does not send pull requests or replace the attested model.

The HTTP exception accepts only a literal loopback IP and must be explicit.
`localhost`, credentials in URLs, query strings, fragments, and non-loopback
plaintext endpoints are rejected. Remote services require verified HTTPS.
Redirects are disabled and environment proxies are ignored.

The optional bounds are:

| Setting | Default | Allowed range |
| --- | ---: | ---: |
| `COMPAIR_BASELINE_GENERATION_TIMEOUT_SECONDS` | 60 seconds | 0.1–300 seconds; use 300 for the supported CPU profile |
| `COMPAIR_BASELINE_GENERATION_MAX_REQUEST_BYTES` | 256,000 | 4,096–8,000,000 |
| `COMPAIR_BASELINE_GENERATION_MAX_RESPONSE_BYTES` | 200,000 | 4,096–1,000,000 |
| `COMPAIR_BASELINE_GENERATION_CONTEXT_TOKENS` | 32,768 | 2,048–131,072 |
| `COMPAIR_BASELINE_GENERATION_OUTPUT_TOKENS` | 1,024 | 64–4,096 |
| `COMPAIR_BASELINE_GENERATION_SEED` | 0 | 0–2,147,483,647 |

Inputs that cannot fit the conservative context bound fail; Core never relies
on hidden model-side truncation.

## Resource guidance

For the recommended CPU/unified-memory profile:

- total memory: 24 GiB minimum, 32 GiB preferred;
- measured 32K inference allocation: approximately 15 GB;
- free storage after acquisition: at least 25 GB; and
- free storage during acquisition or upgrades: at least 40 GB.

`compair-core doctor` reports observed host memory and free storage without
printing a filesystem path or inspecting model-cache contents. When the exact
recommended Ollama profile is selected, capacity below the measured safe floor
is a readiness failure. Falling below the recommended or preferred guidance is
a nonblocking warning. Dedicated-GPU capacity is not claimed unless it can be
explicitly attested; otherwise doctor uses the host memory result
conservatively.

## Verification and readiness

```sh
compair-core-generation verify
compair-core-generation verify --probe
```

Both commands emit exactly one safe JSON value. Default verification checks
configuration, transport, runtime, model, and digest without inference. The
probe sends a minimal private-data-free prompt through the exact schema.

Readiness statuses are `provider_unconfigured`, `endpoint_unavailable`,
`insecure_transport`, `unsupported_runtime`, `model_absent`, `digest_mismatch`,
`structured_output_unavailable`, and `ready`. Output includes only safe model,
digest, runtime, contract and fingerprint fields. It never includes endpoints,
prompts, evidence, findings, raw responses, credentials, leases, or
idempotency keys.

The existing frozen control-plane capability vocabulary continues to collapse
provider setup failures to `worker_unavailable`; use this installed verifier
for the safe detailed diagnosis. Ordinary capability requests perform static
attestation only and never run the inference probe.

`compair-core doctor` incorporates the same runtime/model/digest and packaged
schema checks into whole-workflow readiness. It performs no inference by
default. `compair-core doctor --probe-generation` is the only doctor mode that
sends the existing synthetic, private-data-free schema probe.

## Durable provenance

No migration is required. Existing fields record provider `ollama`, model,
adapter contract, runtime, digest and specification hash in the version,
schema hash/version, provider fingerprint, input/output fingerprints, and
`supports_idempotency=false`. The generation lease, authorization rechecks,
atomic Feedback/outbox transaction, zero-finding success, and replay behavior
are unchanged.
