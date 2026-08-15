# Baseline embedding HTTP service

`baseline_v1` has a separate, fail-closed embedding configuration. It does not
use Core's legacy embedding provider, legacy local-model endpoint, OpenAI
embedding configuration, or deterministic hash fallback.

The provider is disabled by default. A local BGE service can be configured with:

```text
COMPAIR_BASELINE_EMBEDDING_PROVIDER=http
COMPAIR_BASELINE_EMBEDDING_ENDPOINT=http://127.0.0.1:9010
COMPAIR_BASELINE_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
COMPAIR_BASELINE_EMBEDDING_REVISION=<immutable-model-revision>
COMPAIR_BASELINE_EMBEDDING_DIMENSION=384
COMPAIR_BASELINE_EMBEDDING_TIMEOUT_SECONDS=10
COMPAIR_BASELINE_EMBEDDING_BATCH_SIZE=32
COMPAIR_BASELINE_EMBEDDING_ALLOW_INSECURE_LOOPBACK=true
```

Plaintext HTTP is accepted only for `localhost`, a `127.0.0.0/8` address, or
`::1`, and only when the explicit loopback setting is true. Remote endpoints
must use HTTPS. Endpoint userinfo, query strings, and fragments are rejected.

## Protocol: `baseline-embedding-http.v1`

Core first requests `GET <base>/v1/health`. A ready service returns HTTP 200:

```json
{
  "status": "ok",
  "contract_version": "baseline-embedding-http.v1",
  "provider": "baseline_http_v1",
  "model": "BAAI/bge-small-en-v1.5",
  "revision": "<immutable-model-revision>",
  "dimension": 384
}
```

Core verifies every identity field before sending text. It then requests
`POST <base>/v1/embeddings` in configured batches:

```json
{
  "contract_version": "baseline-embedding-http.v1",
  "provider": "baseline_http_v1",
  "model": "BAAI/bge-small-en-v1.5",
  "revision": "<immutable-model-revision>",
  "dimension": 384,
  "texts": ["first ranking document", "second ranking document"]
}
```

The response repeats the attested identity and returns one vector per input, in
the same order:

```json
{
  "contract_version": "baseline-embedding-http.v1",
  "provider": "baseline_http_v1",
  "model": "BAAI/bge-small-en-v1.5",
  "revision": "<immutable-model-revision>",
  "dimension": 384,
  "vectors": [[0.01, 0.02], [0.03, 0.04]]
}
```

The abbreviated example vectors above are illustrative; actual vectors must
contain exactly 384 finite values. Core converts response numbers directly to
little-endian float32, preserves input/vector ordering, and does not normalize
or otherwise transform them. The service must pin the model revision and must
not silently substitute another model.

Neither service nor Core should log request bodies. Core exceptions, retrieval
results, capability responses, task status, and traces contain no submitted
query/document text. Capability output includes status, reason, contract,
provider, model, revision, dimension, identity fingerprint, and a sanitized
transport class; it never includes the configured endpoint or credentials.

Unavailable service, timeout, bad HTTP status, malformed JSON, identity
mismatch, wrong vector count/dimension, and NaN/Inf all fail closed. A compatible
index is not published on document-embedding failure, and query retrieval never
falls back to legacy or lexical-only behavior.

## Manual smoke check

With a compatible BGE/FastEmbed service already running locally:

```bash
python scripts/smoke_baseline_embedding.py \
  --endpoint http://127.0.0.1:9010 \
  --revision '<immutable-model-revision>'
```

The script uses fixed non-sensitive probe strings and prints only the attested
identity, fingerprint, and vector metadata. It does not download a model or
start a service.

## Reproducible loopback FastEmbed validation service

`scripts/live_baseline_embedding_service.py` is an operator-only thin adapter
over the self-hosted FastEmbed/BGE helper. It implements the protocol above;
it is not imported by Core and does not change production provider routing.
The helper accepts only `BAAI/bge-small-en-v1.5`, requires an absolute snapshot
directory whose basename is the attested immutable revision, passes that exact
directory to FastEmbed, and sets `local_files_only=True`. It therefore never
downloads or resolves a different model while serving requests.
Before loading, it also verifies the frozen SHA-256 values of the ONNX model,
configuration, tokenizer, tokenizer configuration, and special-token map.

For the frozen local validation snapshot:

```text
FastEmbed package: fastembed==0.8.0
FastEmbed artifact: qdrant/bge-small-en-v1.5-onnx-Q
Model revision: 52398278842ec682c6f32300af41344b1c0b0bb2
Dimension: 384
```

Create a Python 3.11 environment and reproduce the fully resolved service
package snapshot without changing Core's runtime dependencies:

```bash
python3.11 -m venv /path/to/baseline-embedding-venv
/path/to/baseline-embedding-venv/bin/pip install \
  -r scripts/requirements-baseline-embedding-live.txt
```

Start one worker on loopback and disable access logging so submitted texts are
not recorded:

```bash
SNAPSHOT_ROOT="$HOME/.cache/compair-baseline/models"
SNAPSHOT="$SNAPSHOT_ROOT/models--qdrant--bge-small-en-v1.5-onnx-Q/snapshots/52398278842ec682c6f32300af41344b1c0b0bb2"

COMPAIR_BASELINE_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 \
COMPAIR_BASELINE_EMBEDDING_REVISION=52398278842ec682c6f32300af41344b1c0b0bb2 \
COMPAIR_BASELINE_EMBEDDING_DIMENSION=384 \
COMPAIR_BASELINE_EMBEDDING_SNAPSHOT_DIR="$SNAPSHOT" \
COMPAIR_BASELINE_EMBEDDING_THREADS=8 \
COMPAIR_BASELINE_EMBEDDING_BATCH_SIZE=32 \
HF_HUB_OFFLINE=1 \
uvicorn scripts.live_baseline_embedding_service:app \
  --host 127.0.0.1 --port 9010 --workers 1 --no-access-log
```

The process must be bound to loopback by the operator command. The helper has
no endpoint for legacy `/embed`, performs no normalization, preserves request
and vector order, and returns sanitized error codes without request data.

After the smoke check, build and query a real minimal trusted publication with
the same Core adapter identity. The database path must not already exist:

```bash
python scripts/live_baseline_retrieval_validation.py \
  --endpoint http://127.0.0.1:9010 \
  --revision 52398278842ec682c6f32300af41344b1c0b0bb2 \
  --database /tmp/compair-phase2b2d1-live.sqlite
```

The script prints only corpus/index identifiers, attested identity, selected
paths, and query provenance hash/length/origin. Its in-memory sentinel query is
asserted absent from the retrieval result representation and serialized output.
