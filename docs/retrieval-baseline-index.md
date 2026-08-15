# Durable baseline_v1 index construction

Phase 2B2B constructs durable whole-file lexical and dense artifacts from one
trusted, complete, active corpus generation. It does not accept a retrieval
query and does not retrieve, rank, fuse, select, persist a `Reference`, or call
generation.

Each eligible `supported` corpus file becomes exactly one ranking document in
canonical repository/name/path order:

```text
Repository file: <repository>/<relative-path>

<first 12,000 content characters>
```

The index stores that exact text, its SHA-256, the source content SHA-256, its
ordinal, and its frozen-tokenizer length. Exact sparse term-frequency rows are
stored for every distinct document term. Document lengths and those rows are
sufficient to derive exact document frequency, average document length, and
BM25 with `k1=1.5` and `b=0.75`; SQLite FTS and PostgreSQL text ranking are not
used or represented as `baseline_v1`.

Dense vectors come only from an explicitly supplied adapter whose provider,
model, revision, dimension, and SHA-256 fingerprint exactly match the requested
identity. Vectors are validated after float32 conversion, must be finite and
exactly the pinned dimension, and are stored as portable little-endian float32
bytes with a SHA-256. There is no legacy embedding reuse and no SHA/hash-vector
fallback.

## Publication lifecycle

An index attempt has its own immutable version and status. The builder:

1. records a staging build against the active generation;
2. constructs the canonical documents, lexical statistics, and dense vectors;
3. stages every artifact and records deterministic document, lexical, and dense
   manifest hashes;
4. re-reads and validates the active corpus, file order/count/hash, all token
   frequencies, every vector, and all identity/configuration fingerprints;
5. atomically moves the corpus publication pointer and compatible index-state
   metadata only after validation succeeds.

A failed build records `failed`, `stale`, or `incompatible` without changing
the publication pointer. Activating a new corpus generation marks the old
generation's compatible index metadata and build stale, while retaining its
artifacts for inspection and rollback evidence. Readers therefore observe the
previous complete publication or the new complete publication, never the
staging rows as compatible.

Newly ingested generations still begin with an incomplete index. At the end of
Phase 2B2B, `baseline_v1` remained insufficient because that phase added no
retrieval consumer or evidence-selection path.

## Phase 2B2C read-only consumer

Phase 2B2C adds a dependency-injected, read-only `baseline_v1` adapter. It
requires an explicit `raw_git_diff_v1` query, corpus scope, changed-repository
identity, complete active trusted ingestion, one compatible publication, and
an embedding adapter whose complete pinned identity matches that publication.
There is still no configured production FastEmbed adapter: the adapter protocol
is production code, while concrete vectors in the test suite are fixtures. An
unavailable adapter therefore fails closed.

Before embedding or ranking, the reader verifies the active generation,
ingestion state, changed-repository identity, publication/build status, index
state, corpus and index artifact integrity, configuration fingerprint, and
embedding fingerprint. It derives an index fingerprint from the immutable
corpus, document, lexical, dense, tokenizer, embedding, and configuration
manifests. A second publication/readiness check after selection prevents a
request from returning evidence if activation changes during the read.

BM25 is computed over every persisted eligible document from the frozen query
tokens, document lengths, and term frequencies with `k1=1.5`, `b=0.75`. The
same pinned adapter embeds only the in-memory query; both query and persisted
vectors use finite float32 semantics and a float32 dot product. Full-corpus
BM25 and dense ranks are fused with equal-weight RRF at `k=60`, with repository
path and then stable document identity as deterministic tie keys. Only the
first six fused candidates enter filtering, 12,000-character content
normalization, deduplication, refill, and the four-item/16,000-character budget.
The reader never scans beyond that six-item cut and never invokes legacy.

The returned internal `retrieval-result.v2` contains candidate/evidence lane
scores and ranks plus corpus, index, engine/configuration, embedding, and
query-provenance identifiers. Query provenance contains only SHA-256, character
length, and origin. The raw query is neither stored in the result nor persisted.
Reference persistence, generation, findings, API exposure, CLI behavior, and
production provider configuration remain outside Phase 2B2C.

The deliberate comparator delta remains the secure symlink policy established
earlier: trusted ingestion rejects both in-repository and escaping
symlink-derived inputs instead of following them. The stable document-ID
secondary tie key does not alter valid comparator ordering because trusted
corpus paths are unique.

## Additive migration

The schema adds five tables without altering or rebuilding legacy or Phase
2B2A tables:

- `retrieval_baseline_index_build`
- `retrieval_baseline_index_document`
- `retrieval_baseline_index_term`
- `retrieval_baseline_index_vector`
- `retrieval_baseline_index_publication`

SQLite uses `BLOB` and PostgreSQL uses `BYTEA` for the same float32 bytes. No
table contains retrieval-query text or a legacy `Chunk` foreign key.

## PostgreSQL CI integration

The real PostgreSQL publication/rollback test is skipped unless a dedicated
database URL is supplied:

```bash
COMPAIR_TEST_POSTGRES_URL='postgresql+psycopg2://user:password@host/database' \
  pytest -q tests/test_api_retrieval_indexing_postgres.py
```

It creates missing additive tables, uses a unique corpus scope, proves that a
post-publication exception rolls the pointer back to the prior compatible
build, and deletes only its own corpus rows.
