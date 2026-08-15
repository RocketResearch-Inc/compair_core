# Trusted corpus snapshot ingestion

Phase 2B2A accepts complete manifests from a trusted producer. It does not walk
a filesystem, infer sibling repositories, or accept a partial update. Every
snapshot declares the changed repository, the complete sibling repository set,
each sibling's exact file-record count, and all supported or explicitly skipped
files.

`CorpusSnapshotInput.create(...)` produces the versioned
`corpus-snapshot-input.v1` contract and its canonical SHA-256. Production
callers may deserialize the frozen types directly, but ingestion recomputes and
compares the manifest hash before the first database write.

Paths must already be normalized POSIX-relative paths. Absolute paths, drive
forms, backslashes, NULs, traversal components, normalization aliases, duplicate
repository/path pairs, and content derived by following a symlink are rejected.
A producer may record an unfollowed symlink as a metadata-only
`symlink_rejected` file with the typed `symlink` reason.

The ingestion lifecycle is:

1. Validate the entire source contract in memory.
2. Persist an immutable staging generation and metadata-only provenance
   manifest with `incomplete` status.
3. Re-read and validate repository/file counts, deterministic row order, file
   metadata, the Core generation manifest, and the source manifest hash.
4. Mark it `complete`.
5. Atomically move the corpus active pointer, mark the new ingestion `active`,
   and mark the prior ingestion `stale`.

An activation transaction failure leaves the prior generation active and the
new generation complete but inactive. A validation failure is `failed`.
Generations created before this contract have no trusted ingestion provenance
and fail closed for baseline readiness. A newly active generation still starts
with an incomplete retrieval index, so Phase 2B2A cannot produce baseline
evidence.

The provenance table contains fixed producer/source identifiers, versions,
counts, hashes, typed skip reasons in the canonical manifest, and no raw file
content or retrieval-query field. Supported file content remains in the
existing corpus file table.

## PostgreSQL CI integration

The SQLite lifecycle tests run in the default suite. The real PostgreSQL test
is skipped unless a dedicated database URL is supplied:

```bash
COMPAIR_TEST_POSTGRES_URL='postgresql+psycopg2://user:password@host/database' \
  pytest -q tests/test_api_retrieval_ingestion_postgres.py
```

The test creates the additive retrieval tables if missing, uses a unique corpus
scope, verifies transactional activation rollback, and deletes only its own
corpus rows. It never drops or rebuilds shared tables.
