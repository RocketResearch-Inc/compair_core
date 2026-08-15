# Development validation status

Phase 2B2D.1 live validation succeeded with the production Core HTTP adapter,
FastEmbed 0.8.0, and the pinned `BAAI/bge-small-en-v1.5` ONNX snapshot at
revision `52398278842ec682c6f32300af41344b1c0b0bb2` (384-dimensional float32).
The real smoke check, compatible SQLite index publication, and persistent
baseline retrieval all passed without legacy/hash fallback or raw-query
leakage.

The real PostgreSQL corpus activation and index publication/rollback tests also
passed against PostgreSQL 17: `2 passed`. The current full-suite environment has
exactly two unrelated failures:

- `tests/test_api_load_documents.py::test_load_documents_executes_only_paginated_query`
  (SQLAlchemy environment/import failure);
- `tests/test_reference_reranker.py::ReferenceRerankerTests::test_load_model_resolves_latest_manifest_for_xgboost`
  (scikit-learn is absent from the test environment).

Local model caches, validation databases, service/runtime logs, downloaded
weights, virtual environments, and disposable database state are development
artifacts. They must not be added to Git or committed.
