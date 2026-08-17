from __future__ import annotations

import hashlib
import importlib.util
import os
import pathlib
import sys
import types
import unittest
from dataclasses import dataclass
from unittest import mock


def _load_main_module():
    root = pathlib.Path(__file__).resolve().parents[1]
    package_name = "test_compair_main_module"
    main_path = root / "compair_core" / "compair" / "main.py"
    local_summary_path = root / "compair_core" / "compair" / "local_summary.py"

    package = types.ModuleType(package_name)
    package.__path__ = [str(main_path.parent)]
    sys.modules[package_name] = package

    levenshtein = types.ModuleType("Levenshtein")
    levenshtein.ratio = lambda left, right: 0.0

    sqlalchemy = types.ModuleType("sqlalchemy")
    sqlalchemy.select = lambda *args, **kwargs: None
    sqlalchemy.or_ = lambda *args, **kwargs: ("or", args)

    sqlalchemy_orm = types.ModuleType("sqlalchemy.orm")
    sqlalchemy_orm.Session = object
    sqlalchemy_orm.load_only = lambda *args, **kwargs: ("load_only", args)

    sqlalchemy_orm_attributes = types.ModuleType("sqlalchemy.orm.attributes")
    sqlalchemy_orm_attributes.get_history = lambda *args, **kwargs: types.SimpleNamespace(deleted=[])

    embeddings = types.ModuleType(f"{package_name}.embeddings")
    embeddings.create_embedding = lambda *args, **kwargs: []
    embeddings.create_embeddings = lambda *args, **kwargs: []
    embeddings.Embedder = object
    sys.modules[embeddings.__name__] = embeddings

    feedback = types.ModuleType(f"{package_name}.feedback")
    feedback.get_feedback = lambda *args, **kwargs: "NONE"
    feedback.Reviewer = object
    feedback.split_feedback_items = lambda feedback_text, **kwargs: [feedback_text] if feedback_text and feedback_text != "NONE" else []
    sys.modules[feedback.__name__] = feedback

    logger = types.ModuleType(f"{package_name}.logger")
    logger.log_event = lambda *args, **kwargs: None
    sys.modules[logger.__name__] = logger

    local_summary_spec = importlib.util.spec_from_file_location(
        f"{package_name}.local_summary",
        local_summary_path,
    )
    local_summary_module = importlib.util.module_from_spec(local_summary_spec)
    sys.modules[local_summary_spec.name] = local_summary_module
    assert local_summary_spec.loader is not None
    local_summary_spec.loader.exec_module(local_summary_module)

    models = types.ModuleType(f"{package_name}.models")
    for name in ("Chunk", "Document", "Feedback", "Group", "Note", "Reference", "User"):
        setattr(models, name, type(name, (), {}))
    models.VECTOR_BACKEND = "json"

    def _cosine_similarity(left, right):
        if not left or not right or len(left) != len(right):
            return None
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = sum(a * a for a in left) ** 0.5
        right_norm = sum(b * b for b in right) ** 0.5
        if left_norm == 0.0 or right_norm == 0.0:
            return None
        return numerator / (left_norm * right_norm)

    models.cosine_similarity = _cosine_similarity
    sys.modules[models.__name__] = models

    topic_tags = types.ModuleType(f"{package_name}.topic_tags")
    topic_tags.extract_topic_tags = lambda text: []
    sys.modules[topic_tags.__name__] = topic_tags

    utils = types.ModuleType(f"{package_name}.utils")
    utils.chunk_text_with_mode = lambda text, chunk_mode=None: [text] if text else []
    utils.count_tokens = lambda text: max(1, len(text or "") // 4) if text else 0
    utils.log_activity = lambda *args, **kwargs: None
    utils.sanitize_text_for_database = lambda text: text
    utils.stable_chunk_hash = lambda text: hashlib.sha256((text or "").encode("utf-8")).hexdigest()
    sys.modules[utils.__name__] = utils

    spec = importlib.util.spec_from_file_location(f"{package_name}.main", main_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    isolated_dependencies = {
        "Levenshtein": levenshtein,
        "sqlalchemy": sqlalchemy,
        "sqlalchemy.orm": sqlalchemy_orm,
        "sqlalchemy.orm.attributes": sqlalchemy_orm_attributes,
    }
    missing = object()
    previous_dependencies = {
        name: sys.modules.get(name, missing) for name in isolated_dependencies
    }
    try:
        sys.modules.update(isolated_dependencies)
        spec.loader.exec_module(module)
    finally:
        for name, previous in previous_dependencies.items():
            if previous is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


main = _load_main_module()


@dataclass
class DummyChunk:
    content: str
    chunk_type: str = "document"
    document_id: str | None = None
    note_id: str | None = None
    chunk_id: str | None = None


class _SelectionField:
    def __eq__(self, other):
        return ("eq", other)

    def __ne__(self, other):
        return ("ne", other)

    def in_(self, values):
        return ("in", tuple(values))

    def is_(self, value):
        return ("is", value)


def _install_selection_fields() -> None:
    for name in (
        "chunk_id",
        "hash",
        "embedding",
        "document_id",
        "chunk_type",
        "note_id",
        "content",
        "document",
    ):
        setattr(main.Chunk, name, _SelectionField())
    for name in ("document_id", "groups", "is_published"):
        setattr(main.Document, name, _SelectionField())
    main.Group.group_id = _SelectionField()
    main.User.user_id = _SelectionField()


class _SelectionQuery:
    def __init__(self, rows):
        self.rows = rows

    def options(self, *args, **kwargs):
        return self

    def filter(self, *args, **kwargs):
        return self

    def join(self, *args, **kwargs):
        return self

    def all(self):
        return list(self.rows)

    def first(self):
        return self.rows[0] if self.rows else None


class _BridgeSession:
    def __init__(self, candidates):
        self.candidates = candidates

    def query(self, model):
        if model is main.Chunk:
            return _SelectionQuery(self.candidates)
        raise AssertionError(f"unexpected query model: {model}")


def _bridge_subjects(candidates):
    _install_selection_fields()
    source = types.SimpleNamespace(
        chunk_id="source",
        embedding=[1.0, 0.0],
        document_id="source-doc",
        chunk_type="document",
        content="source text",
    )
    user = types.SimpleNamespace(user_id="user")
    doc = types.SimpleNamespace(
        document_id="source-doc",
        groups=[types.SimpleNamespace(group_id="group")],
    )
    return _BridgeSession(candidates), user, doc, source


class MainRetrievalTests(unittest.TestCase):
    def test_process_text_passes_explicit_query_to_retrieval_engine_unchanged(self) -> None:
        _install_selection_fields()
        text = "source text"
        source = types.SimpleNamespace(
            chunk_id="source",
            hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            embedding=[1.0, 0.0],
            document_id="source-doc",
            note_id=None,
            chunk_type="document",
            content=text,
        )
        user = types.SimpleNamespace(user_id="user")
        doc = types.SimpleNamespace(
            document_id="source-doc",
            author_id="user",
            groups=[types.SimpleNamespace(group_id="group")],
        )

        class Session:
            def query(self, model):
                if model is main.User:
                    return _SelectionQuery([user])
                if model is main.Chunk:
                    return _SelectionQuery([source])
                raise AssertionError(f"unexpected query model: {model}")

            def commit(self):
                return None

        captured = {}

        def retrieval_spy(**kwargs):
            captured.update(kwargs)
            return []

        query = "diff --git a/source.py b/source.py\n-old\n+new\n"
        with mock.patch.object(main, "retrieve_reference_evidence", side_effect=retrieval_spy):
            main.process_text(
                Session(),
                embedder=object(),
                reviewer=object(),
                doc=doc,
                text=text,
                retrieval_query=query,
            )

        request = captured["request"]
        self.assertEqual(request.retrieval_query, query)
        self.assertEqual(request.retrieval_query_origin.value, "explicit")

    def test_process_document_passes_retrieval_query_to_process_text(self) -> None:
        _install_selection_fields()
        main.Feedback.source_chunk_id = _SelectionField()

        class TimestampField:
            def __ge__(self, other):
                return ("ge", other)

        main.Feedback.timestamp = TimestampField()
        query = "diff --git a/source.py b/source.py\n+changed = True\n"
        user = types.SimpleNamespace(user_id="user")
        doc = types.SimpleNamespace(
            document_id="source-doc",
            author_id="user",
            content="a sufficiently long source chunk",
            topic_tags=[],
            groups=[],
        )

        class CountQuery:
            def __iter__(self):
                return iter(())

            def filter(self, *args, **kwargs):
                return self

            def count(self):
                return 0

        session = types.SimpleNamespace(
            query=lambda model: CountQuery(),
            commit=lambda: None,
        )

        with (
            mock.patch.object(main, "get_history", return_value=types.SimpleNamespace(deleted=[])),
            mock.patch.object(main, "chunk_text_with_mode", return_value=[doc.content]),
            mock.patch.object(main, "extract_topic_tags", return_value=[]),
            mock.patch.object(main, "is_code_review_document", return_value=False),
            mock.patch.object(main, "detect_significant_edits", return_value=[0]),
            mock.patch.object(main, "count_tokens", return_value=200),
            mock.patch.object(main, "create_embeddings", return_value=[[1.0, 0.0]]),
            mock.patch.object(main, "process_text") as process_text_spy,
        ):
            main.process_document(
                user,
                session,
                embedder=object(),
                reviewer=object(),
                doc=doc,
                retrieval_query=query,
            )

        self.assertEqual(process_text_spy.call_count, 1)
        self.assertEqual(process_text_spy.call_args.kwargs["retrieval_query"], query)

    def test_legacy_trace_hashes_query_without_recording_text(self) -> None:
        session, user, doc, source = _bridge_subjects([])
        query = "diff --git a/source.py b/source.py\n+private value\n"
        request = main.RetrievalRequest(
            request_id="source",
            changed_repository=None,
            repository_roots=(),
            corpus_version="",
            retrieval_query=query,
            retrieval_query_origin=main.RetrievalQueryOrigin.EXPLICIT,
            corpus_complete=False,
        )
        events = []

        with (
            mock.patch.object(main, "VECTOR_BACKEND", "json"),
            mock.patch.object(main, "_allow_same_document_feedback", return_value=False),
            mock.patch.object(main, "_reference_trace_enabled", return_value=True),
            mock.patch.object(main, "log_event", side_effect=lambda name, **values: events.append((name, values))),
        ):
            main._select_legacy_reference_chunks(
                session,
                embedder=object(),
                user=user,
                doc=doc,
                existing_chunk=source,
                text="source text",
                code_focus=False,
                query_embedding=None,
                focus_text="",
                change_context="",
                reference_doc_ids=None,
                retrieval_request=request,
            )

        trace = next(values for name, values in events if name == "feedback_reference_trace")
        self.assertEqual(trace["retrieval_query_sha256"], hashlib.sha256(query.encode()).hexdigest())
        self.assertEqual(trace["retrieval_query_length"], len(query))
        self.assertEqual(trace["retrieval_query_origin"], "explicit")
        self.assertNotIn(query, repr(trace))

    def test_legacy_trace_marks_its_existing_query_as_derived(self) -> None:
        session, user, doc, source = _bridge_subjects([])
        events = []

        with (
            mock.patch.object(main, "VECTOR_BACKEND", "json"),
            mock.patch.object(main, "_allow_same_document_feedback", return_value=False),
            mock.patch.object(main, "_reference_trace_enabled", return_value=True),
            mock.patch.object(main, "log_event", side_effect=lambda name, **values: events.append((name, values))),
        ):
            main._select_legacy_reference_chunks(
                session,
                embedder=object(),
                user=user,
                doc=doc,
                existing_chunk=source,
                text="source text",
                code_focus=False,
                query_embedding=None,
                focus_text="",
                change_context="",
                reference_doc_ids=None,
            )

        trace = next(values for name, values in events if name == "feedback_reference_trace")
        self.assertEqual(trace["retrieval_query_origin"], "legacy_derived")
        self.assertEqual(trace["retrieval_query_length"], len("source text"))
        self.assertEqual(
            trace["retrieval_query_sha256"],
            hashlib.sha256(b"source text").hexdigest(),
        )

    def test_legacy_selection_bridge_empty_candidates_snapshot(self) -> None:
        session, user, doc, source = _bridge_subjects([])
        with (
            mock.patch.object(main, "VECTOR_BACKEND", "json"),
            mock.patch.object(main, "_is_code_review_chunk", return_value=False),
            mock.patch.object(main, "_allow_same_document_feedback", return_value=False),
            mock.patch.object(main, "_reference_trace_enabled", return_value=False),
        ):
            selected = main._select_legacy_reference_chunks(
                session,
                embedder=object(),
                user=user,
                doc=doc,
                existing_chunk=source,
                text="source text",
                code_focus=False,
                query_embedding=None,
                focus_text="",
                change_context="",
                reference_doc_ids=None,
            )

        self.assertEqual(selected, [])

    def test_process_text_empty_selection_persists_and_generates_nothing(self) -> None:
        _install_selection_fields()
        text = "source text"
        source = types.SimpleNamespace(
            chunk_id="source",
            hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            embedding=[1.0, 0.0],
            document_id="source-doc",
            note_id=None,
            chunk_type="document",
            content=text,
        )
        user = types.SimpleNamespace(user_id="user")
        doc = types.SimpleNamespace(
            document_id="source-doc",
            author_id="user",
            groups=[types.SimpleNamespace(group_id="group")],
        )

        class Session:
            def __init__(self):
                self.persisted = []

            def query(self, model):
                if model is main.User:
                    return _SelectionQuery([user])
                if model is main.Chunk:
                    return _SelectionQuery([source])
                raise AssertionError(f"unexpected query model: {model}")

            def commit(self):
                return None

            def add_all(self, values):
                self.persisted.extend(values)

        session = Session()
        with (
            mock.patch.object(main, "_select_legacy_reference_chunks", return_value=[]),
            mock.patch.object(main, "get_feedback") as feedback,
        ):
            main.process_text(
                session,
                embedder=object(),
                reviewer=object(),
                doc=doc,
                text=text,
            )

        self.assertEqual(session.persisted, [])
        feedback.assert_not_called()

    def test_legacy_selection_bridge_preserves_input_order_for_vector_ties(self) -> None:
        candidates = [
            DummyChunk(content=f"candidate {index}", chunk_id=f"peer-{index}")
            for index in range(1, 6)
        ]
        for candidate in candidates:
            candidate.embedding = [1.0, 0.0]
        session, user, doc, source = _bridge_subjects(candidates)

        with (
            mock.patch.object(main, "VECTOR_BACKEND", "json"),
            mock.patch.object(main, "_is_code_review_chunk", return_value=False),
            mock.patch.object(main, "_allow_same_document_feedback", return_value=False),
            mock.patch.object(main, "_reference_selection_config", return_value=(6, 4, 2)),
            mock.patch.object(main, "_reference_effective_vector_fetch_limit", return_value=6),
            mock.patch.object(
                main,
                "_filter_reference_candidates",
                side_effect=lambda values, **kwargs: (list(values), {}),
            ),
            mock.patch.object(main, "_reference_trace_enabled", return_value=False),
        ):
            selected = main._select_legacy_reference_chunks(
                session,
                embedder=object(),
                user=user,
                doc=doc,
                existing_chunk=source,
                text="source text",
                code_focus=False,
                query_embedding=None,
                focus_text="",
                change_context="",
                reference_doc_ids=None,
            )

        self.assertEqual(selected, candidates[:4])
        self.assertTrue(all(actual is expected for actual, expected in zip(selected, candidates)))

    def test_legacy_selection_bridge_code_hybrid_lane_snapshot(self) -> None:
        candidates = [
            DummyChunk(
                content=f"### File: src/{name}.py\nvalue = '{name}'\n",
                chunk_id=name,
                document_id=f"doc-{name}",
            )
            for name in ("vector-1", "lexical-1", "fts-1", "counterpart-1", "anchor-1")
        ]
        for index, candidate in enumerate(candidates):
            candidate.embedding = [1.0 - (index * 0.1), index * 0.1]
        session, user, doc, source = _bridge_subjects(candidates)
        captured_order = []

        def capture_rerank(target_text, merged, **kwargs):
            captured_order.extend(merged)
            debug_stats = kwargs.get("debug_stats")
            if isinstance(debug_stats, dict):
                debug_stats["hybrid_enabled"] = True
            return list(merged[:4])

        with (
            mock.patch.dict(os.environ, {"COMPAIR_REFERENCE_HYBRID_ENABLED": "1"}),
            mock.patch.object(main, "VECTOR_BACKEND", "json"),
            mock.patch.object(main, "_is_code_review_chunk", return_value=True),
            mock.patch.object(main, "_allow_same_document_feedback", return_value=False),
            mock.patch.object(main, "_reference_query_variants", return_value=[("primary", "source text")]),
            mock.patch.object(main, "_reference_selection_config", return_value=(6, 4, 2)),
            mock.patch.object(main, "_reference_merge_limit", return_value=10),
            mock.patch.object(main, "_reference_effective_vector_fetch_limit", return_value=6),
            mock.patch.object(
                main,
                "_filter_reference_candidates",
                side_effect=lambda values, **kwargs: (list(values), {}),
            ),
            mock.patch.object(main, "_reference_fts_candidates", return_value=[candidates[2]]),
            mock.patch.object(main, "_lexical_reference_candidates", return_value=[candidates[1]]),
            mock.patch.object(main, "_anchor_reference_candidates", return_value=[candidates[4]]),
            mock.patch.object(main, "_reference_counterpart_candidates", return_value=[candidates[3]]),
            mock.patch.object(main, "_rerank_reference_chunks", side_effect=capture_rerank),
            mock.patch.object(main, "_reference_trace_enabled", return_value=False),
        ):
            self.assertTrue(main._reference_hybrid_enabled())
            selected = main._select_legacy_reference_chunks(
                session,
                embedder=object(),
                user=user,
                doc=doc,
                existing_chunk=source,
                text="source text",
                code_focus=True,
                query_embedding=None,
                focus_text="",
                change_context="",
                reference_doc_ids=None,
            )

        expected_order = [
            candidates[0],
            candidates[2],
            candidates[3],
            candidates[4],
            candidates[1],
        ]
        self.assertEqual(captured_order, expected_order)
        self.assertEqual(selected, expected_order[:4])

    def test_process_text_legacy_selection_snapshot_preserves_chunk_identity_and_order(self) -> None:
        class Field:
            def __eq__(self, other):
                return ("eq", other)

            def __ne__(self, other):
                return ("ne", other)

            def in_(self, values):
                return ("in", tuple(values))

            def is_(self, value):
                return ("is", value)

        for name in (
            "chunk_id",
            "hash",
            "embedding",
            "document_id",
            "chunk_type",
            "note_id",
            "content",
            "document",
        ):
            setattr(main.Chunk, name, Field())
        for name in ("document_id", "groups", "is_published"):
            setattr(main.Document, name, Field())
        main.Group.group_id = Field()
        main.User.user_id = Field()

        text = "A plain source chunk used for a stable retrieval snapshot."
        chunk_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        source = types.SimpleNamespace(
            chunk_id="source",
            hash=chunk_hash,
            embedding=[1.0, 0.0],
            document_id="source-doc",
            note_id=None,
            chunk_type="document",
            content=text,
        )
        candidates = [
            types.SimpleNamespace(
                chunk_id="peer-a",
                embedding=[0.0, 1.0],
                document_id="peer-doc-a",
                note_id=None,
                chunk_type="document",
                content="candidate a",
            ),
            types.SimpleNamespace(
                chunk_id="peer-b",
                embedding=[0.6, 0.8],
                document_id="peer-doc-b",
                note_id=None,
                chunk_type="document",
                content="candidate b",
            ),
            types.SimpleNamespace(
                chunk_id="peer-c",
                embedding=[1.0, 0.0],
                document_id="peer-doc-c",
                note_id=None,
                chunk_type="document",
                content="candidate c",
            ),
            types.SimpleNamespace(
                chunk_id="peer-d",
                embedding=[0.8, 0.6],
                document_id="peer-doc-d",
                note_id=None,
                chunk_type="document",
                content="candidate d",
            ),
            types.SimpleNamespace(
                chunk_id="peer-e",
                embedding=[-1.0, 0.0],
                document_id="peer-doc-e",
                note_id=None,
                chunk_type="document",
                content="candidate e",
            ),
        ]
        user = types.SimpleNamespace(user_id="user")
        doc = types.SimpleNamespace(
            document_id="source-doc",
            author_id="user",
            groups=[types.SimpleNamespace(group_id="group")],
        )

        class Query:
            def __init__(self, rows):
                self.rows = rows

            def options(self, *args, **kwargs):
                return self

            def filter(self, *args, **kwargs):
                return self

            def join(self, *args, **kwargs):
                return self

            def all(self):
                return list(self.rows)

            def first(self):
                return self.rows[0] if self.rows else None

        class Session:
            def __init__(self):
                self.chunk_query_count = 0
                self.persisted = []

            def query(self, model):
                if model is main.User:
                    return Query([user])
                if model is main.Chunk:
                    self.chunk_query_count += 1
                    return Query([source] if self.chunk_query_count == 1 else candidates)
                raise AssertionError(f"unexpected query model: {model}")

            def commit(self):
                return None

            def add_all(self, values):
                self.persisted.extend(values)

        class CapturedReference:
            def __init__(self, **values):
                self.__dict__.update(values)

        captured_generation_references = []

        def capture_feedback(reviewer, document, source_text, references, current_user, **kwargs):
            captured_generation_references.extend(references)
            return "NONE"

        session = Session()
        with (
            mock.patch.object(main, "VECTOR_BACKEND", "json"),
            mock.patch.object(main, "Reference", CapturedReference),
            mock.patch.object(main, "get_feedback", side_effect=capture_feedback),
            mock.patch.object(main, "_is_code_review_chunk", return_value=False),
            mock.patch.object(main, "_allow_same_document_feedback", return_value=False),
            mock.patch.object(main, "_reference_selection_config", return_value=(6, 4, 2)),
            mock.patch.object(main, "_reference_effective_vector_fetch_limit", return_value=6),
            mock.patch.object(
                main,
                "_filter_reference_candidates",
                side_effect=lambda values, **kwargs: (list(values), {}),
            ),
            mock.patch.object(main, "_reference_trace_enabled", return_value=False),
        ):
            main.process_text(
                session,
                embedder=object(),
                reviewer=types.SimpleNamespace(model="fixture", provider="fixture"),
                doc=doc,
                text=text,
            )

        expected = [candidates[2], candidates[3], candidates[1], candidates[0]]
        self.assertEqual(captured_generation_references, expected)
        self.assertTrue(
            all(actual is expected_chunk for actual, expected_chunk in zip(captured_generation_references, expected))
        )
        self.assertEqual(
            [reference.reference_chunk_id for reference in session.persisted],
            ["peer-c", "peer-d", "peer-b", "peer-a"],
        )

    def test_reference_scope_allows_same_document_when_explicitly_requested(self) -> None:
        self.assertTrue(
            main._reference_scope_allows_same_document(
                "doc-1",
                allow_same_document=False,
                reference_doc_ids=["doc-1", "doc-2"],
            )
        )

    def test_reference_scope_disables_same_document_when_explicit_scope_excludes_it(self) -> None:
        self.assertFalse(
            main._reference_scope_allows_same_document(
                "doc-1",
                allow_same_document=True,
                reference_doc_ids=["doc-2"],
            )
        )

    def test_reference_candidate_allowed_excludes_same_document_when_self_feedback_disabled(self) -> None:
        doc = types.SimpleNamespace(document_id="doc-1")
        source = DummyChunk(
            chunk_id="source",
            document_id="doc-1",
            content="### File: pyproject.toml\nlicense = { text = \"MIT\" }\n",
        )
        candidate = DummyChunk(
            chunk_id="peer",
            document_id="doc-1",
            content="### File: LICENSE\nGNU GENERAL PUBLIC LICENSE\n",
        )

        allowed = main._reference_candidate_allowed(
            candidate,
            doc=doc,
            source_chunk=source,
            allow_same_document=False,
            code_focus=True,
        )

        self.assertFalse(allowed)

    def test_reference_candidate_allowed_allows_same_document_when_self_feedback_enabled(self) -> None:
        doc = types.SimpleNamespace(document_id="doc-1")
        source = DummyChunk(
            chunk_id="source",
            document_id="doc-1",
            content="### File: pyproject.toml\nlicense = { text = \"MIT\" }\n",
        )
        candidate = DummyChunk(
            chunk_id="peer",
            document_id="doc-1",
            content="### File: LICENSE\nGNU GENERAL PUBLIC LICENSE\n",
        )

        allowed = main._reference_candidate_allowed(
            candidate,
            doc=doc,
            source_chunk=source,
            allow_same_document=True,
            code_focus=True,
        )

        self.assertTrue(allowed)

    def test_reference_candidate_allowed_rejects_same_file_self_feedback_by_default(self) -> None:
        doc = types.SimpleNamespace(document_id="doc-1")
        source = DummyChunk(
            chunk_id="source",
            document_id="doc-1",
            content="### File: src/layouts/BaseLayout.astro (part 2/2)\nconst body = payload;\n",
        )
        candidate = DummyChunk(
            chunk_id="peer",
            document_id="doc-1",
            content="### File: src/layouts/BaseLayout.astro (part 1/2)\nconst sendTrackToEndpoint = () => {};\n",
        )

        allowed = main._reference_candidate_allowed(
            candidate,
            doc=doc,
            source_chunk=source,
            allow_same_document=True,
            code_focus=True,
        )

        self.assertFalse(allowed)

    def test_reference_candidate_allowed_allows_same_file_when_opted_in_and_not_adjacent(self) -> None:
        doc = types.SimpleNamespace(document_id="doc-1")
        source = DummyChunk(
            chunk_id="source",
            document_id="doc-1",
            content=(
                "### File: docs/architecture.md (part 5/9)\n"
                "API tokens are short-lived and rotated by the server.\n"
                "Client caching is disabled.\n"
            ),
        )
        candidate = DummyChunk(
            chunk_id="peer",
            document_id="doc-1",
            content=(
                "### File: docs/architecture.md (part 8/9)\n"
                "API tokens are short-lived and rotated by the server.\n"
                "Clients may cache tokens for 24 hours.\n"
            ),
        )

        original = os.environ.get("COMPAIR_ALLOW_SAME_FILE_SELF_FEEDBACK")
        os.environ["COMPAIR_ALLOW_SAME_FILE_SELF_FEEDBACK"] = "1"
        try:
            allowed = main._reference_candidate_allowed(
                candidate,
                doc=doc,
                source_chunk=source,
                allow_same_document=True,
                code_focus=True,
            )
        finally:
            if original is None:
                os.environ.pop("COMPAIR_ALLOW_SAME_FILE_SELF_FEEDBACK", None)
            else:
                os.environ["COMPAIR_ALLOW_SAME_FILE_SELF_FEEDBACK"] = original

        self.assertTrue(allowed)

    def test_reference_candidate_allowed_never_returns_same_chunk(self) -> None:
        doc = types.SimpleNamespace(document_id="doc-1")
        source = DummyChunk(
            chunk_id="source",
            document_id="doc-1",
            content="### File: pyproject.toml\nlicense = { text = \"MIT\" }\n",
        )

        allowed = main._reference_candidate_allowed(
            source,
            doc=doc,
            source_chunk=source,
            allow_same_document=True,
            code_focus=True,
        )

        self.assertFalse(allowed)

    def test_reference_candidate_allowed_rejects_header_only_snapshot_peer(self) -> None:
        doc = types.SimpleNamespace(document_id="doc-1")
        source = DummyChunk(
            chunk_id="source",
            document_id="doc-1",
            content="### File: docs/CLOUDFLARE_INTEGRATION.md (part 1/2)\n# Cloudflare integration\n",
        )
        candidate = DummyChunk(
            chunk_id="peer",
            document_id="doc-2",
            content="### File: docs/CLOUDFLARE_INTEGRATION.md (part 2/2)\n",
        )

        allowed = main._reference_candidate_allowed(
            candidate,
            doc=doc,
            source_chunk=source,
            allow_same_document=True,
            code_focus=True,
        )

        self.assertFalse(allowed)

    def test_should_reanalyze_existing_chunks_uses_remaining_slots_with_new_chunks(self) -> None:
        self.assertTrue(
            main._should_reanalyze_existing_chunks(
                reanalyze_existing=True,
                meaningful_new_chunk_count=0,
            )
        )
        self.assertFalse(
            main._should_reanalyze_existing_chunks(
                reanalyze_existing=False,
                meaningful_new_chunk_count=0,
            )
        )
        self.assertTrue(
            main._should_reanalyze_existing_chunks(
                reanalyze_existing=True,
                meaningful_new_chunk_count=1,
            )
        )

    def test_reference_query_text_prefers_focus_window_for_large_snapshot_chunk(self) -> None:
        full_chunk = (
            "### File: docs/api_mapping.md\n"
            "The CLI reference describes auth, login, signup, groups, notes, tokens, and delivery.\n"
            "It also includes general setup guidance and install notes for multiple platforms.\n"
            "| `docs list` | `GET /load_documents` |\n"
            "| `activity` | `GET /activity_feed` |\n"
            "| `notifications` | `GET /notification_events` |\n"
        )
        focus_text = "| `activity` | `GET /activity_feed` |"
        self.assertEqual(main._reference_query_text(full_chunk, focus_text, "", code_focus=True), focus_text)

    def test_reference_query_text_keeps_full_chunk_when_focus_is_not_much_smaller(self) -> None:
        full_chunk = (
            "Google OAuth is available on Core and should appear in /capabilities when configured.\n"
            "OAuth cache ready.\n"
        )
        focus_text = "Google OAuth is available on Core and should appear in /capabilities when configured."
        self.assertEqual(main._reference_query_text(full_chunk, focus_text, "", code_focus=True), full_chunk)

    def test_reference_query_text_prefers_before_after_change_context(self) -> None:
        full_chunk = (
            "### File: docs/api_mapping.md\n"
            "| `docs list` | `GET /load_documents` |\n"
            "| `activity` | `GET /activity_feed` |\n"
            "| `notifications` | `GET /notification_events` |\n"
        )
        focus_text = "| `activity` | `GET /activity_feed` |"
        change_context = (
            "### File: docs/api_mapping.md\n"
            "- | `activity` | `GET /get_activity_feed` |\n"
            "+ | `activity` | `GET /activity_feed` |"
        )
        self.assertEqual(
            main._reference_query_text(full_chunk, focus_text, change_context, code_focus=True),
            change_context,
        )

    def test_reference_query_variants_include_full_and_anchor_for_behavioral_docs(self) -> None:
        full_chunk = (
            "### File: docs/user-guide.md\n"
            "Set `COMPAIR_EMAIL_BACKEND=stdout` for local development.\n"
            "Core uses the configured backend to send verification emails.\n"
            "The API returns delivery status from `GET /notification_events`.\n"
        )
        focus_text = "Set `COMPAIR_EMAIL_BACKEND=stdout` for local development."

        variants = main._reference_query_variants(full_chunk, focus_text, "", code_focus=True)

        names = [name for name, _ in variants]
        variant_map = {name: text for name, text in variants}
        self.assertEqual(names[0], "primary")
        self.assertIn("full", names)
        self.assertIn("anchor", names)
        self.assertIn("counterpart", names)
        self.assertIn("COMPAIR_EMAIL_BACKEND", variant_map["anchor"])
        self.assertIn("/notification_events", variant_map["anchor"])
        self.assertIn("terms", variant_map["counterpart"])
        self.assertIn("email", variant_map["counterpart"].lower())

    def test_reference_effective_vector_fetch_limit_boosts_behavioral_docs_and_metadata(self) -> None:
        candidate_limit = 10
        merge_limit = 30
        base_limit = main._reference_vector_fetch_limit(True, candidate_limit, merge_limit)
        behavioral_doc = (
            "### File: docs/user-guide.md\n"
            "Set `COMPAIR_EMAIL_BACKEND=stdout` for local development.\n"
            "Core uses the configured backend to send verification emails.\n"
        )
        metadata_chunk = (
            "### File: pyproject.toml\n"
            'name = "compair-core"\n'
            'license = { text = "MIT" }\n'
        )

        boosted_doc = main._reference_effective_vector_fetch_limit(
            behavioral_doc,
            code_focus=True,
            candidate_limit=candidate_limit,
            merge_limit=merge_limit,
        )
        boosted_metadata = main._reference_effective_vector_fetch_limit(
            metadata_chunk,
            code_focus=True,
            candidate_limit=candidate_limit,
            merge_limit=merge_limit,
        )

        self.assertGreater(boosted_doc, base_limit)
        self.assertGreater(boosted_metadata, base_limit)

    def test_interleave_reference_candidates_preserves_lane_diversity(self) -> None:
        vector = [
            DummyChunk(document_id="vector-1", chunk_id="vector-1", content="### File: docs/a.md\nA\n"),
            DummyChunk(document_id="vector-2", chunk_id="vector-2", content="### File: docs/b.md\nB\n"),
            DummyChunk(document_id="vector-3", chunk_id="vector-3", content="### File: docs/c.md\nC\n"),
        ]
        anchor = [
            DummyChunk(document_id="anchor-1", chunk_id="anchor-1", content="### File: api.py\nroute\n"),
        ]
        lexical = [
            DummyChunk(document_id="lexical-1", chunk_id="lexical-1", content="### File: readme.md\nreadme\n"),
        ]

        merged = main._interleave_reference_candidates(vector, anchor, lexical, limit=4)

        self.assertEqual([chunk.document_id for chunk in merged], ["vector-1", "anchor-1", "lexical-1", "vector-2"])

    def test_change_context_for_chunk_captures_before_and_after_lines(self) -> None:
        prev_chunks = [
            "### File: docs/api_mapping.md\n"
            "| `docs list` | `GET /load_documents` |\n"
            "| `activity` | `GET /get_activity_feed` |\n"
            "| `notifications` | `GET /notification_events` |\n"
        ]
        chunk = (
            "### File: docs/api_mapping.md\n"
            "| `docs list` | `GET /load_documents` |\n"
            "| `activity` | `GET /activity_feed` |\n"
            "| `notifications` | `GET /notification_events` |\n"
        )
        change_context = main._change_context_for_chunk(chunk, prev_chunks, code_focus=True)
        self.assertIn("- | `activity` | `GET /get_activity_feed` |", change_context)
        self.assertIn("+ | `activity` | `GET /activity_feed` |", change_context)

    def test_lexical_reference_candidates_prioritize_exact_route_artifacts(self) -> None:
        target = "| `activity` | `GET /activity_feed` |"
        candidates = [
            DummyChunk(
                document_id="route",
                content=(
                    "### File: desktop/api_mapping.md\n"
                    "| `activity` | `GET /get_activity_feed` |\n"
                    "| `notifications` | `GET /notification_events` |\n"
                ),
            ),
            DummyChunk(
                document_id="auth",
                content=(
                    "### File: docs/core_quickstart.md\n"
                    "Google OAuth is available on Core and should appear in /capabilities when configured.\n"
                ),
            ),
        ]
        selected = main._lexical_reference_candidates(target, candidates, limit=2, code_focus=True)
        self.assertEqual([chunk.document_id for chunk in selected], ["route"])

    def test_lexical_reference_candidates_prioritize_high_signal_metadata_pairs(self) -> None:
        target = (
            "### File: pyproject.toml\n"
            'license = { text = "MIT" }\n'
            'name = "compair-core"\n'
        )
        candidates = [
            DummyChunk(
                document_id="license",
                content=(
                    "### File: LICENSE\n"
                    "GNU GENERAL PUBLIC LICENSE\n"
                ),
            ),
            DummyChunk(
                document_id="docs",
                content=(
                    "### File: README.md\n"
                    "Compair is a context manager for teams.\n"
                ),
            ),
        ]
        selected = main._lexical_reference_candidates(target, candidates, limit=2, code_focus=True)
        self.assertEqual(selected[0].document_id, "license")

    def test_anchor_reference_candidates_prioritize_route_method_conflict(self) -> None:
        target = (
            "### File: desktop-app/src/main.js\n"
            'await fetch(`${base}/delete_group?group_id=${groupId}`, { method: "GET" })\n'
        )
        candidates = [
            DummyChunk(
                document_id="route-conflict",
                content=(
                    "### File: compair_core/api.py\n"
                    '@router.delete("/delete_group")\n'
                    "def delete_group(group_id: str):\n"
                ),
            ),
            DummyChunk(
                document_id="generic",
                content=(
                    "### File: docs/groups.md\n"
                    "Group folders can be deleted from the desktop app.\n"
                ),
            ),
        ]
        selected = main._anchor_reference_candidates(target, candidates, limit=2, code_focus=True)
        self.assertEqual([chunk.document_id for chunk in selected], ["route-conflict"])

    def test_anchor_reference_candidates_prioritize_license_realm_conflict(self) -> None:
        target = (
            "### File: pyproject.toml\n"
            'license = { text = "MIT" }\n'
            'name = "compair-core"\n'
        )
        candidates = [
            DummyChunk(
                document_id="license",
                content=(
                    "### File: LICENSE\n"
                    "GNU GENERAL PUBLIC LICENSE\n"
                    "Version 3, 29 June 2007\n"
                ),
            ),
            DummyChunk(
                document_id="readme",
                content=(
                    "### File: README.md\n"
                    "Compair is a context manager for teams.\n"
                ),
            ),
        ]
        selected = main._anchor_reference_candidates(target, candidates, limit=2, code_focus=True)
        self.assertEqual(selected[0].document_id, "license")

    def test_rerank_reference_chunks_prioritize_structured_delivery_settings_pair(self) -> None:
        target = (
            "### File: compair_ui/components/settings.py\n"
            '"notification_delivery_email_effective": prefs.get("notification_delivery_email_effective"),\n'
            'delivery_endpoint = "/notification_preferences/delivery_email"\n'
        )
        candidates = [
            DummyChunk(
                document_id="generic",
                content=(
                    "### File: README.md\n"
                    "Notification delivery can be configured through the browser UI and CLI.\n"
                ),
            ),
            DummyChunk(
                document_id="api-surface",
                content=(
                    "### File: compair_core/api.py\n"
                    '@router.post("/notification_preferences/delivery_email")\n'
                    'notification_delivery_email_effective = prefs.notification_delivery_email_effective\n'
                ),
            ),
        ]
        ranked = main._rerank_reference_chunks(target, candidates, code_focus=True)
        self.assertGreaterEqual(len(ranked), 2)
        self.assertEqual(ranked[0].document_id, "api-surface")

    def test_rerank_reference_chunks_prioritize_direct_capability_contradiction(self) -> None:
        target = (
            "### File: docs/core_quickstart.md\n"
            "Google OAuth is available on Core and should appear in /capabilities when client credentials are configured.\n"
        )
        candidates = [
            DummyChunk(
                document_id="provider",
                content=(
                    "### File: docs/providers.md\n"
                    "OpenAI is the primary generation provider path for local Core deployments.\n"
                ),
            ),
            DummyChunk(
                document_id="contradiction",
                content=(
                    "### File: README.md\n"
                    "Google OAuth is a Cloud-only path and is not expected on Core.\n"
                    "The /capabilities response should not advertise Google OAuth for a pure Core deployment.\n"
                ),
            ),
        ]
        ranked = main._rerank_reference_chunks(target, candidates, code_focus=True)
        self.assertGreaterEqual(len(ranked), 2)
        self.assertEqual(ranked[0].document_id, "contradiction")

    def test_rerank_reference_chunks_can_diversify_repeated_candidate_paths(self) -> None:
        target = (
            "### File: README.md\n"
            "Set `COMPAIR_EMAIL_BACKEND=stdout` for local development.\n"
            "Core logs verification emails to stdout through the configured mailer backend.\n"
        )
        candidates = [
            DummyChunk(
                chunk_id="docs-a",
                document_id="repo",
                content=(
                    "### File: docs/user-guide.md\n"
                    "Set `COMPAIR_EMAIL_BACKEND=stdout` for local development.\n"
                    "Verification emails are logged to stdout by the mailer backend.\n"
                ),
            ),
            DummyChunk(
                chunk_id="docs-b",
                document_id="repo",
                content=(
                    "### File: docs/user-guide.md\n"
                    "For local development, set `COMPAIR_EMAIL_BACKEND=stdout`.\n"
                    "The mailer backend writes verification emails to stdout.\n"
                ),
            ),
            DummyChunk(
                chunk_id="impl",
                document_id="repo",
                content=(
                    "### File: compair_core/server/providers/console_mailer.py\n"
                    "class ConsoleMailer:\n"
                    "    backend = 'stdout'\n"
                    "    def send_verification_email(self, subject, sender, receivers, html):\n"
                    "        print('[MAIL]', subject)\n"
                ),
            ),
        ]

        env_names = [
            "COMPAIR_CODE_REPO_REFERENCE_LIMIT",
            "COMPAIR_REFERENCE_ADJUDICATOR_ENABLED",
            "COMPAIR_REFERENCE_ADJUDICATOR_TOP_K",
            "COMPAIR_REFERENCE_PATH_DIVERSITY_PENALTY",
            "COMPAIR_REFERENCE_SOURCE_PENALTY_WEIGHT",
        ]
        original_env = {name: os.environ.get(name) for name in env_names}
        try:
            os.environ["COMPAIR_CODE_REPO_REFERENCE_LIMIT"] = "2"
            os.environ["COMPAIR_REFERENCE_ADJUDICATOR_ENABLED"] = "1"
            os.environ["COMPAIR_REFERENCE_ADJUDICATOR_TOP_K"] = "3"
            os.environ["COMPAIR_REFERENCE_PATH_DIVERSITY_PENALTY"] = "8"
            os.environ["COMPAIR_REFERENCE_SOURCE_PENALTY_WEIGHT"] = "0"
            ranked = main._rerank_reference_chunks(target, candidates, code_focus=True)
        finally:
            for name, value in original_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

        selected_paths = [main._extract_snapshot_file_path(chunk.content) for chunk in ranked]
        self.assertEqual(len(selected_paths), 2)
        self.assertEqual(selected_paths.count("docs/user-guide.md"), 1)
        self.assertIn("compair_core/server/providers/console_mailer.py", selected_paths)

    def test_reference_adjudication_payload_detects_docs_vs_impl_mismatch(self) -> None:
        payload = main._reference_adjudication_payload(
            target_text=(
                "### File: README.md\n"
                "Set `COMPAIR_EMAIL_BACKEND=stdout` for local development.\n"
                "Core logs verification emails to stdout.\n"
            ),
            candidate_text=(
                "### File: compair_core/server/providers/console_mailer.py\n"
                "class ConsoleMailer:\n"
                "    def send(self, subject, sender, receivers, html):\n"
                "        print('[MAIL]', subject)\n"
            ),
            candidate_path="compair_core/server/providers/console_mailer.py",
        )

        self.assertEqual(payload["adjudicator_kind"], "docs-vs-impl mismatch")
        self.assertGreater(float(payload["adjudicator_score"]), 0.0)

    def test_reference_adjudication_payload_does_not_treat_license_as_runtime_impl(self) -> None:
        payload = main._reference_adjudication_payload(
            target_text=(
                "### File: docs/user-guide.md\n"
                "Set `COMPAIR_EMAIL_BACKEND=stdout` for local development.\n"
                "Core logs verification emails to stdout through the mailer backend.\n"
            ),
            candidate_text=(
                "### File: LICENSE\n"
                "GNU GENERAL PUBLIC LICENSE\n"
                "Version 3, 29 June 2007\n"
            ),
            candidate_path="LICENSE",
        )

        self.assertNotEqual(payload["adjudicator_kind"], "docs-vs-impl mismatch")
        self.assertLess(float(payload["adjudicator_score"]), 1.0)

    def test_reference_counterpart_signal_boosts_manifest_license_pair(self) -> None:
        manifest = (
            "### File: pyproject.toml\n"
            'name = "compair-core"\n'
            'license = { text = "MIT" }\n'
        )
        license_text = (
            "### File: LICENSE\n"
            "GNU GENERAL PUBLIC LICENSE\n"
            "Version 3, 29 June 2007\n"
        )
        readme = (
            "### File: README.md\n"
            "Compair keeps teams aligned across projects.\n"
            "It reduces drift during review.\n"
        )

        license_score = main._reference_counterpart_signal(manifest, license_text)
        readme_score = main._reference_counterpart_signal(manifest, readme)

        self.assertGreater(license_score, readme_score)
        self.assertGreater(license_score, 1.0)

    def test_reference_counterpart_signal_boosts_docs_to_mailer_impl_pair(self) -> None:
        docs = (
            "### File: docs/user-guide.md\n"
            "Set `COMPAIR_EMAIL_BACKEND=stdout` for local development.\n"
            "Core logs verification emails to stdout through the mailer backend.\n"
        )
        impl = (
            "### File: compair_core/server/providers/console_mailer.py\n"
            "class ConsoleMailer:\n"
            "    backend = 'stdout'\n"
            "    def send_verification_email(self, subject, sender, receivers, html):\n"
            "        print('[MAIL]', subject)\n"
        )
        distractor = (
            "### File: docs/quickstart.md\n"
            "Run `compair login` and configure your API key to begin.\n"
        )

        impl_score = main._reference_counterpart_signal(docs, impl)
        distractor_score = main._reference_counterpart_signal(docs, distractor)

        self.assertGreater(impl_score, distractor_score)
        self.assertGreater(impl_score, 0.0)

    def test_reference_adjudication_payload_detects_manifest_license_mismatch(self) -> None:
        payload = main._reference_adjudication_payload(
            target_text=(
                "### File: pyproject.toml\n"
                'name = "compair-core"\n'
                'license = { text = "MIT" }\n'
            ),
            candidate_text=(
                "### File: LICENSE\n"
                "GNU GENERAL PUBLIC LICENSE\n"
                "Version 3, 29 June 2007\n"
            ),
            candidate_path="LICENSE",
        )

        self.assertEqual(payload["adjudicator_kind"], "value mismatch")
        self.assertGreater(float(payload["adjudicator_score"]), 0.0)

    def test_reference_counterpart_candidates_prioritize_manifest_license_pair(self) -> None:
        target = (
            "### File: pyproject.toml\n"
            'name = "compair-core"\n'
            'license = { text = "MIT" }\n'
        )
        candidates = [
            DummyChunk(
                document_id="manifest-peer",
                content=(
                    "### File: package.json\n"
                    '{\n  "name": "compair-ui",\n  "version": "0.1.0"\n}\n'
                ),
            ),
            DummyChunk(
                document_id="license-peer",
                content=(
                    "### File: LICENSE\n"
                    "GNU GENERAL PUBLIC LICENSE\n"
                    "Version 3, 29 June 2007\n"
                ),
            ),
        ]

        ranked = main._reference_counterpart_candidates(target, candidates, limit=2, code_focus=True)

        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0].document_id, "license-peer")

    def test_reference_counterpart_candidates_prioritize_docs_to_mailer_impl_pair(self) -> None:
        target = (
            "### File: docs/user-guide.md\n"
            "Set `COMPAIR_EMAIL_BACKEND=stdout` for local development.\n"
            "Core logs verification emails to stdout through the mailer backend.\n"
        )
        candidates = [
            DummyChunk(
                document_id="doc-peer",
                content=(
                    "### File: docs/user_guide.md\n"
                    "Set `COMPAIR_EMAIL_BACKEND=stdout` for local development.\n"
                    "See the user guide for more details.\n"
                ),
            ),
            DummyChunk(
                document_id="impl-peer",
                content=(
                    "### File: compair_core/server/providers/console_mailer.py\n"
                    "class ConsoleMailer:\n"
                    "    backend = 'stdout'\n"
                    "    def send_verification_email(self, subject, sender, receivers, html):\n"
                    "        print('[MAIL]', subject)\n"
                ),
            ),
        ]

        ranked = main._reference_counterpart_candidates(target, candidates, limit=2, code_focus=True)

        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0].document_id, "impl-peer")

    def test_reference_fts_queries_expand_manifest_to_legal_terms(self) -> None:
        target = (
            "### File: pyproject.toml\n"
            'name = "compair-core"\n'
            'license = { text = "MIT" }\n'
        )

        queries = main._reference_fts_queries(target, code_focus=True)
        joined = " || ".join(queries)

        self.assertTrue(queries)
        self.assertIn("license*", joined)
        self.assertIn("notice*", joined)
        self.assertIn("copying*", joined)

    def test_reference_fts_queries_expand_behavioral_doc_to_backend_terms(self) -> None:
        target = (
            "### File: docs/user-guide.md\n"
            "Set `COMPAIR_EMAIL_BACKEND=stdout` for local development.\n"
            "Core logs verification emails to stdout through the mailer backend.\n"
        )

        queries = main._reference_fts_queries(target, code_focus=True)
        joined = " || ".join(queries)

        self.assertTrue(queries)
        self.assertIn("email*", joined)
        self.assertIn("backend*", joined)
        self.assertIn("mailer*", joined)
        self.assertIn("provider*", joined)

    def test_reference_fts_candidates_prioritize_manifest_license_pair(self) -> None:
        if not main._reference_fts_available():
            self.skipTest("SQLite FTS5 unavailable")

        target = (
            "### File: pyproject.toml\n"
            'name = "compair-core"\n'
            'license = { text = "MIT" }\n'
        )
        candidates = [
            DummyChunk(
                document_id="manifest-peer",
                content=(
                    "### File: package.json\n"
                    '{\n  "name": "compair-ui",\n  "version": "0.1.0"\n}\n'
                ),
            ),
            DummyChunk(
                document_id="license-peer",
                content=(
                    "### File: LICENSE\n"
                    "GNU GENERAL PUBLIC LICENSE\n"
                    "Version 3, 29 June 2007\n"
                ),
            ),
        ]

        ranked = main._reference_fts_candidates(
            target,
            main._reference_query_variants(target, "", "", code_focus=True),
            candidates,
            limit=2,
            code_focus=True,
        )

        self.assertGreaterEqual(len(ranked), 1)
        self.assertEqual(ranked[0].document_id, "license-peer")

    def test_reference_fts_candidates_prioritize_docs_to_mailer_impl_pair(self) -> None:
        if not main._reference_fts_available():
            self.skipTest("SQLite FTS5 unavailable")

        target = (
            "### File: docs/user-guide.md\n"
            "Set `COMPAIR_EMAIL_BACKEND=stdout` for local development.\n"
            "Core logs verification emails to stdout through the mailer backend.\n"
        )
        candidates = [
            DummyChunk(
                document_id="doc-peer",
                content=(
                    "### File: docs/user_guide.md\n"
                    "Set `COMPAIR_EMAIL_BACKEND=stdout` for local development.\n"
                    "See the user guide for more details.\n"
                ),
            ),
            DummyChunk(
                document_id="impl-peer",
                content=(
                    "### File: compair_core/server/providers/console_mailer.py\n"
                    "class ConsoleMailer:\n"
                    "    backend = 'stdout'\n"
                    "    def send_verification_email(self, subject, sender, receivers, html):\n"
                    "        print('[MAIL]', subject)\n"
                ),
            ),
        ]

        ranked = main._reference_fts_candidates(
            target,
            main._reference_query_variants(target, "", "", code_focus=True),
            candidates,
            limit=2,
            code_focus=True,
        )

        self.assertGreaterEqual(len(ranked), 1)
        self.assertEqual(ranked[0].document_id, "impl-peer")

    def test_rerank_reference_chunks_promote_docs_to_impl_pair_with_adjudicator(self) -> None:
        target = (
            "### File: README.md\n"
            "Set `COMPAIR_EMAIL_BACKEND=stdout` for local development.\n"
            "Core logs verification emails to stdout.\n"
        )
        candidates = [
            DummyChunk(
                chunk_id="docs-peer",
                document_id="docs-peer",
                content=(
                    "### File: docs/quickstart.md\n"
                    "Set `COMPAIR_EMAIL_BACKEND=stdout` for local development.\n"
                    "See the user guide for other mailer backends.\n"
                ),
            ),
            DummyChunk(
                chunk_id="impl-peer",
                document_id="impl-peer",
                content=(
                    "### File: compair_core/server/providers/console_mailer.py\n"
                    "class ConsoleMailer:\n"
                    "    def send(self, subject, sender, receivers, html):\n"
                    "        print('[MAIL]', subject)\n"
                ),
            ),
        ]
        debug_stats: dict[str, object] = {}
        original_hybrid = os.environ.get("COMPAIR_REFERENCE_HYBRID_ENABLED")
        original_adjudicator = os.environ.get("COMPAIR_REFERENCE_ADJUDICATOR_ENABLED")
        try:
            os.environ["COMPAIR_REFERENCE_HYBRID_ENABLED"] = "1"
            os.environ["COMPAIR_REFERENCE_ADJUDICATOR_ENABLED"] = "1"
            ranked = main._rerank_reference_chunks(target, candidates, code_focus=True, debug_stats=debug_stats)
        finally:
            if original_hybrid is None:
                os.environ.pop("COMPAIR_REFERENCE_HYBRID_ENABLED", None)
            else:
                os.environ["COMPAIR_REFERENCE_HYBRID_ENABLED"] = original_hybrid
            if original_adjudicator is None:
                os.environ.pop("COMPAIR_REFERENCE_ADJUDICATOR_ENABLED", None)
            else:
                os.environ["COMPAIR_REFERENCE_ADJUDICATOR_ENABLED"] = original_adjudicator

        self.assertGreaterEqual(len(ranked), 2)
        self.assertEqual(ranked[0].document_id, "impl-peer")
        self.assertEqual(debug_stats.get("trimmed_candidate_count"), 2)
        self.assertIn("adjudicator_top_k", debug_stats)
        row_debug = debug_stats.get("row_debug_by_chunk_id")
        self.assertIsInstance(row_debug, dict)
        assert isinstance(row_debug, dict)
        docs_debug = row_debug.get("docs-peer")
        impl_debug = row_debug.get("impl-peer")
        self.assertIsInstance(docs_debug, dict)
        self.assertIsInstance(impl_debug, dict)
        assert isinstance(docs_debug, dict)
        assert isinstance(impl_debug, dict)
        for entry in (docs_debug, impl_debug):
            self.assertIn("preselection_score", entry)
            self.assertIn("preselection_rank", entry)
            self.assertIn("adjudicated", entry)
            self.assertIn("adjudication_reason", entry)
            self.assertIn("selector_round1_score", entry)
            self.assertIn("selector_round1_rank", entry)
        self.assertTrue(bool(impl_debug.get("adjudicated")))

    def test_rerank_reference_chunks_rescue_high_reranker_docs_to_impl_candidate(self) -> None:
        target = (
            "### File: docs/user-guide.md\n"
            "Set `COMPAIR_EMAIL_BACKEND=stdout` for local development.\n"
            "Core logs verification emails to stdout.\n"
            "The mailer backend controls how verification emails are delivered.\n"
        )
        candidates = [
            DummyChunk(
                chunk_id="docs-peer",
                document_id="docs-peer",
                content=(
                    "### File: docs/user_guide.md\n"
                    "Set `COMPAIR_EMAIL_BACKEND=stdout` for local development.\n"
                    "See the mailer backend guide for additional options.\n"
                ),
            ),
            DummyChunk(
                chunk_id="quickstart-peer",
                document_id="quickstart-peer",
                content=(
                    "### File: docs/quickstart.md\n"
                    "Run `compair login` and configure your API key to begin.\n"
                ),
            ),
            DummyChunk(
                chunk_id="impl-peer",
                document_id="impl-peer",
                content=(
                    "### File: compair_core/server/providers/console_mailer.py\n"
                    "class ConsoleMailer:\n"
                    "    def send(self, subject, sender, receivers, html):\n"
                    "        print('[MAIL]', subject)\n"
                ),
            ),
        ]

        debug_stats: dict[str, object] = {}
        original_hybrid = os.environ.get("COMPAIR_REFERENCE_HYBRID_ENABLED")
        original_adjudicator = os.environ.get("COMPAIR_REFERENCE_ADJUDICATOR_ENABLED")
        original_top_k = os.environ.get("COMPAIR_REFERENCE_ADJUDICATOR_TOP_K")
        original_rescue_count = os.environ.get("COMPAIR_REFERENCE_RERANKER_RESCUE_COUNT")
        original_rescue_min = os.environ.get("COMPAIR_REFERENCE_RERANKER_RESCUE_MIN_SCORE")
        original_metadata = main._reference_reranker_metadata
        original_score = main._reference_reranker_score
        try:
            os.environ["COMPAIR_REFERENCE_HYBRID_ENABLED"] = "1"
            os.environ["COMPAIR_REFERENCE_ADJUDICATOR_ENABLED"] = "1"
            os.environ["COMPAIR_REFERENCE_ADJUDICATOR_TOP_K"] = "1"
            os.environ["COMPAIR_REFERENCE_RERANKER_RESCUE_COUNT"] = "1"
            os.environ["COMPAIR_REFERENCE_RERANKER_RESCUE_MIN_SCORE"] = "0.5"
            main._reference_reranker_metadata = lambda: (True, "test-model", "/tmp/test-model.json")
            main._reference_reranker_score = lambda row: (
                0.6
                if str(row.get("candidate_path") or "").endswith("console_mailer.py")
                else 0.25
            )
            ranked = main._rerank_reference_chunks(target, candidates, code_focus=True, debug_stats=debug_stats)
        finally:
            main._reference_reranker_metadata = original_metadata
            main._reference_reranker_score = original_score
            if original_hybrid is None:
                os.environ.pop("COMPAIR_REFERENCE_HYBRID_ENABLED", None)
            else:
                os.environ["COMPAIR_REFERENCE_HYBRID_ENABLED"] = original_hybrid
            if original_adjudicator is None:
                os.environ.pop("COMPAIR_REFERENCE_ADJUDICATOR_ENABLED", None)
            else:
                os.environ["COMPAIR_REFERENCE_ADJUDICATOR_ENABLED"] = original_adjudicator
            if original_top_k is None:
                os.environ.pop("COMPAIR_REFERENCE_ADJUDICATOR_TOP_K", None)
            else:
                os.environ["COMPAIR_REFERENCE_ADJUDICATOR_TOP_K"] = original_top_k
            if original_rescue_count is None:
                os.environ.pop("COMPAIR_REFERENCE_RERANKER_RESCUE_COUNT", None)
            else:
                os.environ["COMPAIR_REFERENCE_RERANKER_RESCUE_COUNT"] = original_rescue_count
            if original_rescue_min is None:
                os.environ.pop("COMPAIR_REFERENCE_RERANKER_RESCUE_MIN_SCORE", None)
            else:
                os.environ["COMPAIR_REFERENCE_RERANKER_RESCUE_MIN_SCORE"] = original_rescue_min

        self.assertGreaterEqual(len(ranked), 3)
        self.assertEqual(ranked[0].document_id, "impl-peer")
        row_debug = debug_stats.get("row_debug_by_chunk_id")
        self.assertIsInstance(row_debug, dict)
        assert isinstance(row_debug, dict)
        impl_debug = row_debug.get("impl-peer")
        self.assertIsInstance(impl_debug, dict)
        assert isinstance(impl_debug, dict)
        self.assertEqual(impl_debug.get("adjudication_reason"), "rescued")
        self.assertTrue(bool(impl_debug.get("rescued_for_adjudication")))
        self.assertEqual(impl_debug.get("adjudication_rank"), 2)
        self.assertEqual(debug_stats.get("rescued_adjudication_count"), 1)

    def test_chunk_relevance_score_boosts_structured_doc_chunks(self) -> None:
        generic_doc = (
            "### File: docs/overview.md\n"
            "Compair helps teams understand changes across projects.\n"
            "It reduces drift and helps reviewers stay aligned.\n"
        )
        structured_doc = (
            "### File: docs/api_mapping.md\n"
            "| `notifications` | `GET /notification_events` |\n"
            "| `activity` | `GET /activity_feed` |\n"
            "Set COMPAIR_NOTIFICATION_DELIVERY=email to enable email delivery.\n"
        )

        generic_score = main._chunk_relevance_score(generic_doc, 0, True, 1.0)
        structured_score = main._chunk_relevance_score(structured_doc, 1, True, 1.0)

        self.assertGreater(structured_score, generic_score)

    def test_chunk_relevance_score_boosts_behavioral_docs_with_runtime_claims(self) -> None:
        generic_doc = (
            "### File: README.md\n"
            "Compair keeps teams aligned across projects.\n"
            "It helps reduce drift during reviews.\n"
        )
        behavioral_doc = (
            "### File: docs/user-guide.md\n"
            "Set `COMPAIR_EMAIL_BACKEND=stdout` for local development.\n"
            "Core uses the configured backend to send verification emails.\n"
            "The API returns delivery status in the notifications response.\n"
        )

        generic_score = main._chunk_relevance_score(generic_doc, 0, True, 1.0)
        behavioral_score = main._chunk_relevance_score(behavioral_doc, 1, True, 1.0)

        self.assertGreater(behavioral_score, generic_score)

    def test_chunk_relevance_score_boosts_legal_and_manifest_chunks(self) -> None:
        generic_doc = (
            "### File: docs/architecture.md\n"
            "The app contains a frontend, backend, and worker.\n"
            "Deployments can be local or hosted.\n"
        )
        manifest_chunk = (
            "### File: pyproject.toml\n"
            'name = "compair-core"\n'
            'license = { text = "MIT" }\n'
        )
        license_chunk = (
            "### File: LICENSE\n"
            "GNU GENERAL PUBLIC LICENSE\n"
            "Version 3, 29 June 2007\n"
        )

        generic_score = main._chunk_relevance_score(generic_doc, 0, True, 1.0)
        manifest_score = main._chunk_relevance_score(manifest_chunk, 1, True, 1.0)
        license_score = main._chunk_relevance_score(license_chunk, 2, True, 1.0)

        self.assertGreater(manifest_score, generic_score)
        self.assertGreater(license_score, generic_score)

    def test_prioritize_chunks_prefers_structured_public_surface_chunks(self) -> None:
        chunks = [
            (
                "### File: docs/overview.md\n"
                "Compair helps teams understand changes across projects.\n"
                "It reduces drift and helps reviewers stay aligned.\n"
            ),
            (
                "### File: docs/api_mapping.md\n"
                "| `notifications` | `GET /notification_events` |\n"
                "| `activity` | `GET /activity_feed` |\n"
                "Set COMPAIR_NOTIFICATION_DELIVERY=email to enable email delivery.\n"
            ),
            (
                "### File: docs/architecture.md\n"
                "The app contains a frontend, backend, and worker.\n"
                "Deployments can be local or hosted.\n"
            ),
        ]

        selected = main.prioritize_chunks([0, 1, 2], chunks, limit=1, code_focus=True)

        self.assertEqual(selected, [1])

    def test_prioritize_chunks_prefers_behavioral_docs_over_generic_docs(self) -> None:
        chunks = [
            (
                "### File: README.md\n"
                "Compair keeps teams aligned across projects.\n"
                "It helps reduce drift during reviews.\n"
            ),
            (
                "### File: docs/user-guide.md\n"
                "Set `COMPAIR_EMAIL_BACKEND=stdout` for local development.\n"
                "Core uses the configured backend to send verification emails.\n"
                "The API returns delivery status in the notifications response.\n"
            ),
            (
                "### File: docs/quickstart.md\n"
                "Run `compair login` and configure your API key to begin.\n"
            ),
        ]

        selected = main.prioritize_chunks([0, 1, 2], chunks, limit=1, code_focus=True)

        self.assertEqual(selected, [1])

    def test_prioritize_chunks_prefers_manifest_or_license_over_generic_docs(self) -> None:
        chunks = [
            (
                "### File: docs/overview.md\n"
                "Compair keeps teams aligned across projects.\n"
                "It helps reduce drift during reviews.\n"
            ),
            (
                "### File: LICENSE\n"
                "GNU GENERAL PUBLIC LICENSE\n"
                "Version 3, 29 June 2007\n"
            ),
            (
                "### File: pyproject.toml\n"
                'name = "compair-core"\n'
                'license = { text = "MIT" }\n'
            ),
        ]

        selected = main.prioritize_chunks([0, 1, 2], chunks, limit=2, code_focus=True)

        self.assertEqual(selected[:2], [2, 1])

    def test_prioritize_chunks_focus_manifest_boosts_matching_path(self) -> None:
        chunks = [
            (
                "### File: pyproject.toml\n"
                'name = "compair-core"\n'
                'license = { text = "MIT" }\n'
            ),
            (
                "### File: docs/overview.md\n"
                "Compair keeps teams aligned across projects.\n"
            ),
            (
                "### File: packages/admin/billing.ts\n"
                "export function billingUrl() { return '/billing'; }\n"
            ),
        ]

        def fake_relevance(chunk, idx, code_focus, novelty_score, *, chunks=None):
            return 5.0 if idx == 0 else 1.0

        with (
            mock.patch.object(main, "_source_redundancy_penalty", return_value=0.0),
            mock.patch.object(main, "_source_min_selection_score", return_value=-1.0),
            mock.patch.object(main, "_chunk_relevance_score", side_effect=fake_relevance),
        ):
            selected = main.prioritize_chunks(
                [0, 1, 2],
                chunks,
                limit=1,
                code_focus=True,
                focus_manifest={
                    "limits": {"max_boost": 5.0, "min_unfocused_fraction": 0.0},
                    "areas": [{"glob": "packages/admin/**", "weight": 5.0}],
                },
            )

        self.assertEqual(selected, [2])

    def test_prioritize_chunks_without_focus_manifest_keeps_existing_order(self) -> None:
        chunks = [
            "### File: docs/overview.md\nGeneral overview.\n",
            "### File: docs/api_mapping.md\n| `notifications` | `GET /notification_events` |\n",
            "### File: docs/architecture.md\nArchitecture notes.\n",
        ]

        selected_without_arg = main.prioritize_chunks([0, 1, 2], chunks, limit=2, code_focus=True)
        selected_with_none = main.prioritize_chunks([0, 1, 2], chunks, limit=2, code_focus=True, focus_manifest=None)

        self.assertEqual(selected_with_none, selected_without_arg)

    def test_prioritize_chunks_focus_manifest_reserves_unfocused_slots(self) -> None:
        chunks = [
            "### File: focused/a.py\nprint('a')\n",
            "### File: focused/b.py\nprint('b')\n",
            "### File: unfocused/c.py\nprint('c')\n",
        ]

        selected = main.prioritize_chunks(
            [0, 1, 2],
            chunks,
            limit=2,
            code_focus=True,
            focus_manifest={
                "limits": {"min_unfocused_fraction": 0.5},
                "areas": [{"glob": "focused/**", "weight": 5.0}],
            },
        )

        self.assertEqual(len(selected), 2)
        self.assertIn(2, selected)

    def test_source_trace_entries_capture_selected_and_filtered_reasons(self) -> None:
        chunks = [
            (
                "### File: docs/overview.md\n"
                "Compair keeps teams aligned across projects.\n"
            ),
            (
                "### File: docs/user-guide.md\n"
                "Set `COMPAIR_EMAIL_BACKEND=stdout` for local development.\n"
                "Core logs verification emails to stdout.\n"
            ),
            (
                "### File: LICENSE\n"
                "GNU GENERAL PUBLIC LICENSE\n"
            ),
        ]

        entries = main._source_trace_entries(
            new_chunks=chunks,
            code_focus=True,
            novelty_scores={0: 0.2, 1: 0.95, 2: 0.9},
            significant_candidate_indices={1, 2},
            prioritized_indices=[1, 2],
            selected_indices=[1],
            token_lens=[40, 180, 90],
            feedback_min_tokens=100,
            feedback_fallback_min=20,
        )

        by_path = {str(entry.get("path")): entry for entry in entries}
        self.assertEqual(by_path["docs/user-guide.md"]["selection_status"], "selected")
        self.assertEqual(by_path["docs/user-guide.md"]["selected_rank"], 1)
        self.assertEqual(by_path["LICENSE"]["selection_status"], "candidate")
        self.assertEqual(by_path["LICENSE"]["skip_reason"], "below_min_tokens")
        self.assertEqual(by_path["docs/overview.md"]["selection_status"], "filtered")
        self.assertEqual(by_path["docs/overview.md"]["skip_reason"], "below_significance_threshold")


if __name__ == "__main__":
    unittest.main()
