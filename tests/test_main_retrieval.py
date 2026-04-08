from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import pathlib
import sys
import types
import unittest


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
    sys.modules["Levenshtein"] = levenshtein

    sqlalchemy = types.ModuleType("sqlalchemy")
    sqlalchemy.select = lambda *args, **kwargs: None
    sqlalchemy.or_ = lambda *args, **kwargs: ("or", args)
    sys.modules["sqlalchemy"] = sqlalchemy

    sqlalchemy_orm = types.ModuleType("sqlalchemy.orm")
    sqlalchemy_orm.Session = object
    sys.modules["sqlalchemy.orm"] = sqlalchemy_orm

    sqlalchemy_orm_attributes = types.ModuleType("sqlalchemy.orm.attributes")
    sqlalchemy_orm_attributes.get_history = lambda *args, **kwargs: types.SimpleNamespace(deleted=[])
    sys.modules["sqlalchemy.orm.attributes"] = sqlalchemy_orm_attributes

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
    utils.stable_chunk_hash = lambda text: hashlib.sha256((text or "").encode("utf-8")).hexdigest()
    sys.modules[utils.__name__] = utils

    spec = importlib.util.spec_from_file_location(f"{package_name}.main", main_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


main = _load_main_module()


@dataclass
class DummyChunk:
    content: str
    chunk_type: str = "document"
    document_id: str | None = None
    note_id: str | None = None
    chunk_id: str | None = None


class MainRetrievalTests(unittest.TestCase):
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
        )

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
        )

        self.assertFalse(allowed)

    def test_should_reanalyze_existing_chunks_ignores_snapshot_only_new_chunks(self) -> None:
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
        self.assertFalse(
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


if __name__ == "__main__":
    unittest.main()
