from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import os
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
                document_id="docs-peer",
                content=(
                    "### File: docs/quickstart.md\n"
                    "Set `COMPAIR_EMAIL_BACKEND=stdout` for local development.\n"
                    "See the user guide for other mailer backends.\n"
                ),
            ),
            DummyChunk(
                document_id="impl-peer",
                content=(
                    "### File: compair_core/server/providers/console_mailer.py\n"
                    "class ConsoleMailer:\n"
                    "    def send(self, subject, sender, receivers, html):\n"
                    "        print('[MAIL]', subject)\n"
                ),
            ),
        ]
        original_hybrid = os.environ.get("COMPAIR_REFERENCE_HYBRID_ENABLED")
        original_adjudicator = os.environ.get("COMPAIR_REFERENCE_ADJUDICATOR_ENABLED")
        try:
            os.environ["COMPAIR_REFERENCE_HYBRID_ENABLED"] = "1"
            os.environ["COMPAIR_REFERENCE_ADJUDICATOR_ENABLED"] = "1"
            ranked = main._rerank_reference_chunks(target, candidates, code_focus=True)
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
                1.25
                if str(row.get("candidate_path") or "").endswith("console_mailer.py")
                else 0.25
            )
            ranked = main._rerank_reference_chunks(target, candidates, code_focus=True)
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
