from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

import xgboost as xgb

_MODULE_PATH = Path(__file__).resolve().parents[1] / "compair_core" / "compair" / "reference_reranker.py"
_SPEC = importlib.util.spec_from_file_location("compair_reference_reranker_test", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load reference_reranker module from {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

REFERENCE_RERANKER_EMBEDDING_SCHEMA_V2 = _MODULE.REFERENCE_RERANKER_EMBEDDING_SCHEMA_V2
REFERENCE_RERANKER_LATEST_MANIFEST_NAME = _MODULE.REFERENCE_RERANKER_LATEST_MANIFEST_NAME
REFERENCE_RERANKER_MANIFEST_FORMAT = _MODULE.REFERENCE_RERANKER_MANIFEST_FORMAT
combined_feature_vector_from_trace_row = _MODULE.combined_feature_vector_from_trace_row
load_model = _MODULE.load_model
score_trace_row = _MODULE.score_trace_row


def _sample_row(*, source_preview: str, candidate_preview: str, candidate_path: str, same_document: bool = False) -> dict[str, object]:
    return {
        "source_path": "docs/user-guide.md",
        "candidate_path": candidate_path,
        "same_document": same_document,
        "vector_rank": 1,
        "lexical_rank": 1,
        "anchor_rank": 1,
        "lexical_score": 0.7,
        "path_theme_score": 0.3,
        "path_score": 1.0,
        "artifact_score": 0.5,
        "anchor_overlap": 0.4,
        "anchor_conflict": 0.1,
        "combined_signal": 4.2,
        "source_preview": source_preview,
        "candidate_preview": candidate_preview,
        "source_embedding": [0.4, 0.2],
        "candidate_embedding": [0.39, 0.21],
    }


class ReferenceRerankerTests(unittest.TestCase):
    def test_load_model_resolves_latest_manifest_for_xgboost(self) -> None:
        positive = _sample_row(
            source_preview=(
                "### File: docs/user-guide.md\n"
                "Set `COMPAIR_EMAIL_BACKEND=stdout` for local development.\n"
                "The default mailer logs messages to stdout.\n"
            ),
            candidate_preview=(
                "### File: compair_core/server/providers/console_mailer.py\n"
                "class ConsoleMailer:\n"
                "    def send(self, subject, sender, receivers, html):\n"
                "        print('[MAIL]', subject)\n"
            ),
            candidate_path="compair_core/server/providers/console_mailer.py",
            same_document=True,
        )
        negative = _sample_row(
            source_preview=positive["source_preview"],  # type: ignore[index]
            candidate_preview=(
                "### File: docs/quickstart.md\n"
                "Run `compair core up` and `compair core logs --tail 200`.\n"
            ),
            candidate_path="docs/quickstart.md",
            same_document=True,
        )
        positive_vector = combined_feature_vector_from_trace_row(positive)
        negative_vector = combined_feature_vector_from_trace_row(negative)
        self.assertIsNotNone(positive_vector)
        self.assertIsNotNone(negative_vector)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_path = tmp_path / "reference_reranker_xgb_test.json"
            summary_path = tmp_path / "reference_reranker_xgb_test.summary.json"
            manifest_path = tmp_path / REFERENCE_RERANKER_LATEST_MANIFEST_NAME

            model = xgb.XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",
                max_depth=2,
                learning_rate=0.2,
                n_estimators=8,
                random_state=42,
            )
            model.fit([positive_vector, negative_vector], [2.0, 0.0], verbose=False)
            model.save_model(model_path)
            summary_path.write_text(json.dumps({"selected_model_type": "regressor"}, indent=2) + "\n")
            manifest_path.write_text(
                json.dumps(
                    {
                        "model_format": REFERENCE_RERANKER_MANIFEST_FORMAT,
                        "model_kind": "xgboost",
                        "model_version": "test-v1",
                        "selected_model_type": "regressor",
                        "feature_schema": REFERENCE_RERANKER_EMBEDDING_SCHEMA_V2,
                        "artifact_path": model_path.name,
                        "summary_path": summary_path.name,
                    },
                    indent=2,
                )
                + "\n"
            )

            loaded = load_model(tmp_path)
            self.assertEqual(loaded["model_kind"], "xgboost")
            self.assertEqual(loaded["model_version"], "test-v1")

            positive_score = score_trace_row(positive, loaded)
            negative_score = score_trace_row(negative, loaded)
            self.assertGreater(positive_score, negative_score)


if __name__ == "__main__":
    unittest.main()
