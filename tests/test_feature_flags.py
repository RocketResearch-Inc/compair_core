from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import unittest


def _load_feature_flags_module():
    root = pathlib.Path(__file__).resolve().parents[1]
    module_path = root / "compair_core" / "server" / "feature_flags.py"
    spec = importlib.util.spec_from_file_location("test_compair_feature_flags", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


feature_flags = _load_feature_flags_module()


class FeatureFlagTests(unittest.TestCase):
    def test_review_now_enabled_in_core_by_default(self) -> None:
        old_edition = os.environ.get("COMPAIR_EDITION")
        old_flag = os.environ.get("COMPAIR_REVIEW_NOW_ENABLED")
        try:
            os.environ["COMPAIR_EDITION"] = "core"
            os.environ.pop("COMPAIR_REVIEW_NOW_ENABLED", None)

            self.assertTrue(feature_flags.review_now_backend_enabled())
            self.assertIsNone(feature_flags.review_now_disabled_detail())
        finally:
            if old_edition is None:
                os.environ.pop("COMPAIR_EDITION", None)
            else:
                os.environ["COMPAIR_EDITION"] = old_edition
            if old_flag is None:
                os.environ.pop("COMPAIR_REVIEW_NOW_ENABLED", None)
            else:
                os.environ["COMPAIR_REVIEW_NOW_ENABLED"] = old_flag

    def test_review_now_disabled_in_cloud_until_explicitly_enabled(self) -> None:
        old_edition = os.environ.get("COMPAIR_EDITION")
        old_flag = os.environ.get("COMPAIR_REVIEW_NOW_ENABLED")
        try:
            os.environ["COMPAIR_EDITION"] = "cloud"
            os.environ.pop("COMPAIR_REVIEW_NOW_ENABLED", None)

            self.assertFalse(feature_flags.review_now_backend_enabled())
            self.assertIn("COMPAIR_REVIEW_NOW_ENABLED=1", feature_flags.review_now_disabled_detail() or "")

            os.environ["COMPAIR_REVIEW_NOW_ENABLED"] = "1"
            self.assertTrue(feature_flags.review_now_backend_enabled())
            self.assertIsNone(feature_flags.review_now_disabled_detail())
        finally:
            if old_edition is None:
                os.environ.pop("COMPAIR_EDITION", None)
            else:
                os.environ["COMPAIR_EDITION"] = old_edition
            if old_flag is None:
                os.environ.pop("COMPAIR_REVIEW_NOW_ENABLED", None)
            else:
                os.environ["COMPAIR_REVIEW_NOW_ENABLED"] = old_flag


if __name__ == "__main__":
    unittest.main()
