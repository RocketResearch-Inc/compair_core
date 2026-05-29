from __future__ import annotations

import importlib.util
import pathlib
import sys
import types


SPECIAL_TEXT = "literal model marker <|endoftext|> should be source text"
ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE_NAME = "test_compair_tokenization_pkg"


def _load_module(module_name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_core_module(name: str):
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(ROOT / "compair_core" / "compair")]
    sys.modules[PACKAGE_NAME] = package

    models = types.ModuleType(f"{PACKAGE_NAME}.models")
    models.Activity = object
    sys.modules[models.__name__] = models

    logger = types.ModuleType(f"{PACKAGE_NAME}.logger")
    logger.log_event = lambda *args, **kwargs: None
    sys.modules[logger.__name__] = logger

    return _load_module(f"{PACKAGE_NAME}.{name}", ROOT / "compair_core" / "compair" / f"{name}.py")


def test_count_tokens_treats_special_token_markers_as_source_text() -> None:
    utils = _load_core_module("utils")

    assert utils.count_tokens(SPECIAL_TEXT) > 0


def test_chunking_treats_special_token_markers_as_source_text() -> None:
    utils = _load_core_module("utils")

    chunks = utils.chunk_text_smart(
        f"Heading\n\n{SPECIAL_TEXT}\n\nMore content after the marker.",
        target_tokens=8,
        overlap_tokens=2,
        min_tokens=1,
        max_tokens=12,
    )

    assert chunks
    assert any("endoftext" in chunk for chunk in chunks)


def test_embedding_token_helpers_treat_special_token_markers_as_source_text() -> None:
    embeddings = _load_core_module("embeddings")

    assert embeddings._token_count(SPECIAL_TEXT, "text-embedding-3-small") > 0
    assert embeddings._split_text_for_embedding(SPECIAL_TEXT, "text-embedding-3-small", 8)


def test_bundle_token_counter_treats_special_token_markers_as_source_text() -> None:
    bundle_review = _load_module(
        "test_compair_bundle_review_special",
        ROOT / "compair_core" / "compair" / "bundle_review.py",
    )

    quote = bundle_review.count_model_tokens(SPECIAL_TEXT, model="gpt-4o")
    assert quote["tokens"] > 0
