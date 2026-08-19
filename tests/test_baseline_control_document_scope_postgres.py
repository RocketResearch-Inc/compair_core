"""Real PostgreSQL multi-source control-document corpus coverage.

Run with::

    COMPAIR_TEST_POSTGRES_URL=postgresql+psycopg2://... \
      pytest -q tests/test_baseline_control_document_scope_postgres.py
"""

from __future__ import annotations

import pytest
from test_baseline_control_document_scope import (
    test_two_sources_share_a_group_without_corpus_or_publication_collision as _assert_two_source_flow,
)
from test_baseline_control_plane_postgres import (
    postgres_control_environment as _postgres_control_environment_fixture,  # noqa: F401
)


@pytest.fixture
def postgres_control_environment(request: pytest.FixtureRequest):
    return request.getfixturevalue("_postgres_control_environment_fixture")


def test_postgres_two_sources_have_independent_corpora_and_publications(
    postgres_control_environment,
) -> None:
    _assert_two_source_flow(postgres_control_environment)
