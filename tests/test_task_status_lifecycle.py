from datetime import datetime, timedelta, timezone

from compair_core import api


def test_stale_prechunk_parent_recommends_smaller_resubmit(monkeypatch):
    monkeypatch.setenv("COMPAIR_STATUS_STALE_AFTER_SEC", "900")
    stale_progress = (datetime.now(timezone.utc) - timedelta(seconds=901)).isoformat()

    lifecycle, health, retryable, terminal, recommended_action = api._task_lifecycle(
        "PROGRESS",
        {
            "stage": "preparing",
            "last_progress_at": stale_progress,
        },
        None,
    )

    assert lifecycle == "running"
    assert health == "stale_prechunk"
    assert retryable is False
    assert terminal is False
    assert recommended_action == "resubmit_smaller_or_inspect_worker"


def test_stale_chunked_parent_keeps_worker_inspection_guidance(monkeypatch):
    monkeypatch.setenv("COMPAIR_STATUS_STALE_AFTER_SEC", "900")
    stale_progress = (datetime.now(timezone.utc) - timedelta(seconds=901)).isoformat()

    lifecycle, health, retryable, terminal, recommended_action = api._task_lifecycle(
        "PROGRESS",
        {
            "stage": "indexing",
            "last_progress_at": stale_progress,
        },
        None,
    )

    assert lifecycle == "running"
    assert health == "stale"
    assert retryable is False
    assert terminal is False
    assert recommended_action == "inspect_worker"
