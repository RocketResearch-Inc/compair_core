from __future__ import annotations

import base64
import json
import logging
import os
import re
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from compair_core import config_init as config_init_module
from compair_core import doctor as doctor_module
from compair_core.config_init import (
    CONFIG_INIT_ENVIRONMENT_VARIABLE,
    CONFIG_INIT_RESULT_SCHEMA_VERSION,
    EXIT_DESTINATION_EXISTS,
    EXIT_GENERATION,
    EXIT_INSECURE_PATH,
    EXIT_PUBLICATION,
    EXIT_USAGE_OR_PATH,
    ConfigInitError,
    default_config_path,
    initialize_baseline_config,
)
from compair_core.doctor import run_doctor
from compair_core.run_keyring import (
    RUN_KEYRING_VERSION,
    RunKeyringGenerationError,
    parse_run_keyring,
)
from compair_core.runtime_config import attest_keyring, build_runtime_configuration
from compair_core.server.settings import Settings

_ASSIGNMENT = re.compile(
    rf"^{CONFIG_INIT_ENVIRONMENT_VARIABLE}='([^'\n]+)'\n$"
)


def _fragment(path: Path) -> tuple[str, dict[str, object]]:
    rendered = path.read_text(encoding="utf-8")
    matched = _ASSIGNMENT.fullmatch(rendered)
    assert matched is not None
    serialized = matched.group(1)
    parsed = json.loads(serialized)
    assert isinstance(parsed, dict)
    return serialized, parsed


def _secret(payload: dict[str, object]) -> bytes:
    entries = payload["keys"]
    assert isinstance(entries, list) and len(entries) == 1
    entry = entries[0]
    assert isinstance(entry, dict)
    return base64.b64decode(str(entry["key_base64"]), validate=True)


def test_exact_keyring_contract_serialization_and_settings_acceptance(tmp_path) -> None:
    output = tmp_path / "private" / "baseline.env"
    result = initialize_baseline_config(
        output,
        clock=lambda: datetime(2026, 8, 18, tzinfo=timezone.utc),
    )
    serialized, payload = _fragment(output)
    assert payload == {
        "active_key_id": result.active_key_id,
        "keys": [
            {
                "key_base64": payload["keys"][0]["key_base64"],
                "key_id": result.active_key_id,
            }
        ],
        "version": RUN_KEYRING_VERSION,
    }
    assert serialized == json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert len(_secret(payload)) == 32
    parsed = parse_run_keyring(serialized)
    assert parsed.active_key_id == result.active_key_id
    settings = Settings(baseline_run_encryption_keyring=serialized)
    assert attest_keyring(settings.baseline_run_encryption_keyring).valid is True
    assert result.as_dict() == {
        "schema_version": CONFIG_INIT_RESULT_SCHEMA_VERSION,
        "created": True,
        "keyring_schema_version": RUN_KEYRING_VERSION,
        "active_key_id": result.active_key_id,
        "key_count": 1,
        "file_mode": "0600",
        "destination": "explicit",
        "timestamp": "2026-08-18T00:00:00Z",
    }
    rendered = output.read_text(encoding="utf-8")
    assert rendered.endswith("\n")
    assert "$(" not in rendered
    assert "`" not in rendered


def test_repeated_isolated_initializations_use_unique_keys_and_ids(tmp_path) -> None:
    keys: set[bytes] = set()
    key_ids: set[str] = set()
    for ordinal in range(12):
        output = tmp_path / str(ordinal) / "baseline.env"
        initialize_baseline_config(output)
        _, payload = _fragment(output)
        keys.add(_secret(payload))
        key_ids.add(str(payload["active_key_id"]))
    assert len(keys) == 12
    assert len(key_ids) == 12


def test_permissions_private_parent_and_explicit_output(tmp_path) -> None:
    output = tmp_path / "new" / "nested" / "baseline.env"
    result = initialize_baseline_config(output)
    assert result.destination_classification == "explicit"
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert stat.S_IMODE(output.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(output.parent.parent.stat().st_mode) == 0o700


def test_default_xdg_and_home_locations_are_isolated(tmp_path) -> None:
    xdg = tmp_path / "xdg"
    expected_xdg = xdg / "compair-core" / "baseline.env"
    assert default_config_path(environ={"XDG_CONFIG_HOME": str(xdg)}) == expected_xdg
    xdg_result = initialize_baseline_config(
        environ={"XDG_CONFIG_HOME": str(xdg)}
    )
    assert xdg_result.destination_classification == "default"
    assert expected_xdg.exists()

    home = tmp_path / "home"
    expected_home = home / ".config" / "compair-core" / "baseline.env"
    assert default_config_path(environ={}, home_directory=home) == expected_home
    home_result = initialize_baseline_config(environ={}, home_directory=home)
    assert home_result.destination_classification == "default"
    assert expected_home.exists()


def test_relative_and_relative_xdg_paths_are_rejected(tmp_path) -> None:
    with pytest.raises(ConfigInitError) as relative:
        initialize_baseline_config("relative/baseline.env")
    assert relative.value.exit_code == EXIT_USAGE_OR_PATH
    with pytest.raises(ConfigInitError) as xdg:
        initialize_baseline_config(
            environ={"XDG_CONFIG_HOME": "relative"},
            home_directory=tmp_path,
        )
    assert xdg.value.exit_code == EXIT_USAGE_OR_PATH


def test_existing_destination_is_not_read_changed_or_repaired(tmp_path) -> None:
    output = tmp_path / "baseline.env"
    marker = b"existing private bytes\n"
    output.write_bytes(marker)
    output.chmod(0o640)
    before = output.stat()
    with pytest.raises(ConfigInitError) as caught:
        initialize_baseline_config(output)
    assert caught.value.code == "destination_already_exists"
    assert caught.value.exit_code == EXIT_DESTINATION_EXISTS
    assert output.read_bytes() == marker
    assert stat.S_IMODE(output.stat().st_mode) == stat.S_IMODE(before.st_mode)


def test_symlinked_destination_and_parent_are_rejected(tmp_path) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    target = private / "target"
    target.write_text("unchanged", encoding="utf-8")
    destination = private / "baseline.env"
    destination.symlink_to(target)
    with pytest.raises(ConfigInitError) as symlinked_destination:
        initialize_baseline_config(destination)
    assert symlinked_destination.value.code == "destination_symlink_rejected"
    assert symlinked_destination.value.exit_code == EXIT_INSECURE_PATH
    assert target.read_text(encoding="utf-8") == "unchanged"

    actual_parent = tmp_path / "actual-parent"
    actual_parent.mkdir(mode=0o700)
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(actual_parent, target_is_directory=True)
    with pytest.raises(ConfigInitError) as symlinked_parent:
        initialize_baseline_config(linked_parent / "second.env")
    assert symlinked_parent.value.code == "parent_symlink_rejected"
    assert not (actual_parent / "second.env").exists()


def test_insecure_direct_parent_permissions_are_rejected(tmp_path) -> None:
    parent = tmp_path / "shared"
    parent.mkdir()
    parent.chmod(0o777)
    try:
        with pytest.raises(ConfigInitError) as caught:
            initialize_baseline_config(parent / "baseline.env")
        assert caught.value.code == "insecure_parent_permissions"
        assert caught.value.exit_code == EXIT_INSECURE_PATH
    finally:
        parent.chmod(0o700)


def test_concurrent_initialization_has_exactly_one_winner(tmp_path) -> None:
    output = tmp_path / "private" / "baseline.env"

    def invoke() -> tuple[str, str | None]:
        try:
            result = initialize_baseline_config(output)
            return "created", result.active_key_id
        except ConfigInitError as exc:
            return exc.code, None

    with ThreadPoolExecutor(max_workers=8) as executor:
        outcomes = list(executor.map(lambda _value: invoke(), range(8)))
    assert [state for state, _ in outcomes].count("created") == 1
    assert [state for state, _ in outcomes].count("destination_already_exists") == 7
    serialized, payload = _fragment(output)
    assert parse_run_keyring(serialized).active_key_id == payload["active_key_id"]
    assert list(output.parent.glob(".baseline.env.tmp.*")) == []


def test_concurrent_processes_have_exactly_one_publication(tmp_path) -> None:
    root = Path(__file__).parents[1]
    output = tmp_path / "private" / "baseline.env"
    command = [
        sys.executable,
        "-m",
        "compair_core.doctor",
        "config",
        "init",
        "--output",
        str(output),
        "--json",
    ]
    environment = {
        **os.environ,
        "PYTHONPATH": str(root),
        "COMPAIR_DB_DIR": str(tmp_path / "unused-database"),
    }
    processes = [
        subprocess.Popen(
            command,
            cwd=tmp_path,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(6)
    ]
    completed = [
        (process.wait(), process.stdout.read(), process.stderr.read())
        for process in processes
    ]
    assert sorted(code for code, _stdout, _stderr in completed) == [0, 3, 3, 3, 3, 3]
    assert all(stderr == "" for _code, _stdout, stderr in completed)
    payloads = [json.loads(stdout) for _code, stdout, _stderr in completed]
    assert sum(payload["created"] is True for payload in payloads) == 1
    assert sum(payload["created"] is False for payload in payloads) == 5
    serialized, payload = _fragment(output)
    encoded_secret = str(payload["keys"][0]["key_base64"])
    combined = "".join(stdout + stderr for _code, stdout, stderr in completed)
    assert serialized not in combined
    assert encoded_secret not in combined
    assert not (tmp_path / "unused-database").exists()
    assert list(output.parent.glob(".baseline.env.tmp.*")) == []


@pytest.mark.parametrize("stage", ["before_temporary_write", "after_temporary_write"])
def test_injected_write_failures_leave_no_partial_or_temporary_file(
    tmp_path,
    stage: str,
) -> None:
    output = tmp_path / "private" / "baseline.env"

    def fail(observed: str) -> None:
        if observed == stage:
            raise OSError("injected")

    with pytest.raises(ConfigInitError) as caught:
        initialize_baseline_config(output, fault_injector=fail)
    assert caught.value.code == "atomic_publication_failed"
    assert caught.value.exit_code == EXIT_PUBLICATION
    assert not output.exists()
    assert list(output.parent.glob(".baseline.env.tmp.*")) == []


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("random_generation_failed", "random_generation_failed"),
        ("serialization_failed", "serialization_failed"),
    ],
)
def test_generation_failures_are_sanitized_and_write_nothing(
    tmp_path,
    reason: str,
    expected: str,
) -> None:
    output = tmp_path / "private" / "baseline.env"

    def fail():
        raise RunKeyringGenerationError(reason)

    with pytest.raises(ConfigInitError) as caught:
        initialize_baseline_config(output, generator=fail)
    assert caught.value.code == expected
    assert caught.value.exit_code == EXIT_GENERATION
    assert not output.exists()


def test_atomic_link_failure_is_sanitized_and_removes_temporary_file(
    tmp_path,
    monkeypatch,
) -> None:
    output = tmp_path / "private" / "baseline.env"

    def fail_link(*_args, **_kwargs):
        raise OSError("injected link failure with private detail")

    monkeypatch.setattr(config_init_module.os, "link", fail_link)
    with pytest.raises(ConfigInitError) as caught:
        initialize_baseline_config(output)
    assert caught.value.code == "atomic_publication_failed"
    assert caught.value.exit_code == EXIT_PUBLICATION
    assert "private detail" not in str(caught.value)
    assert not output.exists()
    assert list(output.parent.glob(".baseline.env.tmp.*")) == []


def test_success_fsyncs_file_and_containing_directory(tmp_path, monkeypatch) -> None:
    output = tmp_path / "private" / "baseline.env"
    observed: list[int] = []
    original = os.fsync

    def track(file_descriptor: int) -> None:
        observed.append(file_descriptor)
        original(file_descriptor)

    monkeypatch.setattr(config_init_module.os, "fsync", track)
    initialize_baseline_config(output)
    assert len(observed) >= 3


def test_json_command_output_and_errors_never_disclose_secret_or_path(
    tmp_path,
    monkeypatch,
    capsys,
    caplog,
) -> None:
    xdg = tmp_path / "private-xdg"
    output = xdg / "compair-core" / "baseline.env"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.setenv("HOME", str(tmp_path / "private-home"))
    caplog.set_level(logging.DEBUG)
    assert doctor_module.main(["config", "init", "--json"]) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    serialized, payload = _fragment(output)
    encoded_secret = str(payload["keys"][0]["key_base64"])
    assert result["schema_version"] == CONFIG_INIT_RESULT_SCHEMA_VERSION
    assert result["created"] is True
    assert result["destination"] == "default"
    assert captured.err == ""
    combined = captured.out + captured.err + caplog.text
    assert serialized not in combined
    assert encoded_secret not in combined
    assert str(tmp_path) not in captured.out
    assert os.getenv("HOME", "") not in captured.out
    before = output.read_bytes()

    assert doctor_module.main(["config", "init", "--json"]) == (
        EXIT_DESTINATION_EXISTS
    )
    replay = capsys.readouterr()
    replay_payload = json.loads(replay.out)
    assert replay_payload == {
        "schema_version": CONFIG_INIT_RESULT_SCHEMA_VERSION,
        "created": False,
        "reason_code": "destination_already_exists",
    }
    assert replay.err == ""
    assert output.read_bytes() == before
    assert encoded_secret not in replay.out + replay.err + caplog.text


def test_runtime_fingerprints_agree_for_shared_fragment_and_mismatch_for_new_one(
    tmp_path,
) -> None:
    shared = tmp_path / "shared" / "baseline.env"
    separate = tmp_path / "separate" / "baseline.env"
    initialize_baseline_config(shared)
    initialize_baseline_config(separate)
    shared_raw, _ = _fragment(shared)
    separate_raw, _ = _fragment(separate)
    api = build_runtime_configuration(
        Settings(baseline_run_encryption_keyring=shared_raw),
        database_url="sqlite:////private/example/runtime.db",
    )
    worker = build_runtime_configuration(
        Settings(baseline_run_encryption_keyring=shared_raw),
        database_url="sqlite:////private/example/runtime.db",
    )
    mismatched = build_runtime_configuration(
        Settings(baseline_run_encryption_keyring=separate_raw),
        database_url="sqlite:////private/example/runtime.db",
    )
    assert api.keyring_identity_fingerprint == worker.keyring_identity_fingerprint
    assert api.fingerprint == worker.fingerprint
    assert api.keyring_identity_fingerprint != mismatched.keyring_identity_fingerprint
    assert api.fingerprint != mismatched.fingerprint


def test_doctor_reports_generated_keyring_valid_while_other_components_can_fail(
    tmp_path,
) -> None:
    output = tmp_path / "private" / "baseline.env"
    initialize_baseline_config(output)
    serialized, _ = _fragment(output)
    engine = create_engine(f"sqlite:///{tmp_path / 'doctor.db'}")
    try:
        result = run_doctor(
            settings=Settings(baseline_run_encryption_keyring=serialized),
            engine=engine,
        )
    finally:
        engine.dispose()
    assert result.component("keyring").status == "ready"
    assert result.component("keyring").reason_code == "keyring_valid"
    assert result.status != "ready"


def test_config_init_subprocess_does_not_initialize_database(tmp_path) -> None:
    root = Path(__file__).parents[1]
    xdg = tmp_path / "xdg"
    database = tmp_path / "database"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "compair_core.doctor",
            "config",
            "init",
            "--json",
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": str(root),
            "XDG_CONFIG_HOME": str(xdg),
            "HOME": str(tmp_path / "home"),
            "COMPAIR_DB_DIR": str(database),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout)["created"] is True
    assert completed.stderr == ""
    assert not database.exists()
