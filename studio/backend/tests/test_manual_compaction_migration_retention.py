# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import hashlib
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from core import manual_compaction
from models.inference import ChatCompletionRequest
from storage import studio_db


_EMPTY_JSON_HASH = hashlib.sha256(b"[]").hexdigest()
_CLAIM_ID = "3" * 64


def _reset_db(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)


def _thread(thread_id = "thread-1"):
    return {
        "id": thread_id,
        "title": "Compaction retention test",
        "modelType": "base",
        "modelId": "model",
        "archived": False,
        "createdAt": 1,
    }


def _message(
    message_id,
    parent_id,
    role,
    text,
    *,
    thread_id = "thread-1",
):
    return {
        "id": message_id,
        "threadId": thread_id,
        "parentId": parent_id,
        "role": role,
        "content": [{"type": "text", "text": text}],
        "metadata": None,
        "createdAt": 10,
    }


def _seed_branch(tmp_path, monkeypatch):
    _reset_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())
    rows = [
        _message("u1", None, "user", "Explain the migration."),
        _message("a1", "u1", "assistant", "Use a staged rollout."),
        _message("compact-1", "a1", "user", "/compact"),
    ]
    for row in rows:
        studio_db.upsert_chat_message(row)
    return rows


def _wire_messages():
    return ChatCompletionRequest(
        model = "model",
        messages = [
            {"role": "system", "content": "Project rules"},
            {"role": "user", "content": "Explain the migration."},
            {"role": "assistant", "content": "Use a staged rollout."},
            {"role": "user", "content": "/compact"},
        ],
    ).messages


def _prepare(attempt_id):
    command = studio_db.get_chat_message("thread-1", "compact-1")
    metadata = command.get("metadata") if command else None
    current = (
        metadata.get(manual_compaction.MANUAL_COMPACTION_CLIENT_KEY)
        if isinstance(metadata, dict)
        else None
    )
    sequence = 1
    expected_attempt_id = None
    expected_attempt_sequence = None
    if isinstance(current, dict):
        if current.get("attemptId") == attempt_id:
            sequence = current["attemptSequence"]
        else:
            sequence = current["attemptSequence"] + 1
            expected_attempt_id = current["attemptId"]
            expected_attempt_sequence = current["attemptSequence"]
    manual_compaction.bind_manual_compaction_command(
        "thread-1",
        attempt_id = attempt_id,
        command_message_id = "compact-1",
        summary_message_id = "summary-1",
        attempt_sequence = sequence,
        expected_attempt_id = expected_attempt_id,
        expected_attempt_sequence = expected_attempt_sequence,
    )
    return manual_compaction.prepare_manual_compaction(
        "thread-1",
        attempt_id = attempt_id,
        command_message_id = "compact-1",
        expected_head_message_id = "compact-1",
        message_ids = ["u1", "a1", "compact-1"],
        request_messages = _wire_messages(),
    )


def _claim(prepared):
    request = ChatCompletionRequest(
        model = "model",
        thread_id = "thread-1",
        messages = _wire_messages(),
        manual_compaction = {
            "attemptId": prepared["attemptId"],
            "claimId": _CLAIM_ID,
            "threadId": prepared["threadId"],
            "commandMessageId": prepared["commandMessageId"],
            "expectedHeadMessageId": prepared["expectedHeadMessageId"],
            "sourceHash": prepared["sourceHash"],
            "requestHash": prepared["requestHash"],
            "requestMessageCount": prepared["requestMessageCount"],
            "projectInstructionDigest": prepared["projectInstructionDigest"],
            "projectInstructionRevision": prepared["projectInstructionRevision"],
            "contextDigest": prepared["contextDigest"],
            "revision": prepared["revision"],
        },
    )
    manual_compaction.validate_and_rewrite_manual_compaction_request(request)


def _activate(prepared, summary_id = "summary-1"):
    _claim(prepared)
    text = "Durable summary"
    manual_compaction.record_manual_compaction_output(
        prepared["attemptId"], text = text, finish_reason = "stop"
    )
    studio_db.upsert_chat_message(_message(summary_id, "compact-1", "assistant", text))
    return manual_compaction.commit_manual_compaction(
        "thread-1",
        attempt_id = prepared["attemptId"],
        command_message_id = "compact-1",
        summary_message_id = summary_id,
        expected_head_message_id = summary_id,
        expected_revision = prepared["revision"],
        expected_summary_hash = manual_compaction.summary_hash(text),
    )


def _raw_attempt(attempt_id):
    conn = studio_db.get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM manual_compactions WHERE attempt_id = ?", (attempt_id,)
        ).fetchone()
        return dict(row) if row is not None else None
    finally:
        conn.close()


def _force_incompatible_manual_compactions_schema():
    conn = studio_db.get_connection()
    try:
        canonical_schema = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'manual_compactions'"
        ).fetchone()["sql"]
        conn.execute("DROP INDEX IF EXISTS idx_manual_compactions_thread_state")
        conn.execute("DROP INDEX IF EXISTS idx_manual_compactions_live_branch")
        conn.execute("ALTER TABLE manual_compactions RENAME TO discarded_current")
        prefix, closing = canonical_schema.rsplit(")", 1)
        conn.execute(prefix + ", CHECK(archive_status != 'impossible'))" + closing)
        conn.execute("INSERT INTO manual_compactions SELECT * FROM discarded_current")
        conn.execute("DROP TABLE discarded_current")
        conn.commit()
    finally:
        conn.close()


def _run_in_writer_order(monkeypatch, winner_name, winner, loser_name, loser):
    original_get_connection = studio_db.get_connection
    winner_acquired = threading.Event()
    loser_attempted = threading.Event()
    release_winner = threading.Event()
    results = {}

    class GatedConnection:
        def __init__(self, connection):
            self._connection = connection

        def execute(self, statement, *args, **kwargs):
            is_begin = statement.strip().upper() == "BEGIN IMMEDIATE"
            thread_name = threading.current_thread().name
            if is_begin and thread_name == loser_name:
                loser_attempted.set()
            cursor = self._connection.execute(statement, *args, **kwargs)
            if is_begin and thread_name == winner_name:
                winner_acquired.set()
                if not release_winner.wait(timeout = 5):
                    raise AssertionError("timed out holding the SQLite writer lock")
            return cursor

        def __getattr__(self, name):
            return getattr(self._connection, name)

    def gated_get_connection(*args, **kwargs):
        return GatedConnection(original_get_connection(*args, **kwargs))

    def capture(name, operation):
        try:
            results[name] = operation()
        except Exception as exc:  # noqa: BLE001 - the assertion inspects the race outcome
            results[name] = exc

    monkeypatch.setattr(studio_db, "get_connection", gated_get_connection)
    winner_thread = threading.Thread(
        target = capture,
        args = (winner_name, winner),
        name = winner_name,
    )
    loser_thread = threading.Thread(
        target = capture,
        args = (loser_name, loser),
        name = loser_name,
    )
    winner_thread.start()
    assert winner_acquired.wait(timeout = 5)
    loser_thread.start()
    assert loser_attempted.wait(timeout = 5)
    release_winner.set()
    winner_thread.join(timeout = 5)
    loser_thread.join(timeout = 5)
    assert not winner_thread.is_alive()
    assert not loser_thread.is_alive()
    monkeypatch.setattr(studio_db, "get_connection", original_get_connection)
    return results


class _OneShotSchemaFailureConnection:
    def __init__(self, connection, failure):
        self._connection = connection
        self._failure = failure
        self._failed = False
        self._commit_count = 0

    @property
    def row_factory(self):
        return self._connection.row_factory

    @row_factory.setter
    def row_factory(self, value):
        self._connection.row_factory = value

    def commit(self):
        self._commit_count += 1
        fail_first = self._failure == "commit" and self._commit_count == 1
        fail_final = self._failure == "final_commit" and self._commit_count == 2
        if (fail_first or fail_final) and not self._failed:
            self._failed = True
            label = "final commit" if fail_final else "commit"
            raise sqlite3.OperationalError(f"injected schema {label} failure")
        return self._connection.commit()

    def execute(self, statement, *args, **kwargs):
        if (
            self._failure == "begin"
            and not self._failed
            and statement.strip().upper() == "BEGIN IMMEDIATE"
        ):
            self._failed = True
            raise sqlite3.OperationalError("injected schema begin failure")
        return self._connection.execute(statement, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._connection, name)


@pytest.mark.parametrize("failure", ["commit", "begin", "final_commit"])
def test_schema_restores_row_factory_and_transaction_after_boundary_failure(
    tmp_path, monkeypatch, failure
):
    _seed_branch(tmp_path, monkeypatch)
    conn = studio_db.get_connection()
    original_row_factory = lambda _cursor, row: tuple(row)
    conn.row_factory = original_row_factory
    wrapped = _OneShotSchemaFailureConnection(conn, failure)
    label = failure.replace("_", " ")
    try:
        with pytest.raises(sqlite3.OperationalError, match = f"schema {label} failure"):
            studio_db._ensure_manual_compactions_schema(wrapped)
        assert wrapped.row_factory is original_row_factory
        assert conn.in_transaction is False

        studio_db._ensure_manual_compactions_schema(wrapped)
        assert wrapped.row_factory is original_row_factory
        assert conn.in_transaction is False
        assert conn.execute("SELECT 1").fetchone() == (1,)
    finally:
        conn.close()


def test_final_schema_commit_failure_rolls_back_every_migration_write(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare("final-commit-rollback")
    _force_incompatible_manual_compactions_schema()
    conn = studio_db.get_connection()

    def schema_snapshot():
        objects = conn.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE name LIKE 'manual_compaction%' "
            "OR name IN ('idx_manual_compactions_thread_state', "
            "'idx_manual_compactions_live_branch') ORDER BY type, name"
        ).fetchall()
        attempts = conn.execute("SELECT * FROM manual_compactions ORDER BY attempt_id").fetchall()
        reservations = conn.execute(
            "SELECT * FROM manual_compaction_attempt_reservations ORDER BY attempt_id"
        ).fetchall()
        return (
            [tuple(row) for row in objects],
            [tuple(row) for row in attempts],
            [tuple(row) for row in reservations],
        )

    before = schema_snapshot()
    wrapped = _OneShotSchemaFailureConnection(conn, "final_commit")
    try:
        with pytest.raises(sqlite3.OperationalError, match = "schema final commit failure"):
            studio_db._ensure_manual_compactions_schema(wrapped)

        assert conn.in_transaction is False
        assert schema_snapshot() == before
        assert (
            conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name LIKE 'manual_compactions%legacy%'"
            ).fetchall()
            == []
        )

        studio_db._ensure_manual_compactions_schema(wrapped)
        assert conn.in_transaction is False
        assert (
            conn.execute(
                "SELECT attempt_id FROM manual_compactions WHERE attempt_id = ?",
                (prepared["attemptId"],),
            ).fetchone()["attempt_id"]
            == prepared["attemptId"]
        )
    finally:
        conn.close()


def _assert_scrubbed_terminal(
    row,
    *,
    state,
    reason,
    thread_id = "thread-1",
    command_message_id = "compact-1",
):
    assert row["state"] == state
    assert row["terminal_reason"] == reason
    assert row["finished_at"] is not None
    assert row["lease_expires_at"] is None
    assert row["source_message_ids_json"] == "[]"
    assert row["effective_source_message_ids_json"] == "[]"
    assert row["archive_payload_json"] == "[]"
    assert row["archive_payload_hash"] == _EMPTY_JSON_HASH
    assert row["attempt_id"]
    assert row["thread_id"] == thread_id
    assert row["command_message_id"] == command_message_id
    assert row["source_hash"]
    assert row["request_hash"]
    assert row["context_digest"]
    assert row["created_at"] is not None


def test_pending_expiry_fails_and_scrubs_payload_but_keeps_minimal_audit(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare("pending-expiry")
    expires_at = prepared["createdAt"] + manual_compaction.MANUAL_COMPACTION_PENDING_TTL_MS
    monkeypatch.setattr(manual_compaction.time, "time", lambda: (expires_at + 1) / 1000)

    expired = manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])

    assert expired["state"] == "failed"
    assert expired["terminalReason"] == "pending_expired"
    _assert_scrubbed_terminal(
        _raw_attempt(prepared["attemptId"]), state = "failed", reason = "pending_expired"
    )


def test_running_expiry_fails_and_scrubs_payload_but_keeps_started_audit(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare("running-expiry")
    _claim(prepared)
    running = manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])
    monkeypatch.setattr(
        manual_compaction.time,
        "time",
        lambda: (running["leaseExpiresAt"] + 1) / 1000,
    )

    expired = manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])

    assert expired["state"] == "failed"
    assert expired["terminalReason"] == "lease_expired"
    assert expired["startedAt"] == running["startedAt"]
    _assert_scrubbed_terminal(
        _raw_attempt(prepared["attemptId"]), state = "failed", reason = "lease_expired"
    )


def test_repeated_prepare_has_bounded_rows_and_only_scrubbed_terminal_audit(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    now = int(time.time() * 1000)
    monkeypatch.setattr(manual_compaction.time, "time", lambda: now / 1000)
    attempts = [f"attempt-{index:03d}" for index in range(80)]

    for attempt_id in attempts[:-1]:
        prepared = _prepare(attempt_id)
        _claim(prepared)
        manual_compaction.cancel_manual_compaction(
            "thread-1",
            attempt_id = prepared["attemptId"],
            command_message_id = prepared["commandMessageId"],
            claim_id = _CLAIM_ID,
        )
    _prepare(attempts[-1])

    conn = studio_db.get_connection()
    try:
        rows = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM manual_compactions WHERE thread_id = ? "
                "ORDER BY created_at, attempt_id",
                ("thread-1",),
            ).fetchall()
        ]
    finally:
        conn.close()

    assert len(rows) == manual_compaction.MAX_MANUAL_COMPACTION_TERMINAL_ATTEMPTS_PER_THREAD + 1
    live = [row for row in rows if row["state"] == "pending"]
    terminal = [row for row in rows if row["state"] == "cancelled"]
    assert [row["attempt_id"] for row in live] == [attempts[-1]]
    assert len(terminal) == manual_compaction.MAX_MANUAL_COMPACTION_TERMINAL_ATTEMPTS_PER_THREAD
    assert _raw_attempt(attempts[0]) is None
    for row in terminal:
        _assert_scrubbed_terminal(row, state = "cancelled", reason = "cancelled")


def test_retention_keeps_aged_terminal_tombstone_until_exact_fence_replacement(
    tmp_path, monkeypatch
):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare("terminal-referenced")
    _claim(prepared)
    manual_compaction.cancel_manual_compaction(
        "thread-1",
        attempt_id = prepared["attemptId"],
        command_message_id = prepared["commandMessageId"],
        claim_id = _CLAIM_ID,
    )
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "UPDATE manual_compactions SET finished_at = 10, cancelled_at = 10 "
            "WHERE attempt_id = ?",
            (prepared["attemptId"],),
        )
        conn.commit()
    finally:
        conn.close()
    now = 10 + manual_compaction.MANUAL_COMPACTION_TERMINAL_RETENTION_MS + 1
    monkeypatch.setattr(manual_compaction.time, "time", lambda: now / 1000)

    manual_compaction.cleanup_manual_compaction_attempts("thread-1")
    assert _raw_attempt(prepared["attemptId"])["state"] == "cancelled"

    manual_compaction.bind_manual_compaction_command(
        "thread-1",
        attempt_id = "replacement",
        command_message_id = "compact-1",
        summary_message_id = "summary-1",
        attempt_sequence = 2,
        expected_attempt_id = prepared["attemptId"],
        expected_attempt_sequence = 1,
    )
    manual_compaction.cleanup_manual_compaction_attempts("thread-1")
    assert _raw_attempt(prepared["attemptId"]) is None


def test_retention_cap_does_not_count_or_delete_a_referenced_terminal_attempt(
    tmp_path, monkeypatch
):
    _seed_branch(tmp_path, monkeypatch)
    now = int(time.time() * 1000)
    monkeypatch.setattr(manual_compaction.time, "time", lambda: now / 1000)
    attempt_ids = [f"cap-{index:03d}" for index in range(40)]
    prepared = _prepare(attempt_ids[0])
    _claim(prepared)
    manual_compaction.cancel_manual_compaction(
        "thread-1",
        attempt_id = prepared["attemptId"],
        command_message_id = "compact-1",
        claim_id = _CLAIM_ID,
    )
    conn = studio_db.get_connection()
    try:
        columns = [
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(manual_compactions)").fetchall()
        ]
        selected = ["?" if column == "attempt_id" else f'"{column}"' for column in columns]
        for attempt_id in attempt_ids[1:]:
            conn.execute(
                "INSERT INTO manual_compactions ("
                + ",".join(f'"{column}"' for column in columns)
                + ") SELECT "
                + ",".join(selected)
                + " FROM manual_compactions WHERE attempt_id = ?",
                (attempt_id, attempt_ids[0]),
            )
        conn.commit()
    finally:
        conn.close()

    manual_compaction.cleanup_manual_compaction_attempts("thread-1")
    rows = [row for attempt_id in attempt_ids if (row := _raw_attempt(attempt_id)) is not None]
    assert _raw_attempt(attempt_ids[0]) is not None
    assert len(rows) == manual_compaction.MAX_MANUAL_COMPACTION_TERMINAL_ATTEMPTS_PER_THREAD + 1


@pytest.mark.parametrize("malformed_column", ["content_json", "metadata_json"])
def test_retention_reference_discovery_fails_closed_before_any_mutation(
    tmp_path, monkeypatch, malformed_column
):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare("malformed-retention")
    before = _raw_attempt(prepared["attemptId"])
    conn = studio_db.get_connection()
    try:
        conn.execute(
            f"UPDATE chat_messages SET {malformed_column} = ? WHERE id = ?",
            ("{", "compact-1"),
        )
        conn.commit()
    finally:
        conn.close()
    now = prepared["createdAt"] + manual_compaction.MANUAL_COMPACTION_PENDING_TTL_MS + 1
    monkeypatch.setattr(manual_compaction.time, "time", lambda: now / 1000)

    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "Stored"):
        manual_compaction.cleanup_manual_compaction_attempts("thread-1")
    assert _raw_attempt(prepared["attemptId"]) == before


def test_retention_cleanup_and_replacement_race_never_deletes_the_new_owner(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare("race-old")
    _claim(prepared)
    manual_compaction.cancel_manual_compaction(
        "thread-1",
        attempt_id = prepared["attemptId"],
        command_message_id = prepared["commandMessageId"],
        claim_id = _CLAIM_ID,
    )
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "UPDATE manual_compactions SET finished_at = 1, cancelled_at = 1 WHERE attempt_id = ?",
            (prepared["attemptId"],),
        )
        conn.commit()
    finally:
        conn.close()
    now = manual_compaction.MANUAL_COMPACTION_TERMINAL_RETENTION_MS + 2
    monkeypatch.setattr(manual_compaction.time, "time", lambda: now / 1000)
    start = threading.Barrier(2)

    def replace():
        start.wait(timeout = 5)
        return manual_compaction.bind_manual_compaction_command(
            "thread-1",
            attempt_id = "race-new",
            command_message_id = "compact-1",
            summary_message_id = "summary-1",
            attempt_sequence = 2,
            expected_attempt_id = prepared["attemptId"],
            expected_attempt_sequence = 1,
        )

    def cleanup():
        start.wait(timeout = 5)
        manual_compaction.cleanup_manual_compaction_attempts("thread-1")

    with ThreadPoolExecutor(max_workers = 2) as pool:
        replacement = pool.submit(replace)
        cleaned = pool.submit(cleanup)
        rebound = replacement.result(timeout = 10)
        cleaned.result(timeout = 10)

    record = rebound["metadata"][manual_compaction.MANUAL_COMPACTION_CLIENT_KEY]
    assert record["attemptId"] == "race-new"
    assert manual_compaction.get_manual_compaction_attempt("race-new") is None
    old = _raw_attempt(prepared["attemptId"])
    assert old is None or old["state"] == "cancelled"


@pytest.mark.parametrize("first", ["cleanup", "replacement"])
def test_retention_cleanup_and_replacement_have_deterministic_writer_order(
    tmp_path, monkeypatch, first
):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare("ordered-old")
    _claim(prepared)
    manual_compaction.cancel_manual_compaction(
        "thread-1",
        attempt_id = prepared["attemptId"],
        command_message_id = prepared["commandMessageId"],
        claim_id = _CLAIM_ID,
    )
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "UPDATE manual_compactions SET finished_at = 1, cancelled_at = 1 WHERE attempt_id = ?",
            (prepared["attemptId"],),
        )
        conn.commit()
    finally:
        conn.close()
    now = manual_compaction.MANUAL_COMPACTION_TERMINAL_RETENTION_MS + 2
    monkeypatch.setattr(manual_compaction.time, "time", lambda: now / 1000)

    def replace():
        return manual_compaction.bind_manual_compaction_command(
            "thread-1",
            attempt_id = "ordered-new",
            command_message_id = "compact-1",
            summary_message_id = "summary-1",
            attempt_sequence = 2,
            expected_attempt_id = prepared["attemptId"],
            expected_attempt_sequence = 1,
        )

    def cleanup():
        return manual_compaction.cleanup_manual_compaction_attempts("thread-1")

    operations = {"cleanup": cleanup, "replacement": replace}
    second = "replacement" if first == "cleanup" else "cleanup"
    results = _run_in_writer_order(
        monkeypatch,
        f"{first}-winner",
        operations[first],
        f"{second}-loser",
        operations[second],
    )

    assert not isinstance(results[f"{first}-winner"], Exception)
    assert not isinstance(results[f"{second}-loser"], Exception)
    command = studio_db.get_chat_message("thread-1", "compact-1")
    record = command["metadata"][manual_compaction.MANUAL_COMPACTION_CLIENT_KEY]
    assert record["attemptId"] == "ordered-new"
    conn = studio_db.get_connection()
    try:
        reservations = conn.execute(
            "SELECT attempt_id FROM manual_compaction_attempt_reservations "
            "WHERE attempt_id IN ('ordered-old', 'ordered-new') ORDER BY attempt_id"
        ).fetchall()
        assert [row["attempt_id"] for row in reservations] == ["ordered-new", "ordered-old"]
    finally:
        conn.close()
    if first == "replacement":
        assert _raw_attempt(prepared["attemptId"]) is None
    else:
        assert _raw_attempt(prepared["attemptId"])["state"] == "cancelled"


def test_retention_never_scrubs_or_deletes_active_source_and_fork_rows(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare("active-source")
    active = _activate(prepared)
    source_before = _raw_attempt(active["attemptId"])
    ids = iter(["fu1", "fa1", "fcompact", "fsummary"])
    studio_db.fork_chat_thread(
        source_thread_id = "thread-1",
        branch_message_id = "summary-1",
        new_thread_id = "fork-1",
        new_title = "Fork",
        created_at = 20,
        id_factory = lambda: next(ids),
    )
    fork_summary = studio_db.get_chat_message("fork-1", "fsummary")
    fork_attempt_id = fork_summary["metadata"]["manualCompaction"]["attemptId"]
    fork_before = _raw_attempt(fork_attempt_id)
    future = (
        int(time.time() * 1000)
        + manual_compaction.MANUAL_COMPACTION_TERMINAL_RETENTION_MS
        + manual_compaction.MANUAL_COMPACTION_PENDING_TTL_MS
        + 1
    )
    monkeypatch.setattr(manual_compaction.time, "time", lambda: future / 1000)

    manual_compaction.cleanup_manual_compaction_attempts("thread-1")
    manual_compaction.cleanup_manual_compaction_attempts("fork-1")

    source_after = _raw_attempt(active["attemptId"])
    fork_after = _raw_attempt(fork_attempt_id)
    for before, after in ((source_before, source_after), (fork_before, fork_after)):
        assert before is not None
        assert after is not None
        assert after["state"] == "active"
        assert after["source_message_ids_json"] == before["source_message_ids_json"]
        assert (
            after["effective_source_message_ids_json"]
            == before["effective_source_message_ids_json"]
        )
        assert after["archive_payload_json"] == before["archive_payload_json"]
        assert after["archive_payload_hash"] == before["archive_payload_hash"]
        assert after["summary_message_id"] == before["summary_message_id"]
        assert after["summary_hash"] == before["summary_hash"]


def test_startup_reconciliation_expires_and_scrubs_dormant_attempts_globally(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    first = _prepare("terminal-dormant")
    _claim(first)
    manual_compaction.cancel_manual_compaction(
        "thread-1",
        attempt_id = first["attemptId"],
        command_message_id = first["commandMessageId"],
        claim_id = _CLAIM_ID,
    )
    pending = _prepare("pending-dormant")
    studio_db.upsert_chat_thread(_thread("thread-2"))
    for row in (
        _message("u2-1", None, "user", "Explain the migration.", thread_id = "thread-2"),
        _message("a2-1", "u2-1", "assistant", "Use a staged rollout.", thread_id = "thread-2"),
        _message("compact-2", "a2-1", "user", "/compact", thread_id = "thread-2"),
    ):
        studio_db.upsert_chat_message(row)
    manual_compaction.bind_manual_compaction_command(
        "thread-2",
        attempt_id = "running-dormant",
        command_message_id = "compact-2",
        summary_message_id = "summary-2",
        attempt_sequence = 1,
    )
    running = manual_compaction.prepare_manual_compaction(
        "thread-2",
        attempt_id = "running-dormant",
        command_message_id = "compact-2",
        expected_head_message_id = "compact-2",
        message_ids = ["u2-1", "a2-1", "compact-2"],
        request_messages = ChatCompletionRequest(model = "model", messages = _wire_messages()).messages,
    )
    request = ChatCompletionRequest(
        model = "model",
        thread_id = "thread-2",
        messages = _wire_messages(),
        manual_compaction = {
            **{
                key: running[key]
                for key in (
                    "threadId",
                    "commandMessageId",
                    "expectedHeadMessageId",
                    "sourceHash",
                    "requestHash",
                    "requestMessageCount",
                    "projectInstructionDigest",
                    "projectInstructionRevision",
                    "contextDigest",
                    "revision",
                )
            },
            "attemptId": running["attemptId"],
            "claimId": _CLAIM_ID,
        },
    )
    manual_compaction.validate_and_rewrite_manual_compaction_request(request)
    now = int(time.time() * 1000)
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "UPDATE manual_compactions SET terminal_reason = ?, "
            "source_message_ids_json = '[\"u1\"]', "
            "effective_source_message_ids_json = '[\"u1\"]', "
            'archive_payload_json = \'[{"role":"user","content":"secret"}]\', '
            "archive_status = 'archived' WHERE attempt_id = ?",
            ("Bearer legacy-secret", first["attemptId"]),
        )
        conn.execute(
            "UPDATE manual_compactions SET created_at = ? WHERE attempt_id = ?",
            (
                now - manual_compaction.MANUAL_COMPACTION_PENDING_TTL_MS - 1,
                pending["attemptId"],
            ),
        )
        conn.execute(
            "UPDATE manual_compactions SET lease_expires_at = ? WHERE attempt_id = ?",
            (now - 1, running["attemptId"]),
        )
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr(studio_db, "_schema_ready", False)
    conn = studio_db.get_connection()
    conn.close()

    terminal = _raw_attempt(first["attemptId"])
    _assert_scrubbed_terminal(terminal, state = "cancelled", reason = "migrated_cancelled")
    assert terminal["archive_status"] == "pending"
    _assert_scrubbed_terminal(
        _raw_attempt(pending["attemptId"]),
        state = "failed",
        reason = "pending_expired",
    )
    _assert_scrubbed_terminal(
        _raw_attempt(running["attemptId"]),
        state = "failed",
        reason = "inference_failed",
        thread_id = "thread-2",
        command_message_id = "compact-2",
    )
    before = {
        attempt_id: _raw_attempt(attempt_id)
        for attempt_id in (first["attemptId"], pending["attemptId"], running["attemptId"])
    }
    manual_compaction.cleanup_all_manual_compaction_attempts()
    manual_compaction.cleanup_all_manual_compaction_attempts()
    assert {attempt_id: _raw_attempt(attempt_id) for attempt_id in before} == before


def test_startup_reconciliation_failure_retries_on_next_connection(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare("dormant-pending")
    now = int(time.time() * 1000)
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "UPDATE manual_compactions SET created_at = ? WHERE attempt_id = ?",
            (
                now - manual_compaction.MANUAL_COMPACTION_PENDING_TTL_MS - 1,
                prepared["attemptId"],
            ),
        )
        conn.commit()
    finally:
        conn.close()
    original = manual_compaction.reconcile_all_manual_compaction_attempts_in_connection

    def fail_reconciliation(*_args, **_kwargs):
        raise RuntimeError("simulated startup reconciliation failure")

    monkeypatch.setattr(
        manual_compaction,
        "reconcile_all_manual_compaction_attempts_in_connection",
        fail_reconciliation,
    )
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    with pytest.raises(RuntimeError, match = "simulated startup reconciliation failure"):
        studio_db.get_connection()
    assert studio_db._schema_ready is False

    monkeypatch.setattr(
        manual_compaction,
        "reconcile_all_manual_compaction_attempts_in_connection",
        original,
    )
    conn = studio_db.get_connection()
    conn.close()
    _assert_scrubbed_terminal(
        _raw_attempt(prepared["attemptId"]),
        state = "failed",
        reason = "pending_expired",
    )


_LEGACY_SCHEMA = """
    CREATE TABLE manual_compactions (
        attempt_id TEXT NOT NULL PRIMARY KEY,
        thread_id TEXT NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
        command_message_id TEXT NOT NULL,
        source_head_message_id TEXT NOT NULL,
        expected_head_message_id TEXT NOT NULL,
        source_message_ids_json TEXT NOT NULL,
        effective_source_message_ids_json TEXT NOT NULL,
        source_hash TEXT NOT NULL,
        request_hash TEXT NOT NULL,
        request_message_count INTEGER NOT NULL,
        project_instruction_digest TEXT NOT NULL,
        project_instruction_revision INTEGER NOT NULL,
        context_digest TEXT NOT NULL,
        revision INTEGER NOT NULL,
        state TEXT NOT NULL CHECK(state IN ('pending', 'active')),
        summary_message_id TEXT,
        summary_hash TEXT,
        archive_status TEXT NOT NULL DEFAULT 'pending',
        created_at INTEGER NOT NULL,
        committed_at INTEGER,
        UNIQUE(thread_id, revision, command_message_id)
    )
"""


def _install_legacy_table(
    conn,
    *,
    table_name = "manual_compactions",
    attempt_id = "legacy-1",
    command_message_id = "compact-1",
    revision = 1,
):
    conn.execute("DROP INDEX IF EXISTS idx_manual_compactions_thread_state")
    conn.execute("DROP INDEX IF EXISTS idx_manual_compactions_live_branch")
    conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    conn.execute(_LEGACY_SCHEMA.replace("manual_compactions", table_name, 1))
    now = int(time.time() * 1000)
    digest = "a" * 64
    conn.execute(
        f"""
        INSERT INTO "{table_name}" (
            attempt_id, thread_id, command_message_id, source_head_message_id,
            expected_head_message_id, source_message_ids_json,
            effective_source_message_ids_json, source_hash, request_hash,
            request_message_count, project_instruction_digest,
            project_instruction_revision, context_digest, revision, state,
            archive_status, created_at
        ) VALUES (?, 'thread-1', ?, 'a1', ?,
            '["u1","a1"]', '["u1","a1"]', ?, ?, 4, ?, 7, ?, ?,
            'pending', 'pending', ?)
        """,
        (
            attempt_id,
            command_message_id,
            command_message_id,
            digest,
            "b" * 64,
            "c" * 64,
            "d" * 64,
            revision,
            now,
        ),
    )
    conn.commit()


class _FailingConnection:
    def __init__(self, connection, needle):
        self._connection = connection
        self._needle = needle
        self.triggered = False

    def execute(self, statement, *args, **kwargs):
        if not self.triggered and self._needle in " ".join(statement.split()):
            self.triggered = True
            raise RuntimeError("simulated migration interruption")
        return self._connection.execute(statement, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._connection, name)


@pytest.mark.parametrize(
    "failpoint",
    [
        "ALTER TABLE manual_compactions RENAME TO",
        "INSERT OR IGNORE INTO manual_compactions",
        'DROP TABLE "manual_compactions_lifecycle_legacy"',
        "CREATE UNIQUE INDEX idx_manual_compactions_live_branch",
    ],
)
def test_migration_failpoint_rolls_back_and_clean_restart_recovers(
    tmp_path, monkeypatch, failpoint
):
    _seed_branch(tmp_path, monkeypatch)
    conn = studio_db.get_connection()
    _install_legacy_table(conn)
    wrapper = _FailingConnection(conn, failpoint)

    with pytest.raises(RuntimeError, match = "simulated migration interruption"):
        studio_db._ensure_manual_compactions_schema(wrapper)

    assert wrapper.triggered
    schema = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'manual_compactions'"
    ).fetchone()["sql"]
    assert "'running'" not in schema
    assert (
        conn.execute("SELECT attempt_id FROM manual_compactions").fetchone()["attempt_id"]
        == "legacy-1"
    )
    assert (
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name LIKE 'manual_compactions%legacy%'"
        ).fetchall()
        == []
    )
    conn.close()

    monkeypatch.setattr(studio_db, "_schema_ready", False)
    recovered = manual_compaction.get_manual_compaction_attempt("legacy-1")

    assert recovered["state"] == "failed"
    assert recovered["terminalReason"] == "migrated_failed"
    assert recovered["sourceMessageIds"] == []
    assert recovered["effectiveSourceMessageIds"] == []
    assert recovered["requestMessageCount"] == 4
    assert recovered["projectInstructionRevision"] == 7
    conn = studio_db.get_connection()
    try:
        assert (
            conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name LIKE 'manual_compactions%legacy%'"
            ).fetchall()
            == []
        )
    finally:
        conn.close()


def test_restart_rebuilds_full_column_schema_with_stale_lifecycle_checks(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare("stale-check")
    conn = studio_db.get_connection()
    canonical_schema = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'manual_compactions'"
    ).fetchone()["sql"]
    conn.execute("DROP INDEX IF EXISTS idx_manual_compactions_thread_state")
    conn.execute("DROP INDEX IF EXISTS idx_manual_compactions_live_branch")
    conn.execute("ALTER TABLE manual_compactions RENAME TO discarded_current")
    stale_schema = canonical_schema.replace("'cancelled', ", "")
    conn.execute(stale_schema)
    conn.execute("INSERT INTO manual_compactions SELECT * FROM discarded_current")
    conn.execute("DROP TABLE discarded_current")
    conn.commit()
    conn.close()
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    recovered = manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])
    assert recovered["state"] == "pending"
    _claim(recovered)
    cancelled = manual_compaction.cancel_manual_compaction(
        "thread-1",
        attempt_id = prepared["attemptId"],
        command_message_id = prepared["commandMessageId"],
        claim_id = _CLAIM_ID,
    )
    assert cancelled["state"] == "cancelled"
    conn = studio_db.get_connection()
    try:
        schema = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'manual_compactions'"
        ).fetchone()["sql"]
        columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(manual_compactions)").fetchall()
        }
    finally:
        conn.close()
    assert studio_db._manual_compactions_schema_is_compatible(columns, schema)


def test_restart_rebuilds_schema_with_an_unknown_conflicting_check(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare("extra-check")
    conn = studio_db.get_connection()
    canonical_schema = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'manual_compactions'"
    ).fetchone()["sql"]
    conn.execute("DROP INDEX IF EXISTS idx_manual_compactions_thread_state")
    conn.execute("DROP INDEX IF EXISTS idx_manual_compactions_live_branch")
    conn.execute("ALTER TABLE manual_compactions RENAME TO discarded_current")
    prefix, closing = canonical_schema.rsplit(")", 1)
    conn.execute(prefix + ", CHECK(archive_status != 'skipped'))" + closing)
    conn.execute("INSERT INTO manual_compactions SELECT * FROM discarded_current")
    conn.execute("DROP TABLE discarded_current")
    conn.commit()
    conn.close()
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    recovered = manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])
    assert recovered["state"] == "pending"
    conn = studio_db.get_connection()
    try:
        schema = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'manual_compactions'"
        ).fetchone()["sql"]
        columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(manual_compactions)").fetchall()
        }
    finally:
        conn.close()
    assert "archive_status != 'skipped'" not in schema
    assert studio_db._manual_compactions_schema_is_compatible(columns, schema)


@pytest.mark.parametrize(
    "mutation",
    [
        "archive_status = 'unknown'",
        "output_summary_hash = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', "
        "output_finish_reason = 'stop', output_recorded_at = 1",
        "state = 'running', started_at = 1, lease_expires_at = 'not-a-time', "
        "claim_token_hash = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'",
        "state = 'active', command_hash_state = 'summary_saved', started_at = 1, committed_at = 1",
        "state = 'failed', terminal_reason = 'inference_failed', finished_at = 1",
    ],
    ids = [
        "archive-status",
        "pending-output",
        "running-lease",
        "active-summary",
        "terminal-unscrubbed",
    ],
)
def test_strict_attempt_row_decoder_rejects_state_specific_corruption(
    tmp_path, monkeypatch, mutation
):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare("corrupt-state")
    conn = studio_db.get_connection()
    try:
        conn.execute(
            f"UPDATE manual_compactions SET {mutation} WHERE attempt_id = ?",
            (prepared["attemptId"],),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM manual_compactions WHERE attempt_id = ?",
            (prepared["attemptId"],),
        ).fetchone()
    finally:
        conn.close()

    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "Stored"):
        manual_compaction._attempt_from_row(row)


def test_restart_recovers_orphan_legacy_table_without_disturbing_current_rows(
    tmp_path, monkeypatch
):
    _seed_branch(tmp_path, monkeypatch)
    current = _prepare("current-1")
    conn = studio_db.get_connection()
    _install_legacy_table(
        conn,
        table_name = "manual_compactions_lifecycle_legacy",
        attempt_id = "orphan-1",
        command_message_id = "compact-orphan",
        revision = 2,
    )
    conn.close()
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    orphan = manual_compaction.get_manual_compaction_attempt("orphan-1")
    preserved = manual_compaction.get_manual_compaction_attempt(current["attemptId"])

    assert orphan["state"] == "failed"
    assert orphan["terminalReason"] == "migrated_failed"
    assert orphan["sourceMessageIds"] == []
    assert preserved["state"] == "pending"
    assert preserved["sourceHash"] == current["sourceHash"]
    conn = studio_db.get_connection()
    try:
        assert (
            conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name LIKE 'manual_compactions%legacy%'"
            ).fetchall()
            == []
        )
    finally:
        conn.close()


def test_restart_reconciles_orphan_with_a_colliding_live_branch(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    current = _prepare("current-1")
    conn = studio_db.get_connection()
    _install_legacy_table(
        conn,
        table_name = "manual_compactions_lifecycle_legacy",
        attempt_id = "orphan-collision",
    )
    conn.close()
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    preserved = manual_compaction.get_manual_compaction_attempt(current["attemptId"])

    assert preserved["state"] == "pending"
    assert preserved["sourceHash"] == current["sourceHash"]
    orphan = manual_compaction.get_manual_compaction_attempt("orphan-collision")
    assert orphan is None or orphan["state"] in ("cancelled", "failed")


def test_migration_preserves_richer_columns_but_terminalizes_unproven_live_rows(
    tmp_path, monkeypatch
):
    _seed_branch(tmp_path, monkeypatch)
    conn = studio_db.get_connection()
    conn.execute("DROP INDEX IF EXISTS idx_manual_compactions_thread_state")
    conn.execute("DROP INDEX IF EXISTS idx_manual_compactions_live_branch")
    conn.execute("ALTER TABLE manual_compactions RENAME TO discarded_current")
    conn.execute(
        """
        CREATE TABLE manual_compactions (
            attempt_id TEXT NOT NULL PRIMARY KEY,
            thread_id TEXT NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
            command_message_id TEXT NOT NULL,
            source_head_message_id TEXT NOT NULL,
            expected_head_message_id TEXT NOT NULL,
            source_message_ids_json TEXT NOT NULL,
            effective_source_message_ids_json TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            request_hash TEXT NOT NULL,
            request_message_count INTEGER NOT NULL,
            project_instruction_digest TEXT NOT NULL,
            project_instruction_revision INTEGER NOT NULL,
            context_digest TEXT NOT NULL,
            archive_payload_json TEXT NOT NULL,
            archive_payload_hash TEXT NOT NULL,
            output_summary_hash TEXT,
            output_finish_reason TEXT,
            output_recorded_at INTEGER,
            revision INTEGER NOT NULL,
            state TEXT NOT NULL CHECK(state IN ('pending', 'running', 'active', 'cancelled', 'failed')),
            summary_message_id TEXT,
            summary_hash TEXT,
            archive_status TEXT NOT NULL DEFAULT 'pending',
            created_at INTEGER NOT NULL,
            started_at INTEGER,
            lease_expires_at INTEGER,
            cancelled_at INTEGER,
            committed_at INTEGER,
            terminal_reason TEXT,
            finished_at INTEGER,
            UNIQUE(thread_id, revision, command_message_id)
        )
        """
    )
    now = int(time.time() * 1000)
    archive_payload = [{"role": "user", "content": "archived text"}]
    archive_json = json.dumps(
        archive_payload,
        ensure_ascii = False,
        allow_nan = False,
        sort_keys = True,
        separators = (",", ":"),
    )
    archive_hash = hashlib.sha256(archive_json.encode()).hexdigest()
    summary_hash = hashlib.sha256(b"summary").hexdigest()
    rows = [
        ("rich-pending", "pending", None, None, None, None, None, None, None),
        ("rich-running", "running", now - 5, now + 60_000, None, None, None, None, None),
        (
            "rich-active",
            "active",
            now - 10,
            None,
            None,
            now - 1,
            "summary-active",
            summary_hash,
            None,
        ),
        ("rich-cancelled", "cancelled", None, None, now - 3, None, None, None, "user_cancelled"),
        ("rich-failed", "failed", now - 9, None, None, None, None, None, "provider_failed"),
    ]
    for revision, row in enumerate(rows, start = 1):
        (
            attempt_id,
            state,
            started_at,
            lease_expires_at,
            cancelled_at,
            committed_at,
            summary_message_id,
            stored_summary_hash,
            terminal_reason,
        ) = row
        is_terminal = state in ("cancelled", "failed")
        conn.execute(
            """
            INSERT INTO manual_compactions (
                attempt_id, thread_id, command_message_id, source_head_message_id,
                expected_head_message_id, source_message_ids_json,
                effective_source_message_ids_json, source_hash, request_hash,
                request_message_count, project_instruction_digest,
                project_instruction_revision, context_digest, archive_payload_json,
                archive_payload_hash, output_summary_hash, output_finish_reason,
                output_recorded_at, revision, state, summary_message_id, summary_hash,
                archive_status, created_at, started_at, lease_expires_at, cancelled_at,
                committed_at, terminal_reason, finished_at
            ) VALUES (?, 'thread-1', ?, 'a1', 'compact-1', '["u1","a1"]',
                '["u1","a1"]', ?, ?, 4, ?, 7, ?, ?, ?, ?, 'stop', ?, ?, ?, ?, ?,
                'archived', ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                attempt_id,
                f"compact-{revision}",
                "a" * 64,
                "b" * 64,
                "c" * 64,
                "d" * 64,
                archive_json,
                archive_hash,
                summary_hash,
                now - 2,
                revision,
                state,
                summary_message_id,
                stored_summary_hash,
                now - 20,
                started_at,
                lease_expires_at,
                cancelled_at,
                committed_at,
                terminal_reason,
                now - 1 if is_terminal else None,
            ),
        )
    conn.execute("DROP TABLE discarded_current")
    conn.commit()
    conn.close()
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    migrated = {row[0]: manual_compaction.get_manual_compaction_attempt(row[0]) for row in rows}

    assert migrated["rich-pending"]["state"] == "failed"
    assert migrated["rich-pending"]["terminalReason"] == "migrated_failed"
    assert migrated["rich-running"]["state"] == "failed"
    assert migrated["rich-running"]["terminalReason"] == "migrated_failed"
    assert migrated["rich-running"]["startedAt"] == now - 5
    assert migrated["rich-running"]["leaseExpiresAt"] is None
    assert migrated["rich-active"]["state"] == "failed"
    assert migrated["rich-active"]["terminalReason"] == "migrated_failed"
    assert migrated["rich-active"]["summaryMessageId"] is None
    assert migrated["rich-active"]["archivePayload"] == []
    assert migrated["rich-cancelled"]["state"] == "cancelled"
    assert migrated["rich-cancelled"]["terminalReason"] == "migrated_cancelled"
    assert migrated["rich-cancelled"]["cancelledAt"] == now - 3
    assert migrated["rich-failed"]["state"] == "failed"
    assert migrated["rich-failed"]["terminalReason"] == "migrated_failed"
    for attempt in migrated.values():
        assert attempt["requestHash"] == "b" * 64
        assert attempt["requestMessageCount"] == 4
        assert attempt["projectInstructionDigest"] == "c" * 64
        assert attempt["projectInstructionRevision"] == 7
        assert attempt["contextDigest"] == "d" * 64


def test_migration_backfills_state_bound_digest_for_complete_active_checkpoint(
    tmp_path, monkeypatch
):
    _seed_branch(tmp_path, monkeypatch)
    active = _activate(_prepare("complete-active"))
    conn = studio_db.get_connection()
    canonical_schema = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'manual_compactions'"
    ).fetchone()["sql"]
    conn.execute("DROP INDEX IF EXISTS idx_manual_compactions_thread_state")
    conn.execute("DROP INDEX IF EXISTS idx_manual_compactions_live_branch")
    conn.execute("ALTER TABLE manual_compactions RENAME TO discarded_current")
    prefix, closing = canonical_schema.rsplit(")", 1)
    conn.execute(prefix + ", CHECK(archive_status != 'impossible'))" + closing)
    conn.execute("INSERT INTO manual_compactions SELECT * FROM discarded_current")
    conn.execute("DROP TABLE discarded_current")
    conn.commit()
    conn.close()
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    migrated = manual_compaction.get_manual_compaction_attempt(active["attemptId"])

    assert migrated["state"] == "active"
    assert migrated["commandHashState"] == "summary_saved"
    command = studio_db.get_chat_message("thread-1", migrated["commandMessageId"])
    assert migrated["commandHash"] == studio_db.canonical_chat_message_hash(
        command, state = "summary_saved"
    )


@pytest.mark.parametrize("checkpoint_state", ["pending", "active"])
def test_incompatible_schema_rebuild_preserves_exact_empty_string_root_checkpoint(
    tmp_path, monkeypatch, checkpoint_state
):
    _seed_branch(tmp_path, monkeypatch)
    conn = studio_db.get_connection()
    try:
        conn.execute("UPDATE chat_messages SET parent_id = '' WHERE id = 'u1'")
        conn.commit()
    finally:
        conn.close()
    prepared = _prepare(f"empty-root-{checkpoint_state}")
    checkpoint = _activate(prepared) if checkpoint_state == "active" else prepared
    before = _raw_attempt(checkpoint["attemptId"])

    _force_incompatible_manual_compactions_schema()
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    migrated = manual_compaction.get_manual_compaction_attempt(checkpoint["attemptId"])
    after = _raw_attempt(checkpoint["attemptId"])

    assert migrated["state"] == checkpoint_state
    assert migrated["sourceHash"] == checkpoint["sourceHash"]
    assert migrated["commandHash"] == checkpoint["commandHash"]
    assert migrated["commandHashState"] == checkpoint["commandHashState"]
    assert after["source_hash"] == before["source_hash"]
    assert after["command_hash"] == before["command_hash"]
    assert studio_db.get_chat_message("thread-1", "u1")["parentId"] == ""


def _install_cascading_reservation_schema(conn):
    conn.execute("DROP TRIGGER IF EXISTS retire_manual_compaction_attempt_reservations")
    conn.execute(
        "ALTER TABLE manual_compaction_attempt_reservations RENAME TO durable_reservations"
    )
    conn.execute(
        """
        CREATE TABLE manual_compaction_attempt_reservations (
            attempt_id TEXT NOT NULL PRIMARY KEY,
            thread_id TEXT NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
            command_message_id TEXT NOT NULL,
            summary_message_id TEXT NOT NULL,
            attempt_sequence INTEGER NOT NULL
                CHECK(attempt_sequence >= 1 AND attempt_sequence <= 9007199254740991),
            created_at INTEGER NOT NULL,
            UNIQUE(thread_id, command_message_id, attempt_sequence)
        )
        """
    )
    conn.execute(
        "INSERT INTO manual_compaction_attempt_reservations "
        "SELECT attempt_id, thread_id, command_message_id, summary_message_id, "
        "attempt_sequence, created_at FROM durable_reservations"
    )
    conn.execute("DROP TABLE durable_reservations")


def test_cascading_reservation_schema_migrates_without_losing_global_identity(
    tmp_path, monkeypatch
):
    _seed_branch(tmp_path, monkeypatch)
    manual_compaction.bind_manual_compaction_command(
        "thread-1",
        attempt_id = "legacy-reservation",
        command_message_id = "compact-1",
        summary_message_id = "summary-1",
        attempt_sequence = 1,
    )
    conn = studio_db.get_connection()
    try:
        _install_cascading_reservation_schema(conn)
        conn.commit()
    finally:
        conn.close()
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    assert studio_db.get_chat_message("thread-1", "compact-1") is not None
    conn = studio_db.get_connection()
    try:
        assert (
            conn.execute(
                "PRAGMA foreign_key_list(manual_compaction_attempt_reservations)"
            ).fetchall()
            == []
        )
        columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(manual_compaction_attempt_reservations)"
            ).fetchall()
        }
        assert "retired_at" in columns
        reservation = conn.execute(
            "SELECT thread_id, command_message_id, retired_at "
            "FROM manual_compaction_attempt_reservations WHERE attempt_id = ?",
            ("legacy-reservation",),
        ).fetchone()
        assert tuple(reservation) == ("thread-1", "compact-1", None)
    finally:
        conn.close()

    studio_db.delete_chat_threads(["thread-1"])
    conn = studio_db.get_connection()
    try:
        retired = conn.execute(
            "SELECT retired_at FROM manual_compaction_attempt_reservations WHERE attempt_id = ?",
            ("legacy-reservation",),
        ).fetchone()
        assert retired is not None
        assert retired["retired_at"] > 0
    finally:
        conn.close()


def test_reservation_migration_allows_same_command_summary_retries(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare("same-command-first")
    _claim(prepared)
    manual_compaction.cancel_manual_compaction(
        "thread-1",
        attempt_id = prepared["attemptId"],
        command_message_id = prepared["commandMessageId"],
        claim_id = _CLAIM_ID,
    )
    manual_compaction.bind_manual_compaction_command(
        "thread-1",
        attempt_id = "same-command-second",
        command_message_id = "compact-1",
        summary_message_id = "summary-1",
        attempt_sequence = 2,
        expected_attempt_id = prepared["attemptId"],
        expected_attempt_sequence = 1,
    )
    conn = studio_db.get_connection()
    try:
        _install_cascading_reservation_schema(conn)
        conn.commit()
    finally:
        conn.close()
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    conn = studio_db.get_connection()
    try:
        rows = conn.execute(
            "SELECT attempt_id, thread_id, command_message_id, summary_message_id "
            "FROM manual_compaction_attempt_reservations ORDER BY attempt_sequence"
        ).fetchall()
        assert [tuple(row) for row in rows] == [
            ("same-command-first", "thread-1", "compact-1", "summary-1"),
            ("same-command-second", "thread-1", "compact-1", "summary-1"),
        ]
    finally:
        conn.close()


def test_reservation_migration_rejects_cross_command_duplicate_summary_identity(
    tmp_path, monkeypatch
):
    _seed_branch(tmp_path, monkeypatch)
    manual_compaction.bind_manual_compaction_command(
        "thread-1",
        attempt_id = "duplicate-first",
        command_message_id = "compact-1",
        summary_message_id = "shared-summary",
        attempt_sequence = 1,
    )
    studio_db.upsert_chat_thread(_thread("thread-2"))
    for message in (
        _message("u2", None, "user", "Other question", thread_id = "thread-2"),
        _message("a2", "u2", "assistant", "Other answer", thread_id = "thread-2"),
        _message("compact-2", "a2", "user", "/compact", thread_id = "thread-2"),
    ):
        studio_db.upsert_chat_message(message)
    manual_compaction.bind_manual_compaction_command(
        "thread-2",
        attempt_id = "duplicate-second",
        command_message_id = "compact-2",
        summary_message_id = "second-summary",
        attempt_sequence = 1,
    )
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "UPDATE manual_compaction_attempt_reservations SET summary_message_id = ? "
            "WHERE attempt_id = ?",
            ("shared-summary", "duplicate-second"),
        )
        _install_cascading_reservation_schema(conn)
        before = [
            tuple(row)
            for row in conn.execute(
                "SELECT * FROM manual_compaction_attempt_reservations ORDER BY attempt_id"
            ).fetchall()
        ]
        conn.commit()
    finally:
        conn.close()
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    with pytest.raises(ValueError, match = "summary reservation conflicts"):
        studio_db.get_connection()

    raw = sqlite3.connect(str(studio_db.studio_db_path()))
    try:
        assert raw.execute(
            "PRAGMA foreign_key_list(manual_compaction_attempt_reservations)"
        ).fetchall()
        assert [
            tuple(row)
            for row in raw.execute(
                "SELECT * FROM manual_compaction_attempt_reservations ORDER BY attempt_id"
            ).fetchall()
        ] == before
        assert (
            raw.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name LIKE 'manual_compaction_attempt_reservations_legacy%'"
            ).fetchall()
            == []
        )
    finally:
        raw.close()
    assert studio_db._schema_ready is False


def test_reservation_backfill_rejects_cross_command_duplicate_summary_identity(
    tmp_path, monkeypatch
):
    _seed_branch(tmp_path, monkeypatch)
    manual_compaction.bind_manual_compaction_command(
        "thread-1",
        attempt_id = "backfill-first",
        command_message_id = "compact-1",
        summary_message_id = "shared-summary",
        attempt_sequence = 1,
    )
    studio_db.upsert_chat_thread(_thread("thread-2"))
    for message in (
        _message("u2", None, "user", "Other question", thread_id = "thread-2"),
        _message("a2", "u2", "assistant", "Other answer", thread_id = "thread-2"),
        _message("compact-2", "a2", "user", "/compact", thread_id = "thread-2"),
    ):
        studio_db.upsert_chat_message(message)
    manual_compaction.bind_manual_compaction_command(
        "thread-2",
        attempt_id = "backfill-second",
        command_message_id = "compact-2",
        summary_message_id = "second-summary",
        attempt_sequence = 1,
    )
    conn = studio_db.get_connection()
    try:
        row = conn.execute(
            "SELECT metadata_json FROM chat_messages WHERE id = ?",
            ("compact-2",),
        ).fetchone()
        metadata = json.loads(row["metadata_json"])
        metadata[manual_compaction.MANUAL_COMPACTION_CLIENT_KEY]["summaryMessageId"] = (
            "shared-summary"
        )
        conn.execute(
            "UPDATE chat_messages SET metadata_json = ? WHERE id = ?",
            (json.dumps(metadata, separators = (",", ":")), "compact-2"),
        )
        conn.execute("DELETE FROM manual_compaction_attempt_reservations")
        conn.commit()
    finally:
        conn.close()
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    with pytest.raises(ValueError, match = "summary reservation conflicts"):
        studio_db.get_connection()

    raw = sqlite3.connect(str(studio_db.studio_db_path()))
    try:
        assert raw.execute("SELECT * FROM manual_compaction_attempt_reservations").fetchall() == []
        assert (
            raw.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name LIKE 'manual_compaction_attempt_reservations_legacy%'"
            ).fetchall()
            == []
        )
    finally:
        raw.close()
    assert studio_db._schema_ready is False


@pytest.mark.parametrize("terminal_state", ["cancelled", "failed"])
def test_incompatible_schema_rebuild_preserves_terminal_claim_owner_replay(
    tmp_path, monkeypatch, terminal_state
):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare(f"terminal-claim-{terminal_state}")
    _claim(prepared)
    if terminal_state == "cancelled":
        manual_compaction.cancel_manual_compaction(
            "thread-1",
            attempt_id = prepared["attemptId"],
            command_message_id = prepared["commandMessageId"],
            claim_id = _CLAIM_ID,
        )
    else:
        manual_compaction.fail_manual_compaction_attempt(prepared["attemptId"], "provider_failed")
    before = _raw_attempt(prepared["attemptId"])
    conn = studio_db.get_connection()
    canonical_schema = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'manual_compactions'"
    ).fetchone()["sql"]
    conn.execute("DROP INDEX IF EXISTS idx_manual_compactions_thread_state")
    conn.execute("DROP INDEX IF EXISTS idx_manual_compactions_live_branch")
    conn.execute("ALTER TABLE manual_compactions RENAME TO discarded_current")
    prefix, closing = canonical_schema.rsplit(")", 1)
    conn.execute(prefix + ", CHECK(archive_status != 'impossible'))" + closing)
    conn.execute("INSERT INTO manual_compactions SELECT * FROM discarded_current")
    conn.execute("DROP TABLE discarded_current")
    conn.commit()
    conn.close()
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    migrated = manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])
    after = _raw_attempt(prepared["attemptId"])

    assert migrated["state"] == terminal_state
    assert after["claim_token_hash"] == before["claim_token_hash"]
    assert after["claim_token_hash"] != _CLAIM_ID
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "not owned"):
        manual_compaction.cancel_manual_compaction(
            "thread-1",
            attempt_id = prepared["attemptId"],
            command_message_id = prepared["commandMessageId"],
            claim_id = "4" * 64,
        )
    replayed = manual_compaction.cancel_manual_compaction(
        "thread-1",
        attempt_id = prepared["attemptId"],
        command_message_id = prepared["commandMessageId"],
        claim_id = _CLAIM_ID,
    )
    assert replayed["state"] == terminal_state


def test_migration_maps_incomplete_running_row_to_a_valid_failed_audit(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    conn = studio_db.get_connection()
    conn.execute("DROP INDEX IF EXISTS idx_manual_compactions_thread_state")
    conn.execute("DROP INDEX IF EXISTS idx_manual_compactions_live_branch")
    conn.execute("ALTER TABLE manual_compactions RENAME TO discarded_current")
    conn.execute(
        _LEGACY_SCHEMA.replace(
            "CHECK(state IN ('pending', 'active'))",
            "CHECK(state IN ('pending', 'running', 'active'))",
        )
    )
    now = int(time.time() * 1000)
    conn.execute(
        """
        INSERT INTO manual_compactions (
            attempt_id, thread_id, command_message_id, source_head_message_id,
            expected_head_message_id, source_message_ids_json,
            effective_source_message_ids_json, source_hash, request_hash,
            request_message_count, project_instruction_digest,
            project_instruction_revision, context_digest, revision, state,
            archive_status, created_at
        ) VALUES ('broken-running', 'thread-1', 'compact-1', 'a1', 'compact-1',
            '["u1","a1"]', '["u1","a1"]', ?, ?, 4, ?, 7, ?, 1,
            'running', 'pending', ?)
        """,
        ("a" * 64, "b" * 64, "c" * 64, "d" * 64, now),
    )
    conn.execute("DROP TABLE discarded_current")
    conn.commit()
    conn.close()
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    recovered = manual_compaction.get_manual_compaction_attempt("broken-running")

    assert recovered["state"] == "failed"
    assert recovered["terminalReason"] == "migrated_failed"
    assert recovered["finishedAt"] == now
    _assert_scrubbed_terminal(
        _raw_attempt("broken-running"), state = "failed", reason = "migrated_failed"
    )
