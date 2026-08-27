# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import hashlib
import json
import time

import pytest

from core import manual_compaction
from models.inference import ChatCompletionRequest
from storage import studio_db


_EMPTY_JSON_HASH = hashlib.sha256(b"[]").hexdigest()


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

    for attempt_id in attempts:
        _prepare(attempt_id)

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
        _assert_scrubbed_terminal(row, state = "cancelled", reason = "replaced")


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
    pending = _prepare("pending-dormant")
    studio_db.upsert_chat_thread(_thread("thread-2"))
    for row in (
        _message("u2-1", None, "user", "Explain the migration.", thread_id = "thread-2"),
        _message("a2-1", "u2-1", "assistant", "Use a staged rollout.", thread_id = "thread-2"),
        _message("compact-2", "a2-1", "user", "/compact", thread_id = "thread-2"),
    ):
        studio_db.upsert_chat_message(row)
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
        reason = "lease_expired",
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

    assert recovered["state"] == "pending"
    assert recovered["sourceMessageIds"] == ["u1", "a1"]
    assert recovered["effectiveSourceMessageIds"] == ["u1", "a1"]
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
    cancelled = manual_compaction.cancel_manual_compaction(
        "thread-1",
        attempt_id = prepared["attemptId"],
        command_message_id = prepared["commandMessageId"],
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
        "state = 'running', started_at = 1, lease_expires_at = 'not-a-time'",
        "state = 'active', started_at = 1, committed_at = 1",
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

    assert orphan["state"] == "pending"
    assert orphan["sourceMessageIds"] == ["u1", "a1"]
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


def test_migration_preserves_richer_columns_and_every_valid_lifecycle_state(tmp_path, monkeypatch):
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

    assert migrated["rich-pending"]["state"] == "pending"
    assert migrated["rich-running"]["state"] == "running"
    assert migrated["rich-running"]["startedAt"] == now - 5
    assert migrated["rich-running"]["leaseExpiresAt"] == now + 60_000
    assert migrated["rich-active"]["state"] == "active"
    assert migrated["rich-active"]["summaryMessageId"] == "summary-active"
    assert migrated["rich-active"]["summaryHash"] == summary_hash
    assert migrated["rich-active"]["archivePayload"] == archive_payload
    assert migrated["rich-active"]["outputSummaryHash"] == summary_hash
    assert migrated["rich-active"]["outputFinishReason"] == "stop"
    assert migrated["rich-active"]["archiveStatus"] == "archived"
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
