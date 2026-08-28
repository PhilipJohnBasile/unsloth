# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from core import manual_compaction
from core.rag import conversation_archive
from models.inference import ChatCompletionRequest
from routes import chat_history, inference
from storage import chat_generation_runs_db, research_runs_db, studio_db


_CLAIM_ID = "1" * 64
_prepare_core = manual_compaction.prepare_manual_compaction


def _reset_db(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)


def _thread(thread_id = "thread-1"):
    return {
        "id": thread_id,
        "title": "Compaction test",
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
    metadata = None,
):
    return {
        "id": message_id,
        "threadId": thread_id,
        "parentId": parent_id,
        "role": role,
        "content": [{"type": "text", "text": text}],
        "metadata": metadata,
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


def _set_raw_message_json(message_id, column, value):
    statements = {
        "content_json": "UPDATE chat_messages SET content_json = ? WHERE id = ?",
        "attachments_json": "UPDATE chat_messages SET attachments_json = ? WHERE id = ?",
        "metadata_json": "UPDATE chat_messages SET metadata_json = ? WHERE id = ?",
    }
    conn = studio_db.get_connection()
    try:
        conn.execute(statements[column], (value, message_id))
        conn.commit()
    finally:
        conn.close()


def _set_raw_parent(message_id, parent_id):
    conn = studio_db.get_connection()
    try:
        conn.execute("UPDATE chat_messages SET parent_id = ? WHERE id = ?", (parent_id, message_id))
        conn.commit()
    finally:
        conn.close()


def _raw_message_row(message_id):
    conn = studio_db.get_connection()
    try:
        row = conn.execute(
            "SELECT id, thread_id, parent_id, role, content_json, attachments_json, "
            "metadata_json, created_at FROM chat_messages WHERE id = ?",
            (message_id,),
        ).fetchone()
        return tuple(row) if row is not None else None
    finally:
        conn.close()


def _raw_attempt_row(attempt_id):
    conn = studio_db.get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM manual_compactions WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        return tuple(row) if row is not None else None
    finally:
        conn.close()


def _raw_reservation_row(attempt_id):
    conn = studio_db.get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM manual_compaction_attempt_reservations WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        return tuple(row) if row is not None else None
    finally:
        conn.close()


def _raw_research_state(run_id, thread_id = "thread-1"):
    conn = studio_db.get_connection()
    try:
        run = conn.execute("SELECT * FROM research_runs WHERE id = ?", (run_id,)).fetchone()
        claims = conn.execute(
            "SELECT * FROM research_thread_claims WHERE thread_id = ? ORDER BY owner_subject",
            (thread_id,),
        ).fetchall()
        events = conn.execute(
            "SELECT * FROM research_events WHERE run_id = ? ORDER BY seq",
            (run_id,),
        ).fetchall()
        return (
            tuple(run) if run is not None else None,
            [tuple(row) for row in claims],
            [tuple(row) for row in events],
        )
    finally:
        conn.close()


def _raw_generation_state(run_id):
    conn = studio_db.get_connection()
    try:
        run = conn.execute(
            "SELECT * FROM chat_generation_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        events = conn.execute(
            "SELECT * FROM chat_generation_events WHERE run_id = ? ORDER BY seq",
            (run_id,),
        ).fetchall()
        return tuple(run) if run is not None else None, [tuple(row) for row in events]
    finally:
        conn.close()


def _raw_thread_and_messages(thread_id = "thread-1"):
    conn = studio_db.get_connection()
    try:
        thread = conn.execute(
            "SELECT * FROM chat_threads WHERE id = ?",
            (thread_id,),
        ).fetchone()
        messages = conn.execute(
            "SELECT * FROM chat_messages WHERE thread_id = ? ORDER BY id",
            (thread_id,),
        ).fetchall()
        return (
            tuple(thread) if thread is not None else None,
            [tuple(row) for row in messages],
        )
    finally:
        conn.close()


def _seed_prunable_branch(
    tmp_path,
    monkeypatch,
    assistant_content,
    *,
    metadata = None,
):
    _reset_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())
    rows = [
        _message("u1", None, "user", "Explain the migration."),
        _message("a1", "u1", "assistant", "Use a staged rollout."),
        _message("u2", "a1", "user", "Repeat the unsafe operation."),
        {
            **_message("a2", "u2", "assistant", ""),
            "content": assistant_content,
            "metadata": metadata,
        },
        _message("compact-1", "a2", "user", "/compact"),
    ]
    for row in rows:
        studio_db.upsert_chat_message(row)
    return rows


def _wire_messages():
    return [
        {"role": "user", "content": "Explain the migration."},
        {"role": "assistant", "content": "Use a staged rollout."},
        {"role": "user", "content": "/compact"},
    ]


def _full_wire_messages():
    return [{"role": "system", "content": "Project rules"}, *_wire_messages()]


def _prepare_bound(
    thread_id,
    *,
    attempt_id,
    command_message_id,
    expected_head_message_id,
    message_ids,
    request_messages,
    summary_message_id = None,
):
    if summary_message_id is None:
        summary_message_id = (
            "summary-1" if command_message_id == "compact-1" else f"summary-{command_message_id}"
        )
    command = studio_db.get_chat_message(thread_id, command_message_id)
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
            sequence = current.get("attemptSequence", 1)
            summary_message_id = current.get("summaryMessageId", summary_message_id)
        else:
            sequence = current.get("attemptSequence", 0) + 1
            expected_attempt_id = current.get("attemptId")
            expected_attempt_sequence = current.get("attemptSequence")
    manual_compaction.bind_manual_compaction_command(
        thread_id,
        attempt_id = attempt_id,
        command_message_id = command_message_id,
        summary_message_id = summary_message_id,
        attempt_sequence = sequence,
        expected_attempt_id = expected_attempt_id,
        expected_attempt_sequence = expected_attempt_sequence,
    )
    return _prepare_core(
        thread_id,
        attempt_id = attempt_id,
        command_message_id = command_message_id,
        expected_head_message_id = expected_head_message_id,
        message_ids = message_ids,
        request_messages = request_messages,
    )


def _prepare(messages = None, *, summary_message_id = "summary-1"):
    request_messages = ChatCompletionRequest(
        model = "model", messages = messages or _full_wire_messages()
    ).messages
    return _prepare_bound(
        "thread-1",
        attempt_id = "attempt-1",
        command_message_id = "compact-1",
        expected_head_message_id = "compact-1",
        message_ids = ["u1", "a1", "compact-1"],
        request_messages = request_messages,
        summary_message_id = summary_message_id,
    )


def _client_record(thread_id = "thread-1", command_message_id = "compact-1"):
    command = studio_db.get_chat_message(thread_id, command_message_id)
    metadata = command.get("metadata") if command else None
    return (
        metadata.get(manual_compaction.MANUAL_COMPACTION_CLIENT_KEY)
        if isinstance(metadata, dict)
        else None
    )


def _request(
    prepared,
    messages = None,
    *,
    claim_id = _CLAIM_ID,
):
    return ChatCompletionRequest(
        model = "model",
        thread_id = "thread-1",
        messages = messages or _full_wire_messages(),
        tools = [{"type": "function", "function": {"name": "terminal"}}],
        tool_choice = "auto",
        enable_tools = True,
        enabled_tools = ["terminal"],
        mcp_enabled = True,
        deep_research_armed = True,
        confirm_tool_calls = True,
        max_tool_calls_per_message = 25,
        run_tools_locally = True,
        rag_scope = {"thread_id": "thread-1"},
        context_overflow = "truncate_oldest",
        context_policy = "checkpoint",
        compaction_headroom_ratio = 0.25,
        compaction_threshold = 50_000,
        continue_final_message = True,
        response_format = {"type": "json_object"},
        n = 2,
        stop = "STOP",
        logprobs = True,
        top_logprobs = 3,
        parallel_tool_calls = True,
        context_management = [{"type": "compaction"}],
        manual_compaction = {
            "attemptId": prepared["attemptId"],
            "claimId": claim_id,
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


def _claim(
    prepared,
    messages = None,
    *,
    claim_id = _CLAIM_ID,
):
    request = _request(prepared, messages = messages, claim_id = claim_id)
    manual_compaction.validate_and_rewrite_manual_compaction_request(request)
    return request


def _record_output(
    prepared,
    text,
    finish_reason = "stop",
):
    return manual_compaction.record_manual_compaction_output(
        prepared["attemptId"],
        text = text,
        finish_reason = finish_reason,
    )


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
    # These lifecycle modules bind the storage function at import time, while manual
    # compaction resolves it through the storage module. Gate every direct writer so
    # this helper controls the same SQLite transaction boundary for either operation.
    monkeypatch.setattr(chat_generation_runs_db, "get_connection", gated_get_connection)
    monkeypatch.setattr(research_runs_db, "get_connection", gated_get_connection)
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


def test_prepare_pins_exact_branch_and_excludes_literal_command(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()

    assert prepared["sourceMessageIds"] == ["u1", "a1"]
    assert prepared["sourceHeadMessageId"] == "a1"
    assert prepared["expectedHeadMessageId"] == "compact-1"
    assert prepared["revision"] == 1
    assert prepared["state"] == "pending"
    assert prepared["requestMessageCount"] == 4
    assert prepared["requestHash"] == manual_compaction.canonical_request_hash(
        _full_wire_messages()
    )
    assert prepared["sourceHash"] == manual_compaction.canonical_source_hash(
        [
            studio_db.get_chat_message("thread-1", "u1"),
            studio_db.get_chat_message("thread-1", "a1"),
        ]
    )
    assert _prepare() == prepared


def test_prepare_prunes_anthropic_refusal_and_pins_full_source(tmp_path, monkeypatch):
    _seed_prunable_branch(
        tmp_path,
        monkeypatch,
        [{"type": "text", "text": "I cannot help with that."}],
        metadata = {"custom": {"anthropicRefusal": True}},
    )
    messages = [
        {"role": "system", "content": "Project rules"},
        {"role": "user", "content": "Explain the migration."},
        {"role": "assistant", "content": "Use a staged rollout."},
        {"role": "user", "content": "/compact"},
    ]

    prepared = _prepare_bound(
        "thread-1",
        attempt_id = "attempt-refusal",
        command_message_id = "compact-1",
        expected_head_message_id = "compact-1",
        message_ids = ["u1", "a1", "u2", "a2", "compact-1"],
        request_messages = messages,
    )

    full_source = [
        studio_db.get_chat_message("thread-1", message_id)
        for message_id in ("u1", "a1", "u2", "a2")
    ]
    assert prepared["sourceMessageIds"] == ["u1", "a1", "u2", "a2"]
    assert prepared["effectiveSourceMessageIds"] == ["u1", "a1", "u2", "a2"]
    assert prepared["sourceHash"] == manual_compaction.canonical_source_hash(full_source)
    assert [message["role"] for message in prepared["archivePayload"]] == ["user", "assistant"]
    manual_compaction.validate_and_rewrite_manual_compaction_request(
        _request(prepared, messages = messages)
    )


@pytest.mark.parametrize("include_reasoning", [False, True])
def test_prepare_prunes_abandoned_turn_in_both_reasoning_modes(
    tmp_path, monkeypatch, include_reasoning
):
    _seed_prunable_branch(
        tmp_path,
        monkeypatch,
        [{"type": "reasoning", "text": "partial"}],
        metadata = {"custom": {"incomplete": {"reason": "cancelled"}}},
    )
    stored_branch = [
        studio_db.get_chat_message("thread-1", message_id)
        for message_id in ("u1", "a1", "u2", "a2", "compact-1")
    ]
    assert [
        message["id"]
        for message in manual_compaction._prune_stored_branch(
            stored_branch, include_reasoning = include_reasoning
        )
    ] == ["u1", "a1", "compact-1"]
    messages = [
        {"role": "system", "content": "Project rules"},
        {"role": "user", "content": "Explain the migration."},
        {"role": "assistant", "content": "Use a staged rollout."},
        {"role": "user", "content": "/compact"},
    ]

    prepared = _prepare_bound(
        "thread-1",
        attempt_id = f"attempt-abandoned-{include_reasoning}",
        command_message_id = "compact-1",
        expected_head_message_id = "compact-1",
        message_ids = ["u1", "a1", "u2", "a2", "compact-1"],
        request_messages = messages,
    )

    assert prepared["sourceMessageIds"] == ["u1", "a1", "u2", "a2"]
    assert [message["role"] for message in prepared["archivePayload"]] == ["user", "assistant"]
    manual_compaction.validate_and_rewrite_manual_compaction_request(
        _request(prepared, messages = messages)
    )


@pytest.mark.parametrize(
    ("assistant_content", "metadata", "attempt_id"),
    [
        (
            [{"type": "text", "text": "I cannot help with that."}],
            {"custom": {"anthropicRefusal": True}},
            "attempt-archive-refusal",
        ),
        (
            [{"type": "reasoning", "text": "partial"}],
            {"custom": {"incomplete": {"reason": "cancelled"}}},
            "attempt-archive-abandoned",
        ),
    ],
    ids = ["refusal", "abandoned"],
)
def test_archive_uses_persisted_pruned_payload(
    tmp_path, monkeypatch, assistant_content, metadata, attempt_id
):
    _seed_prunable_branch(
        tmp_path,
        monkeypatch,
        assistant_content,
        metadata = metadata,
    )
    messages = [
        {"role": "system", "content": "Project rules"},
        {"role": "user", "content": "Explain the migration."},
        {"role": "assistant", "content": "Use a staged rollout."},
        {"role": "user", "content": "/compact"},
    ]
    prepared = _prepare_bound(
        "thread-1",
        attempt_id = attempt_id,
        command_message_id = "compact-1",
        expected_head_message_id = "compact-1",
        message_ids = ["u1", "a1", "u2", "a2", "compact-1"],
        request_messages = messages,
    )
    manual_compaction.validate_and_rewrite_manual_compaction_request(
        _request(prepared, messages = messages)
    )
    _record_output(prepared, "summary")
    studio_db.upsert_chat_message(_message("summary-1", "compact-1", "assistant", "summary"))
    result = manual_compaction.commit_manual_compaction(
        "thread-1",
        attempt_id = attempt_id,
        command_message_id = "compact-1",
        summary_message_id = "summary-1",
        expected_head_message_id = "summary-1",
        expected_revision = 1,
        expected_summary_hash = manual_compaction.summary_hash("summary"),
    )
    archived = []
    monkeypatch.setattr(conversation_archive, "enabled", lambda: True)
    monkeypatch.setattr(conversation_archive, "can_archive", lambda _thread_id: True)

    def capture_archive(_thread_id, source, **_kwargs):
        archived.append(source)
        return 1

    monkeypatch.setattr(conversation_archive, "archive_turns", capture_archive)
    monkeypatch.setattr(conversation_archive, "degraded", lambda: False)

    assert manual_compaction.archive_manual_compaction_best_effort(result) == "archived"
    assert archived == [prepared["archivePayload"]]
    assert "Repeat the unsafe operation" not in json.dumps(archived)


@pytest.mark.parametrize("include_reasoning", [False, True])
def test_stored_branch_keeps_prompt_before_trailing_abandoned_turn(
    tmp_path, monkeypatch, include_reasoning
):
    _seed_prunable_branch(
        tmp_path,
        monkeypatch,
        [],
        metadata = {"custom": {"incomplete": {"reason": "interrupted"}}},
    )
    trailing_branch = [
        studio_db.get_chat_message("thread-1", message_id)
        for message_id in ("u1", "a1", "u2", "a2")
    ]

    pruned = manual_compaction._prune_stored_branch(
        trailing_branch, include_reasoning = include_reasoning
    )

    assert [message["id"] for message in pruned] == ["u1", "a1", "u2"]


@pytest.mark.parametrize("include_reasoning", [False, True])
def test_prepare_keeps_complete_reasoning_only_turn_in_both_modes(
    tmp_path, monkeypatch, include_reasoning
):
    _seed_prunable_branch(
        tmp_path,
        monkeypatch,
        [{"type": "reasoning", "text": "hidden analysis"}],
    )
    # This is the exact frontend serializer shape for a complete reasoning-only turn.
    reasoning_turn = {"role": "assistant", "content": ""}
    if include_reasoning:
        reasoning_turn["reasoning_content"] = "hidden analysis"
    messages = [
        {"role": "system", "content": "Project rules"},
        {"role": "user", "content": "Explain the migration."},
        {"role": "assistant", "content": "Use a staged rollout."},
        {"role": "user", "content": "Repeat the unsafe operation."},
        reasoning_turn,
        {"role": "user", "content": "/compact"},
    ]

    prepared = _prepare_bound(
        "thread-1",
        attempt_id = f"attempt-reasoning-{include_reasoning}",
        command_message_id = "compact-1",
        expected_head_message_id = "compact-1",
        message_ids = ["u1", "a1", "u2", "a2", "compact-1"],
        request_messages = messages,
    )

    assert prepared["sourceMessageIds"] == ["u1", "a1", "u2", "a2"]
    manual_compaction.validate_and_rewrite_manual_compaction_request(
        _request(prepared, messages = messages)
    )


@pytest.mark.parametrize(
    "column",
    ["content_json", "attachments_json", "metadata_json"],
)
def test_prepare_fails_closed_on_malformed_stored_message_json(tmp_path, monkeypatch, column):
    _seed_branch(tmp_path, monkeypatch)
    _set_raw_message_json("u1", column, "{")

    with pytest.raises(
        manual_compaction.ManualCompactionConflict,
        match = r"Stored message .* JSON is invalid",
    ):
        _prepare()


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("content_json", '[{"type":"text","text":"hello","value":NaN}]'),
        ("attachments_json", '[{"value":Infinity}]'),
        ("metadata_json", '{"value":-Infinity}'),
    ],
    ids = ["content-nan", "attachments-infinity", "metadata-negative-infinity"],
)
def test_prepare_fails_closed_on_non_rfc_stored_message_json(tmp_path, monkeypatch, column, value):
    _seed_branch(tmp_path, monkeypatch)
    _set_raw_message_json("u1", column, value)

    with pytest.raises(
        manual_compaction.ManualCompactionConflict,
        match = r"Stored message .* JSON is invalid",
    ):
        _prepare()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("content", [{"type": "text", "text": "hello", "value": float("nan")}]),
        ("attachments", [{"value": float("inf")}]),
        ("metadata", {"value": float("-inf")}),
    ],
    ids = ["content-nan", "attachments-infinity", "metadata-negative-infinity"],
)
def test_message_writes_reject_non_rfc_json(tmp_path, monkeypatch, field, value):
    _reset_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())
    message = _message("u1", None, "user", "hello")
    message[field] = value

    with pytest.raises(ValueError):
        studio_db.upsert_chat_message(message)
    assert studio_db.get_chat_message("thread-1", "u1") is None
    with pytest.raises(ValueError, match = "RFC JSON"):
        chat_history.ChatMessage(**message)


@pytest.mark.parametrize(
    "column",
    ["content_json", "attachments_json", "metadata_json"],
)
def test_active_retry_fails_closed_on_malformed_stored_message_json(tmp_path, monkeypatch, column):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    _claim(prepared)
    _record_output(prepared, "summary")
    studio_db.upsert_chat_message(_message("summary-1", "compact-1", "assistant", "summary"))
    manual_compaction.commit_manual_compaction(
        "thread-1",
        attempt_id = prepared["attemptId"],
        command_message_id = prepared["commandMessageId"],
        summary_message_id = "summary-1",
        expected_head_message_id = "summary-1",
        expected_revision = prepared["revision"],
        expected_summary_hash = manual_compaction.summary_hash("summary"),
    )
    _set_raw_message_json("u1", column, "{")

    with pytest.raises(
        manual_compaction.ManualCompactionConflict,
        match = r"Stored message .* JSON is invalid",
    ):
        manual_compaction.commit_manual_compaction(
            "thread-1",
            attempt_id = prepared["attemptId"],
            command_message_id = prepared["commandMessageId"],
            summary_message_id = "summary-1",
            expected_head_message_id = "summary-1",
            expected_revision = prepared["revision"],
            expected_summary_hash = manual_compaction.summary_hash("summary"),
        )


@pytest.mark.parametrize(
    "column",
    ["source_message_ids_json", "effective_source_message_ids_json"],
)
def test_attempt_reads_fail_closed_on_malformed_stored_json(tmp_path, monkeypatch, column):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    statements = {
        "source_message_ids_json": (
            "UPDATE manual_compactions SET source_message_ids_json = ? WHERE attempt_id = ?"
        ),
        "effective_source_message_ids_json": (
            "UPDATE manual_compactions SET effective_source_message_ids_json = ? "
            "WHERE attempt_id = ?"
        ),
    }
    conn = studio_db.get_connection()
    try:
        conn.execute(statements[column], ("{", prepared["attemptId"]))
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(
        manual_compaction.ManualCompactionConflict,
        match = r"Stored manual compaction .* JSON is invalid",
    ):
        manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("source_message_ids_json", '["u1",NaN]'),
        ("effective_source_message_ids_json", '["u1",Infinity]'),
        ("archive_payload_json", '[{"role":"user","content":-Infinity}]'),
    ],
    ids = ["source-nan", "effective-infinity", "archive-negative-infinity"],
)
def test_attempt_reads_fail_closed_on_non_rfc_stored_json(tmp_path, monkeypatch, column, value):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    conn = studio_db.get_connection()
    try:
        conn.execute(
            f"UPDATE manual_compactions SET {column} = ? WHERE attempt_id = ?",
            (value, prepared["attemptId"]),
        )
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(
        manual_compaction.ManualCompactionConflict,
        match = r"Stored manual compaction .* JSON is invalid",
    ):
        manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])


@pytest.mark.parametrize(
    ("statement", "values", "detail"),
    [
        (
            "UPDATE manual_compactions SET archive_payload_json = ? WHERE attempt_id = ?",
            ("{",),
            "archive payload JSON is invalid",
        ),
        (
            "UPDATE manual_compactions SET archive_payload_hash = ? WHERE attempt_id = ?",
            ("0" * 64,),
            "archive payload hash is invalid",
        ),
        (
            "UPDATE manual_compactions SET output_summary_hash = ? WHERE attempt_id = ?",
            ("0" * 64,),
            "output provenance is invalid",
        ),
        (
            "UPDATE manual_compactions SET output_finish_reason = ?, output_recorded_at = ? "
            "WHERE attempt_id = ?",
            ("unknown", 1),
            "finish reason is invalid",
        ),
        (
            "UPDATE manual_compactions SET output_finish_reason = ?, output_recorded_at = ? "
            "WHERE attempt_id = ?",
            ("stop", "bad"),
            "output provenance is invalid",
        ),
    ],
    ids = [
        "archive-json",
        "archive-hash",
        "orphan-output-hash",
        "finish-reason",
        "recorded-at",
    ],
)
def test_attempt_reads_fail_closed_on_malformed_archive_and_output_provenance(
    tmp_path, monkeypatch, statement, values, detail
):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    conn = studio_db.get_connection()
    try:
        conn.execute(statement, (*values, prepared["attemptId"]))
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(manual_compaction.ManualCompactionConflict, match = detail):
        manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])


@pytest.mark.parametrize(
    ("message_ids", "command_id", "detail"),
    [
        (["u1", "u1"], "u1", "cycle"),
        (["u1", "missing"], "missing", "missing"),
    ],
)
def test_prepare_rejects_non_exact_ancestry(tmp_path, monkeypatch, message_ids, command_id, detail):
    _seed_branch(tmp_path, monkeypatch)
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = detail):
        _prepare_core(
            "thread-1",
            attempt_id = "bad-attempt",
            command_message_id = command_id,
            expected_head_message_id = command_id,
            message_ids = message_ids,
            request_messages = _full_wire_messages(),
        )


def test_prepare_rejects_cross_thread_ids_and_branch_bounds(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread("thread-2"))
    studio_db.upsert_chat_message(
        _message("foreign", None, "user", "/compact", thread_id = "thread-2")
    )
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "crosses"):
        _prepare_core(
            "thread-1",
            attempt_id = "cross-thread",
            command_message_id = "foreign",
            expected_head_message_id = "foreign",
            message_ids = ["u1", "foreign"],
            request_messages = _full_wire_messages(),
        )
    monkeypatch.setattr(manual_compaction, "MAX_MANUAL_COMPACTION_MESSAGES", 2)
    with pytest.raises(manual_compaction.ManualCompactionError, match = "2 to 2"):
        _prepare_bound(
            "thread-1",
            attempt_id = "too-long",
            command_message_id = "compact-1",
            expected_head_message_id = "compact-1",
            message_ids = ["u1", "a1", "compact-1"],
            request_messages = _full_wire_messages(),
        )


def test_prepare_rejects_attempt_collision_and_non_literal_head(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    _prepare()
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "live /compact"):
        _prepare_bound(
            "thread-1",
            attempt_id = "attempt-same-branch",
            command_message_id = "compact-1",
            expected_head_message_id = "compact-1",
            message_ids = ["u1", "a1", "compact-1"],
            request_messages = ChatCompletionRequest(
                model = "model", messages = _full_wire_messages()
            ).messages,
        )
    assert manual_compaction.get_manual_compaction_attempt("attempt-1")["state"] == "pending"
    studio_db.upsert_chat_message(_message("compact-other", "a1", "user", "/compact"))
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "already used"):
        _prepare_bound(
            "thread-1",
            attempt_id = "attempt-1",
            command_message_id = "compact-other",
            expected_head_message_id = "compact-other",
            message_ids = ["u1", "a1", "compact-other"],
            request_messages = _full_wire_messages(),
        )
    studio_db.upsert_chat_message(_message("compact-invalid", "a1", "user", "/compact now"))
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "literal"):
        _prepare_bound(
            "thread-1",
            attempt_id = "not-literal",
            command_message_id = "compact-invalid",
            expected_head_message_id = "compact-invalid",
            message_ids = ["u1", "a1", "compact-invalid"],
            request_messages = _full_wire_messages(),
        )


def test_inference_revalidates_and_forces_a_bounded_no_tool_handoff(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    request = _request(prepared)
    request.image_base64 = "forged-image"
    request.audio_base64 = "forged-audio"
    request.video_base64 = "forged-video"
    request.max_tokens = 1
    request.max_completion_tokens = 2
    request.studio_tool_history = True
    request.openai_code_exec_container_id = "forged-openai-container"
    request.anthropic_code_exec_container_id = "forged-anthropic-container"
    request.permission_mode = "full"
    request.bypass_permissions = True
    request.auto_heal_tool_calls = True
    request.nudge_tool_calls = True
    request.tool_call_timeout = 9999

    manual_compaction.validate_and_rewrite_manual_compaction_request(request)

    assert request.messages[-1].content == manual_compaction.MANUAL_COMPACTION_HANDOFF_INSTRUCTION
    assert request.tools is None
    assert request.tool_choice == "none"
    assert request.enable_tools is False
    assert request.enabled_tools == []
    assert request.mcp_enabled is False
    assert request.deep_research_armed is False
    assert request.confirm_tool_calls is False
    assert request.permission_mode == "off"
    assert request.bypass_permissions is False
    assert request.auto_heal_tool_calls is False
    assert request.nudge_tool_calls is False
    assert request.max_tool_calls_per_message == 0
    assert request.tool_call_timeout == 1
    assert request.run_tools_locally is False
    assert request.studio_tool_history is None
    assert request.openai_code_exec_container_id is None
    assert request.anthropic_code_exec_container_id is None
    assert request.rag_scope is None
    assert request.image_base64 is None
    assert request.audio_base64 is None
    assert request.video_base64 is None
    assert request.context_overflow == "error"
    assert request.context_policy is None
    assert request.compaction_headroom_ratio is None
    assert request.compaction_threshold is None
    assert request.max_tokens is None
    assert request.max_completion_tokens == manual_compaction.MAX_MANUAL_COMPACTION_SUMMARY_TOKENS
    assert request.continue_final_message is False
    assert request.response_format is None
    assert request.n == 1
    assert request.stop is None
    assert request.logprobs is False
    assert request.top_logprobs is None
    assert request.parallel_tool_calls is False
    assert "context_management" not in (request.model_extra or {})


def test_post_claim_rewrite_failure_terminalizes_attempt(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()

    def fail_rewrite(_payload, **_kwargs):
        raise RuntimeError("rewrite failed")

    monkeypatch.setattr(manual_compaction, "_rewrite_claimed_payload", fail_rewrite)

    with pytest.raises(RuntimeError, match = "rewrite failed"):
        manual_compaction.validate_and_rewrite_manual_compaction_request(_request(prepared))

    stored = manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])
    assert stored["state"] == "failed"
    assert stored["terminalReason"] == "request_rewrite_failed"
    assert stored["leaseExpiresAt"] is None


def test_inference_pins_an_unspecified_output_budget(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    request = _request(prepared)
    assert request.max_tokens is None
    assert request.max_completion_tokens is None

    manual_compaction.validate_and_rewrite_manual_compaction_request(request)

    assert request.max_tokens is None
    assert request.max_completion_tokens == manual_compaction.MAX_MANUAL_COMPACTION_SUMMARY_TOKENS


def test_inference_pins_plain_text_and_removes_all_reasoning_controls(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    raw = _request(prepared).model_dump(by_alias = True, exclude_none = True)
    raw.update(
        {
            "enable_thinking": True,
            "reasoning_effort": "high",
            "preserve_thinking": True,
            "thinking": {"type": "enabled"},
            "chat_template_kwargs": {"enable_thinking": True},
            "reasoning": {"effort": "high"},
            "thinking_budget": 8192,
        }
    )
    request = ChatCompletionRequest(**raw)

    manual_compaction.validate_and_rewrite_manual_compaction_request(request)

    assert request.enable_thinking is False
    assert request.reasoning_effort == "none"
    assert request.preserve_thinking is False
    assert request.thinking is None
    assert request.model_extra is not None
    assert not any("reason" in key.lower() or "think" in key.lower() for key in request.model_extra)
    assert "chat_template_kwargs" not in request.model_extra


@pytest.mark.parametrize(
    ("provenance", "requested_marker", "expected_marker", "expected_passthrough"),
    [
        ({"source": "local"}, False, True, False),
        ({"source": "external"}, True, None, True),
        (None, True, None, True),
    ],
    ids = ["stored-local", "stored-external", "stored-unknown"],
)
def test_prepare_and_inference_pin_expanded_tool_history(
    tmp_path, monkeypatch, provenance, requested_marker, expected_marker, expected_passthrough
):
    _seed_branch(tmp_path, monkeypatch)
    stored_assistant = studio_db.get_chat_message("thread-1", "a1")
    tool_part = {
        "type": "tool-call",
        "toolCallId": "call-1",
        "toolName": "terminal",
        "argsText": '{"cmd":"pwd"}',
        "args": {"cmd": "pwd"},
        "result": "/workspace",
    }
    if provenance is not None:
        tool_part["provenance"] = provenance
    stored_assistant["content"] = [
        tool_part,
        {"type": "text", "text": "Use a staged rollout."},
    ]
    studio_db.upsert_chat_message(stored_assistant, allow_generation_edit = True)
    messages = [
        {"role": "system", "content": "Project rules"},
        {"role": "user", "content": "Explain the migration."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": '{"cmd":"pwd"}'},
                }
            ],
        },
        {
            "role": "tool",
            "content": "/workspace",
            "tool_call_id": "call-1",
            "name": "terminal",
        },
        {"role": "assistant", "content": "Use a staged rollout."},
        {"role": "user", "content": "/compact"},
    ]
    prepared = _prepare(messages)
    request = _request(prepared, messages = messages)
    request.studio_tool_history = requested_marker

    manual_compaction.validate_and_rewrite_manual_compaction_request(request)

    assert prepared["requestMessageCount"] == 6
    assert request.messages[-1].content == manual_compaction.MANUAL_COMPACTION_HANDOFF_INSTRUCTION
    assert request.studio_tool_history is expected_marker
    assert inference._only_studio_tool_history(request) is (expected_marker is True)
    backend = SimpleNamespace(supports_tools = True, supports_tool_passthrough = True)
    assert inference._takes_tool_passthrough(request, backend) is expected_passthrough


def test_prepare_matches_frontend_search_image_and_assistant_markup_replay(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    image_id = "0123456789ab"
    stored_assistant = studio_db.get_chat_message("thread-1", "a1")
    stored_assistant["content"] = [
        {
            "type": "tool-call",
            "toolCallId": "call-search",
            "toolName": "web_search",
            "args": {"query": "migration diagram"},
            "result": {
                "text": f"found\n\n[[img:{image_id}]]",
                "webImages": [
                    {
                        "id": image_id,
                        "title": "Migration",
                        "domain": "example.com",
                        "source": "https://example.com/migration.png",
                    }
                ],
            },
        },
        {
            "type": "text",
            "text": (
                f"Answer [[img:{image_id}]] and `[[img:{image_id}]]` "
                "with data:audio/wav;base64,AAAA"
            ),
        },
    ]
    studio_db.upsert_chat_message(stored_assistant, allow_generation_edit = True)
    messages = [
        {"role": "system", "content": "Project rules"},
        {"role": "user", "content": "Explain the migration."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-search",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"query":"migration diagram"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "found",
            "tool_call_id": "call-search",
            "name": "web_search",
        },
        {
            "role": "assistant",
            "content": f"Answer  and `[[img:{image_id}]]` with [audio]",
        },
        {"role": "user", "content": "/compact"},
    ]

    prepared = _prepare(messages)
    request = _request(prepared, messages = messages)
    manual_compaction.validate_and_rewrite_manual_compaction_request(request)

    assert prepared["requestMessageCount"] == 6
    assert request.messages[-1].content == manual_compaction.MANUAL_COMPACTION_HANDOFF_INSTRUCTION


def test_prepare_omits_valid_stored_assistant_source_citations(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    assistant = studio_db.get_chat_message("thread-1", "a1")
    assistant["content"] = [
        {"type": "text", "text": "Use a staged rollout."},
        {
            "type": "source",
            "sourceType": "url",
            "id": "source-1",
            "url": "https://example.com/migration",
            "parentId": "source-parent",
            "metadata": {"description": "A source citation."},
        },
    ]
    studio_db.upsert_chat_message(assistant, allow_generation_edit = True)

    prepared = _prepare()
    manual_compaction.validate_and_rewrite_manual_compaction_request(_request(prepared))


@pytest.mark.parametrize(
    "source_part",
    [
        {"type": "source", "sourceType": "url", "url": "https://example.com", "title": "x"},
        {"type": "source", "sourceType": "file", "id": "s", "url": "#doc", "title": "x"},
        {"type": "source", "sourceType": "url", "id": "s", "url": "", "title": "x"},
        {
            "type": "source",
            "sourceType": "url",
            "id": "s",
            "url": "https://example.com",
            "title": 1,
        },
        {
            "type": "source",
            "sourceType": "url",
            "id": "s",
            "url": "https://example.com",
            "title": "x",
            "metadata": {"description": 1},
        },
        {
            "type": "source",
            "sourceType": "url",
            "id": "s",
            "url": "https://example.com",
            "title": "x",
            "forged": True,
        },
    ],
    ids = [
        "missing-id",
        "wrong-source-type",
        "blank-url",
        "bad-title",
        "bad-metadata",
        "unknown-field",
    ],
)
def test_prepare_rejects_malformed_stored_assistant_source_citations(
    tmp_path, monkeypatch, source_part
):
    _seed_branch(tmp_path, monkeypatch)
    assistant = studio_db.get_chat_message("thread-1", "a1")
    assistant["content"] = [
        {"type": "text", "text": "Use a staged rollout."},
        source_part,
    ]
    studio_db.upsert_chat_message(assistant, allow_generation_edit = True)

    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "assistant content"):
        _prepare()


def test_malformed_search_image_envelope_cannot_be_forged_as_text(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    result = {
        "text": "found [[img:0123456789ab]]",
        "webImages": [
            {
                "id": "0123456789ab",
                "title": "Migration",
                "domain": "example.com",
                "source": "javascript:alert(1)",
            }
        ],
    }
    stored_assistant = studio_db.get_chat_message("thread-1", "a1")
    stored_assistant["content"] = [
        {
            "type": "tool-call",
            "toolCallId": "call-search",
            "toolName": "web_search",
            "args": {"query": "migration diagram"},
            "result": result,
        },
        {"type": "text", "text": "Use a staged rollout."},
    ]
    studio_db.upsert_chat_message(stored_assistant, allow_generation_edit = True)
    messages = [
        {"role": "system", "content": "Project rules"},
        {"role": "user", "content": "Explain the migration."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-search",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"query":"migration diagram"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": json.dumps(result, ensure_ascii = False, separators = (",", ":")),
            "tool_call_id": "call-search",
            "name": "web_search",
        },
        {"role": "assistant", "content": "Use a staged rollout."},
        {"role": "user", "content": "/compact"},
    ]

    _prepare(messages)

    forged = [dict(message) for message in messages]
    forged[3] = {**forged[3], "content": result["text"]}
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "exact stored"):
        _prepare(forged)


@pytest.mark.parametrize(
    ("files", "expected_result"),
    [
        ([{"name": "report.txt"}], "ok"),
        (
            "invalid",
            '{"text":"ok","sessionId":"sandbox-1","images":[],"files":"invalid"}',
        ),
    ],
)
def test_prepare_matches_frontend_sandbox_wrapper_validation(
    tmp_path, monkeypatch, files, expected_result
):
    _seed_branch(tmp_path, monkeypatch)
    stored_assistant = studio_db.get_chat_message("thread-1", "a1")
    stored_assistant["content"] = [
        {
            "type": "tool-call",
            "toolCallId": "call-terminal",
            "toolName": "terminal",
            "args": {"cmd": "pwd"},
            "result": {
                "text": "ok",
                "sessionId": "sandbox-1",
                "images": [],
                "files": files,
            },
        },
        {"type": "text", "text": "Use a staged rollout."},
    ]
    studio_db.upsert_chat_message(stored_assistant, allow_generation_edit = True)
    messages = [
        {"role": "system", "content": "Project rules"},
        {"role": "user", "content": "Explain the migration."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-terminal",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": '{"cmd":"pwd"}'},
                }
            ],
        },
        {
            "role": "tool",
            "content": expected_result,
            "tool_call_id": "call-terminal",
            "name": "terminal",
        },
        {"role": "assistant", "content": "Use a staged rollout."},
        {"role": "user", "content": "/compact"},
    ]

    prepared = _prepare(messages)
    manual_compaction.validate_and_rewrite_manual_compaction_request(
        _request(prepared, messages = messages)
    )

    if files == "invalid":
        forged = [dict(message) for message in messages]
        forged[3] = {**forged[3], "content": "ok"}
        with pytest.raises(manual_compaction.ManualCompactionConflict, match = "exact stored"):
            _prepare(forged)


def test_prepare_binds_codex_reasoning_replay_metadata(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    assistant = studio_db.get_chat_message("thread-1", "a1")
    reasoning = [{"type": "reasoning", "id": "reasoning-1", "summary": []}]
    assistant["metadata"] = {"custom": {"openaiCodexReasoning": reasoning}}
    studio_db.upsert_chat_message(assistant, allow_generation_edit = True)
    messages = _full_wire_messages()
    messages[2] = {
        "role": "assistant",
        "content": "Use a staged rollout.",
        "extra_content": {"openai_codex_reasoning": reasoning},
    }
    prepared = _prepare(messages)

    manual_compaction.validate_and_rewrite_manual_compaction_request(
        _request(prepared, messages = messages)
    )

    assistant["metadata"] = {
        "custom": {"openaiCodexReasoning": [{"type": "reasoning", "id": "changed", "summary": []}]}
    }
    before = _raw_message_row("a1")
    with pytest.raises(studio_db.ChatMessageProtectedError, match = "checkpoint"):
        studio_db.upsert_chat_message(assistant, allow_generation_edit = True)
    assert _raw_message_row("a1") == before
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "already running"):
        manual_compaction.validate_and_rewrite_manual_compaction_request(
            _request(prepared, messages = messages)
        )


def test_prepare_requires_current_stored_project_instructions(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    studio_db.upsert_chat_project(
        {
            "id": "project-1",
            "name": "Project",
            "instructions": "Never skip the release gate.",
            "archived": False,
            "createdAt": 1,
            "updatedAt": 1,
        }
    )
    studio_db.update_chat_thread("thread-1", {"projectId": "project-1"})
    messages = _full_wire_messages()
    messages[0] = {
        "role": "system",
        "content": (
            "<project_instructions>\nNever skip the release gate.\n"
            "</project_instructions>\n\nProject rules"
        ),
    }
    prepared = _prepare(messages)
    manual_compaction.validate_and_rewrite_manual_compaction_request(
        _request(prepared, messages = messages)
    )

    changed = list(messages)
    changed[0] = {
        "role": "system",
        "content": "<project_instructions>\nSkip it.\n</project_instructions>",
    }
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "project instructions"):
        _prepare(ChatCompletionRequest(model = "model", messages = changed).messages)


def test_prepare_rejects_a_truncated_wire_branch(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "exact stored"):
        _prepare(
            [
                {"role": "assistant", "content": "Use a staged rollout."},
                {"role": "user", "content": "/compact"},
            ]
        )


def test_prepare_rejects_same_role_source_text_drift(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    changed = _full_wire_messages()
    changed[1] = {"role": "user", "content": "A different migration request."}

    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "exact stored"):
        _prepare(changed)


def test_prepare_binds_text_and_image_attachments_to_wire_content(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    user = studio_db.get_chat_message("thread-1", "u1")
    user["attachments"] = [
        {
            "content": [
                {"type": "text", "text": "attached context"},
                {"type": "image", "image": "aGVsbG8="},
            ]
        }
    ]
    studio_db.upsert_chat_message(user)
    messages = [
        {"role": "system", "content": "Project rules"},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Explain the migration.\nattached context",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,aGVsbG8="},
                },
            ],
        },
        {"role": "assistant", "content": "Use a staged rollout."},
        {"role": "user", "content": "/compact"},
    ]
    prepared = _prepare(messages)

    manual_compaction.validate_and_rewrite_manual_compaction_request(
        _request(prepared, messages = messages)
    )

    changed = list(messages)
    changed[1] = {
        "role": "user",
        "content": [{"type": "text", "text": "Explain the migration."}],
    }
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "exact stored"):
        _prepare(changed)


@pytest.mark.parametrize(
    "media_part",
    [
        {"type": "audio", "audio": {"data": "AAAA", "format": "wav"}},
        {
            "type": "file",
            "filename": "clip.mp4",
            "data": "AAAA",
            "mimeType": "video/mp4",
        },
    ],
    ids = ["audio", "video"],
)
def test_prepare_omits_valid_stored_user_media_attachments(tmp_path, monkeypatch, media_part):
    _seed_branch(tmp_path, monkeypatch)
    user = studio_db.get_chat_message("thread-1", "u1")
    user["attachments"] = [{"content": [media_part]}]
    studio_db.upsert_chat_message(user)

    prepared = _prepare()

    assert prepared["sourceHash"] == manual_compaction.canonical_source_hash(
        [
            studio_db.get_chat_message("thread-1", "u1"),
            studio_db.get_chat_message("thread-1", "a1"),
        ]
    )
    manual_compaction.validate_and_rewrite_manual_compaction_request(_request(prepared))


@pytest.mark.parametrize(
    "media_part",
    [
        {"type": "audio", "audio": {"data": "", "format": "wav"}},
        {"type": "audio", "audio": {"data": "AAAA"}},
        {"type": "file", "data": "AAAA", "mimeType": "text/plain"},
        {"type": "unknown", "data": "AAAA"},
    ],
    ids = ["blank-audio", "missing-audio-format", "non-video-file", "unknown"],
)
def test_prepare_rejects_malformed_or_unknown_stored_user_media(tmp_path, monkeypatch, media_part):
    _seed_branch(tmp_path, monkeypatch)
    user = studio_db.get_chat_message("thread-1", "u1")
    user["attachments"] = [{"content": [media_part]}]
    studio_db.upsert_chat_message(user)

    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "user content"):
        _prepare()


def test_inference_rejects_truncation_and_prepare_drift(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    truncated = _request(
        prepared,
        messages = [
            {"role": "assistant", "content": "Use a staged rollout."},
            {"role": "user", "content": "/compact"},
        ],
    )
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "complete untruncated"):
        manual_compaction.validate_and_rewrite_manual_compaction_request(truncated)

    drifted = _request(prepared)
    drifted.manual_compaction.source_hash = "0" * 64
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "drifted"):
        manual_compaction.validate_and_rewrite_manual_compaction_request(drifted)

    content_drifted = _request(prepared)
    content_drifted.messages[1].content = "Different source prompt."
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "request changed"):
        manual_compaction.validate_and_rewrite_manual_compaction_request(content_drifted)

    system_drifted = _request(prepared)
    system_drifted.messages[0].content = "Different project rules."
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "request changed"):
        manual_compaction.validate_and_rewrite_manual_compaction_request(system_drifted)

    changed_source = _message("a1", "u1", "assistant", "Changed answer.")
    before = _raw_message_row("a1")
    with pytest.raises(studio_db.ChatMessageProtectedError, match = "checkpoint"):
        studio_db.upsert_chat_message(changed_source)
    assert _raw_message_row("a1") == before
    _set_raw_message_json(
        "a1",
        "content_json",
        json.dumps(changed_source["content"], separators = (",", ":")),
    )
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "source changed"):
        manual_compaction.validate_and_rewrite_manual_compaction_request(_request(prepared))


@pytest.mark.asyncio
async def test_nonstream_response_records_server_observed_output_and_rejects_forgery(
    tmp_path, monkeypatch
):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    _claim(prepared)
    body = {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "server summary"},
                "finish_reason": "stop",
            }
        ]
    }
    response = inference.Response(
        content = json.dumps(body),
        media_type = "application/json",
    )
    await inference._observe_manual_compaction_response(
        response,
        attempt_id = prepared["attemptId"],
        cancel_event = threading.Event(),
    )
    stored = manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])
    assert stored["outputSummaryHash"] == manual_compaction.summary_hash("server summary")
    assert stored["outputFinishReason"] == "stop"

    studio_db.upsert_chat_message(_message("summary-1", "compact-1", "assistant", "forged summary"))
    with pytest.raises(
        manual_compaction.ManualCompactionConflict,
        match = "server-observed output",
    ):
        manual_compaction.commit_manual_compaction(
            "thread-1",
            attempt_id = prepared["attemptId"],
            command_message_id = prepared["commandMessageId"],
            summary_message_id = "summary-1",
            expected_head_message_id = "summary-1",
            expected_revision = prepared["revision"],
            expected_summary_hash = manual_compaction.summary_hash("forged summary"),
        )


def _manual_sse_chunk(*, content = None, finish_reason = None):
    delta = {} if content is None else {"content": content}
    return (
        "data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": finish_reason,
                    }
                ]
            },
            separators = (",", ":"),
        )
        + "\n\n"
    )


async def _observe_sse(
    prepared,
    chunks,
    *,
    cancelled = False,
):
    async def body():
        for chunk in chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    cancel_event = threading.Event()
    if cancelled:
        cancel_event.set()
    response = inference.StreamingResponse(body(), media_type = "text/event-stream")
    await inference._observe_manual_compaction_response(
        response,
        attempt_id = prepared["attemptId"],
        cancel_event = cancel_event,
    )
    iterator = response.body_iterator
    chunks = [chunk async for chunk in iterator]
    await iterator.aclose()
    return chunks


@pytest.mark.asyncio
async def test_stream_records_only_a_clean_terminal_server_event(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    _claim(prepared)
    first = _manual_sse_chunk(content = "stream ")
    second = _manual_sse_chunk(content = "summary")
    terminal = _manual_sse_chunk(finish_reason = "stop")

    chunks = await _observe_sse(
        prepared,
        [first[:9], first[9:] + second, terminal + "data: [DONE]\n\n"],
    )

    assert "".join(chunks) == first + second + terminal + "data: [DONE]\n\n"
    stored = manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])
    assert stored["outputSummaryHash"] == manual_compaction.summary_hash("stream summary")
    assert stored["outputFinishReason"] == "stop"


def _buffered_stream_observer(
    prepared,
    chunks,
    cancel_event = None,
):
    async def body():
        for chunk in chunks:
            yield chunk

    return inference._observe_manual_compaction_stream(
        body(),
        attempt_id = prepared["attemptId"],
        cancel_event = cancel_event or threading.Event(),
    )


@pytest.mark.asyncio
async def test_downstream_close_after_first_released_chunk_cancels_recorded_output(
    tmp_path, monkeypatch
):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    _claim(prepared)
    chunks = [
        _manual_sse_chunk(content = "stream "),
        _manual_sse_chunk(content = "summary"),
        _manual_sse_chunk(finish_reason = "stop"),
        "data: [DONE]\n\n",
    ]
    cancel_event = threading.Event()
    observed = _buffered_stream_observer(prepared, chunks, cancel_event)

    assert await observed.__anext__() == chunks[0]
    recorded = manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])
    assert recorded["state"] == "running"
    assert recorded["outputSummaryHash"] is None
    assert observed._validated is True

    await observed.aclose()
    cancelled = manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])
    assert cancel_event.is_set()
    assert cancelled["state"] == "cancelled"
    assert cancelled["terminalReason"] == "inference_cancelled"
    await observed.aclose()
    assert (
        manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])["cancelledAt"]
        == cancelled["cancelledAt"]
    )

    studio_db.upsert_chat_message(_message("summary-1", "compact-1", "assistant", "stream summary"))
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "cancelled"):
        manual_compaction.commit_manual_compaction(
            "thread-1",
            attempt_id = prepared["attemptId"],
            command_message_id = prepared["commandMessageId"],
            summary_message_id = "summary-1",
            expected_head_message_id = "summary-1",
            expected_revision = prepared["revision"],
            expected_summary_hash = manual_compaction.summary_hash("stream summary"),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "delivered_chunks",
    [1, 2, 3, 4],
    ids = ["first-content", "last-content", "finish-choice", "done-marker"],
)
async def test_downstream_close_cancels_at_every_buffered_chunk_boundary(
    tmp_path, monkeypatch, delivered_chunks
):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    _claim(prepared)
    chunks = [
        _manual_sse_chunk(content = "stream "),
        _manual_sse_chunk(content = "summary"),
        _manual_sse_chunk(finish_reason = "stop"),
        "data: [DONE]\n\n",
    ]
    cancel_event = threading.Event()
    observed = _buffered_stream_observer(prepared, chunks, cancel_event)

    delivered = [await observed.__anext__() for _ in range(delivered_chunks)]
    assert delivered == chunks[:delivered_chunks]
    await observed.aclose()

    stored = manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])
    assert cancel_event.is_set()
    assert stored["state"] == "cancelled"
    assert stored["terminalReason"] == "inference_cancelled"


@pytest.mark.parametrize(
    ("chunks", "cancelled", "error"),
    [
        (
            [_manual_sse_chunk(content = "partial"), "data: [DONE]\n\n"],
            False,
            "inference_failed",
        ),
        (
            [
                _manual_sse_chunk(content = "partial"),
                _manual_sse_chunk(finish_reason = "stop"),
                "data: [DONE]\n\n",
            ],
            True,
            "cancelled",
        ),
        (
            [
                _manual_sse_chunk(content = "partial"),
                _manual_sse_chunk(finish_reason = "length"),
                "data: [DONE]\n\n",
            ],
            False,
            "inference_failed",
        ),
    ],
    ids = ["missing-terminal", "cancelled", "length"],
)
@pytest.mark.asyncio
async def test_stream_provenance_blocks_incomplete_outputs(
    tmp_path, monkeypatch, chunks, cancelled, error
):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    _claim(prepared)
    await _observe_sse(prepared, chunks, cancelled = cancelled)
    studio_db.upsert_chat_message(_message("summary-1", "compact-1", "assistant", "partial"))

    with pytest.raises(manual_compaction.ManualCompactionConflict, match = error):
        manual_compaction.commit_manual_compaction(
            "thread-1",
            attempt_id = prepared["attemptId"],
            command_message_id = prepared["commandMessageId"],
            summary_message_id = "summary-1",
            expected_head_message_id = "summary-1",
            expected_revision = prepared["revision"],
            expected_summary_hash = manual_compaction.summary_hash("partial"),
        )


@pytest.mark.asyncio
async def test_interrupted_stream_does_not_finalize_output(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    _claim(prepared)
    chunks = await _observe_sse(
        prepared,
        [_manual_sse_chunk(content = "partial"), RuntimeError("interrupted")],
    )
    assert len(chunks) == 1
    assert "manual_compaction_failed" in chunks[0]
    assert "interrupted" not in chunks[0]
    stored = manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])
    assert stored["outputSummaryHash"] is None
    assert stored["outputFinishReason"] is None


@pytest.mark.asyncio
async def test_stream_observer_preserves_disconnect_cancellation(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    _claim(prepared)
    cancelled = False

    async def body():
        nonlocal cancelled
        try:
            yield _manual_sse_chunk(content = "partial")
            raise asyncio.CancelledError()
        except asyncio.CancelledError:
            cancelled = True
            raise

    observed = inference._observe_manual_compaction_stream(
        body(),
        attempt_id = prepared["attemptId"],
        cancel_event = threading.Event(),
    )
    with pytest.raises(asyncio.CancelledError):
        await observed.__anext__()

    assert cancelled is True
    stored = manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])
    assert stored["outputSummaryHash"] is None
    assert stored["outputFinishReason"] is None


def test_output_provenance_exact_retry_is_idempotent_and_mismatch_is_rejected(
    tmp_path, monkeypatch
):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    _claim(prepared)
    first = _record_output(prepared, "summary")
    retried = _record_output(prepared, "summary")
    assert retried["outputRecordedAt"] == first["outputRecordedAt"]
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "differently"):
        _record_output(prepared, "other summary")


def test_commit_activates_summary_keeps_originals_and_is_idempotent(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    _claim(prepared)
    summary_text = "Goal: migrate safely. Decision: use a staged rollout."
    _record_output(prepared, summary_text)
    studio_db.upsert_chat_message(_message("summary-1", "compact-1", "assistant", summary_text))
    digest = manual_compaction.summary_hash(summary_text)

    committed = manual_compaction.commit_manual_compaction(
        "thread-1",
        attempt_id = "attempt-1",
        command_message_id = "compact-1",
        summary_message_id = "summary-1",
        expected_head_message_id = "summary-1",
        expected_revision = 1,
        expected_summary_hash = digest,
    )
    assert committed["state"] == "active"
    assert committed["summaryHash"] == digest
    assert {message["id"] for message in studio_db.list_chat_messages("thread-1")} == {
        "u1",
        "a1",
        "compact-1",
        "summary-1",
    }
    metadata = studio_db.get_chat_message("thread-1", "summary-1")["metadata"]
    assert metadata["manualCompaction"] == {
        "schemaVersion": 1,
        "state": "active",
        "attemptId": "attempt-1",
        "threadId": "thread-1",
        "revision": 1,
        "commandMessageId": "compact-1",
        "sourceHeadMessageId": "a1",
        "summaryMessageId": "summary-1",
        "sourceHash": prepared["sourceHash"],
        "requestHash": prepared["requestHash"],
        "requestMessageCount": prepared["requestMessageCount"],
        "projectInstructionDigest": prepared["projectInstructionDigest"],
        "projectInstructionRevision": prepared["projectInstructionRevision"],
        "contextDigest": prepared["contextDigest"],
        "archivePayloadHash": prepared["archivePayloadHash"],
        "outputSummaryHash": digest,
        "outputFinishReason": "stop",
        "summaryHash": digest,
    }
    retried = manual_compaction.commit_manual_compaction(
        "thread-1",
        attempt_id = "attempt-1",
        command_message_id = "compact-1",
        summary_message_id = "summary-1",
        expected_head_message_id = "summary-1",
        expected_revision = 1,
        expected_summary_hash = digest,
    )
    assert retried["state"] == "active"
    assert retried["summaryMessageId"] == "summary-1"


def test_commit_rejects_stale_revision_wrong_hash_and_non_child(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    _claim(prepared)
    _record_output(prepared, "summary")
    studio_db.upsert_chat_message(_message("summary-1", "compact-1", "assistant", "summary"))
    digest = manual_compaction.summary_hash("summary")
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "stale"):
        manual_compaction.commit_manual_compaction(
            "thread-1",
            attempt_id = "attempt-1",
            command_message_id = "compact-1",
            summary_message_id = "summary-1",
            expected_head_message_id = "summary-1",
            expected_revision = 2,
            expected_summary_hash = digest,
        )
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "hash"):
        manual_compaction.commit_manual_compaction(
            "thread-1",
            attempt_id = "attempt-1",
            command_message_id = "compact-1",
            summary_message_id = "summary-1",
            expected_head_message_id = "summary-1",
            expected_revision = 1,
            expected_summary_hash = "0" * 64,
        )
    studio_db.upsert_chat_message(_message("summary-1", "a1", "assistant", "summary"))
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "exact stored ancestry"):
        manual_compaction.commit_manual_compaction(
            "thread-1",
            attempt_id = "attempt-1",
            command_message_id = "compact-1",
            summary_message_id = "summary-1",
            expected_head_message_id = "summary-1",
            expected_revision = 1,
            expected_summary_hash = digest,
        )


def test_commit_rejects_a_summary_that_is_no_longer_the_leaf(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    _claim(prepared)
    _record_output(prepared, "summary")
    studio_db.upsert_chat_message(_message("summary-1", "compact-1", "assistant", "summary"))
    studio_db.upsert_chat_message(_message("later", "summary-1", "user", "continue"))

    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "no longer"):
        manual_compaction.commit_manual_compaction(
            "thread-1",
            attempt_id = "attempt-1",
            command_message_id = "compact-1",
            summary_message_id = "summary-1",
            expected_head_message_id = "summary-1",
            expected_revision = 1,
            expected_summary_hash = manual_compaction.summary_hash("summary"),
        )


def test_prepare_and_commit_route_models_drive_archive_after_activation(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    bound = chat_history.bind_thread_manual_compaction(
        "thread-1",
        chat_history.ManualCompactionBindRequest(
            attemptId = "route-attempt",
            commandMessageId = "compact-1",
            summaryMessageId = "summary-route",
            attemptSequence = 1,
        ),
        current_subject = "test-user",
    )
    assert bound.metadata[manual_compaction.MANUAL_COMPACTION_CLIENT_KEY]["state"] == "bound"
    prepared = chat_history.prepare_thread_manual_compaction(
        "thread-1",
        chat_history.ManualCompactionPrepareRequest(
            attemptId = "route-attempt",
            commandMessageId = "compact-1",
            expectedHeadMessageId = "compact-1",
            messageIds = ["u1", "a1", "compact-1"],
            messages = _full_wire_messages(),
        ),
        current_subject = "test-user",
    )
    _claim(prepared.model_dump())
    _record_output(prepared.model_dump(), "route summary")
    studio_db.upsert_chat_message(
        _message("summary-route", "compact-1", "assistant", "route summary")
    )
    archived = []

    def mark_archived(result):
        archived.append(result["attemptId"])
        conn = studio_db.get_connection()
        try:
            conn.execute(
                "UPDATE manual_compactions SET archive_status = 'archived' WHERE attempt_id = ?",
                (result["attemptId"],),
            )
            conn.commit()
        finally:
            conn.close()
        return "archived"

    monkeypatch.setattr(
        chat_history,
        "archive_manual_compaction_best_effort",
        mark_archived,
    )

    committed = chat_history.commit_thread_manual_compaction(
        "thread-1",
        chat_history.ManualCompactionCommitRequest(
            attemptId = prepared.attemptId,
            commandMessageId = prepared.commandMessageId,
            summaryMessageId = "summary-route",
            expectedHeadMessageId = "summary-route",
            expectedRevision = prepared.revision,
            summaryHash = manual_compaction.summary_hash("route summary"),
        ),
        current_subject = "test-user",
    )

    assert committed.state == "active"
    assert committed.archiveStatus == "archived"
    assert archived == ["route-attempt"]
    retried = chat_history.commit_thread_manual_compaction(
        "thread-1",
        chat_history.ManualCompactionCommitRequest(
            attemptId = prepared.attemptId,
            commandMessageId = prepared.commandMessageId,
            summaryMessageId = "summary-route",
            expectedHeadMessageId = "summary-route",
            expectedRevision = prepared.revision,
            summaryHash = manual_compaction.summary_hash("route summary"),
        ),
        current_subject = "test-user",
    )
    assert retried.archiveStatus == "archived"
    assert archived == ["route-attempt"]


def test_pending_archive_recovers_after_activation_crash_and_stays_idempotent(
    tmp_path, monkeypatch
):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare(summary_message_id = "summary-crash")
    _claim(prepared)
    _record_output(prepared, "crash-safe summary")
    studio_db.upsert_chat_message(
        _message("summary-crash", "compact-1", "assistant", "crash-safe summary")
    )
    digest = manual_compaction.summary_hash("crash-safe summary")

    activated = manual_compaction.commit_manual_compaction(
        "thread-1",
        attempt_id = prepared["attemptId"],
        command_message_id = prepared["commandMessageId"],
        summary_message_id = "summary-crash",
        expected_head_message_id = "summary-crash",
        expected_revision = prepared["revision"],
        expected_summary_hash = digest,
    )
    assert activated["archiveStatus"] == "pending"
    assert activated["archivePayload"]

    archived = []

    def mark_archived(result):
        archived.append(result["attemptId"])
        conn = studio_db.get_connection()
        try:
            conn.execute(
                "UPDATE manual_compactions SET archive_status = 'archived' WHERE attempt_id = ?",
                (result["attemptId"],),
            )
            conn.commit()
        finally:
            conn.close()
        return "archived"

    monkeypatch.setattr(
        chat_history,
        "archive_manual_compaction_best_effort",
        mark_archived,
    )
    payload = chat_history.ManualCompactionCommitRequest(
        attemptId = prepared["attemptId"],
        commandMessageId = prepared["commandMessageId"],
        summaryMessageId = "summary-crash",
        expectedHeadMessageId = "summary-crash",
        expectedRevision = prepared["revision"],
        summaryHash = digest,
    )

    recovered = chat_history.commit_thread_manual_compaction(
        "thread-1", payload, current_subject = "test-user"
    )
    assert recovered.archiveStatus == "archived"
    assert archived == [prepared["attemptId"]]

    retried = chat_history.commit_thread_manual_compaction(
        "thread-1", payload, current_subject = "test-user"
    )
    assert retried.archiveStatus == "archived"
    assert archived == [prepared["attemptId"]]


@pytest.mark.parametrize(
    ("written", "degraded", "verified", "expected"),
    [
        (1, False, False, "archived"),
        (0, False, True, "archived"),
        (0, False, False, "failed"),
        (0, True, True, "failed"),
    ],
)
def test_archive_status_requires_a_verified_non_degraded_write(
    tmp_path, monkeypatch, written, degraded, verified, expected
):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    _claim(prepared)
    _record_output(prepared, "summary")
    studio_db.upsert_chat_message(_message("summary-1", "compact-1", "assistant", "summary"))
    result = manual_compaction.commit_manual_compaction(
        "thread-1",
        attempt_id = "attempt-1",
        command_message_id = "compact-1",
        summary_message_id = "summary-1",
        expected_head_message_id = "summary-1",
        expected_revision = 1,
        expected_summary_hash = manual_compaction.summary_hash("summary"),
    )
    monkeypatch.setattr(conversation_archive, "enabled", lambda: True)
    monkeypatch.setattr(conversation_archive, "can_archive", lambda _thread_id: True)
    monkeypatch.setattr(
        conversation_archive,
        "archive_turns",
        lambda *_args, **_kwargs: written,
    )
    monkeypatch.setattr(conversation_archive, "degraded", lambda: degraded)
    monkeypatch.setattr(
        conversation_archive,
        "turns_archived",
        lambda *_args, **_kwargs: verified,
    )

    assert manual_compaction.archive_manual_compaction_best_effort(result) == expected
    retried = manual_compaction.commit_manual_compaction(
        "thread-1",
        attempt_id = "attempt-1",
        command_message_id = "compact-1",
        summary_message_id = "summary-1",
        expected_head_message_id = "summary-1",
        expected_revision = 1,
        expected_summary_hash = manual_compaction.summary_hash("summary"),
    )
    assert retried["archiveStatus"] == expected
    assert retried["archivePayload"] == result["archivePayload"]


@pytest.mark.parametrize("failure_point", ["open", "update", "commit"])
def test_archive_db_failures_remain_retryable_and_never_escape(
    tmp_path, monkeypatch, failure_point
):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    _claim(prepared)
    _record_output(prepared, "summary")
    studio_db.upsert_chat_message(_message("summary-1", "compact-1", "assistant", "summary"))
    result = manual_compaction.commit_manual_compaction(
        "thread-1",
        attempt_id = "attempt-1",
        command_message_id = "compact-1",
        summary_message_id = "summary-1",
        expected_head_message_id = "summary-1",
        expected_revision = 1,
        expected_summary_hash = manual_compaction.summary_hash("summary"),
    )
    monkeypatch.setattr(conversation_archive, "enabled", lambda: True)
    monkeypatch.setattr(conversation_archive, "can_archive", lambda _thread_id: True)
    monkeypatch.setattr(conversation_archive, "archive_turns", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(conversation_archive, "degraded", lambda: False)
    original_get_connection = studio_db.get_connection

    class FailingConnection:
        def __init__(self, connection):
            self.connection = connection

        def execute(self, statement, *args, **kwargs):
            if failure_point == "update" and statement.startswith("UPDATE manual_compactions"):
                raise OSError("update failed")
            return self.connection.execute(statement, *args, **kwargs)

        def commit(self):
            if failure_point == "commit":
                raise OSError("commit failed")
            return self.connection.commit()

        def __getattr__(self, name):
            return getattr(self.connection, name)

    class FailingDb:
        @staticmethod
        def get_connection():
            if failure_point == "open":
                raise OSError("open failed")
            return FailingConnection(original_get_connection())

    monkeypatch.setattr(manual_compaction, "_db", lambda: FailingDb)
    assert manual_compaction.archive_manual_compaction_best_effort(result) == "failed"


def test_fork_archive_reopen_failure_does_not_fail_the_committed_fork(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    _claim(prepared)
    _record_output(prepared, "summary")
    studio_db.upsert_chat_message(_message("summary-1", "compact-1", "assistant", "summary"))
    manual_compaction.commit_manual_compaction(
        "thread-1",
        attempt_id = prepared["attemptId"],
        command_message_id = prepared["commandMessageId"],
        summary_message_id = "summary-1",
        expected_head_message_id = "summary-1",
        expected_revision = prepared["revision"],
        expected_summary_hash = manual_compaction.summary_hash("summary"),
    )
    original_get_attempt = chat_history.get_manual_compaction_attempt
    calls = 0

    def fail_once(attempt_id):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("reopen failed")
        return original_get_attempt(attempt_id)

    monkeypatch.setattr(chat_history, "get_manual_compaction_attempt", fail_once)
    response = chat_history.fork_thread(
        "thread-1",
        chat_history.ChatForkRequest(
            messageId = "summary-1",
            newThreadId = "fork-route-db-failure",
            createdAt = 20,
        ),
        current_subject = "test-user",
    )

    assert response.thread.id == "fork-route-db-failure"
    assert studio_db.get_chat_thread("fork-route-db-failure") is not None


@pytest.mark.parametrize("terminal_status", ["archived", "skipped"])
def test_archive_retry_cannot_regress_a_terminal_status(tmp_path, monkeypatch, terminal_status):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    _claim(prepared)
    _record_output(prepared, "summary")
    studio_db.upsert_chat_message(_message("summary-1", "compact-1", "assistant", "summary"))
    result = manual_compaction.commit_manual_compaction(
        "thread-1",
        attempt_id = "attempt-1",
        command_message_id = "compact-1",
        summary_message_id = "summary-1",
        expected_head_message_id = "summary-1",
        expected_revision = 1,
        expected_summary_hash = manual_compaction.summary_hash("summary"),
    )
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "UPDATE manual_compactions SET archive_status = ? WHERE attempt_id = ?",
            (terminal_status, result["attemptId"]),
        )
        conn.commit()
    finally:
        conn.close()
    monkeypatch.setattr(conversation_archive, "enabled", lambda: True)
    monkeypatch.setattr(conversation_archive, "can_archive", lambda _thread_id: True)
    monkeypatch.setattr(
        conversation_archive,
        "archive_turns",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("late retry failed")),
    )

    assert manual_compaction.archive_manual_compaction_best_effort(result) == terminal_status
    assert (
        manual_compaction.get_manual_compaction_attempt("attempt-1")["archiveStatus"]
        == terminal_status
    )


def test_next_prepare_uses_branch_local_active_revision(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    _claim(prepared)
    text = "summary"
    _record_output(prepared, text)
    studio_db.upsert_chat_message(_message("summary-1", "compact-1", "assistant", text))
    manual_compaction.commit_manual_compaction(
        "thread-1",
        attempt_id = "attempt-1",
        command_message_id = "compact-1",
        summary_message_id = "summary-1",
        expected_head_message_id = "summary-1",
        expected_revision = 1,
        expected_summary_hash = manual_compaction.summary_hash(text),
    )
    studio_db.upsert_chat_message(_message("compact-2", "summary-1", "user", "/compact"))
    next_attempt = _prepare_bound(
        "thread-1",
        attempt_id = "attempt-2",
        command_message_id = "compact-2",
        expected_head_message_id = "compact-2",
        message_ids = ["u1", "a1", "compact-1", "summary-1", "compact-2"],
        request_messages = [
            {"role": "system", "content": "Project rules"},
            {"role": "assistant", "content": text},
            {"role": "user", "content": "/compact"},
        ],
    )
    assert next_attempt["revision"] == 2
    assert next_attempt["sourceMessageIds"] == ["u1", "a1", "compact-1", "summary-1"]
    assert next_attempt["effectiveSourceMessageIds"] == ["summary-1"]
    assert next_attempt["requestMessageCount"] == 3
    reduced_messages = [
        {"role": "system", "content": "Project rules"},
        {"role": "assistant", "content": text},
        {"role": "user", "content": "/compact"},
    ]
    request = _request(next_attempt, messages = reduced_messages)
    manual_compaction.validate_and_rewrite_manual_compaction_request(request)
    assert len(request.messages) == 3
    assert request.messages[1].content == text
    assert all("Explain the migration" not in str(message.content) for message in request.messages)


def test_summary_rejects_hidden_payloads_then_normalizes_plain_text(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    _claim(prepared)
    _record_output(prepared, "summary")
    invalid_messages = [
        {
            **_message("summary-1", "compact-1", "assistant", "summary"),
            "content": [{"type": "reasoning", "text": "hidden"}],
        },
        {
            **_message("summary-1", "compact-1", "assistant", "summary"),
            "content": [
                {
                    "type": "tool-call",
                    "toolName": "terminal",
                    "toolCallId": "call-1",
                    "args": {},
                }
            ],
        },
        {
            **_message("summary-1", "compact-1", "assistant", "summary"),
            "content": [{"type": "image", "image": "data:image/png;base64,AA=="}],
        },
        {
            **_message("summary-1", "compact-1", "assistant", "summary"),
            "attachments": [{"id": "attachment-1", "content": []}],
        },
        _message(
            "summary-1",
            "compact-1",
            "assistant",
            "summary",
            metadata = {"custom": {"openaiCodexReasoning": []}},
        ),
        {
            **_message("summary-1", "compact-1", "assistant", "summary"),
            "content": [{"type": "text", "text": "summary", "provider": "hidden"}],
        },
    ]
    for invalid in invalid_messages:
        studio_db.upsert_chat_message(invalid, allow_generation_edit = True)
        with pytest.raises(manual_compaction.ManualCompactionConflict, match = "summary"):
            manual_compaction.commit_manual_compaction(
                "thread-1",
                attempt_id = "attempt-1",
                command_message_id = "compact-1",
                summary_message_id = "summary-1",
                expected_head_message_id = "summary-1",
                expected_revision = 1,
                expected_summary_hash = manual_compaction.summary_hash("summary"),
            )

    plain = _message("summary-1", "compact-1", "assistant", "summary")
    plain["content"] = "summary"
    studio_db.upsert_chat_message(plain, allow_generation_edit = True)
    committed = manual_compaction.commit_manual_compaction(
        "thread-1",
        attempt_id = "attempt-1",
        command_message_id = "compact-1",
        summary_message_id = "summary-1",
        expected_head_message_id = "summary-1",
        expected_revision = 1,
        expected_summary_hash = manual_compaction.summary_hash("summary"),
    )
    assert committed["state"] == "active"
    assert studio_db.get_chat_message("thread-1", "summary-1")["content"] == [
        {"type": "text", "text": "summary"}
    ]


def test_active_checkpoint_rejects_edit_delete_and_prune(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    _claim(prepared)
    _record_output(prepared, "summary")
    studio_db.upsert_chat_message(_message("summary-1", "compact-1", "assistant", "summary"))
    manual_compaction.commit_manual_compaction(
        "thread-1",
        attempt_id = "attempt-1",
        command_message_id = "compact-1",
        summary_message_id = "summary-1",
        expected_head_message_id = "summary-1",
        expected_revision = 1,
        expected_summary_hash = manual_compaction.summary_hash("summary"),
    )

    for message_id in ("u1", "compact-1", "summary-1"):
        changed = studio_db.get_chat_message("thread-1", message_id)
        changed["content"] = [{"type": "text", "text": "changed"}]
        with pytest.raises(studio_db.ChatMessageProtectedError, match = "checkpoint"):
            studio_db.upsert_chat_message(changed, allow_generation_edit = True)

        remaining = [
            message
            for message in studio_db.list_chat_messages("thread-1")
            if message["id"] != message_id
        ]
        with pytest.raises(studio_db.ChatMessageProtectedError, match = "pruned"):
            studio_db.sync_chat_messages("thread-1", remaining, prune_missing = True)

    attempt = manual_compaction.get_manual_compaction_attempt("attempt-1")
    assert attempt["state"] == "active"
    assert attempt["revision"] == 1


def test_project_instruction_context_drift_blocks_inference_and_commit(tmp_path, monkeypatch):
    _reset_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_project(
        {
            "id": "project-1",
            "name": "Project",
            "instructions": "Keep migrations reversible.",
            "archived": False,
            "createdAt": 1,
            "updatedAt": 7,
        }
    )
    thread = _thread()
    thread["projectId"] = "project-1"
    studio_db.upsert_chat_thread(thread)
    for row in [
        _message("u1", None, "user", "Explain the migration."),
        _message("a1", "u1", "assistant", "Use a staged rollout."),
        _message("compact-1", "a1", "user", "/compact"),
    ]:
        studio_db.upsert_chat_message(row)
    messages = [
        {
            "role": "system",
            "content": (
                "<project_instructions>\nKeep migrations reversible.\n"
                "</project_instructions>\nOther guidance"
            ),
        },
        *_wire_messages(),
    ]
    prepared = _prepare(messages)
    assert prepared["projectInstructionRevision"] == 7
    assert len(prepared["projectInstructionDigest"]) == 64
    assert len(prepared["contextDigest"]) == 64

    studio_db.update_chat_project(
        "project-1",
        {"instructions": "Use destructive migrations.", "updatedAt": 8},
    )
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "instructions changed"):
        manual_compaction.validate_and_rewrite_manual_compaction_request(
            _request(prepared, messages = messages)
        )

    studio_db.update_chat_project(
        "project-1",
        {"instructions": "Keep migrations reversible.", "updatedAt": 9},
    )
    refreshed = _prepare_bound(
        "thread-1",
        attempt_id = "attempt-refreshed",
        command_message_id = "compact-1",
        expected_head_message_id = "compact-1",
        message_ids = ["u1", "a1", "compact-1"],
        request_messages = messages,
    )
    invalidated = manual_compaction.get_manual_compaction_attempt("attempt-1")
    assert invalidated["state"] == "failed"
    assert invalidated["terminalReason"] == "prepare_invalidated"
    _claim(refreshed, messages = messages)
    studio_db.update_chat_project(
        "project-1",
        {"instructions": "Use destructive migrations.", "updatedAt": 10},
    )
    studio_db.upsert_chat_message(_message("summary-1", "compact-1", "assistant", "summary"))
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "instructions changed"):
        manual_compaction.commit_manual_compaction(
            "thread-1",
            attempt_id = "attempt-refreshed",
            command_message_id = "compact-1",
            summary_message_id = "summary-1",
            expected_head_message_id = "summary-1",
            expected_revision = 1,
            expected_summary_hash = manual_compaction.summary_hash("summary"),
        )


@pytest.mark.parametrize("archive_status", ["pending", "failed"])
def test_active_identical_retry_recovers_archive_after_project_instructions_change(
    tmp_path, monkeypatch, archive_status
):
    _reset_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_project(
        {
            "id": "project-1",
            "name": "Project",
            "instructions": "Keep migrations reversible.",
            "archived": False,
            "createdAt": 1,
            "updatedAt": 7,
        }
    )
    thread = _thread()
    thread["projectId"] = "project-1"
    studio_db.upsert_chat_thread(thread)
    for row in [
        _message("u1", None, "user", "Explain the migration."),
        _message("a1", "u1", "assistant", "Use a staged rollout."),
        _message("compact-1", "a1", "user", "/compact"),
    ]:
        studio_db.upsert_chat_message(row)
    messages = [
        {
            "role": "system",
            "content": (
                "<project_instructions>\nKeep migrations reversible.\n"
                "</project_instructions>\nOther guidance"
            ),
        },
        *_wire_messages(),
    ]
    prepared = _prepare(messages)
    _claim(prepared, messages = messages)
    _record_output(prepared, "summary")
    studio_db.upsert_chat_message(_message("summary-1", "compact-1", "assistant", "summary"))
    manual_compaction.commit_manual_compaction(
        "thread-1",
        attempt_id = prepared["attemptId"],
        command_message_id = prepared["commandMessageId"],
        summary_message_id = "summary-1",
        expected_head_message_id = "summary-1",
        expected_revision = prepared["revision"],
        expected_summary_hash = manual_compaction.summary_hash("summary"),
    )
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "UPDATE manual_compactions SET archive_status = ? WHERE attempt_id = ?",
            (archive_status, prepared["attemptId"]),
        )
        conn.commit()
    finally:
        conn.close()
    studio_db.update_chat_project(
        "project-1",
        {"instructions": "Use the revised policy.", "updatedAt": 8},
    )
    archived = []

    def mark_archived(result):
        archived.append(result["archivePayload"])
        conn = studio_db.get_connection()
        try:
            conn.execute(
                "UPDATE manual_compactions SET archive_status = 'archived' WHERE attempt_id = ?",
                (result["attemptId"],),
            )
            conn.commit()
        finally:
            conn.close()
        return "archived"

    monkeypatch.setattr(
        chat_history,
        "archive_manual_compaction_best_effort",
        mark_archived,
    )
    payload = chat_history.ManualCompactionCommitRequest(
        attemptId = prepared["attemptId"],
        commandMessageId = prepared["commandMessageId"],
        summaryMessageId = "summary-1",
        expectedHeadMessageId = "summary-1",
        expectedRevision = prepared["revision"],
        summaryHash = manual_compaction.summary_hash("summary"),
    )

    recovered = chat_history.commit_thread_manual_compaction(
        "thread-1", payload, current_subject = "test-user"
    )
    assert recovered.archiveStatus == "archived"
    assert archived == [prepared["archivePayload"]]

    retried = chat_history.commit_thread_manual_compaction(
        "thread-1", payload, current_subject = "test-user"
    )
    assert retried.archiveStatus == "archived"
    assert archived == [prepared["archivePayload"]]

    mismatched = payload.model_copy(update = {"summaryHash": "0" * 64})
    with pytest.raises(HTTPException, match = "already committed differently"):
        chat_history.commit_thread_manual_compaction(
            "thread-1", mismatched, current_subject = "test-user"
        )


def test_missing_tool_call_ids_normalize_deterministically_across_prepare_and_inference(
    tmp_path, monkeypatch
):
    _seed_branch(tmp_path, monkeypatch)
    stored = studio_db.get_chat_message("thread-1", "a1")
    stored["content"] = [
        {
            "type": "tool-call",
            "toolName": "terminal",
            "argsText": '{"cmd":"pwd"}',
            "args": {"cmd": "pwd"},
            "result": "/workspace",
        },
        {"type": "text", "text": "Use a staged rollout."},
    ]
    studio_db.upsert_chat_message(stored, allow_generation_edit = True)

    def messages():
        return [
            {"role": "system", "content": "Project rules"},
            {"role": "user", "content": "Explain the migration."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "terminal", "arguments": '{"cmd":"pwd"}'},
                    }
                ],
            },
            {"role": "tool", "content": "/workspace", "name": "terminal"},
            {"role": "assistant", "content": "Use a staged rollout."},
            {"role": "user", "content": "/compact"},
        ]

    first_hash = manual_compaction.canonical_request_hash(messages())
    assert first_hash == manual_compaction.canonical_request_hash(messages())
    prepared = _prepare(messages())
    request = _request(prepared, messages = messages())
    call_id = request.messages[2].tool_calls[0]["id"]
    assert call_id.startswith("call_")
    assert request.messages[3].tool_call_id == call_id
    manual_compaction.validate_and_rewrite_manual_compaction_request(request)


def test_identical_missing_id_calls_in_two_rounds_get_distinct_stable_ids(tmp_path, monkeypatch):
    _reset_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())
    tool_content = [
        {
            "type": "tool-call",
            "toolName": "terminal",
            "argsText": '{"cmd":"pwd"}',
            "args": {"cmd": "pwd"},
            "result": "/workspace",
        }
    ]
    rows = [
        _message("u1", None, "user", "First round"),
        {**_message("a1", "u1", "assistant", ""), "content": tool_content},
        _message("u2", "a1", "user", "Second round"),
        {**_message("a2", "u2", "assistant", ""), "content": tool_content},
        _message("compact-1", "a2", "user", "/compact"),
    ]
    for row in rows:
        studio_db.upsert_chat_message(row)

    def messages():
        call = {
            "type": "function",
            "function": {"name": "terminal", "arguments": '{"cmd":"pwd"}'},
        }
        return [
            {"role": "system", "content": "Project rules"},
            {"role": "user", "content": "First round"},
            {"role": "assistant", "content": None, "tool_calls": [dict(call)]},
            {"role": "tool", "content": "/workspace", "name": "terminal"},
            {"role": "user", "content": "Second round"},
            {"role": "assistant", "content": None, "tool_calls": [dict(call)]},
            {"role": "tool", "content": "/workspace", "name": "terminal"},
            {"role": "user", "content": "/compact"},
        ]

    normalized = ChatCompletionRequest(model = "model", messages = messages())
    first_id = normalized.messages[2].tool_calls[0]["id"]
    second_id = normalized.messages[5].tool_calls[0]["id"]
    assert first_id != second_id
    assert normalized.messages[3].tool_call_id == first_id
    assert normalized.messages[6].tool_call_id == second_id

    prepared = _prepare_bound(
        "thread-1",
        attempt_id = "attempt-1",
        command_message_id = "compact-1",
        expected_head_message_id = "compact-1",
        message_ids = ["u1", "a1", "u2", "a2", "compact-1"],
        request_messages = messages(),
    )
    request = _request(prepared, messages = messages())
    manual_compaction.validate_and_rewrite_manual_compaction_request(request)
    assert request.messages[2].tool_calls[0]["id"] == first_id
    assert request.messages[5].tool_calls[0]["id"] == second_id


def test_concurrent_prepare_and_commit_serialize_without_revision_split(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)

    def prepare(attempt_id):
        try:
            return _prepare_with_id(attempt_id)
        except manual_compaction.ManualCompactionConflict as exc:
            return exc

    def _prepare_with_id(attempt_id):
        return _prepare_bound(
            "thread-1",
            attempt_id = attempt_id,
            command_message_id = "compact-1",
            expected_head_message_id = "compact-1",
            message_ids = ["u1", "a1", "compact-1"],
            request_messages = ChatCompletionRequest(
                model = "model", messages = _full_wire_messages()
            ).messages,
        )

    with ThreadPoolExecutor(max_workers = 2) as pool:
        prepared_results = list(pool.map(prepare, ("attempt-a", "attempt-b")))
    winners = [result for result in prepared_results if isinstance(result, dict)]
    losers = [result for result in prepared_results if isinstance(result, Exception)]
    assert len(winners) == 1
    assert len(losers) == 1
    assert any(detail in str(losers[0]) for detail in ("binding CAS is stale", "live /compact"))
    winner = winners[0]
    assert winner["state"] == "pending"
    losing_attempt_id = "attempt-b" if winner["attemptId"] == "attempt-a" else "attempt-a"
    assert manual_compaction.get_manual_compaction_attempt(losing_attempt_id) is None
    _claim(winner)
    _record_output(winner, "summary")
    studio_db.upsert_chat_message(_message("summary-1", "compact-1", "assistant", "summary"))

    def commit():
        return manual_compaction.commit_manual_compaction(
            "thread-1",
            attempt_id = winner["attemptId"],
            command_message_id = "compact-1",
            summary_message_id = "summary-1",
            expected_head_message_id = "summary-1",
            expected_revision = 1,
            expected_summary_hash = manual_compaction.summary_hash("summary"),
        )

    with ThreadPoolExecutor(max_workers = 2) as pool:
        committed = list(pool.map(lambda _index: commit(), range(2)))
    assert [result["state"] for result in committed] == ["active", "active"]
    assert {result["summaryHash"] for result in committed} == {
        manual_compaction.summary_hash("summary")
    }


def test_pending_source_drift_cannot_replace_the_fenced_ancestry(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    first = _prepare()
    user = studio_db.get_chat_message("thread-1", "u1")
    user["content"] = [{"type": "text", "text": "Explain the revised migration."}]
    before = _raw_message_row("u1")
    with pytest.raises(studio_db.ChatMessageProtectedError, match = "checkpoint"):
        studio_db.upsert_chat_message(user)
    assert _raw_message_row("u1") == before
    _set_raw_message_json(
        "u1",
        "content_json",
        json.dumps(user["content"], separators = (",", ":")),
    )
    messages = [
        {"role": "system", "content": "Project rules"},
        {"role": "user", "content": "Explain the revised migration."},
        {"role": "assistant", "content": "Use a staged rollout."},
        {"role": "user", "content": "/compact"},
    ]

    command_before = _raw_message_row("compact-1")
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "ancestry changed"):
        _prepare_bound(
            "thread-1",
            attempt_id = "attempt-2",
            command_message_id = "compact-1",
            expected_head_message_id = "compact-1",
            message_ids = ["u1", "a1", "compact-1"],
            request_messages = messages,
        )

    assert _raw_message_row("compact-1") == command_before
    assert manual_compaction.get_manual_compaction_attempt(first["attemptId"])["state"] == "pending"
    assert manual_compaction.get_manual_compaction_attempt("attempt-2") is None


def test_child_after_prepare_claim_terminalizes_and_new_head_can_bind(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    studio_db.upsert_chat_message(_message("late-child", "compact-1", "assistant", "Late response"))

    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "no longer the branch"):
        _claim(prepared)

    invalidated = manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])
    assert invalidated["state"] == "failed"
    assert invalidated["terminalReason"] == "prepare_invalidated"
    studio_db.upsert_chat_message(_message("compact-2", "late-child", "user", "/compact"))
    bound = manual_compaction.bind_manual_compaction_command(
        "thread-1",
        attempt_id = "attempt-2",
        command_message_id = "compact-2",
        summary_message_id = "summary-2",
        attempt_sequence = 1,
    )
    assert bound["metadata"][manual_compaction.MANUAL_COMPACTION_CLIENT_KEY]["attemptId"] == (
        "attempt-2"
    )


def test_child_after_prepare_replacement_terminalizes_before_branch_head_conflict(
    tmp_path, monkeypatch
):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    studio_db.upsert_chat_message(_message("late-child", "compact-1", "assistant", "Late response"))
    fence_before = _client_record()

    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "no longer the branch"):
        manual_compaction.bind_manual_compaction_command(
            "thread-1",
            attempt_id = "attempt-2",
            command_message_id = "compact-1",
            summary_message_id = "summary-2",
            attempt_sequence = 2,
            expected_attempt_id = prepared["attemptId"],
            expected_attempt_sequence = 1,
        )

    invalidated = manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])
    assert invalidated["state"] == "failed"
    assert invalidated["terminalReason"] == "prepare_invalidated"
    assert _client_record() == fence_before
    assert manual_compaction.get_manual_compaction_attempt("attempt-2") is None
    assert _raw_reservation_row("attempt-2") is None


@pytest.mark.parametrize("first", ["claim", "replacement"])
def test_child_after_prepare_claim_and_replacement_serialize_to_one_invalidation(
    tmp_path, monkeypatch, first
):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    studio_db.upsert_chat_message(_message("late-child", "compact-1", "assistant", "Late response"))

    def claim():
        return _claim(prepared)

    def replacement():
        return manual_compaction.bind_manual_compaction_command(
            "thread-1",
            attempt_id = "attempt-2",
            command_message_id = "compact-1",
            summary_message_id = "summary-2",
            attempt_sequence = 2,
            expected_attempt_id = prepared["attemptId"],
            expected_attempt_sequence = 1,
        )

    operations = {"claim": claim, "replacement": replacement}
    second = "replacement" if first == "claim" else "claim"
    results = _run_in_writer_order(
        monkeypatch,
        f"{first}-winner",
        operations[first],
        f"{second}-loser",
        operations[second],
    )

    assert all(
        isinstance(result, manual_compaction.ManualCompactionConflict)
        for result in results.values()
    )
    invalidated = manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])
    assert invalidated["state"] == "failed"
    assert invalidated["terminalReason"] == "prepare_invalidated"
    assert _client_record()["attemptId"] == prepared["attemptId"]
    assert manual_compaction.get_manual_compaction_attempt("attempt-2") is None
    assert _raw_reservation_row("attempt-2") is None


def test_expired_running_attempt_can_be_replaced(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    _claim(prepared)
    stored = manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])
    monkeypatch.setattr(
        manual_compaction.time,
        "time",
        lambda: (stored["leaseExpiresAt"] + 1) / 1000,
    )
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "lease expired"):
        _claim(prepared)
    assert manual_compaction.get_manual_compaction_attempt("attempt-1")["state"] == "failed"
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "terminal"):
        _prepare()

    replacement = _prepare_bound(
        "thread-1",
        attempt_id = "attempt-after-expiry",
        command_message_id = "compact-1",
        expected_head_message_id = "compact-1",
        message_ids = ["u1", "a1", "compact-1"],
        request_messages = _full_wire_messages(),
    )

    assert replacement["state"] == "pending"
    assert manual_compaction.get_manual_compaction_attempt("attempt-1")["state"] == "failed"


def test_expired_commit_cancels_before_summary_activation(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    _claim(prepared)
    running = manual_compaction.get_manual_compaction_attempt("attempt-1")
    studio_db.upsert_chat_message(_message("summary-1", "compact-1", "assistant", "summary"))
    monkeypatch.setattr(
        manual_compaction.time,
        "time",
        lambda: (running["leaseExpiresAt"] + 1) / 1000,
    )

    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "lease expired"):
        manual_compaction.commit_manual_compaction(
            "thread-1",
            attempt_id = "attempt-1",
            command_message_id = "compact-1",
            summary_message_id = "summary-1",
            expected_head_message_id = "summary-1",
            expected_revision = 1,
            expected_summary_hash = manual_compaction.summary_hash("summary"),
        )

    terminal = manual_compaction.get_manual_compaction_attempt("attempt-1")
    assert terminal["state"] == "failed"
    assert terminal["leaseExpiresAt"] is None
    assert terminal["committedAt"] is None
    summary = studio_db.get_chat_message("thread-1", "summary-1")
    assert summary.get("metadata") is None


def test_expired_commit_wins_writer_lock_over_replacement(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    _claim(prepared)
    running = manual_compaction.get_manual_compaction_attempt("attempt-1")
    studio_db.upsert_chat_message(_message("summary-1", "compact-1", "assistant", "summary"))
    monkeypatch.setattr(
        manual_compaction.time,
        "time",
        lambda: (running["leaseExpiresAt"] + 1) / 1000,
    )

    def commit():
        return manual_compaction.commit_manual_compaction(
            "thread-1",
            attempt_id = "attempt-1",
            command_message_id = "compact-1",
            summary_message_id = "summary-1",
            expected_head_message_id = "summary-1",
            expected_revision = 1,
            expected_summary_hash = manual_compaction.summary_hash("summary"),
        )

    def replace():
        return _prepare_bound(
            "thread-1",
            attempt_id = "attempt-2",
            command_message_id = "compact-1",
            expected_head_message_id = "compact-1",
            message_ids = ["u1", "a1", "compact-1"],
            request_messages = _full_wire_messages(),
        )

    results = _run_in_writer_order(
        monkeypatch,
        "commit-winner",
        commit,
        "replacement-loser",
        replace,
    )

    assert isinstance(results["commit-winner"], manual_compaction.ManualCompactionConflict)
    assert "lease expired" in str(results["commit-winner"])
    assert isinstance(results["replacement-loser"], manual_compaction.ManualCompactionConflict)
    assert "no longer the branch head" in str(results["replacement-loser"])
    assert manual_compaction.get_manual_compaction_attempt("attempt-1")["state"] == "failed"
    assert manual_compaction.get_manual_compaction_attempt("attempt-2") is None


def test_expired_replacement_binding_rejects_late_commit_and_recovers_stale_child(
    tmp_path, monkeypatch
):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    _claim(prepared)
    running = manual_compaction.get_manual_compaction_attempt("attempt-1")
    monkeypatch.setattr(
        manual_compaction.time,
        "time",
        lambda: (running["leaseExpiresAt"] + 1) / 1000,
    )

    replacement = manual_compaction.bind_manual_compaction_command(
        "thread-1",
        attempt_id = "attempt-2",
        command_message_id = "compact-1",
        summary_message_id = "summary-1",
        attempt_sequence = 2,
        expected_attempt_id = "attempt-1",
        expected_attempt_sequence = 1,
    )
    assert (
        replacement["metadata"][manual_compaction.MANUAL_COMPACTION_CLIENT_KEY]["attemptId"]
        == "attempt-2"
    )

    studio_db.upsert_chat_message(_message("summary-1", "compact-1", "assistant", "summary"))
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "failed: lease_expired"):
        manual_compaction.commit_manual_compaction(
            "thread-1",
            attempt_id = "attempt-1",
            command_message_id = "compact-1",
            summary_message_id = "summary-1",
            expected_head_message_id = "summary-1",
            expected_revision = 1,
            expected_summary_hash = manual_compaction.summary_hash("summary"),
        )
    assert _client_record()["attemptId"] == "attempt-2"

    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "branch head"):
        _prepare_core(
            "thread-1",
            attempt_id = "attempt-2",
            command_message_id = "compact-1",
            expected_head_message_id = "compact-1",
            message_ids = ["u1", "a1", "compact-1"],
            request_messages = _full_wire_messages(),
        )

    # The stale child has no server authority. Removing it cannot remove or rewrite the
    # fenced command ancestry, and an exact retry then promotes the durable reservation.
    studio_db.sync_chat_messages(
        "thread-1",
        [
            studio_db.get_chat_message("thread-1", "u1"),
            studio_db.get_chat_message("thread-1", "a1"),
            studio_db.get_chat_message("thread-1", "compact-1"),
        ],
        prune_missing = True,
    )
    recovered = _prepare_core(
        "thread-1",
        attempt_id = "attempt-2",
        command_message_id = "compact-1",
        expected_head_message_id = "compact-1",
        message_ids = ["u1", "a1", "compact-1"],
        request_messages = _full_wire_messages(),
    )
    assert recovered["state"] == "pending"
    assert manual_compaction.get_manual_compaction_attempt("attempt-1")["state"] == "failed"


def test_expired_replacement_wins_writer_lock_before_late_summary_and_commit(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    _claim(prepared)
    running = manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])
    monkeypatch.setattr(
        manual_compaction.time,
        "time",
        lambda: (running["leaseExpiresAt"] + 1) / 1000,
    )

    def replace():
        return manual_compaction.bind_manual_compaction_command(
            "thread-1",
            attempt_id = "attempt-2",
            command_message_id = "compact-1",
            summary_message_id = "summary-1",
            attempt_sequence = 2,
            expected_attempt_id = prepared["attemptId"],
            expected_attempt_sequence = 1,
        )

    def late_commit():
        studio_db.upsert_chat_message(_message("summary-1", "compact-1", "assistant", "summary"))
        return manual_compaction.commit_manual_compaction(
            "thread-1",
            attempt_id = prepared["attemptId"],
            command_message_id = prepared["commandMessageId"],
            summary_message_id = "summary-1",
            expected_head_message_id = "summary-1",
            expected_revision = prepared["revision"],
            expected_summary_hash = manual_compaction.summary_hash("summary"),
        )

    results = _run_in_writer_order(
        monkeypatch,
        "replacement-winner",
        replace,
        "late-commit-loser",
        late_commit,
    )

    assert isinstance(results["replacement-winner"], dict)
    assert isinstance(results["late-commit-loser"], manual_compaction.ManualCompactionConflict)
    assert "failed: lease_expired" in str(results["late-commit-loser"])
    assert _client_record()["attemptId"] == "attempt-2"
    assert _raw_reservation_row("attempt-2") is not None
    assert manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])["state"] == (
        "failed"
    )
    assert manual_compaction.get_manual_compaction_attempt("attempt-2") is None
    summary = studio_db.get_chat_message("thread-1", "summary-1")
    assert summary.get("metadata") is None


def test_only_one_inference_request_can_claim_an_attempt(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()

    def claim():
        try:
            return _claim(prepared)
        except manual_compaction.ManualCompactionConflict as exc:
            return exc

    with ThreadPoolExecutor(max_workers = 2) as pool:
        results = list(pool.map(lambda _index: claim(), range(2)))

    winners = [result for result in results if not isinstance(result, Exception)]
    losers = [result for result in results if isinstance(result, Exception)]
    assert len(winners) == 1
    assert len(losers) == 1
    assert "already running" in str(losers[0])
    stored = manual_compaction.get_manual_compaction_attempt("attempt-1")
    assert stored["state"] == "running"
    assert stored["startedAt"] is not None
    assert stored["leaseExpiresAt"] > stored["startedAt"]


def test_cancelled_attempt_cannot_commit_and_releases_the_branch(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    _claim(prepared)
    cancelled = manual_compaction.cancel_manual_compaction(
        "thread-1",
        attempt_id = "attempt-1",
        command_message_id = "compact-1",
        claim_id = _CLAIM_ID,
    )
    assert cancelled["state"] == "cancelled"
    assert cancelled["cancelledAt"] is not None
    assert cancelled["leaseExpiresAt"] is None
    studio_db.upsert_chat_message(_message("summary-1", "compact-1", "assistant", "summary"))
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "cancelled"):
        manual_compaction.commit_manual_compaction(
            "thread-1",
            attempt_id = "attempt-1",
            command_message_id = "compact-1",
            summary_message_id = "summary-1",
            expected_head_message_id = "summary-1",
            expected_revision = 1,
            expected_summary_hash = manual_compaction.summary_hash("summary"),
        )
    studio_db.sync_chat_messages(
        "thread-1",
        [
            message
            for message in studio_db.list_chat_messages("thread-1")
            if message["id"] != "summary-1"
        ],
        prune_missing = True,
    )
    replacement = _prepare_bound(
        "thread-1",
        attempt_id = "attempt-2",
        command_message_id = "compact-1",
        expected_head_message_id = "compact-1",
        message_ids = ["u1", "a1", "compact-1"],
        request_messages = _full_wire_messages(),
    )
    assert replacement["state"] == "pending"


def test_commit_requires_running_then_transitions_to_active(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    studio_db.upsert_chat_message(_message("summary-1", "compact-1", "assistant", "summary"))
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "claim inference"):
        manual_compaction.commit_manual_compaction(
            "thread-1",
            attempt_id = "attempt-1",
            command_message_id = "compact-1",
            summary_message_id = "summary-1",
            expected_head_message_id = "summary-1",
            expected_revision = 1,
            expected_summary_hash = manual_compaction.summary_hash("summary"),
        )
    studio_db.sync_chat_messages(
        "thread-1",
        [
            message
            for message in studio_db.list_chat_messages("thread-1")
            if message["id"] != "summary-1"
        ],
        prune_missing = True,
    )
    _claim(prepared)
    _record_output(prepared, "summary")
    studio_db.upsert_chat_message(_message("summary-1", "compact-1", "assistant", "summary"))
    committed = manual_compaction.commit_manual_compaction(
        "thread-1",
        attempt_id = "attempt-1",
        command_message_id = "compact-1",
        summary_message_id = "summary-1",
        expected_head_message_id = "summary-1",
        expected_revision = 1,
        expected_summary_hash = manual_compaction.summary_hash("summary"),
    )
    assert committed["state"] == "active"
    assert committed["leaseExpiresAt"] is None
    assert manual_compaction.get_manual_compaction_attempt("attempt-1")["leaseExpiresAt"] is None


def test_fork_rekeys_active_summary_ids_attempt_and_source_hash(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    _claim(prepared)
    text = "summary"
    _record_output(prepared, text)
    studio_db.upsert_chat_message(_message("summary-1", "compact-1", "assistant", text))
    manual_compaction.commit_manual_compaction(
        "thread-1",
        attempt_id = "attempt-1",
        command_message_id = "compact-1",
        summary_message_id = "summary-1",
        expected_head_message_id = "summary-1",
        expected_revision = 1,
        expected_summary_hash = manual_compaction.summary_hash(text),
    )
    ids = iter(["fu1", "fa1", "fcompact", "fsummary"])
    studio_db.fork_chat_thread(
        source_thread_id = "thread-1",
        branch_message_id = "summary-1",
        new_thread_id = "fork-1",
        new_title = "Fork",
        created_at = 20,
        id_factory = lambda: next(ids),
    )
    forked = studio_db.get_chat_message("fork-1", "fsummary")["metadata"]["manualCompaction"]
    assert forked["threadId"] == "fork-1"
    assert forked["commandMessageId"] == "fcompact"
    assert forked["sourceHeadMessageId"] == "fa1"
    assert forked["summaryMessageId"] == "fsummary"
    assert forked["attemptId"].startswith("fork-")
    assert forked["attemptId"] != "attempt-1"
    assert forked["sourceHash"] != prepared["sourceHash"]
    assert forked["sourceHash"] == manual_compaction.canonical_source_hash(
        [
            studio_db.get_chat_message("fork-1", "fu1"),
            studio_db.get_chat_message("fork-1", "fa1"),
        ]
    )
    forked_attempt = manual_compaction.get_manual_compaction_attempt(forked["attemptId"])
    assert forked_attempt["archivePayloadHash"] == prepared["archivePayloadHash"]
    assert forked_attempt["outputSummaryHash"] == manual_compaction.summary_hash(text)
    assert forked_attempt["outputFinishReason"] == "stop"
    assert forked_attempt["archiveStatus"] == "pending"

    nested_ids = iter(["nfu1", "nfa1", "nfcompact", "nfsummary"])
    studio_db.fork_chat_thread(
        source_thread_id = "fork-1",
        branch_message_id = "fsummary",
        new_thread_id = "fork-2",
        new_title = "Nested fork",
        created_at = 30,
        id_factory = lambda: next(nested_ids),
    )
    nested_summary = studio_db.get_chat_message("fork-2", "nfsummary")
    nested_metadata = nested_summary["metadata"]["manualCompaction"]
    assert nested_metadata["threadId"] == "fork-2"
    assert nested_metadata["revision"] == 1
    changed = dict(nested_summary)
    changed["content"] = [{"type": "text", "text": "changed"}]
    with pytest.raises(studio_db.ChatMessageProtectedError, match = "checkpoint"):
        studio_db.upsert_chat_message(changed)

    studio_db.upsert_chat_message(
        _message("compact-2", "nfsummary", "user", "/compact", thread_id = "fork-2")
    )
    nested_prepare = _prepare_bound(
        "fork-2",
        attempt_id = "attempt-nested",
        command_message_id = "compact-2",
        expected_head_message_id = "compact-2",
        message_ids = ["nfu1", "nfa1", "nfcompact", "nfsummary", "compact-2"],
        request_messages = [
            {"role": "system", "content": "Project rules"},
            {"role": "assistant", "content": text},
            {"role": "user", "content": "/compact"},
        ],
    )
    assert nested_prepare["revision"] == 2
    assert nested_prepare["effectiveSourceMessageIds"] == ["nfsummary"]


def test_fork_keeps_complete_active_checkpoint_but_strips_newer_live_authority(
    tmp_path, monkeypatch
):
    _seed_branch(tmp_path, monkeypatch)
    first = _prepare()
    _claim(first)
    _record_output(first, "summary")
    studio_db.upsert_chat_message(_message("summary-1", "compact-1", "assistant", "summary"))
    manual_compaction.commit_manual_compaction(
        "thread-1",
        attempt_id = first["attemptId"],
        command_message_id = first["commandMessageId"],
        summary_message_id = "summary-1",
        expected_head_message_id = "summary-1",
        expected_revision = first["revision"],
        expected_summary_hash = manual_compaction.summary_hash("summary"),
    )
    for message in (
        _message("u2", "summary-1", "user", "Continue."),
        _message("a2", "u2", "assistant", "Continuing."),
        _message("compact-2", "a2", "user", "/compact"),
    ):
        studio_db.upsert_chat_message(message)
    second_messages = [
        {"role": "system", "content": "Project rules"},
        {"role": "assistant", "content": "summary"},
        {"role": "user", "content": "Continue."},
        {"role": "assistant", "content": "Continuing."},
        {"role": "user", "content": "/compact"},
    ]
    second = _prepare_bound(
        "thread-1",
        attempt_id = "attempt-2",
        command_message_id = "compact-2",
        expected_head_message_id = "compact-2",
        message_ids = ["u1", "a1", "compact-1", "summary-1", "u2", "a2", "compact-2"],
        request_messages = ChatCompletionRequest(model = "model", messages = second_messages).messages,
        summary_message_id = "summary-2",
    )
    _claim(second, messages = second_messages, claim_id = "9" * 64)
    source_attempt_before = manual_compaction.get_manual_compaction_attempt("attempt-2")
    ids = iter(["fu1", "fa1", "fc1", "fs1", "fu2", "fa2", "fc2"])

    studio_db.fork_chat_thread(
        source_thread_id = "thread-1",
        branch_message_id = "compact-2",
        new_thread_id = "fork-live",
        new_title = "Fork live",
        created_at = 30,
        id_factory = lambda: next(ids),
    )

    fork_active = studio_db.get_chat_message("fork-live", "fs1")["metadata"]["manualCompaction"]
    assert fork_active["state"] == "active"
    assert fork_active["threadId"] == "fork-live"
    assert (
        manual_compaction.get_manual_compaction_attempt(fork_active["attemptId"])["state"]
        == "active"
    )
    fork_live_command = studio_db.get_chat_message("fork-live", "fc2")
    assert manual_compaction.MANUAL_COMPACTION_CLIENT_KEY not in (
        fork_live_command.get("metadata") or {}
    )
    rebound = manual_compaction.bind_manual_compaction_command(
        "fork-live",
        attempt_id = "fork-fresh-attempt",
        command_message_id = "fc2",
        summary_message_id = "fork-fresh-summary",
        attempt_sequence = 1,
    )
    assert (
        rebound["metadata"][manual_compaction.MANUAL_COMPACTION_CLIENT_KEY]["attemptSequence"] == 1
    )
    assert manual_compaction.get_manual_compaction_attempt("attempt-2") == source_attempt_before


def test_fork_strips_malformed_and_incomplete_client_authority_without_touching_source(
    tmp_path, monkeypatch
):
    _seed_branch(tmp_path, monkeypatch)
    malformed_metadata = {
        manual_compaction.MANUAL_COMPACTION_CLIENT_KEY: {
            "schemaVersion": 1,
            "state": "running",
            "attemptId": "stale",
        }
    }
    _set_raw_message_json(
        "compact-1",
        "metadata_json",
        json.dumps(malformed_metadata, separators = (",", ":")),
    )
    source_before = _raw_message_row("compact-1")
    ids = iter(["fu1", "fa1", "fcompact"])

    studio_db.fork_chat_thread(
        source_thread_id = "thread-1",
        branch_message_id = "compact-1",
        new_thread_id = "fork-malformed",
        new_title = "Fork malformed",
        created_at = 30,
        id_factory = lambda: next(ids),
    )

    forked = studio_db.get_chat_message("fork-malformed", "fcompact")
    assert manual_compaction.MANUAL_COMPACTION_CLIENT_KEY not in (forked.get("metadata") or {})
    assert _raw_message_row("compact-1") == source_before


def test_fork_route_retries_the_cloned_effective_archive(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    _claim(prepared)
    _record_output(prepared, "summary")
    studio_db.upsert_chat_message(_message("summary-1", "compact-1", "assistant", "summary"))
    manual_compaction.commit_manual_compaction(
        "thread-1",
        attempt_id = prepared["attemptId"],
        command_message_id = prepared["commandMessageId"],
        summary_message_id = "summary-1",
        expected_head_message_id = "summary-1",
        expected_revision = prepared["revision"],
        expected_summary_hash = manual_compaction.summary_hash("summary"),
    )
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "UPDATE manual_compactions SET archive_status = 'archived' WHERE attempt_id = ?",
            (prepared["attemptId"],),
        )
        conn.commit()
    finally:
        conn.close()
    archived = []

    def capture_archive(attempt):
        archived.append(attempt)
        return "archived"

    monkeypatch.setattr(
        chat_history,
        "archive_manual_compaction_best_effort",
        capture_archive,
    )
    response = chat_history.fork_thread(
        "thread-1",
        chat_history.ChatForkRequest(
            messageId = "summary-1",
            newThreadId = "fork-route",
            createdAt = 20,
        ),
        current_subject = "test-user",
    )

    assert response.thread.id == "fork-route"
    assert len(archived) == 1
    assert archived[0]["threadId"] == "fork-route"
    assert archived[0]["archiveStatus"] == "pending"
    assert archived[0]["archivePayload"] == prepared["archivePayload"]


def test_summary_rejects_blank_oversize_and_surrogate_utf8(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    _claim(prepared)
    studio_db.upsert_chat_message(_message("summary-1", "compact-1", "assistant", " "))
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "blank"):
        manual_compaction.commit_manual_compaction(
            "thread-1",
            attempt_id = "attempt-1",
            command_message_id = "compact-1",
            summary_message_id = "summary-1",
            expected_head_message_id = "summary-1",
            expected_revision = 1,
            expected_summary_hash = hashlib.sha256(b" ").hexdigest(),
        )
    with pytest.raises(manual_compaction.ManualCompactionError, match = "valid UTF-8"):
        manual_compaction.summary_hash("bad\udcff")
    with pytest.raises(manual_compaction.ManualCompactionError, match = "too large"):
        manual_compaction.summary_hash(
            "x" * (manual_compaction.MAX_MANUAL_COMPACTION_SUMMARY_BYTES + 1)
        )


def test_schema_persists_attempt_fields_as_canonical_json(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    conn = studio_db.get_connection()
    try:
        row = conn.execute(
            "SELECT source_message_ids_json, effective_source_message_ids_json, "
            "source_hash, revision, state, request_hash, request_message_count, "
            "project_instruction_digest, project_instruction_revision, context_digest, "
            "archive_payload_json, archive_payload_hash, output_summary_hash, "
            "output_finish_reason, output_recorded_at "
            "FROM manual_compactions "
            "WHERE attempt_id = 'attempt-1'"
        ).fetchone()
    finally:
        conn.close()
    assert json.loads(row["source_message_ids_json"]) == ["u1", "a1"]
    assert json.loads(row["effective_source_message_ids_json"]) == ["u1", "a1"]
    assert row["source_hash"] == prepared["sourceHash"]
    assert row["revision"] == 1
    assert row["state"] == "pending"
    assert row["request_hash"] == prepared["requestHash"]
    assert row["request_message_count"] == prepared["requestMessageCount"]
    assert row["project_instruction_digest"] == prepared["projectInstructionDigest"]
    assert row["project_instruction_revision"] == prepared["projectInstructionRevision"]
    assert row["context_digest"] == prepared["contextDigest"]
    assert json.loads(row["archive_payload_json"]) == prepared["archivePayload"]
    assert row["archive_payload_hash"] == prepared["archivePayloadHash"]
    assert row["output_summary_hash"] is None
    assert row["output_finish_reason"] is None
    assert row["output_recorded_at"] is None


def test_schema_migrates_pre_lifecycle_table_but_terminalizes_incomplete_authority(
    tmp_path, monkeypatch
):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    conn = studio_db.get_connection()
    try:
        conn.execute("DROP INDEX IF EXISTS idx_manual_compactions_thread_state")
        conn.execute("DROP INDEX IF EXISTS idx_manual_compactions_live_branch")
        conn.execute("ALTER TABLE manual_compactions RENAME TO manual_compactions_new")
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
        )
        conn.execute(
            """
            INSERT INTO manual_compactions (
                attempt_id, thread_id, command_message_id, source_head_message_id,
                expected_head_message_id, source_message_ids_json,
                effective_source_message_ids_json, source_hash, request_hash,
                request_message_count, project_instruction_digest,
                project_instruction_revision, context_digest, revision, state,
                summary_message_id, summary_hash, archive_status, created_at,
                committed_at
            )
            SELECT
                attempt_id, thread_id, command_message_id, source_head_message_id,
                expected_head_message_id, source_message_ids_json,
                effective_source_message_ids_json, source_hash, request_hash,
                request_message_count, project_instruction_digest,
                project_instruction_revision, context_digest, revision, state,
                summary_message_id, summary_hash, archive_status, created_at,
                committed_at
            FROM manual_compactions_new
            """
        )
        conn.execute("DROP TABLE manual_compactions_new")
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr(studio_db, "_schema_ready", False)
    migrated = manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])
    assert migrated["state"] == "failed"
    assert migrated["terminalReason"] == "migrated_failed"
    assert migrated["startedAt"] is None
    assert migrated["leaseExpiresAt"] is None
    assert migrated["archivePayload"] == []
    assert migrated["sourceMessageIds"] == []
    assert migrated["outputSummaryHash"] is None
    assert migrated["outputFinishReason"] is None
    conn = studio_db.get_connection()
    try:
        schema = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'manual_compactions'"
        ).fetchone()["sql"]
        indexes = {
            row["name"] for row in conn.execute("PRAGMA index_list(manual_compactions)").fetchall()
        }
    finally:
        conn.close()
    assert "'running'" in schema
    assert "archive_payload_json" in schema
    assert "output_finish_reason" in schema
    assert "idx_manual_compactions_live_branch" in indexes


def test_two_initial_bind_clients_converge_on_one_attempt_and_summary(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    claim_a = "a" * 64
    claim_b = "b" * 64

    def bind(attempt_id, summary_message_id):
        return manual_compaction.bind_manual_compaction_command(
            "thread-1",
            attempt_id = attempt_id,
            command_message_id = "compact-1",
            summary_message_id = summary_message_id,
            attempt_sequence = 1,
        )

    results = _run_in_writer_order(
        monkeypatch,
        "client-a",
        lambda: bind("attempt-a", "summary-a"),
        "client-b",
        lambda: bind("attempt-b", "summary-b"),
    )

    assert (
        results["client-a"]["metadata"][manual_compaction.MANUAL_COMPACTION_CLIENT_KEY]["attemptId"]
        == "attempt-a"
    )
    assert isinstance(results["client-b"], manual_compaction.ManualCompactionConflict)
    assert "binding CAS is stale" in str(results["client-b"])
    record = _client_record()
    assert record["attemptId"] == "attempt-a"
    assert record["summaryMessageId"] == "summary-a"

    prepared = _prepare_core(
        "thread-1",
        attempt_id = record["attemptId"],
        command_message_id = "compact-1",
        expected_head_message_id = "compact-1",
        message_ids = ["u1", "a1", "compact-1"],
        request_messages = ChatCompletionRequest(
            model = "model", messages = _full_wire_messages()
        ).messages,
    )
    recovered = _prepare_core(
        "thread-1",
        attempt_id = record["attemptId"],
        command_message_id = "compact-1",
        expected_head_message_id = "compact-1",
        message_ids = ["u1", "a1", "compact-1"],
        request_messages = ChatCompletionRequest(
            model = "model", messages = _full_wire_messages()
        ).messages,
    )
    assert recovered == prepared
    _claim(prepared, claim_id = claim_a)
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "not owned"):
        manual_compaction.cancel_manual_compaction(
            "thread-1",
            attempt_id = prepared["attemptId"],
            command_message_id = prepared["commandMessageId"],
            claim_id = claim_b,
        )
    _record_output(prepared, "one summary")
    studio_db.upsert_chat_message(_message("summary-a", "compact-1", "assistant", "one summary"))
    committed = manual_compaction.commit_manual_compaction(
        "thread-1",
        attempt_id = prepared["attemptId"],
        command_message_id = prepared["commandMessageId"],
        summary_message_id = "summary-a",
        expected_head_message_id = "summary-a",
        expected_revision = prepared["revision"],
        expected_summary_hash = manual_compaction.summary_hash("one summary"),
    )
    assert committed["state"] == "active"
    assert _client_record()["state"] == "summary_saved"
    conn = studio_db.get_connection()
    try:
        assert (
            conn.execute(
                "SELECT COUNT(*) AS count FROM manual_compactions WHERE thread_id = ?",
                ("thread-1",),
            ).fetchone()["count"]
            == 1
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) AS count FROM chat_messages WHERE thread_id = ? AND parent_id = ?",
                ("thread-1", "compact-1"),
            ).fetchone()["count"]
            == 1
        )
    finally:
        conn.close()


def test_late_initial_bind_cannot_downgrade_prepared_or_summary_saved(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    manual_compaction.bind_manual_compaction_command(
        "thread-1",
        attempt_id = "attempt-a",
        command_message_id = "compact-1",
        summary_message_id = "summary-a",
        attempt_sequence = 1,
    )
    prepared = _prepare_core(
        "thread-1",
        attempt_id = "attempt-a",
        command_message_id = "compact-1",
        expected_head_message_id = "compact-1",
        message_ids = ["u1", "a1", "compact-1"],
        request_messages = ChatCompletionRequest(
            model = "model", messages = _full_wire_messages()
        ).messages,
    )
    authoritative_prepared = json.loads(json.dumps(_client_record()))
    assert authoritative_prepared["state"] == "prepared"
    assert "claimId" not in authoritative_prepared["envelope"]

    for state in ("prepared", "summary_saved"):
        with pytest.raises(manual_compaction.ManualCompactionConflict):
            manual_compaction.bind_manual_compaction_command(
                "thread-1",
                attempt_id = "attempt-late",
                command_message_id = "compact-1",
                summary_message_id = "summary-late",
                attempt_sequence = 1,
            )
        assert _client_record()["state"] == state
        if state == "prepared":
            _claim(prepared, claim_id = "a" * 64)
            _record_output(prepared, "durable summary")
            studio_db.upsert_chat_message(
                _message("summary-a", "compact-1", "assistant", "durable summary")
            )
            manual_compaction.commit_manual_compaction(
                "thread-1",
                attempt_id = prepared["attemptId"],
                command_message_id = prepared["commandMessageId"],
                summary_message_id = "summary-a",
                expected_head_message_id = "summary-a",
                expected_revision = prepared["revision"],
                expected_summary_hash = manual_compaction.summary_hash("durable summary"),
            )
    saved = _client_record()
    assert saved["attemptId"] == "attempt-a"
    assert saved["summaryMessageId"] == "summary-a"
    assert saved["summaryHash"] == manual_compaction.summary_hash("durable summary")


def test_request_drift_cannot_replace_or_cancel_the_prepared_winner(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    command_before = json.loads(json.dumps(_client_record()))
    drifted = _full_wire_messages()
    drifted[1] = {"role": "user", "content": "Different request"}
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "exact stored"):
        _prepare(drifted)
    assert _client_record() == command_before
    assert manual_compaction.get_manual_compaction_attempt("attempt-1")["state"] == "pending"
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "not owned"):
        manual_compaction.cancel_manual_compaction(
            "thread-1",
            attempt_id = "attempt-1",
            command_message_id = "compact-1",
            claim_id = "b" * 64,
        )
    assert manual_compaction.get_manual_compaction_attempt("attempt-1")["state"] == "pending"


def test_claim_secret_owns_cancel_replay_and_never_crosses_the_request_boundary(
    tmp_path, monkeypatch
):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    claim_id = "c" * 64
    request = _claim(prepared, claim_id = claim_id)
    assert request.manual_compaction is None
    conn = studio_db.get_connection()
    try:
        raw = dict(
            conn.execute(
                "SELECT * FROM manual_compactions WHERE attempt_id = ?",
                (prepared["attemptId"],),
            ).fetchone()
        )
    finally:
        conn.close()
    assert raw["claim_token_hash"] == hashlib.sha256(claim_id.encode("ascii")).hexdigest()
    assert claim_id not in json.dumps(raw, sort_keys = True)
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "not owned"):
        manual_compaction.cancel_manual_compaction(
            "thread-1",
            attempt_id = prepared["attemptId"],
            command_message_id = prepared["commandMessageId"],
            claim_id = "d" * 64,
        )
    cancelled = manual_compaction.cancel_manual_compaction(
        "thread-1",
        attempt_id = prepared["attemptId"],
        command_message_id = prepared["commandMessageId"],
        claim_id = claim_id,
    )
    replayed = manual_compaction.cancel_manual_compaction(
        "thread-1",
        attempt_id = prepared["attemptId"],
        command_message_id = prepared["commandMessageId"],
        claim_id = claim_id,
    )
    assert cancelled["state"] == replayed["state"] == "cancelled"
    assert claim_id not in json.dumps(cancelled, sort_keys = True)
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "not owned"):
        manual_compaction.cancel_manual_compaction(
            "thread-1",
            attempt_id = prepared["attemptId"],
            command_message_id = prepared["commandMessageId"],
            claim_id = "d" * 64,
        )


def test_two_concurrent_claims_have_one_owner_and_only_that_owner_can_cancel(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    claims = ("e" * 64, "f" * 64)

    def claim(claim_id):
        try:
            return claim_id, _claim(prepared, claim_id = claim_id)
        except manual_compaction.ManualCompactionConflict as exc:
            return claim_id, exc

    with ThreadPoolExecutor(max_workers = 2) as pool:
        results = list(pool.map(claim, claims))
    winners = [result for result in results if not isinstance(result[1], Exception)]
    losers = [result for result in results if isinstance(result[1], Exception)]
    assert len(winners) == len(losers) == 1
    winner_claim = winners[0][0]
    loser_claim = losers[0][0]
    assert "already running" in str(losers[0][1])
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "not owned"):
        manual_compaction.cancel_manual_compaction(
            "thread-1",
            attempt_id = prepared["attemptId"],
            command_message_id = prepared["commandMessageId"],
            claim_id = loser_claim,
        )
    cancelled = manual_compaction.cancel_manual_compaction(
        "thread-1",
        attempt_id = prepared["attemptId"],
        command_message_id = prepared["commandMessageId"],
        claim_id = winner_claim,
    )
    assert cancelled["state"] == "cancelled"


def test_process_restart_terminalizes_owned_running_attempt_without_plaintext(
    tmp_path, monkeypatch
):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    claim_id = "9" * 64
    _claim(prepared, claim_id = claim_id)

    monkeypatch.setattr(studio_db, "_schema_ready", False)
    recovered = manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])

    assert recovered["state"] == "failed"
    assert recovered["terminalReason"] == "inference_failed"
    assert claim_id not in json.dumps(recovered, sort_keys = True)
    replayed = manual_compaction.cancel_manual_compaction(
        "thread-1",
        attempt_id = prepared["attemptId"],
        command_message_id = prepared["commandMessageId"],
        claim_id = claim_id,
    )
    assert replayed["state"] == "failed"
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "not owned"):
        manual_compaction.cancel_manual_compaction(
            "thread-1",
            attempt_id = prepared["attemptId"],
            command_message_id = prepared["commandMessageId"],
            claim_id = "8" * 64,
        )


def test_ordinary_message_writes_cannot_bypass_client_fence(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    manual_compaction.bind_manual_compaction_command(
        "thread-1",
        attempt_id = "attempt-1",
        command_message_id = "compact-1",
        summary_message_id = "summary-1",
        attempt_sequence = 1,
    )
    command = studio_db.get_chat_message("thread-1", "compact-1")
    record = command["metadata"][manual_compaction.MANUAL_COMPACTION_CLIENT_KEY]
    reordered = {key: record[key] for key in reversed(list(record))}
    command["metadata"] = {manual_compaction.MANUAL_COMPACTION_CLIENT_KEY: reordered}
    saved = studio_db.upsert_chat_message(command)
    assert saved["metadata"][manual_compaction.MANUAL_COMPACTION_CLIENT_KEY] == record

    unrelated = json.loads(json.dumps(saved))
    unrelated["metadata"]["unrelated"] = {"value": 1}
    before = _raw_message_row("compact-1")
    with pytest.raises(studio_db.ChatMessageProtectedError, match = "checkpoint"):
        studio_db.upsert_chat_message(unrelated)
    assert _raw_message_row("compact-1") == before

    changed = json.loads(json.dumps(saved))
    changed["metadata"][manual_compaction.MANUAL_COMPACTION_CLIENT_KEY]["attemptId"] = "forged"
    with pytest.raises(studio_db.ChatMessageProtectedError, match = "fenced transition"):
        studio_db.upsert_chat_message(changed)
    removed = json.loads(json.dumps(saved))
    removed["metadata"].pop(manual_compaction.MANUAL_COMPACTION_CLIENT_KEY)
    with pytest.raises(studio_db.ChatMessageProtectedError, match = "fenced transition"):
        studio_db.upsert_chat_message(removed)

    studio_db.upsert_chat_message(_message("compact-2", "a1", "user", "/compact"))
    introduced = studio_db.get_chat_message("thread-1", "compact-2")
    introduced["metadata"] = {
        manual_compaction.MANUAL_COMPACTION_CLIENT_KEY: record,
    }
    with pytest.raises(studio_db.ChatMessageProtectedError, match = "fenced transition"):
        studio_db.upsert_chat_message(introduced)

    stale_batch = studio_db.list_chat_messages("thread-1")
    for message in stale_batch:
        if message["id"] == "compact-1":
            message["metadata"].pop(manual_compaction.MANUAL_COMPACTION_CLIENT_KEY)
    with pytest.raises(studio_db.ChatMessageProtectedError, match = "fenced transition"):
        studio_db.sync_chat_messages("thread-1", stale_batch)
    without_command = [
        message
        for message in studio_db.list_chat_messages("thread-1")
        if message["id"] != "compact-1"
    ]
    with pytest.raises(studio_db.ChatMessageProtectedError, match = "cannot be pruned"):
        studio_db.sync_chat_messages("thread-1", without_command, prune_missing = True)


@pytest.mark.parametrize(
    "mutation",
    ["parent", "role", "content", "content-parts", "attachments", "metadata", "created-at"],
)
def test_live_command_fence_rejects_every_persisted_row_mutation_without_writes(
    tmp_path, monkeypatch, mutation
):
    _seed_branch(tmp_path, monkeypatch)
    manual_compaction.bind_manual_compaction_command(
        "thread-1",
        attempt_id = "attempt-1",
        command_message_id = "compact-1",
        summary_message_id = "summary-1",
        attempt_sequence = 1,
    )
    changed = json.loads(json.dumps(studio_db.get_chat_message("thread-1", "compact-1")))
    if mutation == "parent":
        changed["parentId"] = "u1"
    elif mutation == "role":
        changed["role"] = "assistant"
    elif mutation == "content":
        changed["content"][0]["text"] = "/compact stale"
    elif mutation == "content-parts":
        changed["content"].append({"type": "text", "text": "stale"})
    elif mutation == "attachments":
        changed["attachments"] = [{"id": "attachment-1", "name": "stale.txt"}]
    elif mutation == "metadata":
        changed["metadata"]["stale"] = True
    else:
        changed["createdAt"] += 1

    before = _raw_message_row("compact-1")
    with pytest.raises(studio_db.ChatMessageProtectedError, match = "checkpoint"):
        studio_db.upsert_chat_message(changed)
    assert _raw_message_row("compact-1") == before


@pytest.mark.parametrize(
    ("message_id", "mutation"),
    [("u1", "content"), ("u1", "metadata"), ("a1", "parent"), ("a1", "created-at")],
)
def test_prepared_fence_protects_complete_source_ancestry_without_writes(
    tmp_path, monkeypatch, message_id, mutation
):
    _seed_branch(tmp_path, monkeypatch)
    _prepare()
    changed = json.loads(json.dumps(studio_db.get_chat_message("thread-1", message_id)))
    if mutation == "content":
        changed["content"][0]["text"] = "stale"
    elif mutation == "metadata":
        changed["metadata"] = {"stale": True}
    elif mutation == "parent":
        changed["parentId"] = None
    else:
        changed["createdAt"] += 1

    before = _raw_message_row(message_id)
    with pytest.raises(studio_db.ChatMessageProtectedError, match = "checkpoint"):
        studio_db.upsert_chat_message(changed)
    assert _raw_message_row(message_id) == before


def test_live_fence_rejects_source_relink_and_prune_atomically(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    _prepare()
    before = {message_id: _raw_message_row(message_id) for message_id in ("u1", "a1", "compact-1")}

    relinked = studio_db.list_chat_messages("thread-1")
    next(message for message in relinked if message["id"] == "a1")["parentId"] = None
    with pytest.raises(studio_db.ChatMessageProtectedError, match = "checkpoint"):
        studio_db.sync_chat_messages("thread-1", relinked, prune_missing = True)
    assert {message_id: _raw_message_row(message_id) for message_id in before} == before

    pruned = [
        message for message in studio_db.list_chat_messages("thread-1") if message["id"] != "u1"
    ]
    with pytest.raises(studio_db.ChatMessageProtectedError, match = "cannot be pruned"):
        studio_db.sync_chat_messages("thread-1", pruned, prune_missing = True)
    assert {message_id: _raw_message_row(message_id) for message_id in before} == before


def test_bound_fence_blocks_attachment_and_content_blob_deletion(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    user = studio_db.get_chat_message("thread-1", "u1")
    user["attachments"] = [{"id": "attachment-u1", "name": "notes.txt"}]
    studio_db.upsert_chat_message(user)
    command = studio_db.get_chat_message("thread-1", "compact-1")
    command["attachments"] = [{"id": "attachment-command", "name": "command.txt"}]
    studio_db.upsert_chat_message(command)
    assistant = studio_db.get_chat_message("thread-1", "a1")
    image_part = {"type": "image", "image": "data:image/png;base64,YQ=="}
    assistant["content"].append(image_part)
    studio_db.upsert_chat_message(assistant)
    content_part_id = studio_db._content_part_id(image_part)
    assert content_part_id is not None
    manual_compaction.bind_manual_compaction_command(
        "thread-1",
        attempt_id = "attempt-1",
        command_message_id = "compact-1",
        summary_message_id = "summary-1",
        attempt_sequence = 1,
    )

    for message_id, attachment_id in (
        ("u1", "attachment-u1"),
        ("compact-1", "attachment-command"),
        ("a1", content_part_id),
    ):
        before = _raw_message_row(message_id)
        with pytest.raises(studio_db.ChatMessageProtectedError, match = "cannot be edited"):
            studio_db.delete_chat_attachment(message_id, attachment_id)
        assert _raw_message_row(message_id) == before


def test_claim_rechecks_server_derived_prepared_command_hash(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    command = studio_db.get_chat_message("thread-1", "compact-1")
    command["metadata"]["raw-drift"] = True
    _set_raw_message_json(
        "compact-1",
        "metadata_json",
        json.dumps(command["metadata"], separators = (",", ":")),
    )

    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "changed after prepare"):
        _claim(prepared)
    raw_attempt_row = None
    conn = studio_db.get_connection()
    try:
        raw_attempt_row = dict(
            conn.execute(
                "SELECT state, claim_token_hash FROM manual_compactions WHERE attempt_id = ?",
                (prepared["attemptId"],),
            ).fetchone()
        )
    finally:
        conn.close()
    assert raw_attempt_row == {"state": "pending", "claim_token_hash": None}


def test_commit_rechecks_server_derived_prepared_command_hash(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    _claim(prepared)
    _record_output(prepared, "summary")
    studio_db.upsert_chat_message(_message("summary-1", "compact-1", "assistant", "summary"))
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "UPDATE chat_messages SET created_at = created_at + 1 WHERE id = ?",
            ("compact-1",),
        )
        conn.commit()
    finally:
        conn.close()
    before = _raw_message_row("compact-1")

    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "changed after prepare"):
        manual_compaction.commit_manual_compaction(
            "thread-1",
            attempt_id = prepared["attemptId"],
            command_message_id = prepared["commandMessageId"],
            summary_message_id = "summary-1",
            expected_head_message_id = "summary-1",
            expected_revision = prepared["revision"],
            expected_summary_hash = manual_compaction.summary_hash("summary"),
        )
    assert _raw_message_row("compact-1") == before
    assert (
        manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])["state"] == "running"
    )


def test_prepared_command_digest_cannot_be_replayed_as_summary_saved(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    _claim(prepared)
    _record_output(prepared, "summary")
    studio_db.upsert_chat_message(_message("summary-1", "compact-1", "assistant", "summary"))
    summary_hash = manual_compaction.summary_hash("summary")
    command = studio_db.get_chat_message("thread-1", "compact-1")
    client = command["metadata"][manual_compaction.MANUAL_COMPACTION_CLIENT_KEY]
    client = {
        **client,
        "state": "summary_saved",
        "summaryHash": summary_hash,
    }
    command["metadata"][manual_compaction.MANUAL_COMPACTION_CLIENT_KEY] = client
    prepared_hash = prepared["commandHash"]
    assert prepared_hash != studio_db.canonical_chat_message_hash(command, state = "summary_saved")
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "UPDATE chat_messages SET metadata_json = ? WHERE id = ?",
            (json.dumps(command["metadata"], separators = (",", ":")), "compact-1"),
        )
        conn.execute(
            "UPDATE manual_compactions SET state = 'active', summary_message_id = ?, "
            "summary_hash = ?, command_hash_state = 'summary_saved', lease_expires_at = NULL, "
            "claim_token_hash = NULL, committed_at = ? WHERE attempt_id = ?",
            ("summary-1", summary_hash, 20, prepared["attemptId"]),
        )
        conn.commit()
    finally:
        conn.close()
    before = _raw_message_row("compact-1")

    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "changed after prepare"):
        manual_compaction.commit_manual_compaction(
            "thread-1",
            attempt_id = prepared["attemptId"],
            command_message_id = prepared["commandMessageId"],
            summary_message_id = "summary-1",
            expected_head_message_id = "summary-1",
            expected_revision = prepared["revision"],
            expected_summary_hash = summary_hash,
        )
    assert _raw_message_row("compact-1") == before


@pytest.mark.parametrize(
    "path",
    [
        ("threadId",),
        ("commandMessageId",),
        ("attemptId",),
        ("summaryMessageId",),
        ("expectedHeadMessageId",),
        ("sourceHeadMessageId",),
        ("envelope", "attemptId"),
        ("envelope", "threadId"),
        ("envelope", "commandMessageId"),
        ("envelope", "expectedHeadMessageId"),
    ],
    ids = lambda path: "-".join(path),
)
def test_route_maps_every_invalid_stored_client_id_to_409_without_mutation(
    tmp_path, monkeypatch, path
):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    command = studio_db.get_chat_message("thread-1", "compact-1")
    client = command["metadata"][manual_compaction.MANUAL_COMPACTION_CLIENT_KEY]
    target = client
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = "x" * 129
    _set_raw_message_json(
        "compact-1",
        "metadata_json",
        json.dumps(command["metadata"], separators = (",", ":")),
    )
    before = _raw_message_row("compact-1")

    with pytest.raises(HTTPException) as exc_info:
        chat_history.bind_thread_manual_compaction(
            "thread-1",
            chat_history.ManualCompactionBindRequest(
                attemptId = prepared["attemptId"],
                commandMessageId = prepared["commandMessageId"],
                summaryMessageId = "summary-1",
                attemptSequence = 1,
            ),
            current_subject = "test-user",
        )

    assert exc_info.value.status_code == 409
    assert "Stored manual compaction" in str(exc_info.value.detail)
    assert _raw_message_row("compact-1") == before


def test_authenticated_asgi_bind_maps_corrupt_stored_id_to_409_without_mutation(
    tmp_path, monkeypatch
):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    command = studio_db.get_chat_message("thread-1", "compact-1")
    command["metadata"][manual_compaction.MANUAL_COMPACTION_CLIENT_KEY]["envelope"]["threadId"] = (
        "x" * 129
    )
    _set_raw_message_json(
        "compact-1",
        "metadata_json",
        json.dumps(command["metadata"], separators = (",", ":")),
    )
    before = _raw_message_row("compact-1")
    app = FastAPI()
    app.include_router(chat_history.router, prefix = "/api/chat")
    app.dependency_overrides[chat_history.get_current_subject] = lambda: "test-user"

    response = TestClient(app, raise_server_exceptions = False).post(
        "/api/chat/threads/thread-1/compactions:bind",
        headers = {"Authorization": "Bearer authenticated-test-session"},
        json = {
            "attemptId": prepared["attemptId"],
            "commandMessageId": prepared["commandMessageId"],
            "summaryMessageId": "summary-1",
            "attemptSequence": 1,
        },
    )

    assert response.status_code == 409
    assert "Stored manual compaction" in response.json()["detail"]
    assert _raw_message_row("compact-1") == before


@pytest.mark.parametrize("operation", ["bind", "prepare", "commit", "cancel"])
def test_mounted_manual_compaction_routes_require_auth_and_reject_corrupt_fence_without_mutation(
    tmp_path, monkeypatch, operation
):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    claim_id = "a" * 64
    summary_text = "summary"
    if operation in ("commit", "cancel"):
        _claim(prepared, claim_id = claim_id)
    if operation == "commit":
        _record_output(prepared, summary_text)
        studio_db.upsert_chat_message(_message("summary-1", "compact-1", "assistant", summary_text))

    command = studio_db.get_chat_message("thread-1", "compact-1")
    command["metadata"][manual_compaction.MANUAL_COMPACTION_CLIENT_KEY]["envelope"]["threadId"] = (
        "x" * 129
    )
    _set_raw_message_json(
        "compact-1",
        "metadata_json",
        json.dumps(command["metadata"], separators = (",", ":")),
    )
    command_before = _raw_message_row("compact-1")
    attempt_before = _raw_attempt_row(prepared["attemptId"])
    reservation_before = _raw_reservation_row(prepared["attemptId"])
    summary_before = _raw_message_row("summary-1")

    requests = {
        "bind": {
            "attemptId": prepared["attemptId"],
            "commandMessageId": prepared["commandMessageId"],
            "summaryMessageId": "summary-1",
            "attemptSequence": 1,
        },
        "prepare": {
            "attemptId": prepared["attemptId"],
            "commandMessageId": prepared["commandMessageId"],
            "expectedHeadMessageId": prepared["expectedHeadMessageId"],
            "messageIds": ["u1", "a1", "compact-1"],
            "messages": _full_wire_messages(),
        },
        "commit": {
            "attemptId": prepared["attemptId"],
            "commandMessageId": prepared["commandMessageId"],
            "summaryMessageId": "summary-1",
            "expectedHeadMessageId": "summary-1",
            "expectedRevision": prepared["revision"],
            "summaryHash": manual_compaction.summary_hash(summary_text),
        },
        "cancel": {
            "attemptId": prepared["attemptId"],
            "commandMessageId": prepared["commandMessageId"],
            "claimId": claim_id,
        },
    }
    app = FastAPI()
    app.include_router(chat_history.router, prefix = "/api/chat")
    client = TestClient(app, raise_server_exceptions = False)
    path = f"/api/chat/threads/thread-1/compactions:{operation}"

    unauthenticated = client.post(path, json = requests[operation])
    assert unauthenticated.status_code in (401, 403)
    assert _raw_message_row("compact-1") == command_before
    assert _raw_attempt_row(prepared["attemptId"]) == attempt_before
    assert _raw_reservation_row(prepared["attemptId"]) == reservation_before
    assert _raw_message_row("summary-1") == summary_before

    app.dependency_overrides[chat_history.get_current_subject] = lambda: "test-user"
    authenticated = client.post(
        path,
        headers = {"Authorization": "Bearer authenticated-test-session"},
        json = requests[operation],
    )
    assert authenticated.status_code == 409
    assert "Stored manual compaction" in authenticated.json()["detail"]
    assert _raw_message_row("compact-1") == command_before
    assert _raw_attempt_row(prepared["attemptId"]) == attempt_before
    assert _raw_reservation_row(prepared["attemptId"]) == reservation_before
    assert _raw_message_row("summary-1") == summary_before


def test_malformed_legacy_client_metadata_fails_closed_but_preserves_exact_fence(
    tmp_path, monkeypatch
):
    _seed_branch(tmp_path, monkeypatch)
    malformed = {"schemaVersion": 1}
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "UPDATE chat_messages SET metadata_json = ? WHERE thread_id = ? AND id = ?",
            (
                json.dumps(
                    {manual_compaction.MANUAL_COMPACTION_CLIENT_KEY: malformed},
                    separators = (",", ":"),
                ),
                "thread-1",
                "compact-1",
            ),
        )
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "invalid"):
        manual_compaction.bind_manual_compaction_command(
            "thread-1",
            attempt_id = "attempt-1",
            command_message_id = "compact-1",
            summary_message_id = "summary-1",
            attempt_sequence = 1,
        )
    command = studio_db.get_chat_message("thread-1", "compact-1")
    command["metadata"]["unrelated"] = True
    before = _raw_message_row("compact-1")
    with pytest.raises(studio_db.ChatMessageProtectedError, match = "metadata is corrupt"):
        studio_db.upsert_chat_message(command)
    assert _raw_message_row("compact-1") == before
    command["metadata"][manual_compaction.MANUAL_COMPACTION_CLIENT_KEY] = {"schemaVersion": 2}
    with pytest.raises(studio_db.ChatMessageProtectedError, match = "metadata is corrupt"):
        studio_db.upsert_chat_message(command)


def test_terminal_replacement_requires_exact_sequence_and_blocks_late_owner(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    _claim(prepared, claim_id = "4" * 64)
    manual_compaction.cancel_manual_compaction(
        "thread-1",
        attempt_id = prepared["attemptId"],
        command_message_id = prepared["commandMessageId"],
        claim_id = "4" * 64,
    )
    replacement_message = manual_compaction.bind_manual_compaction_command(
        "thread-1",
        attempt_id = "attempt-2",
        command_message_id = "compact-1",
        summary_message_id = "summary-1",
        attempt_sequence = 2,
        expected_attempt_id = "attempt-1",
        expected_attempt_sequence = 1,
    )
    replacement = replacement_message["metadata"][manual_compaction.MANUAL_COMPACTION_CLIENT_KEY]
    assert replacement["state"] == "bound"
    assert replacement["attemptSequence"] == 2
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "CAS is stale"):
        manual_compaction.bind_manual_compaction_command(
            "thread-1",
            attempt_id = "attempt-1",
            command_message_id = "compact-1",
            summary_message_id = "summary-1",
            attempt_sequence = 1,
        )
    second = _prepare_core(
        "thread-1",
        attempt_id = "attempt-2",
        command_message_id = "compact-1",
        expected_head_message_id = "compact-1",
        message_ids = ["u1", "a1", "compact-1"],
        request_messages = ChatCompletionRequest(
            model = "model", messages = _full_wire_messages()
        ).messages,
    )
    assert second["state"] == "pending"
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "binding is stale"):
        manual_compaction.cancel_manual_compaction(
            "thread-1",
            attempt_id = "attempt-1",
            command_message_id = "compact-1",
            claim_id = "4" * 64,
        )
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "live /compact"):
        manual_compaction.bind_manual_compaction_command(
            "thread-1",
            attempt_id = "attempt-3",
            command_message_id = "compact-1",
            summary_message_id = "summary-1",
            attempt_sequence = 3,
            expected_attempt_id = "attempt-2",
            expected_attempt_sequence = 2,
        )


@pytest.mark.parametrize(
    ("stored_parent", "stale_parent"),
    [(None, ""), ("", None)],
    ids = ["null-to-empty", "empty-to-null"],
)
def test_fenced_root_parent_representation_is_exact_for_upsert_sync_and_digest(
    tmp_path, monkeypatch, stored_parent, stale_parent
):
    rows = _seed_branch(tmp_path, monkeypatch)
    if stored_parent == "":
        _set_raw_parent("u1", "")
    prepared = _prepare()
    before = _raw_message_row("u1")
    stale = studio_db.get_chat_message("thread-1", "u1")
    stale["parentId"] = stale_parent

    with pytest.raises(studio_db.ChatMessageProtectedError, match = "checkpoint"):
        studio_db.upsert_chat_message(stale)
    assert _raw_message_row("u1") == before

    snapshot = [studio_db.get_chat_message("thread-1", row["id"]) for row in rows]
    snapshot[0]["parentId"] = stale_parent
    with pytest.raises(studio_db.ChatMessageProtectedError, match = "checkpoint"):
        studio_db.sync_chat_messages("thread-1", snapshot)
    assert _raw_message_row("u1") == before

    stored_root = studio_db.get_chat_message("thread-1", "u1")
    stale_root = {**stored_root, "parentId": stale_parent}
    assert studio_db.canonical_chat_message_hash(
        stored_root, state = "prepared"
    ) != studio_db.canonical_chat_message_hash(stale_root, state = "prepared")

    _claim(prepared)
    assert (
        manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])["state"] == "running"
    )


@pytest.mark.parametrize("corruption", ["missing", "cycle", "cross-thread"])
def test_initial_bind_requires_complete_acyclic_same_thread_ancestry(
    tmp_path, monkeypatch, corruption
):
    _seed_branch(tmp_path, monkeypatch)
    if corruption == "cross-thread":
        studio_db.upsert_chat_thread(_thread("thread-2"))
        studio_db.upsert_chat_message(
            _message("foreign-parent", None, "user", "Foreign", thread_id = "thread-2")
        )
        invalid_parent = "foreign-parent"
    else:
        invalid_parent = "missing-parent" if corruption == "missing" else "compact-1"
    _set_raw_parent("a1", invalid_parent)
    before = _raw_message_row("compact-1")

    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "ancestry"):
        manual_compaction.bind_manual_compaction_command(
            "thread-1",
            attempt_id = "attempt-invalid",
            command_message_id = "compact-1",
            summary_message_id = "summary-invalid",
            attempt_sequence = 1,
        )

    assert _raw_message_row("compact-1") == before
    conn = studio_db.get_connection()
    try:
        assert (
            conn.execute(
                "SELECT 1 FROM manual_compaction_attempt_reservations WHERE attempt_id = ?",
                ("attempt-invalid",),
            ).fetchone()
            is None
        )
    finally:
        conn.close()


@pytest.mark.parametrize("corruption", ["missing", "cycle", "cross-thread"])
def test_terminal_replacement_requires_complete_original_ancestry(
    tmp_path, monkeypatch, corruption
):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    _claim(prepared, claim_id = "5" * 64)
    manual_compaction.cancel_manual_compaction(
        "thread-1",
        attempt_id = prepared["attemptId"],
        command_message_id = prepared["commandMessageId"],
        claim_id = "5" * 64,
    )
    if corruption == "cross-thread":
        studio_db.upsert_chat_thread(_thread("thread-2"))
        studio_db.upsert_chat_message(
            _message("foreign-parent", None, "user", "Foreign", thread_id = "thread-2")
        )
        invalid_parent = "foreign-parent"
    else:
        invalid_parent = "missing-parent" if corruption == "missing" else "compact-1"
    _set_raw_parent("a1", invalid_parent)
    command_before = _raw_message_row("compact-1")

    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "ancestry"):
        manual_compaction.bind_manual_compaction_command(
            "thread-1",
            attempt_id = "attempt-replacement",
            command_message_id = "compact-1",
            summary_message_id = "summary-replacement",
            attempt_sequence = 2,
            expected_attempt_id = prepared["attemptId"],
            expected_attempt_sequence = 1,
        )

    assert _raw_message_row("compact-1") == command_before
    assert _client_record()["attemptId"] == prepared["attemptId"]


def test_terminal_fence_blocks_full_row_edits_prune_relink_and_attachment_delete(
    tmp_path, monkeypatch
):
    rows = _seed_branch(tmp_path, monkeypatch)
    root = studio_db.get_chat_message("thread-1", "u1")
    root["attachments"] = [{"id": "file-1", "name": "notes.txt"}]
    studio_db.upsert_chat_message(root)
    prepared = _prepare()
    _claim(prepared, claim_id = "6" * 64)
    manual_compaction.cancel_manual_compaction(
        "thread-1",
        attempt_id = prepared["attemptId"],
        command_message_id = prepared["commandMessageId"],
        claim_id = "6" * 64,
    )
    command_before = _raw_message_row("compact-1")
    source_before = _raw_message_row("u1")

    command = studio_db.get_chat_message("thread-1", "compact-1")
    command.update(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "changed"}],
            "attachments": [{"id": "file-2"}],
            "createdAt": 99,
        }
    )
    with pytest.raises(studio_db.ChatMessageProtectedError, match = "checkpoint"):
        studio_db.upsert_chat_message(command)
    assert _raw_message_row("compact-1") == command_before

    snapshot = [studio_db.get_chat_message("thread-1", row["id"]) for row in rows]
    snapshot[1]["parentId"] = None
    with pytest.raises(studio_db.ChatMessageProtectedError, match = "checkpoint"):
        studio_db.sync_chat_messages("thread-1", snapshot[1:], prune_missing = True)
    assert _raw_message_row("u1") == source_before
    assert _raw_message_row("compact-1") == command_before

    with pytest.raises(studio_db.ChatMessageProtectedError):
        studio_db.delete_chat_attachment("u1", "file-1")
    assert _raw_message_row("u1") == source_before


@pytest.mark.parametrize("same_thread", [False, True], ids = ["cross-thread", "same-thread"])
@pytest.mark.parametrize("winner_is_second", [False, True], ids = ["first-wins", "second-wins"])
def test_attempt_reservation_collision_has_one_winner_and_loser_can_recover(
    tmp_path, monkeypatch, same_thread, winner_is_second
):
    _seed_branch(tmp_path, monkeypatch)
    if same_thread:
        second_thread = "thread-1"
        second_parent = "a1"
    else:
        second_thread = "thread-2"
        studio_db.upsert_chat_thread(_thread(second_thread))
        studio_db.upsert_chat_message(
            _message("u2", None, "user", "Other branch", thread_id = second_thread)
        )
        studio_db.upsert_chat_message(
            _message("a2", "u2", "assistant", "Other answer", thread_id = second_thread)
        )
        second_parent = "a2"
    studio_db.upsert_chat_message(
        _message(
            "compact-2",
            second_parent,
            "user",
            "/compact",
            thread_id = second_thread,
        )
    )

    def bind(
        thread_id,
        command_id,
        summary_id,
        attempt_id = "shared-attempt",
    ):
        return manual_compaction.bind_manual_compaction_command(
            thread_id,
            attempt_id = attempt_id,
            command_message_id = command_id,
            summary_message_id = summary_id,
            attempt_sequence = 1,
        )

    first = lambda: bind("thread-1", "compact-1", "summary-1")
    second = lambda: bind(second_thread, "compact-2", "summary-2")
    winner = second if winner_is_second else first
    loser = first if winner_is_second else second
    winner_identity = (
        (second_thread, "compact-2", "summary-2")
        if winner_is_second
        else ("thread-1", "compact-1", "summary-1")
    )
    loser_identity = (
        ("thread-1", "compact-1", "summary-1")
        if winner_is_second
        else (second_thread, "compact-2", "summary-2")
    )
    results = _run_in_writer_order(
        monkeypatch,
        "reservation-winner",
        winner,
        "reservation-loser",
        loser,
    )
    assert isinstance(results["reservation-winner"], dict)
    assert isinstance(results["reservation-loser"], manual_compaction.ManualCompactionConflict)
    assert _client_record(winner_identity[0], winner_identity[1])["attemptId"] == "shared-attempt"
    assert _client_record(loser_identity[0], loser_identity[1]) is None

    recovered = bind(*loser_identity, "recovered-attempt")
    assert recovered["metadata"][manual_compaction.MANUAL_COMPACTION_CLIENT_KEY]["attemptId"] == (
        "recovered-attempt"
    )
    conn = studio_db.get_connection()
    try:
        rows = conn.execute(
            "SELECT attempt_id, thread_id, command_message_id FROM "
            "manual_compaction_attempt_reservations ORDER BY attempt_id"
        ).fetchall()
        assert [tuple(row) for row in rows] == sorted(
            [
                ("recovered-attempt", loser_identity[0], loser_identity[1]),
                ("shared-attempt", winner_identity[0], winner_identity[1]),
            ]
        )
    finally:
        conn.close()


def test_reservation_restart_backfill_and_cross_command_summary_fence(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    manual_compaction.bind_manual_compaction_command(
        "thread-1",
        attempt_id = "attempt-1",
        command_message_id = "compact-1",
        summary_message_id = "shared-summary",
        attempt_sequence = 1,
    )
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "DELETE FROM manual_compaction_attempt_reservations WHERE attempt_id = 'attempt-1'"
        )
        conn.commit()
    finally:
        conn.close()
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    assert studio_db.get_chat_message("thread-1", "compact-1") is not None
    conn = studio_db.get_connection()
    try:
        restored = conn.execute(
            "SELECT thread_id, command_message_id, summary_message_id, attempt_sequence "
            "FROM manual_compaction_attempt_reservations WHERE attempt_id = 'attempt-1'"
        ).fetchone()
        assert tuple(restored) == ("thread-1", "compact-1", "shared-summary", 1)
    finally:
        conn.close()

    studio_db.upsert_chat_message(_message("compact-2", "a1", "user", "/compact"))
    before = _raw_message_row("compact-2")
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "Summary message id"):
        manual_compaction.bind_manual_compaction_command(
            "thread-1",
            attempt_id = "attempt-2",
            command_message_id = "compact-2",
            summary_message_id = "shared-summary",
            attempt_sequence = 1,
        )
    assert _raw_message_row("compact-2") == before


@pytest.mark.parametrize("deletion", ["thread", "project", "history"])
def test_attempt_reservation_survives_every_history_deletion_and_blocks_late_callbacks(
    tmp_path, monkeypatch, deletion
):
    _seed_branch(tmp_path, monkeypatch)
    if deletion == "project":
        studio_db.upsert_chat_project(
            {
                "id": "project-1",
                "name": "Project",
                "instructions": "",
                "archived": False,
                "createdAt": 1,
                "updatedAt": 1,
            }
        )
        studio_db.update_chat_thread("thread-1", {"projectId": "project-1"})
    prepared = _prepare()
    _claim(prepared, claim_id = "7" * 64)

    if deletion == "thread":
        studio_db.delete_chat_threads(["thread-1"])
    elif deletion == "project":
        studio_db.delete_chat_project("project-1")
    else:
        studio_db.clear_chat_history()

    assert manual_compaction.get_manual_compaction_attempt(prepared["attemptId"]) is None
    conn = studio_db.get_connection()
    try:
        reservation = conn.execute(
            "SELECT thread_id, command_message_id, retired_at "
            "FROM manual_compaction_attempt_reservations WHERE attempt_id = ?",
            (prepared["attemptId"],),
        ).fetchone()
        assert reservation is not None
        assert tuple(reservation[:2]) == ("thread-1", "compact-1")
        assert reservation["retired_at"] > 0
    finally:
        conn.close()

    studio_db.upsert_chat_thread(_thread("thread-2"))
    for message in [
        _message("u2", None, "user", "Explain the migration.", thread_id = "thread-2"),
        _message(
            "a2",
            "u2",
            "assistant",
            "Use a staged rollout.",
            thread_id = "thread-2",
        ),
        _message("compact-2", "a2", "user", "/compact", thread_id = "thread-2"),
    ]:
        studio_db.upsert_chat_message(message)
    command_before = _raw_message_row("compact-2")
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "permanently retired"):
        manual_compaction.bind_manual_compaction_command(
            "thread-2",
            attempt_id = prepared["attemptId"],
            command_message_id = "compact-2",
            summary_message_id = "summary-2",
            attempt_sequence = 1,
        )
    assert _raw_message_row("compact-2") == command_before
    assert (
        manual_compaction.fail_manual_compaction_attempt(prepared["attemptId"], "provider_failed")
        is None
    )
    with pytest.raises(manual_compaction.ManualCompactionNotFound):
        manual_compaction.record_manual_compaction_output(
            prepared["attemptId"], text = "late output", finish_reason = "stop"
        )

    recovered = manual_compaction.bind_manual_compaction_command(
        "thread-2",
        attempt_id = f"fresh-{deletion}",
        command_message_id = "compact-2",
        summary_message_id = "summary-2",
        attempt_sequence = 1,
    )
    assert recovered["metadata"][manual_compaction.MANUAL_COMPACTION_CLIENT_KEY]["attemptId"] == (
        f"fresh-{deletion}"
    )


@pytest.mark.asyncio
async def test_real_nonstream_callback_after_deletion_cannot_restore_retired_attempt(
    tmp_path, monkeypatch
):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    _claim(prepared, claim_id = "8" * 64)
    studio_db.delete_chat_threads(["thread-1"])
    reservation_before = _raw_reservation_row(prepared["attemptId"])

    response = inference.Response(
        content = json.dumps(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "late summary"},
                        "finish_reason": "stop",
                    }
                ]
            }
        ),
        media_type = "application/json",
    )
    observed = await inference._observe_manual_compaction_response(
        response,
        attempt_id = prepared["attemptId"],
        cancel_event = threading.Event(),
    )

    assert observed.status_code == 502
    assert json.loads(observed.body)["error"]["code"] == "manual_compaction_failed"
    assert _raw_reservation_row(prepared["attemptId"]) == reservation_before
    assert manual_compaction.get_manual_compaction_attempt(prepared["attemptId"]) is None


@pytest.mark.parametrize("first", ["callback", "reuse"])
def test_post_deletion_callback_and_attempt_reuse_recheck_retired_reservation_in_both_orders(
    tmp_path, monkeypatch, first
):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    _claim(prepared, claim_id = "9" * 64)
    studio_db.delete_chat_threads(["thread-1"])
    reservation_before = _raw_reservation_row(prepared["attemptId"])
    studio_db.upsert_chat_thread(_thread("thread-2"))
    for message in (
        _message("u2", None, "user", "Other question", thread_id = "thread-2"),
        _message("a2", "u2", "assistant", "Other answer", thread_id = "thread-2"),
        _message("compact-2", "a2", "user", "/compact", thread_id = "thread-2"),
    ):
        studio_db.upsert_chat_message(message)
    command_before = _raw_message_row("compact-2")

    def callback():
        return manual_compaction.record_manual_compaction_output(
            prepared["attemptId"],
            text = "late summary",
            finish_reason = "stop",
        )

    def reuse():
        return manual_compaction.bind_manual_compaction_command(
            "thread-2",
            attempt_id = prepared["attemptId"],
            command_message_id = "compact-2",
            summary_message_id = "summary-2",
            attempt_sequence = 1,
        )

    operations = {"callback": callback, "reuse": reuse}
    second = "reuse" if first == "callback" else "callback"
    results = _run_in_writer_order(
        monkeypatch,
        f"{first}-winner",
        operations[first],
        f"{second}-loser",
        operations[second],
    )

    assert isinstance(
        results["callback-winner" if first == "callback" else "callback-loser"],
        manual_compaction.ManualCompactionNotFound,
    )
    reuse_result = results["reuse-winner" if first == "reuse" else "reuse-loser"]
    assert isinstance(reuse_result, manual_compaction.ManualCompactionConflict)
    assert "permanently retired" in str(reuse_result)
    assert _raw_reservation_row(prepared["attemptId"]) == reservation_before
    assert _raw_message_row("compact-2") == command_before
    assert _client_record("thread-2", "compact-2") is None


def test_generation_existing_assistant_bind_loses_after_command_fence(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    _set_raw_message_json("a1", "content_json", "[]")
    manual_compaction.bind_manual_compaction_command(
        "thread-1",
        attempt_id = "attempt-generation",
        command_message_id = "compact-1",
        summary_message_id = "summary-generation",
        attempt_sequence = 1,
    )
    before = _raw_message_row("a1")

    with pytest.raises(
        chat_generation_runs_db.ChatGenerationConflictError, match = "Manual compaction"
    ):
        chat_generation_runs_db.create_run(
            run_id = "generation-run",
            owner_subject = "test-user",
            thread_id = "thread-1",
            user_message_id = "u1",
            assistant_message_id = "a1",
            request_payload = {"model": "model", "messages": []},
        )

    assert _raw_message_row("a1") == before
    assert chat_generation_runs_db.get_run("generation-run") is None


def test_generation_status_transition_rolls_back_after_later_command_fence(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    _set_raw_message_json("a1", "content_json", "[]")
    chat_generation_runs_db.create_run(
        run_id = "generation-run",
        owner_subject = "test-user",
        thread_id = "thread-1",
        user_message_id = "u1",
        assistant_message_id = "a1",
        request_payload = {"model": "model", "messages": []},
    )
    monkeypatch.setattr(
        manual_compaction,
        "_require_branch_lifecycle_quiescent",
        lambda *_args, **_kwargs: None,
    )
    manual_compaction.bind_manual_compaction_command(
        "thread-1",
        attempt_id = "attempt-generation",
        command_message_id = "compact-1",
        summary_message_id = "summary-generation",
        attempt_sequence = 1,
    )
    before = _raw_message_row("a1")
    worker_token = chat_generation_runs_db.get_worker_token("generation-run")

    with pytest.raises(
        chat_generation_runs_db.ChatGenerationConflictError, match = "Manual compaction"
    ):
        chat_generation_runs_db.mark_running("generation-run", worker_token)

    assert _raw_message_row("a1") == before
    assert chat_generation_runs_db.get_run("generation-run")["status"] == "queued"


def test_active_generation_writer_wins_before_bind_without_partial_fence(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    _set_raw_message_json("a1", "content_json", "[]")
    chat_generation_runs_db.create_run(
        run_id = "generation-run",
        owner_subject = "test-user",
        thread_id = "thread-1",
        user_message_id = "u1",
        assistant_message_id = "a1",
        request_payload = {"model": "model", "messages": []},
    )
    command_before = _raw_message_row("compact-1")

    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "active generation"):
        manual_compaction.bind_manual_compaction_command(
            "thread-1",
            attempt_id = "attempt-generation",
            command_message_id = "compact-1",
            summary_message_id = "summary-generation",
            attempt_sequence = 1,
        )

    assert _raw_message_row("compact-1") == command_before
    assert _client_record() is None
    worker_token = chat_generation_runs_db.get_worker_token("generation-run")
    assert chat_generation_runs_db.mark_running("generation-run", worker_token) is True
    assert chat_generation_runs_db.get_run("generation-run")["status"] == "running"


def test_research_existing_assistant_bind_loses_after_command_fence(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    _set_raw_message_json("a1", "content_json", "[]")
    manual_compaction.bind_manual_compaction_command(
        "thread-1",
        attempt_id = "attempt-research",
        command_message_id = "compact-1",
        summary_message_id = "summary-research",
        attempt_sequence = 1,
    )
    before = _raw_message_row("a1")

    with pytest.raises(research_runs_db.ResearchConflictError, match = "Manual compaction"):
        research_runs_db.create_run(
            run_id = "research-run",
            owner_subject = "test-user",
            thread_id = "thread-1",
            user_message_id = "u1",
            assistant_message_id = "a1",
            config = {"model": "model"},
            created_at = 20,
        )

    assert _raw_message_row("a1") == before
    assert research_runs_db.get_run("research-run") is None


def test_active_research_writer_wins_before_bind_without_partial_fence(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    _set_raw_message_json("a1", "content_json", "[]")
    research_runs_db.create_run(
        run_id = "research-run",
        owner_subject = "test-user",
        thread_id = "thread-1",
        user_message_id = "u1",
        assistant_message_id = "a1",
        config = {"model": "model"},
        created_at = 20,
    )
    command_before = _raw_message_row("compact-1")

    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "active research"):
        manual_compaction.bind_manual_compaction_command(
            "thread-1",
            attempt_id = "attempt-research",
            command_message_id = "compact-1",
            summary_message_id = "summary-research",
            attempt_sequence = 1,
        )

    assert _raw_message_row("compact-1") == command_before
    assert _client_record() is None
    assert research_runs_db.request_cancel("research-run") == "cancelling"


@pytest.mark.parametrize(
    ("lifecycle", "assistant_mode"),
    [
        ("generation", "new"),
        ("generation", "off-branch"),
        ("research", "new"),
        ("research", "none"),
        ("research", "off-branch"),
    ],
)
@pytest.mark.parametrize("first", ["fence", "lifecycle"])
def test_lifecycle_activation_and_command_fence_serialize_for_protected_user_ancestry(
    tmp_path, monkeypatch, lifecycle, assistant_mode, first
):
    _seed_branch(tmp_path, monkeypatch)
    assistant_id = None if assistant_mode == "none" else f"{lifecycle}-{assistant_mode}-reply"
    if assistant_mode == "off-branch":
        studio_db.upsert_chat_message(_message(assistant_id, "u1", "assistant", ""))
    run_id = f"{lifecycle}-{assistant_mode}-run"
    command_before = _raw_message_row("compact-1")
    thread_before, messages_before = _raw_thread_and_messages()

    def bind():
        return manual_compaction.bind_manual_compaction_command(
            "thread-1",
            attempt_id = f"{run_id}-attempt",
            command_message_id = "compact-1",
            summary_message_id = f"{run_id}-summary",
            attempt_sequence = 1,
        )

    if lifecycle == "generation":
        expected_conflict = chat_generation_runs_db.ChatGenerationConflictError

        def activate():
            return chat_generation_runs_db.create_run(
                run_id = run_id,
                owner_subject = "test-user",
                thread_id = "thread-1",
                user_message_id = "u1",
                assistant_message_id = assistant_id,
                request_payload = {"model": "model", "messages": []},
            )

        def assert_no_lifecycle_mutation():
            assert _raw_generation_state(run_id) == (None, [])

    else:
        expected_conflict = research_runs_db.ResearchConflictError

        def activate():
            return research_runs_db.create_run(
                run_id = run_id,
                owner_subject = "test-user",
                thread_id = "thread-1",
                user_message_id = "u1",
                assistant_message_id = assistant_id,
                config = {"model": "model"},
                created_at = 20,
            )

        def assert_no_lifecycle_mutation():
            assert _raw_research_state(run_id) == (None, [], [])

    operations = {"fence": bind, "lifecycle": activate}
    second = "lifecycle" if first == "fence" else "fence"
    results = _run_in_writer_order(
        monkeypatch,
        f"{first}-winner",
        operations[first],
        f"{second}-loser",
        operations[second],
    )

    if first == "fence":
        assert isinstance(results["fence-winner"], dict)
        assert isinstance(results["lifecycle-loser"], expected_conflict)
        assert "Manual compaction" in str(results["lifecycle-loser"])
        assert_no_lifecycle_mutation()
        thread_after, messages_after = _raw_thread_and_messages()
        assert thread_after == thread_before
        before_by_id = {row[0]: row for row in messages_before}
        after_by_id = {row[0]: row for row in messages_after}
        assert set(after_by_id) == set(before_by_id)
        for message_id, row in before_by_id.items():
            if message_id != "compact-1":
                assert after_by_id[message_id] == row
        assert _client_record()["attemptId"] == f"{run_id}-attempt"
        assert _raw_reservation_row(f"{run_id}-attempt") is not None
    else:
        assert not isinstance(results["lifecycle-winner"], Exception)
        assert isinstance(results["fence-loser"], manual_compaction.ManualCompactionConflict)
        assert _raw_message_row("compact-1") == command_before
        assert _client_record() is None
        assert _raw_reservation_row(f"{run_id}-attempt") is None


@pytest.mark.parametrize("lifecycle", ["generation", "research"])
@pytest.mark.parametrize("lifecycle_wins", [False, True], ids = ["fence-wins", "lifecycle-wins"])
def test_lifecycle_creation_and_command_bind_serialize_without_partial_authority(
    tmp_path, monkeypatch, lifecycle, lifecycle_wins
):
    _seed_branch(tmp_path, monkeypatch)
    _set_raw_message_json("a1", "content_json", "[]")
    command_before = _raw_message_row("compact-1")

    def bind():
        return manual_compaction.bind_manual_compaction_command(
            "thread-1",
            attempt_id = "attempt-lifecycle-race",
            command_message_id = "compact-1",
            summary_message_id = "summary-lifecycle-race",
            attempt_sequence = 1,
        )

    if lifecycle == "generation":
        expected_conflict = chat_generation_runs_db.ChatGenerationConflictError

        def create_lifecycle():
            return chat_generation_runs_db.create_run(
                run_id = "lifecycle-race",
                owner_subject = "test-user",
                thread_id = "thread-1",
                user_message_id = "u1",
                assistant_message_id = "a1",
                request_payload = {"model": "model", "messages": []},
            )

        get_run = chat_generation_runs_db.get_run
    else:
        expected_conflict = research_runs_db.ResearchConflictError

        def create_lifecycle():
            return research_runs_db.create_run(
                run_id = "lifecycle-race",
                owner_subject = "test-user",
                thread_id = "thread-1",
                user_message_id = "u1",
                assistant_message_id = "a1",
                config = {"model": "model"},
                created_at = 20,
            )

        get_run = research_runs_db.get_run

    winner_name = "lifecycle-winner" if lifecycle_wins else "fence-winner"
    loser_name = "fence-loser" if lifecycle_wins else "lifecycle-loser"
    winner = create_lifecycle if lifecycle_wins else bind
    loser = bind if lifecycle_wins else create_lifecycle
    results = _run_in_writer_order(
        monkeypatch,
        winner_name,
        winner,
        loser_name,
        loser,
    )

    if lifecycle_wins:
        assert not isinstance(results[winner_name], Exception)
        assert isinstance(results[loser_name], manual_compaction.ManualCompactionConflict)
        assert _client_record() is None
        assert _raw_message_row("compact-1") == command_before
        assert get_run("lifecycle-race") is not None
    else:
        assert isinstance(results[winner_name], dict)
        assert isinstance(results[loser_name], expected_conflict)
        assert _client_record()["attemptId"] == "attempt-lifecycle-race"
        assert get_run("lifecycle-race") is None

    conn = studio_db.get_connection()
    try:
        reservation = conn.execute(
            "SELECT thread_id, command_message_id FROM "
            "manual_compaction_attempt_reservations WHERE attempt_id = ?",
            ("attempt-lifecycle-race",),
        ).fetchone()
        assert (reservation is None) is lifecycle_wins
    finally:
        conn.close()


def test_research_rebind_unbind_rolls_back_after_later_command_fence(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    _set_raw_message_json("a1", "content_json", "[]")
    research_runs_db.create_run(
        run_id = "research-run",
        owner_subject = "test-user",
        thread_id = "thread-1",
        user_message_id = "u1",
        assistant_message_id = "a1",
        config = {"model": "model"},
        created_at = 20,
    )
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "UPDATE research_runs SET status = 'cancelled', cancel_requested = 1, "
            "completed_at = 21 WHERE id = 'research-run'"
        )
        conn.commit()
    finally:
        conn.close()
    manual_compaction.bind_manual_compaction_command(
        "thread-1",
        attempt_id = "attempt-research",
        command_message_id = "compact-1",
        summary_message_id = "summary-research",
        attempt_sequence = 1,
    )
    studio_db.upsert_chat_message(_message("u2", "compact-1", "user", "Next question"))
    studio_db.upsert_chat_message(_message("a2", "u2", "assistant", ""))
    before = _raw_message_row("a1")

    with pytest.raises(research_runs_db.ResearchConflictError, match = "Manual compaction"):
        research_runs_db.rebind_cancelled(
            thread_id = "thread-1",
            user_message_id = "u2",
            assistant_message_id = "a2",
            config = {"model": "model"},
        )

    assert _raw_message_row("a1") == before
    stored = research_runs_db.get_run("research-run")
    assert stored["status"] == "cancelled"
    assert stored["assistantMessageId"] == "a1"


@pytest.mark.parametrize("assistant_mode", ["none", "off-branch"])
def test_research_retry_rejects_protected_user_ancestry_without_mutation(
    tmp_path, monkeypatch, assistant_mode
):
    _seed_branch(tmp_path, monkeypatch)
    assistant_id = None if assistant_mode == "none" else "research-retry-reply"
    research_runs_db.create_run(
        run_id = "research-retry",
        owner_subject = "test-user",
        thread_id = "thread-1",
        user_message_id = "u1",
        assistant_message_id = assistant_id,
        config = {"model": "model"},
        created_at = 20,
    )
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "UPDATE research_runs SET status = 'failed', completed_at = 21, "
            "error_message = 'stopped' WHERE id = 'research-retry'"
        )
        conn.commit()
    finally:
        conn.close()
    manual_compaction.bind_manual_compaction_command(
        "thread-1",
        attempt_id = "attempt-retry-fence",
        command_message_id = "compact-1",
        summary_message_id = "summary-retry-fence",
        attempt_sequence = 1,
    )
    research_before = _raw_research_state("research-retry")
    storage_before = _raw_thread_and_messages()

    with pytest.raises(research_runs_db.ResearchConflictError, match = "Manual compaction"):
        research_runs_db.retry("research-retry")

    assert _raw_research_state("research-retry") == research_before
    assert _raw_thread_and_messages() == storage_before


@pytest.mark.parametrize("assistant_mode", ["none", "off-branch"])
def test_research_rebind_rejects_protected_target_user_without_mutation(
    tmp_path, monkeypatch, assistant_mode
):
    _seed_branch(tmp_path, monkeypatch)
    studio_db.upsert_chat_message(_message("research-old-user", None, "user", "Old question"))
    research_runs_db.create_run(
        run_id = "research-rebind",
        owner_subject = "test-user",
        thread_id = "thread-1",
        user_message_id = "research-old-user",
        assistant_message_id = None,
        config = {"model": "model"},
        created_at = 20,
    )
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "UPDATE research_runs SET status = 'cancelled', cancel_requested = 1, "
            "completed_at = 21 WHERE id = 'research-rebind'"
        )
        conn.commit()
    finally:
        conn.close()
    assistant_id = None if assistant_mode == "none" else "research-rebind-reply"
    if assistant_id is not None:
        studio_db.upsert_chat_message(_message(assistant_id, "u1", "assistant", ""))
    manual_compaction.bind_manual_compaction_command(
        "thread-1",
        attempt_id = "attempt-rebind-fence",
        command_message_id = "compact-1",
        summary_message_id = "summary-rebind-fence",
        attempt_sequence = 1,
    )
    research_before = _raw_research_state("research-rebind")
    storage_before = _raw_thread_and_messages()

    with pytest.raises(research_runs_db.ResearchConflictError, match = "Manual compaction"):
        research_runs_db.rebind_cancelled(
            thread_id = "thread-1",
            user_message_id = "u1",
            assistant_message_id = assistant_id,
            config = {"model": "model"},
        )

    assert _raw_research_state("research-rebind") == research_before
    assert _raw_thread_and_messages() == storage_before


def test_terminal_research_fallback_cannot_insert_after_user_ancestry_is_fenced(
    tmp_path, monkeypatch
):
    _seed_branch(tmp_path, monkeypatch)
    research_runs_db.create_run(
        run_id = "research-fallback",
        owner_subject = "test-user",
        thread_id = "thread-1",
        user_message_id = "u1",
        assistant_message_id = None,
        config = {"model": "model"},
        created_at = 20,
    )
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "UPDATE research_runs SET status = 'failed', completed_at = 21, "
            "error_message = 'stopped' WHERE id = 'research-fallback'"
        )
        conn.commit()
    finally:
        conn.close()
    manual_compaction.bind_manual_compaction_command(
        "thread-1",
        attempt_id = "attempt-fallback-fence",
        command_message_id = "compact-1",
        summary_message_id = "summary-fallback-fence",
        attempt_sequence = 1,
    )
    research_before = _raw_research_state("research-fallback")
    storage_before = _raw_thread_and_messages()

    with pytest.raises(research_runs_db.ResearchConflictError, match = "Manual compaction"):
        research_runs_db.create_and_bind_terminal_fallback(
            "research-fallback",
            text = "Failed safely",
            status = "failed",
        )

    assert _raw_research_state("research-fallback") == research_before
    assert _raw_thread_and_messages() == storage_before
    assert studio_db.get_chat_message("thread-1", "research-research-fallback") is None


@pytest.mark.parametrize("operation", ["direct-discovery", "terminal-fallback"])
@pytest.mark.parametrize("child_state", ["discovered", "already-bound"])
def test_existing_research_child_discovery_fails_closed_after_user_ancestry_is_fenced(
    tmp_path, monkeypatch, operation, child_state
):
    _seed_branch(tmp_path, monkeypatch)
    run_id = f"research-{child_state}"
    child_id = f"{run_id}-reply"
    research_runs_db.create_run(
        run_id = run_id,
        owner_subject = "test-user",
        thread_id = "thread-1",
        user_message_id = "u1",
        assistant_message_id = child_id if child_state == "already-bound" else None,
        config = {"model": "model"},
        created_at = 20,
    )
    if child_state == "discovered":
        studio_db.upsert_chat_message(
            _message(
                child_id,
                "u1",
                "assistant",
                "",
                metadata = {"researchRunId": run_id},
            )
        )
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "UPDATE research_runs SET status = 'failed', completed_at = 21, "
            "error_message = 'stopped' WHERE id = ?",
            (run_id,),
        )
        conn.commit()
    finally:
        conn.close()
    manual_compaction.bind_manual_compaction_command(
        "thread-1",
        attempt_id = f"attempt-{child_state}-discovery-fence",
        command_message_id = "compact-1",
        summary_message_id = f"summary-{child_state}-discovery-fence",
        attempt_sequence = 1,
    )
    research_before = _raw_research_state(run_id)
    storage_before = _raw_thread_and_messages()

    with pytest.raises(research_runs_db.ResearchConflictError, match = "Manual compaction"):
        if operation == "direct-discovery":
            research_runs_db.discover_and_bind_assistant_message(run_id)
        else:
            research_runs_db.create_and_bind_terminal_fallback(
                run_id,
                text = "Failed safely",
                status = "failed",
            )

    assert _raw_research_state(run_id) == research_before
    assert _raw_thread_and_messages() == storage_before


def test_legacy_active_checkpoint_does_not_block_a_distinct_new_command(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    _claim(prepared)
    _record_output(prepared, "old summary")
    studio_db.upsert_chat_message(_message("summary-1", "compact-1", "assistant", "old summary"))
    active = manual_compaction.commit_manual_compaction(
        "thread-1",
        attempt_id = prepared["attemptId"],
        command_message_id = prepared["commandMessageId"],
        summary_message_id = "summary-1",
        expected_head_message_id = "summary-1",
        expected_revision = prepared["revision"],
        expected_summary_hash = manual_compaction.summary_hash("old summary"),
    )
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "UPDATE chat_messages SET metadata_json = NULL WHERE thread_id = ? AND id = ?",
            ("thread-1", "compact-1"),
        )
        conn.commit()
    finally:
        conn.close()
    studio_db.upsert_chat_message(_message("compact-2", "summary-1", "user", "/compact"))
    next_attempt = _prepare_bound(
        "thread-1",
        attempt_id = "attempt-2",
        command_message_id = "compact-2",
        expected_head_message_id = "compact-2",
        message_ids = ["u1", "a1", "compact-1", "summary-1", "compact-2"],
        request_messages = [
            {"role": "system", "content": "Project rules"},
            {"role": "assistant", "content": "old summary"},
            {"role": "user", "content": "/compact"},
        ],
        summary_message_id = "summary-2",
    )
    assert active["state"] == "active"
    assert next_attempt["state"] == "pending"
    assert next_attempt["revision"] == 2
    assert manual_compaction.get_manual_compaction_attempt(active["attemptId"])["state"] == "active"
