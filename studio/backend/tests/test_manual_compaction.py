# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from core import manual_compaction
from core.rag import conversation_archive
from models.inference import ChatCompletionRequest
from routes import chat_history, inference
from storage import studio_db


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


def _prepare(messages = None):
    request_messages = ChatCompletionRequest(
        model = "model", messages = messages or _full_wire_messages()
    ).messages
    return manual_compaction.prepare_manual_compaction(
        "thread-1",
        attempt_id = "attempt-1",
        command_message_id = "compact-1",
        expected_head_message_id = "compact-1",
        message_ids = ["u1", "a1", "compact-1"],
        request_messages = request_messages,
    )


def _request(prepared, messages = None):
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


def _claim(prepared, messages = None):
    request = _request(prepared, messages = messages)
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

    prepared = manual_compaction.prepare_manual_compaction(
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

    prepared = manual_compaction.prepare_manual_compaction(
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
    prepared = manual_compaction.prepare_manual_compaction(
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

    prepared = manual_compaction.prepare_manual_compaction(
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
        manual_compaction.prepare_manual_compaction(
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
        manual_compaction.prepare_manual_compaction(
            "thread-1",
            attempt_id = "cross-thread",
            command_message_id = "foreign",
            expected_head_message_id = "foreign",
            message_ids = ["u1", "foreign"],
            request_messages = _full_wire_messages(),
        )
    monkeypatch.setattr(manual_compaction, "MAX_MANUAL_COMPACTION_MESSAGES", 2)
    with pytest.raises(manual_compaction.ManualCompactionError, match = "2 to 2"):
        manual_compaction.prepare_manual_compaction(
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
    replacement = manual_compaction.prepare_manual_compaction(
        "thread-1",
        attempt_id = "attempt-same-branch",
        command_message_id = "compact-1",
        expected_head_message_id = "compact-1",
        message_ids = ["u1", "a1", "compact-1"],
        request_messages = ChatCompletionRequest(
            model = "model", messages = _full_wire_messages()
        ).messages,
    )
    assert replacement["state"] == "pending"
    assert manual_compaction.get_manual_compaction_attempt("attempt-1")["state"] == "cancelled"
    studio_db.upsert_chat_message(_message("compact-other", "a1", "user", "/compact"))
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "another branch"):
        manual_compaction.prepare_manual_compaction(
            "thread-1",
            attempt_id = "attempt-1",
            command_message_id = "compact-other",
            expected_head_message_id = "compact-other",
            message_ids = ["u1", "a1", "compact-other"],
            request_messages = _full_wire_messages(),
        )
    studio_db.upsert_chat_message(_message("compact-1", "a1", "user", "/compact now"))
    with pytest.raises(manual_compaction.ManualCompactionConflict, match = "literal"):
        manual_compaction.prepare_manual_compaction(
            "thread-1",
            attempt_id = "not-literal",
            command_message_id = "compact-1",
            expected_head_message_id = "compact-1",
            message_ids = ["u1", "a1", "compact-1"],
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
        manual_compaction.prepare_manual_compaction(
            "thread-1",
            attempt_id = "forged-search-images",
            command_message_id = "compact-1",
            expected_head_message_id = "compact-1",
            message_ids = ["u1", "a1", "compact-1"],
            request_messages = forged,
        )


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
            manual_compaction.prepare_manual_compaction(
                "thread-1",
                attempt_id = "forged-wrapper",
                command_message_id = "compact-1",
                expected_head_message_id = "compact-1",
                message_ids = ["u1", "a1", "compact-1"],
                request_messages = forged,
            )


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
    studio_db.upsert_chat_message(assistant, allow_generation_edit = True)
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
        manual_compaction.prepare_manual_compaction(
            "thread-1",
            attempt_id = "altered-project",
            command_message_id = "compact-1",
            expected_head_message_id = "compact-1",
            message_ids = ["u1", "a1", "compact-1"],
            request_messages = ChatCompletionRequest(model = "model", messages = changed).messages,
        )


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
        manual_compaction.prepare_manual_compaction(
            "thread-1",
            attempt_id = "attachment-drift",
            command_message_id = "compact-1",
            expected_head_message_id = "compact-1",
            message_ids = ["u1", "a1", "compact-1"],
            request_messages = changed,
        )


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

    studio_db.upsert_chat_message(_message("a1", "u1", "assistant", "Changed answer."))
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
    prepared = _prepare()
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
    next_attempt = manual_compaction.prepare_manual_compaction(
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
    refreshed = manual_compaction.prepare_manual_compaction(
        "thread-1",
        attempt_id = "attempt-refreshed",
        command_message_id = "compact-1",
        expected_head_message_id = "compact-1",
        message_ids = ["u1", "a1", "compact-1"],
        request_messages = messages,
    )
    assert manual_compaction.get_manual_compaction_attempt("attempt-1")["state"] == "cancelled"
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

    prepared = manual_compaction.prepare_manual_compaction(
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
        return manual_compaction.prepare_manual_compaction(
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
    assert all(isinstance(result, dict) for result in prepared_results)
    stored = [
        manual_compaction.get_manual_compaction_attempt(result["attemptId"])
        for result in prepared_results
    ]
    live = [attempt for attempt in stored if attempt["state"] == "pending"]
    cancelled = [attempt for attempt in stored if attempt["state"] == "cancelled"]
    assert len(live) == 1
    assert len(cancelled) == 1

    winner = live[0]
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


def test_pending_source_drift_is_replaced_without_bricking_the_branch(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    first = _prepare()
    user = studio_db.get_chat_message("thread-1", "u1")
    user["content"] = [{"type": "text", "text": "Explain the revised migration."}]
    studio_db.upsert_chat_message(user)
    messages = [
        {"role": "system", "content": "Project rules"},
        {"role": "user", "content": "Explain the revised migration."},
        {"role": "assistant", "content": "Use a staged rollout."},
        {"role": "user", "content": "/compact"},
    ]

    replacement = manual_compaction.prepare_manual_compaction(
        "thread-1",
        attempt_id = "attempt-2",
        command_message_id = "compact-1",
        expected_head_message_id = "compact-1",
        message_ids = ["u1", "a1", "compact-1"],
        request_messages = messages,
    )

    assert replacement["state"] == "pending"
    assert replacement["sourceHash"] != first["sourceHash"]
    assert manual_compaction.get_manual_compaction_attempt("attempt-1")["state"] == "cancelled"
    _claim(replacement, messages = messages)


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

    replacement = manual_compaction.prepare_manual_compaction(
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
        return manual_compaction.prepare_manual_compaction(
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


def test_expired_replacement_wins_writer_lock_over_late_commit(tmp_path, monkeypatch):
    _seed_branch(tmp_path, monkeypatch)
    prepared = _prepare()
    _claim(prepared)
    running = manual_compaction.get_manual_compaction_attempt("attempt-1")
    monkeypatch.setattr(
        manual_compaction.time,
        "time",
        lambda: (running["leaseExpiresAt"] + 1) / 1000,
    )

    def replace():
        return manual_compaction.prepare_manual_compaction(
            "thread-1",
            attempt_id = "attempt-2",
            command_message_id = "compact-1",
            expected_head_message_id = "compact-1",
            message_ids = ["u1", "a1", "compact-1"],
            request_messages = _full_wire_messages(),
        )

    def store_late_summary_then_commit():
        studio_db.upsert_chat_message(_message("summary-1", "compact-1", "assistant", "summary"))
        return manual_compaction.commit_manual_compaction(
            "thread-1",
            attempt_id = "attempt-1",
            command_message_id = "compact-1",
            summary_message_id = "summary-1",
            expected_head_message_id = "summary-1",
            expected_revision = 1,
            expected_summary_hash = manual_compaction.summary_hash("summary"),
        )

    results = _run_in_writer_order(
        monkeypatch,
        "replacement-winner",
        replace,
        "late-response-loser",
        store_late_summary_then_commit,
    )

    assert results["replacement-winner"]["state"] == "pending"
    assert isinstance(results["late-response-loser"], manual_compaction.ManualCompactionConflict)
    assert "failed" in str(results["late-response-loser"])
    assert manual_compaction.get_manual_compaction_attempt("attempt-2")["state"] == "pending"
    assert manual_compaction.get_manual_compaction_attempt("attempt-1")["state"] == "failed"


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
    replacement = manual_compaction.prepare_manual_compaction(
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
    nested_prepare = manual_compaction.prepare_manual_compaction(
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


def test_schema_migrates_the_pre_lifecycle_compaction_table(tmp_path, monkeypatch):
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
    assert migrated["state"] == "pending"
    assert migrated["startedAt"] is None
    assert migrated["leaseExpiresAt"] is None
    assert migrated["archivePayload"] == []
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
