# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import inspect
import json
import os
import re
import sqlite3
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from core.inference.llama_server_args import BATCH_MAX
from routes import chat_history


def _message(message_id: str, thread_id: str) -> chat_history.ChatMessage:
    return chat_history.ChatMessage(
        id = message_id,
        threadId = thread_id,
        parentId = None,
        role = "user",
        content = [{"type": "text", "text": "hello"}],
        createdAt = 1_700_000_000_000,
    )


def test_async_delete_handlers_dispatch_sqlite_to_the_threadpool():
    """Cleanup handlers may be async, but their blocking sqlite work must leave the event loop."""
    coroutine_handlers = sorted(
        route.endpoint.__name__
        for route in chat_history.router.routes
        if inspect.iscoroutinefunction(route.endpoint)
    )
    assert coroutine_handlers == ["clear_history", "delete_project", "delete_threads"]
    for handler in (
        chat_history.clear_history,
        chat_history.delete_project,
        chat_history.delete_threads,
    ):
        assert "run_in_threadpool" in inspect.getsource(handler)


def test_replace_thread_messages_rejects_body_thread_mismatch(monkeypatch):
    called = False

    def fake_get_chat_thread(thread_id: str):
        return {"id": thread_id}

    def fake_sync_chat_messages(*args, **kwargs):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(chat_history, "get_chat_thread", fake_get_chat_thread)
    monkeypatch.setattr(chat_history, "sync_chat_messages", fake_sync_chat_messages)

    with pytest.raises(HTTPException) as exc_info:
        chat_history.replace_thread_messages(
            "thread-1",
            chat_history.ChatMessageSyncRequest(
                messages = [_message("msg-1", "thread-2")],
                pruneMissing = True,
            ),
            current_subject = "test-user",
        )

    assert exc_info.value.status_code == 400
    assert "Message threadId mismatch" in str(exc_info.value.detail)
    assert called is False


def test_replace_thread_messages_reports_protected_research_turn(monkeypatch):
    monkeypatch.setattr(chat_history, "get_chat_thread", lambda _thread_id: {"id": "thread-1"})

    def reject_prune(*_args, **_kwargs):
        raise chat_history.ChatMessageProtectedError(
            "Research prompts and responses cannot be deleted from their original thread"
        )

    monkeypatch.setattr(chat_history, "sync_chat_messages", reject_prune)

    with pytest.raises(HTTPException) as exc_info:
        chat_history.replace_thread_messages(
            "thread-1",
            chat_history.ChatMessageSyncRequest(messages = [], pruneMissing = True),
            current_subject = "test-user",
        )

    assert exc_info.value.status_code == 409
    assert "Research prompts and responses" in str(exc_info.value.detail)


def test_save_thread_message_forwards_explicit_generation_edit(monkeypatch):
    monkeypatch.setattr(chat_history, "get_chat_thread", lambda _thread_id: {"id": "thread-1"})
    captured = {}

    def save(message, *, allow_generation_edit = False):
        captured["allow_generation_edit"] = allow_generation_edit
        return message

    monkeypatch.setattr(chat_history, "upsert_chat_message", save)
    payload = _message("assistant-1", "thread-1").model_copy(update = {"role": "assistant"})
    chat_history.save_thread_message(
        "thread-1",
        "assistant-1",
        payload,
        allow_generation_edit = True,
        current_subject = "test-user",
    )
    assert captured == {"allow_generation_edit": True}


def test_save_thread_message_returns_404_when_thread_is_deleted_during_write(monkeypatch):
    parent_reads = iter(({"id": "thread-1"}, None))
    monkeypatch.setattr(chat_history, "get_chat_thread", lambda _thread_id: next(parent_reads))

    def missing_parent(*_args, **_kwargs):
        raise sqlite3.IntegrityError("FOREIGN KEY constraint failed")

    monkeypatch.setattr(chat_history, "upsert_chat_message", missing_parent)

    with pytest.raises(HTTPException) as exc_info:
        chat_history.save_thread_message(
            "thread-1",
            "msg-1",
            _message("msg-1", "thread-1"),
            current_subject = "test-user",
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Thread thread-1 not found"


def test_replace_thread_messages_returns_404_when_thread_is_deleted_during_write(monkeypatch):
    parent_reads = iter(({"id": "thread-1"}, None))
    monkeypatch.setattr(chat_history, "get_chat_thread", lambda _thread_id: next(parent_reads))

    def missing_parent(*_args, **_kwargs):
        raise sqlite3.IntegrityError("FOREIGN KEY constraint failed")

    monkeypatch.setattr(chat_history, "sync_chat_messages", missing_parent)

    with pytest.raises(HTTPException) as exc_info:
        chat_history.replace_thread_messages(
            "thread-1",
            chat_history.ChatMessageSyncRequest(
                messages = [_message("msg-1", "thread-1")],
                pruneMissing = True,
            ),
            current_subject = "test-user",
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Thread thread-1 not found"


def test_save_thread_message_does_not_mask_an_unrelated_integrity_error(monkeypatch):
    monkeypatch.setattr(
        chat_history,
        "get_chat_thread",
        lambda _thread_id: {"id": "thread-1"},
    )
    failure = sqlite3.IntegrityError("unrelated constraint")

    def raise_unrelated(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(chat_history, "upsert_chat_message", raise_unrelated)

    with pytest.raises(sqlite3.IntegrityError) as exc_info:
        chat_history.save_thread_message(
            "thread-1",
            "msg-1",
            _message("msg-1", "thread-1"),
            current_subject = "test-user",
        )

    assert exc_info.value is failure


def test_save_thread_distinguishes_a_tombstone_from_an_unknown_id(monkeypatch):
    def reject_deleted_thread(_thread, **_kwargs):
        raise chat_history.ChatThreadDeletedError("thread-1")

    monkeypatch.setattr(chat_history, "upsert_chat_thread", reject_deleted_thread)
    payload = chat_history.ChatThread(
        id = "thread-1",
        title = "Deleted",
        modelType = "base",
        modelId = "model-1",
        createdAt = 1,
    )

    with pytest.raises(HTTPException) as exc_info:
        chat_history.save_thread(payload, current_subject = "test-user")

    assert exc_info.value.status_code == 410
    assert exc_info.value.detail == "Thread thread-1 was deleted"


def test_chat_thread_payload_carries_gguf_variant():
    thread = chat_history.ChatThread(
        id = "thread-1",
        title = "GGUF chat",
        modelType = "base",
        modelId = "unsloth/Qwen3-GGUF",
        modelGgufVariant = "Q6_K",
        createdAt = 1,
    )
    patch = chat_history.ChatThreadPatch(modelGgufVariant = "Q8_0")

    assert thread.model_dump()["modelGgufVariant"] == "Q6_K"
    assert patch.model_dump(exclude_unset = True) == {"modelGgufVariant": "Q8_0"}


def test_clear_history_fences_pending_thread_ids(monkeypatch):
    captured: list[str] = []
    captured_operation_ids: list[str | None] = []

    def clear_with_ids(
        thread_ids = (),
        operation_id = None,
        include_chat_generation_runs = False,
        hook_session_ledger = None,
    ):
        captured.extend(thread_ids)
        captured_operation_ids.append(operation_id)
        result = (list(thread_ids), [])
        if include_chat_generation_runs:
            response = (*result, [], False)
        else:
            response = (*result, False)
        return (*response, []) if hook_session_ledger is not None else response

    async def remove_sandboxes(_thread_ids, _delete_files):
        return 0, []

    monkeypatch.setattr(chat_history, "clear_chat_history_with_replay_status", clear_with_ids)
    monkeypatch.setattr(chat_history, "_remove_sandboxes", remove_sandboxes)
    monkeypatch.setattr(chat_history, "_cancel_active_generations", lambda _ids: None)
    monkeypatch.setattr(chat_history, "_cancel_research_runs", lambda _request, _ids: None)
    request = SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace()))

    response = asyncio.run(
        chat_history.clear_history(
            request,
            chat_history.ChatClearRequest(ids = ["pending-thread"], operationId = "clear-operation-1"),
            current_subject = "test-user",
        )
    )

    assert response == {
        "status": "deleted",
        "deletedThreadIds": ["pending-thread"],
        "sandboxes_removed": 0,
        "sandboxes_kept": [],
    }
    assert captured == ["pending-thread"]
    assert captured_operation_ids == ["clear-operation-1"]


def test_clear_history_reaps_search_thumbnails_with_a_body(monkeypatch):
    """DELETE /api/chat is clear-all either way, and the frontend always sends a body.

    Gating the thumbnail reap on `payload is None` meant it never ran, so "Clear all
    chats" left every cached thumbnail — which says what was searched for — on disk.
    """
    from core.inference import search_images

    reaped: list[bool] = []
    monkeypatch.setattr(search_images, "clear_cache", lambda only_ids = None: reaped.append(True))

    async def remove_sandboxes(_thread_ids, _delete_files):
        return 0, []

    monkeypatch.setattr(
        chat_history,
        "clear_chat_history_with_replay_status",
        lambda thread_ids = (), operation_id = None, include_chat_generation_runs = False, hook_session_ledger = None: (
            ([], [], [], False, [])
            if include_chat_generation_runs and hook_session_ledger is not None
            else ([], [], [], False)
            if include_chat_generation_runs
            else ([], [], False, [])
            if hook_session_ledger is not None
            else ([], [], False)
        ),
    )
    monkeypatch.setattr(chat_history, "_remove_sandboxes", remove_sandboxes)
    monkeypatch.setattr(chat_history, "_cancel_active_generations", lambda _ids: None)
    monkeypatch.setattr(chat_history, "_cancel_research_runs", lambda _request, _ids: None)
    monkeypatch.setattr(
        chat_history, "_remove_conversation_archives", lambda _ids, cutoff = None: None
    )
    request = SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace()))

    asyncio.run(
        chat_history.clear_history(
            request,
            chat_history.ChatClearRequest(ids = [], operationId = "clear-operation-2"),
            current_subject = "test-user",
        )
    )

    assert reaped == [True]


def test_project_delete_cancels_research_before_workspace_cleanup(monkeypatch):
    project = {
        "id": "project-1",
        "name": "Project",
        "createdAt": 1,
        "updatedAt": 1,
        "memberIds": ["thread-1"],
        "activeResearchRunIds": ["run-1"],
    }
    cancelled: list[str] = []
    monkeypatch.setattr(
        chat_history,
        "delete_chat_project",
        lambda _project_id, delete_files = False, **_kwargs: project,
    )
    monkeypatch.setattr(
        chat_history,
        "_cancel_research_runs",
        lambda _request, run_ids: cancelled.extend(run_ids),
    )
    monkeypatch.setattr(chat_history, "_cancel_active_generations", lambda _ids: None)

    async def fail_workspace_cleanup(_ids, _delete_files):
        raise OSError("workspace is busy")

    monkeypatch.setattr(chat_history, "_remove_sandboxes", fail_workspace_cleanup)

    with pytest.raises(OSError, match = "workspace is busy"):
        asyncio.run(
            chat_history.delete_project(
                "project-1",
                SimpleNamespace(),
                delete_files = True,
                current_subject = "test-user",
            )
        )

    assert cancelled == ["run-1"]


def test_project_delete_always_ends_captured_hook_sessions(monkeypatch):
    captured = [{"project_id": "project-1", "session_id": "thread-1"}]
    project = {
        "id": "project-1",
        "name": "Project",
        "createdAt": 1,
        "updatedAt": 1,
        "memberIds": ["thread-1"],
        "activeResearchRunIds": ["run-1"],
        "hookSessions": captured,
    }
    ended = []
    monkeypatch.setattr(
        chat_history,
        "delete_chat_project",
        lambda _project_id, delete_files = False, **_kwargs: project,
    )
    monkeypatch.setattr(
        chat_history,
        "_cancel_research_runs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("cancel failed")),
    )
    monkeypatch.setattr(
        chat_history,
        "_end_project_hook_sessions",
        lambda records, **kwargs: ended.append((records, kwargs["reason"])),
    )

    with pytest.raises(RuntimeError, match = "cancel failed"):
        asyncio.run(
            chat_history.delete_project(
                "project-1",
                SimpleNamespace(),
                current_subject = "test-user",
            )
        )

    assert ended == [(captured, "delete")]


@pytest.mark.parametrize("outbox_state", ["failed", "recovery-owned"])
def test_project_delete_retains_managed_root_until_session_end_consumption(
    tmp_path, monkeypatch, outbox_state
):
    from core.inference import tools
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio-home"))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "studio-home" / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    snapshot = json.dumps(
        {
            "token": {
                "project_id": "project-1",
                "session_id": "thread-1",
                "generation": "generation-1",
            }
        }
    )
    connection = studio_db.get_connection()
    try:
        studio_db.enqueue_project_hook_session_end_outbox(
            connection,
            [("generation-1:SessionEnd", snapshot, "project")],
        )
        connection.commit()
    finally:
        connection.close()
    if outbox_state == "failed":
        assert studio_db.mark_project_hook_session_end_outbox_failed(
            "generation-1:SessionEnd",
            "finalizer failed",
        )
    else:
        assert studio_db.claim_project_hook_session_end_outbox(
            "generation-1:SessionEnd",
            "recovery-worker",
        )

    managed_root = tmp_path / "Projects" / "project-project-1"
    managed_root.mkdir(parents = True)
    project = {
        "id": "project-1",
        "name": "Project",
        "createdAt": 1,
        "updatedAt": 1,
        "workspaceKind": "managed",
        "managedRootPath": str(managed_root),
        "rootPath": str(managed_root),
        "memberIds": ["thread-1"],
        "activeResearchRunIds": [],
        "activeChatGenerationRunIds": [],
        "hookSessions": [],
    }
    deleted = []
    orphaned = []
    deferred = []
    monkeypatch.setattr(
        chat_history,
        "delete_chat_project",
        lambda _project_id, delete_files = False, **_kwargs: project,
    )
    monkeypatch.setattr(chat_history, "get_chat_project", lambda _project_id: None)
    monkeypatch.setattr(chat_history, "_delete_project_rag_sources", lambda _id: None)
    monkeypatch.setattr(chat_history, "_cancel_research_runs", lambda *_args: None)
    monkeypatch.setattr(chat_history, "_cancel_chat_generation_runs", lambda *_args: None)
    monkeypatch.setattr(chat_history, "_cancel_active_generations", lambda *_args: None)
    monkeypatch.setattr(
        chat_history, "_remove_conversation_archives", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(chat_history, "_end_project_hook_sessions", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tools, "wait_for_sessions_idle", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(tools, "live_project_owns", lambda *_args: False)
    monkeypatch.setattr(
        tools,
        "record_orphaned_project",
        lambda *args: orphaned.append(args),
    )
    monkeypatch.setattr(
        tools,
        "finish_workspace_delete_when_idle",
        lambda project_id: deferred.append(project_id),
    )
    monkeypatch.setattr(
        studio_db,
        "delete_project_workspace",
        lambda value: deleted.append(value),
    )
    monkeypatch.setattr(
        studio_db,
        "sandbox_is_referenced_elsewhere",
        lambda *_args: False,
    )

    async def remove_sandboxes(_thread_ids, _delete_files):
        return 0, []

    monkeypatch.setattr(chat_history, "_remove_sandboxes", remove_sandboxes)
    request = SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace()))

    asyncio.run(
        chat_history.delete_project(
            "project-1",
            request,
            delete_files = True,
            current_subject = "test-user",
        )
    )

    assert deleted == []
    assert orphaned == [("project-1", str(managed_root / "sandbox"), True, str(managed_root))]
    assert deferred == ["project-1"]


# ---------------------------------------------------------------------------
# /api/chat/settings
# ---------------------------------------------------------------------------


def test_chat_settings_payload_accepts_fast_mode_presets():
    payload = chat_history.ChatSettingsPayload.model_validate(
        {
            "inferenceParams": {"fastMode": False},
            "customPresets": [
                {
                    "name": "Fast Opus",
                    "params": {
                        "temperature": 0.6,
                        "topP": 0.95,
                        "topK": 20,
                        "minP": 0.01,
                        "repetitionPenalty": 1.0,
                        "presencePenalty": 0.0,
                        "maxTokens": 8192,
                        "systemPrompt": "",
                        "trustRemoteCode": False,
                        "fastMode": True,
                    },
                },
            ],
        }
    )

    dumped = payload.model_dump(exclude_unset = True)
    assert dumped["inferenceParams"]["fastMode"] is False
    assert dumped["customPresets"][0]["params"]["fastMode"] is True


def test_chat_settings_payload_carries_per_model_params():
    """The payload is extra="forbid", so per-model memory only reaches the DB if the
    schema names both keys. A patch also has to survive exclude_unset with just the
    one model it touched, since that is what keeps other models' settings intact."""
    payload = chat_history.ChatSettingsPayload.model_validate(
        {
            "rememberParamsPerModel": True,
            "inferenceParamsByModel": {
                "unsloth/Qwen3.5-9B-GGUF": {"temperature": 0.2, "maxTokens": 4096},
            },
        }
    )

    dumped = payload.model_dump(exclude_unset = True)
    assert dumped["rememberParamsPerModel"] is True
    assert dumped["inferenceParamsByModel"] == {
        "unsloth/Qwen3.5-9B-GGUF": {"temperature": 0.2, "maxTokens": 4096},
    }
    # Nothing else is implied by the patch, so the merge cannot clobber it.
    assert "inferenceParams" not in dumped


def test_chat_settings_payload_rejects_junk_per_model_params():
    """Provider-qualified ids are opaque keys, but the values are still real
    inference settings -- an unknown field must 400 rather than reach the DB."""
    with pytest.raises(ValidationError):
        chat_history.ChatSettingsPayload.model_validate(
            {"inferenceParamsByModel": {"openai:gpt-x": {"notAParam": 1}}}
        )


def test_chat_settings_payload_accepts_preset_load_config():
    payload = chat_history.ChatSettingsPayload.model_validate(
        {
            "customPresets": [
                {
                    "name": "GGUF preset",
                    "params": {"temperature": 0.7, "maxTokens": 512},
                    "loadConfig": {
                        "customContextLength": 256,
                        "kvCacheDtype": "q8_0",
                        "tensorParallel": False,
                    },
                },
            ],
        }
    )

    dumped = payload.model_dump(exclude_unset = True)
    assert dumped["customPresets"][0]["loadConfig"]["customContextLength"] == 256
    assert dumped["customPresets"][0]["loadConfig"]["kvCacheDtype"] == "q8_0"


def test_chat_settings_payload_accepts_preset_batch_sizes():
    from pydantic import ValidationError

    # extra="forbid" 400s the whole settings write, and the normalizer emits both keys on
    # every preset (null included), so a preset that only pinned nParallel would stop saving.
    payload = chat_history.ChatSettingsPayload.model_validate(
        {
            "customPresets": [
                {
                    "name": "batch preset",
                    "params": {"temperature": 0.7},
                    "loadConfig": {"nParallel": 4, "nBatch": 4096, "nUbatch": 1024},
                },
            ],
        }
    )
    dumped = payload.model_dump(exclude_unset = True)
    assert dumped["customPresets"][0]["loadConfig"]["nBatch"] == 4096
    assert dumped["customPresets"][0]["loadConfig"]["nUbatch"] == 1024

    # The unset shape the normalizer sends alongside an untouched knob.
    chat_history.ChatPresetLoadConfig.model_validate(
        {"nParallel": 4, "nBatch": None, "nUbatch": None}
    )
    for bad in ({"nBatch": 0}, {"nUbatch": BATCH_MAX + 1}, {"nBatch": True}):
        with pytest.raises(ValidationError):
            chat_history.ChatPresetLoadConfig.model_validate(bad)


def test_chat_settings_payload_accepts_mlx_kv_bits():
    from pydantic import ValidationError

    # extra="forbid" rejects the whole settings write on an undeclared key.
    payload = chat_history.ChatSettingsPayload.model_validate(
        {
            "customPresets": [
                {
                    "name": "MLX preset",
                    "params": {"temperature": 0.7},
                    "loadConfig": {"mlxKvBits": 8},
                },
            ],
        }
    )
    dumped = payload.model_dump(exclude_unset = True)
    assert dumped["customPresets"][0]["loadConfig"]["mlxKvBits"] == 8

    for width in (4, None):
        chat_history.ChatPresetLoadConfig.model_validate({"mlxKvBits": width})
    # Only the widths MLX supports.
    with pytest.raises(ValidationError):
        chat_history.ChatPresetLoadConfig.model_validate({"mlxKvBits": 7})


def test_chat_settings_payload_accepts_nudge_tool_calls():
    # extra="forbid" 400s PUT /api/chat/settings on unknown keys, so the
    # frontend's persisted nudgeToolCalls needs a payload field (like
    # autoHealToolCalls).
    payload = chat_history.ChatSettingsPayload.model_validate(
        {"autoHealToolCalls": True, "nudgeToolCalls": False}
    )
    dumped = payload.model_dump(exclude_unset = True)
    assert dumped == {"autoHealToolCalls": True, "nudgeToolCalls": False}


def test_chat_inference_settings_covers_frontend_persisted_fields():
    # Drift guard: every InferenceParams field the UI persists (all but
    # checkpoint) must exist on ChatInferenceSettings, else extra="forbid"
    # 400s PUT /api/chat/settings on the next added field (issue #5862).
    runtime_ts = os.path.join(
        _backend,
        "..",
        "frontend",
        "src",
        "features",
        "chat",
        "types",
        "runtime.ts",
    )
    if not os.path.exists(runtime_ts):
        pytest.skip("frontend runtime.ts not present")

    with open(runtime_ts, encoding = "utf-8") as fh:
        block = re.search(r"interface InferenceParams \{(.*?)\n\}", fh.read(), re.DOTALL)
    assert block, "InferenceParams interface not found in runtime.ts"
    persisted = set(re.findall(r"^\s*(\w+)\??:", block.group(1), re.M)) - {"checkpoint"}

    backend = set(chat_history.ChatInferenceSettings.model_fields)
    assert persisted == backend, (
        f"schema drift: frontend-only {persisted - backend}, backend-only {backend - persisted}"
    )


# ---------------------------------------------------------------------------
# /api/chat/import-ledger
# ---------------------------------------------------------------------------


def test_get_import_ledger_round_trips_through_storage(monkeypatch):
    seen: list[str] = []

    def fake_list():
        return list(seen)

    monkeypatch.setattr(chat_history, "list_chat_legacy_imports", fake_list)

    response = chat_history.get_import_ledger(current_subject = "test-user")
    assert response.threadIds == []

    seen.extend(["legacy-a", "legacy-b"])
    response = chat_history.get_import_ledger(current_subject = "test-user")
    assert response.threadIds == ["legacy-a", "legacy-b"]


def test_record_import_ledger_returns_accepted_and_inserted(monkeypatch):
    captured: list[list[str]] = []

    def fake_upsert(thread_ids):
        captured.append(list(thread_ids))
        # Pretend two of the three were already in the ledger.
        return (len(thread_ids), max(0, len(thread_ids) - 2))

    monkeypatch.setattr(chat_history, "upsert_chat_legacy_imports", fake_upsert)

    response = chat_history.record_import_ledger(
        payload = chat_history.ChatImportLedgerRecordRequest(
            threadIds = ["a", "b", "c"],
        ),
        current_subject = "test-user",
    )
    assert response.accepted == 3
    assert response.inserted == 1
    assert captured == [["a", "b", "c"]]


def test_record_import_ledger_rejects_oversize_payload():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        chat_history.ChatImportLedgerRecordRequest(
            threadIds = [f"id-{i}" for i in range(10_001)],
        )


# ---------------------------------------------------------------------------
# /api/chat/threads/{id}/fork
# ---------------------------------------------------------------------------


def test_fork_thread_404_when_source_missing(monkeypatch):
    monkeypatch.setattr(chat_history, "get_chat_thread", lambda _id: None)
    with pytest.raises(HTTPException) as exc:
        chat_history.fork_thread(
            thread_id = "missing",
            payload = chat_history.ChatForkRequest(
                messageId = "m1",
                newThreadId = "new",
                createdAt = 1,
            ),
            current_subject = "test-user",
        )
    assert exc.value.status_code == 404


def test_fork_thread_404_when_branch_message_missing(monkeypatch):
    monkeypatch.setattr(chat_history, "get_chat_thread", lambda _id: {"id": _id, "title": "T"})
    monkeypatch.setattr(chat_history, "get_chat_message", lambda _t, _m: None)
    with pytest.raises(HTTPException) as exc:
        chat_history.fork_thread(
            thread_id = "src",
            payload = chat_history.ChatForkRequest(
                messageId = "missing",
                newThreadId = "new",
                createdAt = 1,
            ),
            current_subject = "test-user",
        )
    assert exc.value.status_code == 404


def test_fork_thread_happy_path(monkeypatch):
    source = {
        "id": "src",
        "title": "Original",
        "modelType": "base",
        "modelId": "m",
        "pairId": None,
        "archived": False,
        "createdAt": 1,
        "openaiCodeExecContainerId": None,
        "anthropicCodeExecContainerId": None,
        "forkedFromThreadId": None,
        "forkedFromMessageId": None,
    }
    forked = {
        **source,
        "id": "new",
        "title": "fork · Original",
        "createdAt": 2,
        "forkedFromThreadId": "src",
        "forkedFromMessageId": "m1",
    }
    monkeypatch.setattr(chat_history, "get_chat_thread", lambda _id: source)
    monkeypatch.setattr(
        chat_history,
        "get_chat_message",
        lambda _t, _m: {
            "id": _m,
            "threadId": _t,
            "role": "user",
            "content": [],
            "createdAt": 1,
        },
    )
    monkeypatch.setattr(chat_history, "fork_chat_thread", lambda **_: forked)
    monkeypatch.setattr(
        chat_history,
        "list_chat_messages",
        lambda _id: [
            {
                "id": "n1",
                "threadId": "new",
                "parentId": None,
                "role": "user",
                "content": [],
                "createdAt": 1,
            }
        ],
    )
    response = chat_history.fork_thread(
        thread_id = "src",
        payload = chat_history.ChatForkRequest(
            messageId = "m1",
            newThreadId = "new",
            createdAt = 2,
        ),
        current_subject = "test-user",
    )
    assert response.thread.id == "new"
    assert response.thread.title == "fork · Original"
    assert response.thread.forkedFromThreadId == "src"
    assert response.thread.forkedFromMessageId == "m1"
    assert len(response.messages) == 1
    assert response.containerSnapshotWarning is None


def test_fork_thread_warns_when_parent_had_container(monkeypatch):
    source = {
        "id": "src",
        "title": "T",
        "modelType": "base",
        "modelId": "",
        "pairId": None,
        "archived": False,
        "createdAt": 1,
        "openaiCodeExecContainerId": "cnt_123",
        "anthropicCodeExecContainerId": None,
        "forkedFromThreadId": None,
        "forkedFromMessageId": None,
    }
    monkeypatch.setattr(chat_history, "get_chat_thread", lambda _id: source)
    monkeypatch.setattr(
        chat_history,
        "get_chat_message",
        lambda _t, _m: {
            "id": _m,
            "threadId": _t,
            "role": "user",
            "content": [],
            "createdAt": 1,
        },
    )
    monkeypatch.setattr(
        chat_history,
        "fork_chat_thread",
        lambda **_: {
            **source,
            "id": "new",
            "title": "fork · T",
            "forkedFromThreadId": "src",
            "forkedFromMessageId": "m1",
            "openaiCodeExecContainerId": None,
        },
    )
    monkeypatch.setattr(chat_history, "list_chat_messages", lambda _id: [])
    response = chat_history.fork_thread(
        thread_id = "src",
        payload = chat_history.ChatForkRequest(
            messageId = "m1",
            newThreadId = "new",
            createdAt = 2,
        ),
        current_subject = "test-user",
    )
    assert response.containerSnapshotWarning is not None
    assert "fresh" in response.containerSnapshotWarning.lower()


def test_get_fork_count(monkeypatch):
    monkeypatch.setattr(chat_history, "count_forks_for_message", lambda _t, _m: 3)
    response = chat_history.get_fork_count(
        thread_id = "t",
        message_id = "m",
        current_subject = "test-user",
    )
    assert response.count == 3


def test_get_thread_fork_counts(monkeypatch):
    monkeypatch.setattr(chat_history, "fork_counts_for_thread", lambda _t: {"m1": 2, "m2": 1})
    response = chat_history.get_thread_fork_counts(
        thread_id = "t",
        current_subject = "test-user",
    )
    assert response.counts == {"m1": 2, "m2": 1}


def _clear_thread_row(thread_id: str) -> dict:
    return {
        "id": thread_id,
        "title": "Test Chat",
        "modelType": "base",
        "modelId": "test-model",
        "pairId": None,
        "archived": False,
        "createdAt": 1_700_000_000_000,
    }


def _install_active_hook_turn(project_id: str, thread_id: str):
    from core.agent_workspace import hook_runtime

    token = hook_runtime.HookSessionToken(project_id, thread_id, "live-generation")
    turn_cancel = threading.Event()
    with hook_runtime._BACKGROUND_LOCK:
        hook_runtime._ACTIVE_SESSIONS[(project_id, thread_id)] = token
        hook_runtime._SESSION_TURNS[token] = {1: turn_cancel}
    return token, turn_cancel


def _remove_active_hook_turn(token) -> None:
    from core.agent_workspace import hook_runtime
    with hook_runtime._BACKGROUND_LOCK:
        hook_runtime._SESSION_TURNS.pop(token, None)
        hook_runtime._ENDING_SESSIONS.discard(token)
        if hook_runtime._ACTIVE_SESSIONS.get((token.project_id, token.session_id)) == token:
            hook_runtime._ACTIVE_SESSIONS.pop((token.project_id, token.session_id), None)


def _destructive_test_ledger(token, turn_cancel, calls):
    @contextmanager
    def ledger(_connection, _thread_ids, *, reason):
        calls.append(reason)
        with hook_runtime._BACKGROUND_LOCK:
            hook_runtime._ENDING_SESSIONS.add(token)
        turn_cancel.set()
        yield []

    from core.agent_workspace import hook_runtime
    return ledger


def _capturing_test_ledger(token, workspace):
    from core.agent_workspace import hook_runtime

    @contextmanager
    def ledger(
        _connection,
        thread_ids,
        *,
        reason = "clear",
    ):
        records = [
            {
                "project_id": token.project_id,
                "session_id": thread_id,
                "model": "model",
            }
            for thread_id in sorted(thread_ids)
        ]
        with hook_runtime.capture_project_hook_session_ledgers(
            records,
            session_ids = tuple(sorted(thread_ids)),
            reason = reason,
            workspace_snapshots = {token.project_id: workspace},
        ) as captured:
            yield captured

    return ledger


class _FailingDatabaseConnection:
    def __init__(
        self,
        connection,
        *,
        fail_commit = False,
        fail_sql = None,
    ):
        self._connection = connection
        self._fail_commit = fail_commit
        self._fail_sql = fail_sql

    def __getattr__(self, name):
        return getattr(self._connection, name)

    def execute(self, sql, *args, **kwargs):
        if self._fail_sql and self._fail_sql in sql:
            raise sqlite3.OperationalError("injected SQL failure")
        return self._connection.execute(sql, *args, **kwargs)

    def commit(self):
        if self._fail_commit:
            raise sqlite3.OperationalError("injected commit failure")
        return self._connection.commit()


def _insert_folder_project(studio_db, tmp_path) -> SimpleNamespace:
    managed = tmp_path / "managed"
    folder = tmp_path / "folder"
    managed.mkdir()
    folder.mkdir()
    metadata = folder.stat()
    connection = studio_db.get_connection()
    try:
        connection.execute(
            """
            INSERT INTO chat_projects (
                id, name, instructions, root_path, workspace_kind,
                folder_path, folder_device_id, folder_file_id,
                workspace_revision, archived, created_at, updated_at
            ) VALUES (?, ?, '', ?, 'folder', ?, ?, ?, 0, 0, 1, 1)
            """,
            (
                "project",
                "Project",
                str(managed),
                str(folder),
                str(metadata.st_dev),
                str(metadata.st_ino),
            ),
        )
        connection.commit()
    finally:
        connection.close()
    studio_db.upsert_chat_thread({**_clear_thread_row("thread"), "projectId": "project"})
    return SimpleNamespace(
        project_id = "project",
        root = folder,
        kind = "folder",
        device_id = metadata.st_dev,
        file_id = metadata.st_ino,
        revision = 0,
    )


def test_rejected_thread_patch_does_not_capture_or_cancel_active_generation(tmp_path, monkeypatch):
    from core.agent_workspace import hook_runtime
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    studio_db.upsert_chat_thread(_clear_thread_row("thread"))
    token, turn_cancel = _install_active_hook_turn("project", "thread")
    calls = []
    try:
        with pytest.raises(studio_db.ChatThreadPreconditionFailed):
            studio_db.update_chat_thread(
                "thread",
                {"archived": True},
                expected_title = "stale-title",
                hook_session_ledger = _destructive_test_ledger(token, turn_cancel, calls),
            )
        assert calls == []
        assert not turn_cancel.is_set()
        assert token not in hook_runtime._ENDING_SESSIONS
        assert hook_runtime._ACTIVE_SESSIONS[("project", "thread")] == token
    finally:
        _remove_active_hook_turn(token)


def test_missing_thread_patch_does_not_capture_or_cancel_active_generation(tmp_path, monkeypatch):
    from core.agent_workspace import hook_runtime
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    studio_db.get_connection().close()
    token, turn_cancel = _install_active_hook_turn("project", "missing-thread")
    calls = []
    try:
        assert (
            studio_db.update_chat_thread(
                "missing-thread",
                {"archived": True},
                hook_session_ledger = _destructive_test_ledger(token, turn_cancel, calls),
            )
            is None
        )
        assert calls == []
        assert not turn_cancel.is_set()
        assert token not in hook_runtime._ENDING_SESSIONS
        assert hook_runtime._ACTIVE_SESSIONS[("project", "missing-thread")] == token
    finally:
        _remove_active_hook_turn(token)


def test_missing_settings_write_does_not_capture_or_cancel_active_generation(tmp_path, monkeypatch):
    from core.agent_workspace import hook_runtime
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    studio_db.upsert_chat_thread(_clear_thread_row("thread"))
    monkeypatch.setattr(
        studio_db, "_write_chat_thread_settings_in_conn", lambda *_args, **_kwargs: None
    )
    token, turn_cancel = _install_active_hook_turn("project", "thread")
    calls = []
    try:
        assert (
            studio_db.update_chat_thread(
                "thread",
                {"archived": True},
                settings_write = {"replace": {"toolsEnabled": True}},
                hook_session_ledger = _destructive_test_ledger(token, turn_cancel, calls),
            )
            is None
        )
        assert calls == []
        assert not turn_cancel.is_set()
        assert token not in hook_runtime._ENDING_SESSIONS
        assert hook_runtime._ACTIVE_SESSIONS[("project", "thread")] == token
    finally:
        _remove_active_hook_turn(token)


def test_thread_archive_commit_failure_rolls_back_without_cancelling_live_turn(
    tmp_path, monkeypatch
):
    from core.agent_workspace import hook_runtime
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    workspace = _insert_folder_project(studio_db, tmp_path)
    token, turn_cancel = _install_active_hook_turn("project", "thread")
    monkeypatch.setattr(hook_runtime, "_trusted_invocations", lambda *_args, **_kwargs: ((), ()))
    real_get_connection = studio_db.get_connection
    monkeypatch.setattr(
        studio_db,
        "get_connection",
        lambda *args, **kwargs: _FailingDatabaseConnection(
            real_get_connection(*args, **kwargs), fail_commit = True
        ),
    )
    try:
        with pytest.raises(sqlite3.OperationalError, match = "commit failure"):
            studio_db.update_chat_thread(
                "thread",
                {"archived": True},
                hook_session_ledger = _capturing_test_ledger(token, workspace),
            )
        assert not turn_cancel.is_set()
        assert token not in hook_runtime._ENDING_SESSIONS
        assert hook_runtime._ACTIVE_SESSIONS[("project", "thread")] == token
        connection = real_get_connection()
        try:
            row = connection.execute(
                "SELECT archived FROM chat_threads WHERE id = 'thread'"
            ).fetchone()
        finally:
            connection.close()
        assert row["archived"] == 0
    finally:
        _remove_active_hook_turn(token)


def test_folder_project_delete_sql_failure_rolls_back_without_cancelling_live_turn(
    tmp_path, monkeypatch
):
    from core.agent_workspace import hook_runtime
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    workspace = _insert_folder_project(studio_db, tmp_path)
    token, turn_cancel = _install_active_hook_turn("project", "thread")
    monkeypatch.setattr(hook_runtime, "_trusted_invocations", lambda *_args, **_kwargs: ((), ()))
    real_get_connection = studio_db.get_connection
    monkeypatch.setattr(
        studio_db,
        "get_connection",
        lambda *args, **kwargs: _FailingDatabaseConnection(
            real_get_connection(*args, **kwargs),
            fail_sql = "DELETE FROM chat_projects",
        ),
    )
    try:
        with pytest.raises(sqlite3.OperationalError, match = "SQL failure"):
            studio_db.delete_chat_project(
                "project",
                hook_session_ledger = _capturing_test_ledger(token, workspace),
            )
        assert not turn_cancel.is_set()
        assert token not in hook_runtime._ENDING_SESSIONS
        assert hook_runtime._ACTIVE_SESSIONS[("project", "thread")] == token
        connection = real_get_connection()
        try:
            project = connection.execute(
                "SELECT id FROM chat_projects WHERE id = 'project'"
            ).fetchone()
            thread = connection.execute(
                "SELECT id FROM chat_threads WHERE id = 'thread'"
            ).fetchone()
        finally:
            connection.close()
        assert project is not None
        assert thread is not None
    finally:
        _remove_active_hook_turn(token)


def test_cross_project_destructive_transaction_and_generation_admission_do_not_invert(
    tmp_path, monkeypatch
):
    from core.agent_workspace import hook_runtime
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    for project_id in ("project-a", "project-b"):
        studio_db.upsert_chat_project(
            {
                "id": project_id,
                "name": project_id,
                "createdAt": 1,
                "updatedAt": 1,
            }
        )
    studio_db.upsert_chat_thread({**_clear_thread_row("thread-a"), "projectId": "project-a"})
    studio_db.upsert_chat_thread({**_clear_thread_row("thread-b"), "projectId": "project-b"})
    token_a = hook_runtime._admit_project_hook_session(
        "project-a",
        "thread-a",
        None,
        create = True,
        deadline = time.monotonic() + 2,
    )
    assert token_a is not None
    monkeypatch.setattr(hook_runtime, "_trusted_invocations", lambda *_args, **_kwargs: ((), ()))

    real_ledger = chat_history._project_hook_session_ledger
    ledger_entered = threading.Event()
    release_transaction = threading.Event()
    transaction_committed = []

    @contextmanager
    def coordinated_ledger(connection, thread_ids, **kwargs):
        assert connection.in_transaction
        with real_ledger(connection, thread_ids, **kwargs) as records:
            ledger_entered.set()
            assert release_transaction.wait(2)
            yield records
            transaction_committed.append(not connection.in_transaction)

    monkeypatch.setattr(chat_history, "_project_hook_session_ledger", coordinated_ledger)
    real_end = hook_runtime.end_project_hook_session
    finalized = []

    def observe_finalizer(*args, **kwargs):
        assert transaction_committed == [True]
        is_owned = getattr(hook_runtime._ADMISSION_FENCE, "_is_owned", None)
        assert is_owned is not None and not is_owned()
        connection = studio_db.get_connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.rollback()
        finally:
            connection.close()
        finalized.append(kwargs["session_token"])
        return real_end(*args, **kwargs)

    monkeypatch.setattr(hook_runtime, "end_project_hook_session", observe_finalizer)
    move_result = []
    admitted_b = []
    failures = []
    admission_started = threading.Event()
    admission_done = threading.Event()

    def move_project_a_thread():
        try:
            move_result.append(
                chat_history.save_thread(
                    chat_history.ChatThread(
                        **{
                            **_clear_thread_row("thread-a"),
                            "projectId": "project-b",
                        }
                    ),
                    current_subject = "test-user",
                )
            )
        except BaseException as exc:  # noqa: BLE001 - surface worker failures in the test
            failures.append(exc)

    def admit_project_b_generation():
        admission_started.set()
        try:
            admitted_b.append(
                hook_runtime._admit_project_hook_session(
                    "project-b",
                    "thread-b",
                    None,
                    create = True,
                    deadline = time.monotonic() + 2,
                )
            )
        except BaseException as exc:  # noqa: BLE001 - surface worker failures in the test
            failures.append(exc)
        finally:
            admission_done.set()

    destructive = threading.Thread(target = move_project_a_thread)
    admission = threading.Thread(target = admit_project_b_generation)
    try:
        destructive.start()
        assert ledger_entered.wait(1)
        admission.start()
        assert admission_started.wait(1)
        assert not admission_done.wait(0.1)
        release_transaction.set()
        destructive.join(3)
        admission.join(3)
        assert not destructive.is_alive()
        assert not admission.is_alive()
        assert failures == []
        assert move_result[0].projectId == "project-b"
        assert finalized == [token_a]
        assert transaction_committed == [True]
        assert admitted_b[0] is not None
        assert admitted_b[0].project_id == "project-b"
        assert admitted_b[0].session_id == "thread-b"
        assert hook_runtime.snapshot_project_hook_session("project-a", "thread-a") is None
        assert hook_runtime.snapshot_project_hook_session("project-b", "thread-b") == admitted_b[0]
        assert hook_runtime.snapshot_project_hook_session("project-b", "thread-a") is None
        assert studio_db.get_chat_thread("thread-a")["projectId"] == "project-b"
        connection = studio_db.get_connection()
        try:
            outbox = connection.execute(
                "SELECT consumed_at FROM project_hook_session_end_outbox WHERE id = ?",
                (f"{token_a.generation}:SessionEnd",),
            ).fetchone()
        finally:
            connection.close()
        assert outbox is not None and outbox["consumed_at"] is not None
    finally:
        release_transaction.set()
        destructive.join(3)
        if admission.ident is not None:
            admission.join(3)
        _remove_active_hook_turn(token_a)
        for token in admitted_b:
            if token is not None:
                _remove_active_hook_turn(token)


def test_clear_hook_ledger_includes_pending_ids_and_replay_emits_no_new_end(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    studio_db.upsert_chat_thread(_clear_thread_row("stored-thread"))
    captured = []

    @contextmanager
    def ledger(_connection, thread_ids):
        captured.append(set(thread_ids))
        yield [{"project_id": "project", "session_id": "synthetic"}]

    first = studio_db.clear_chat_history_with_replay_status(
        ["pending-thread"],
        operation_id = "clear-hook-ledger",
        hook_session_ledger = ledger,
    )
    replay = studio_db.clear_chat_history_with_replay_status(
        ["reused-thread"],
        operation_id = "clear-hook-ledger",
        hook_session_ledger = ledger,
    )

    assert captured == [{"stored-thread", "pending-thread"}]
    assert first[2] is False
    assert first[3] == [{"project_id": "project", "session_id": "synthetic"}]
    assert replay[0] == ["stored-thread"]
    assert replay[2] is True
    assert replay[3] == []


def test_session_end_outbox_claim_is_exclusive_against_immediate_delivery(tmp_path, monkeypatch):
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    connection = studio_db.get_connection()
    try:
        snapshot = json.dumps(
            {
                "token": {
                    "project_id": "project",
                    "session_id": "thread",
                    "generation": "generation",
                }
            }
        )
        studio_db.enqueue_project_hook_session_end_outbox(
            connection,
            [("generation:SessionEnd", snapshot)],
        )
        connection.commit()
    finally:
        connection.close()

    barrier = threading.Barrier(3)
    claims = []

    def claim(owner):
        barrier.wait()
        claims.append(
            (
                owner,
                studio_db.claim_project_hook_session_end_outbox(
                    "generation:SessionEnd",
                    owner,
                ),
            )
        )

    workers = [
        threading.Thread(target = claim, args = ("recovery",)),
        threading.Thread(target = claim, args = ("immediate",)),
    ]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(2)

    assert len(claims) == 2
    [winner] = [owner for owner, claimed in claims if claimed]
    assert sorted(claimed for _owner, claimed in claims) == [False, True]
    assert studio_db.pending_project_hook_session_end_outbox() == []
    assert studio_db.mark_project_hook_session_end_outbox_consumed(
        "generation:SessionEnd",
        claim_owner = winner,
    )


def test_immediate_session_end_defers_to_existing_outbox_claim(monkeypatch):
    from core.agent_workspace import hook_runtime

    executed = []
    monkeypatch.setattr(
        chat_history,
        "claim_project_hook_session_end_outbox",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        hook_runtime,
        "end_project_hook_session",
        lambda *_args, **_kwargs: executed.append(True),
    )

    chat_history._end_project_hook_sessions(
        [
            {
                "project_id": "project",
                "session_id": "thread",
                "model": "model",
                "session_end_outbox_id": "generation:SessionEnd",
            }
        ]
    )

    assert executed == []


def test_session_end_claim_heartbeat_prevents_live_owner_reclaim(tmp_path, monkeypatch):
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    snapshot = json.dumps(
        {
            "token": {
                "project_id": "project",
                "session_id": "thread",
                "generation": "generation",
            }
        }
    )
    connection = studio_db.get_connection()
    try:
        studio_db.enqueue_project_hook_session_end_outbox(
            connection,
            [("generation:SessionEnd", snapshot)],
        )
        connection.commit()
    finally:
        connection.close()

    assert studio_db.claim_project_hook_session_end_outbox(
        "generation:SessionEnd", "owner-a", lease_ms = 1_000
    )
    with studio_db.project_hook_session_end_claim_heartbeat(
        "generation:SessionEnd", "owner-a", lease_ms = 1_000
    ):
        time.sleep(1.2)
        assert not studio_db.claim_project_hook_session_end_outbox(
            "generation:SessionEnd", "owner-b", lease_ms = 1_000
        )

    time.sleep(1.1)
    assert studio_db.claim_project_hook_session_end_outbox(
        "generation:SessionEnd", "owner-b", lease_ms = 1_000
    )


def test_session_end_claim_heartbeat_reports_forced_ownership_loss(monkeypatch):
    from storage import studio_db
    monkeypatch.setattr(
        studio_db,
        "renew_project_hook_session_end_outbox_claim",
        lambda *_args, **_kwargs: False,
    )

    with studio_db.project_hook_session_end_claim_heartbeat(
        "generation:SessionEnd", "owner", lease_ms = 50
    ) as ownership_lost:
        assert ownership_lost.wait(0.5)


def test_recovery_claims_only_current_delivery_while_first_finalizer_is_slow(tmp_path, monkeypatch):
    from core.agent_workspace import hook_runtime
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    connection = studio_db.get_connection()
    try:
        records = []
        for generation in ("generation-a", "generation-b"):
            records.append(
                (
                    f"{generation}:SessionEnd",
                    json.dumps(
                        {
                            "token": {
                                "project_id": "project",
                                "session_id": f"thread-{generation}",
                                "generation": generation,
                            }
                        }
                    ),
                )
            )
        studio_db.enqueue_project_hook_session_end_outbox(connection, records)
        connection.commit()
    finally:
        connection.close()

    first_started = threading.Event()
    release_first = threading.Event()

    def deserialize(raw):
        token_data = json.loads(raw)["token"]
        return SimpleNamespace(token = hook_runtime.HookSessionToken(**token_data))

    def end(_project_id, *, session_token, **_kwargs):
        if session_token.generation == "generation-a":
            first_started.set()
            assert release_first.wait(2)
        return hook_runtime.HookEventResult(event = "SessionEnd")

    monkeypatch.setattr(hook_runtime, "_deserialize_project_hook_session_end_snapshot", deserialize)
    monkeypatch.setattr(hook_runtime, "end_project_hook_session", end)
    recovered = []
    worker = threading.Thread(
        target = lambda: recovered.extend(
            hook_runtime.recover_pending_project_hook_session_ends(limit = 2)
        )
    )
    worker.start()
    assert first_started.wait(2)

    # The recovery worker owns only generation-a. A competing worker may claim
    # generation-b immediately instead of reclaiming an expired batch lease.
    assert studio_db.claim_project_hook_session_end_outbox(
        "generation-b:SessionEnd", "competing-recovery"
    )
    release_first.set()
    worker.join(3)

    assert not worker.is_alive()
    assert len(recovered) == 1


def test_recovery_heartbeats_claim_during_snapshot_preparation(tmp_path, monkeypatch):
    from core.agent_workspace import hook_runtime
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    snapshot = json.dumps(
        {
            "token": {
                "project_id": "project",
                "session_id": "thread",
                "generation": "generation",
            }
        }
    )
    connection = studio_db.get_connection()
    try:
        studio_db.enqueue_project_hook_session_end_outbox(
            connection, [("generation:SessionEnd", snapshot)]
        )
        connection.commit()
    finally:
        connection.close()

    original_heartbeat = studio_db.project_hook_session_end_claim_heartbeat
    monkeypatch.setattr(
        studio_db,
        "project_hook_session_end_claim_heartbeat",
        lambda record_id, owner: original_heartbeat(record_id, owner, lease_ms = 1_000),
    )
    preparation_started = threading.Event()
    release_preparation = threading.Event()
    token = hook_runtime.HookSessionToken("project", "thread", "generation")

    def deserialize(_raw):
        preparation_started.set()
        assert release_preparation.wait(3)
        return SimpleNamespace(token = token)

    monkeypatch.setattr(hook_runtime, "_deserialize_project_hook_session_end_snapshot", deserialize)
    monkeypatch.setattr(
        hook_runtime,
        "end_project_hook_session",
        lambda *_args, **_kwargs: hook_runtime.HookEventResult(event = "SessionEnd"),
    )
    worker = threading.Thread(
        target = hook_runtime.recover_pending_project_hook_session_ends,
        kwargs = {"limit": 1},
    )
    worker.start()
    assert preparation_started.wait(2)
    time.sleep(1.2)
    assert not studio_db.claim_project_hook_session_end_outbox(
        "generation:SessionEnd", "competing-recovery", lease_ms = 1_000
    )
    release_preparation.set()
    worker.join(3)

    assert not worker.is_alive()


def test_recovery_ownership_loss_cancels_old_owner_before_side_effect(tmp_path, monkeypatch):
    from core.agent_workspace import hook_runtime
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    connection = studio_db.get_connection()
    try:
        studio_db.enqueue_project_hook_session_end_outbox(
            connection,
            [
                (
                    "generation:SessionEnd",
                    json.dumps(
                        {
                            "token": {
                                "project_id": "project",
                                "session_id": "thread",
                                "generation": "generation",
                            }
                        }
                    ),
                )
            ],
        )
        connection.commit()
    finally:
        connection.close()

    token = hook_runtime.HookSessionToken("project", "thread", "generation")
    monkeypatch.setattr(
        hook_runtime,
        "_deserialize_project_hook_session_end_snapshot",
        lambda _raw: SimpleNamespace(token = token),
    )
    original_heartbeat = studio_db.project_hook_session_end_claim_heartbeat
    monkeypatch.setattr(
        studio_db,
        "project_hook_session_end_claim_heartbeat",
        lambda record_id, owner: original_heartbeat(record_id, owner, lease_ms = 1_000),
    )
    real_renew = studio_db.renew_project_hook_session_end_outbox_claim
    renewal_started = threading.Event()
    release_renewal = threading.Event()

    def blocked_renew(*args, **kwargs):
        renewal_started.set()
        assert release_renewal.wait(2)
        return real_renew(*args, **kwargs)

    monkeypatch.setattr(studio_db, "renew_project_hook_session_end_outbox_claim", blocked_renew)
    side_effects = []

    def end(_project_id, *, delivery_cancel_event, **_kwargs):
        assert delivery_cancel_event.wait(2)
        if not delivery_cancel_event.is_set():
            side_effects.append("started")
        return hook_runtime.HookEventResult(
            event = "SessionEnd",
            errors = ("Project SessionEnd delivery ownership was lost.",),
        )

    monkeypatch.setattr(hook_runtime, "end_project_hook_session", end)
    recovered = []
    worker = threading.Thread(
        target = lambda: recovered.extend(
            hook_runtime.recover_pending_project_hook_session_ends(limit = 1)
        )
    )
    worker.start()
    assert renewal_started.wait(2)

    connection = studio_db.get_connection()
    try:
        connection.execute(
            "UPDATE project_hook_session_end_outbox SET claim_expires_at = 0 WHERE id = ?",
            ("generation:SessionEnd",),
        )
        connection.commit()
    finally:
        connection.close()
    assert studio_db.claim_project_hook_session_end_outbox(
        "generation:SessionEnd", "replacement-owner"
    )
    release_renewal.set()
    worker.join(3)

    assert not worker.is_alive()
    assert side_effects == []
    assert recovered and recovered[0].errors
    connection = studio_db.get_connection()
    try:
        row = connection.execute(
            "SELECT claim_owner, consumed_at FROM project_hook_session_end_outbox WHERE id = ?",
            ("generation:SessionEnd",),
        ).fetchone()
    finally:
        connection.close()
    assert row["claim_owner"] == "replacement-owner"
    assert row["consumed_at"] is None


def test_pending_session_end_fences_cross_project_crash_handoff(tmp_path, monkeypatch):
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    snapshot = json.dumps(
        {
            "token": {
                "project_id": "project-a",
                "session_id": "thread",
                "generation": "generation-a",
            }
        }
    )
    connection = studio_db.get_connection()
    try:
        studio_db.enqueue_project_hook_session_end_outbox(
            connection,
            [("generation-a:SessionEnd", snapshot)],
        )
        connection.commit()
    finally:
        connection.close()

    assert studio_db.project_hook_session_end_pending_for("project-b", "thread")


def test_project_wide_session_end_fence_blocks_different_member_session(tmp_path, monkeypatch):
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    snapshot = json.dumps(
        {
            "token": {
                "project_id": "project-a",
                "session_id": "old-thread",
                "generation": "generation-a",
            }
        }
    )
    connection = studio_db.get_connection()
    try:
        studio_db.enqueue_project_hook_session_end_outbox(
            connection,
            [("generation-a:SessionEnd", snapshot)],
        )
        connection.commit()
        studio_db.enqueue_project_hook_session_end_outbox(
            connection,
            [("generation-a:SessionEnd", snapshot, "project")],
        )
        connection.commit()
    finally:
        connection.close()

    assert studio_db.project_hook_session_end_pending_for("project-a", "new-thread")
    assert not studio_db.project_hook_session_end_pending_for("project-b", "new-thread")


def test_clear_all_persists_global_session_end_fence_across_restart(tmp_path, monkeypatch):
    from core.agent_workspace import hook_runtime
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    monkeypatch.setattr(
        chat_history,
        "_project_hook_session_records",
        lambda *_args, **_kwargs: [
            {"project_id": "project-a", "session_id": "old-thread", "model": "model"}
        ],
    )
    snapshot = SimpleNamespace(invocations = (object(),))

    @contextmanager
    def capture(records, **kwargs):
        assert kwargs["include_all"] is True
        assert kwargs["reason"] == "clear"
        yield [{**records[0], "session_end_snapshot": snapshot}]

    snapshot_json = json.dumps(
        {
            "token": {
                "project_id": "project-a",
                "session_id": "old-thread",
                "generation": "generation-clear-all",
            }
        }
    )
    monkeypatch.setattr(hook_runtime, "capture_project_hook_session_ledgers", capture)
    monkeypatch.setattr(
        hook_runtime,
        "serialize_project_hook_session_end_snapshot",
        lambda _snapshot: ("generation-clear-all:SessionEnd", snapshot_json),
    )
    connection = studio_db.get_connection()
    try:
        connection.execute("BEGIN IMMEDIATE")
        with chat_history._project_hook_session_ledger(
            connection,
            {"old-thread"},
            include_all = True,
        ):
            connection.commit()
    finally:
        connection.close()

    monkeypatch.setattr(studio_db, "_schema_ready", False)
    connection = studio_db.get_connection()
    try:
        scope = connection.execute(
            "SELECT fence_scope FROM project_hook_session_end_outbox"
        ).fetchone()["fence_scope"]
    finally:
        connection.close()

    assert scope == "global"
    assert studio_db.project_hook_session_end_pending_for("project-b", "new-thread")


def test_session_end_ledger_persists_empty_sealed_snapshot_as_quiescence_fence(
    tmp_path, monkeypatch
):
    from core.agent_workspace import hook_runtime
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    monkeypatch.setattr(
        chat_history,
        "_project_hook_session_records",
        lambda *_args, **_kwargs: [
            {"project_id": "project", "session_id": "thread", "model": "model"}
        ],
    )
    snapshot = SimpleNamespace(invocations = ())

    @contextmanager
    def capture(records, **_kwargs):
        yield [{**records[0], "session_end_snapshot": snapshot}]

    monkeypatch.setattr(hook_runtime, "capture_project_hook_session_ledgers", capture)
    monkeypatch.setattr(
        hook_runtime,
        "serialize_project_hook_session_end_snapshot",
        lambda _snapshot: (
            "generation-empty:SessionEnd",
            json.dumps(
                {
                    "token": {
                        "project_id": "project",
                        "session_id": "thread",
                        "generation": "generation-empty",
                    }
                }
            ),
        ),
    )
    connection = studio_db.get_connection()
    try:
        connection.execute("BEGIN IMMEDIATE")
        with chat_history._project_hook_session_ledger(connection, {"thread"}):
            connection.commit()
        row = connection.execute(
            "SELECT id, consumed_at FROM project_hook_session_end_outbox"
        ).fetchone()
    finally:
        connection.close()

    assert tuple(row) == ("generation-empty:SessionEnd", None)


@pytest.mark.parametrize("reason", ["archive", "other", "delete"])
def test_project_lifecycle_ledgers_persist_project_wide_fence_scope(tmp_path, monkeypatch, reason):
    from core.agent_workspace import hook_runtime
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    monkeypatch.setattr(
        chat_history,
        "_project_hook_session_records",
        lambda *_args, **_kwargs: [
            {"project_id": "project", "session_id": "old-thread", "model": "model"}
        ],
    )
    snapshot = SimpleNamespace(invocations = (object(),))

    @contextmanager
    def capture(records, **kwargs):
        assert kwargs["project_ids"] == ("project",)
        assert kwargs["reason"] == reason
        yield [{**records[0], "session_end_snapshot": snapshot}]

    snapshot_json = json.dumps(
        {
            "token": {
                "project_id": "project",
                "session_id": "old-thread",
                "generation": f"generation-{reason}",
            }
        }
    )
    monkeypatch.setattr(hook_runtime, "capture_project_hook_session_ledgers", capture)
    monkeypatch.setattr(
        hook_runtime,
        "serialize_project_hook_session_end_snapshot",
        lambda _snapshot: (f"generation-{reason}:SessionEnd", snapshot_json),
    )
    connection = studio_db.get_connection()
    try:
        connection.execute("BEGIN IMMEDIATE")
        with chat_history._project_hook_session_ledger(
            connection,
            {"old-thread"},
            reason = reason,
            project_ids = ("project",),
        ):
            connection.commit()
        row = connection.execute(
            "SELECT fence_scope FROM project_hook_session_end_outbox"
        ).fetchone()
    finally:
        connection.close()

    assert row["fence_scope"] == "project"
    assert studio_db.project_hook_session_end_pending_for("project", "new-thread")


def test_managed_root_survives_pending_claimed_and_failed_end_then_deletes_on_consumption(
    tmp_path, linkable_temp_base, monkeypatch
):
    from core.inference import tools
    from storage import studio_db

    safe_root = linkable_temp_base / tmp_path.name
    safe_root.mkdir(parents = True, exist_ok = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(safe_root / "studio-home"))
    monkeypatch.setenv(
        "UNSLOTH_STUDIO_PROJECTS_HOME",
        str(safe_root / "studio-home" / "Projects"),
    )
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    project = {"id": "project-root", "name": "Project Root"}
    managed_root = Path(studio_db._default_project_root(project))
    sandbox = managed_root / "sandbox"
    sandbox.mkdir(parents = True)
    marker = sandbox / "finalizer-visible.txt"
    marker.write_text("keep", encoding = "utf-8")
    record_id = "generation-root:SessionEnd"
    snapshot = json.dumps(
        {
            "token": {
                "project_id": project["id"],
                "session_id": "thread-root",
                "generation": "generation-root",
            }
        }
    )
    connection = studio_db.get_connection()
    try:
        studio_db.enqueue_project_hook_session_end_outbox(
            connection,
            [(record_id, snapshot, "project")],
        )
        connection.commit()
    finally:
        connection.close()
    tools.record_orphaned_project(
        project["id"],
        str(sandbox),
        True,
        str(managed_root),
    )
    monkeypatch.setattr(tools, "live_project_owns", lambda *_args: False)
    monkeypatch.setattr(tools, "wait_for_sessions_idle", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(studio_db, "sandbox_is_referenced_elsewhere", lambda *_args: False)

    tools.collect_orphaned_project_workspaces()
    assert marker.exists()
    assert studio_db.claim_project_hook_session_end_outbox(record_id, "first-owner")
    tools.collect_orphaned_project_workspaces()
    assert marker.exists()
    assert studio_db.mark_project_hook_session_end_outbox_failed(
        record_id,
        "finalizer failed",
        claim_owner = "first-owner",
    )
    tools.collect_orphaned_project_workspaces()
    assert marker.exists()

    connection = studio_db.get_connection()
    try:
        connection.execute(
            "UPDATE project_hook_session_end_outbox SET next_attempt_at = 0 WHERE id = ?",
            (record_id,),
        )
        connection.commit()
    finally:
        connection.close()
    assert studio_db.claim_project_hook_session_end_outbox(record_id, "recovery-owner")
    assert studio_db.mark_project_hook_session_end_outbox_consumed(
        record_id,
        claim_owner = "recovery-owner",
    )
    tools.collect_orphaned_project_workspaces()

    assert not managed_root.exists()
    assert tools.list_orphaned_projects() == []


def test_dead_letter_pressure_never_deletes_unresolved_lifecycle_fences(tmp_path, monkeypatch):
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    records = []
    for index in range(257):
        generation = f"generation-{index}"
        records.append(
            (
                f"{generation}:SessionEnd",
                json.dumps(
                    {
                        "token": {
                            "project_id": "project",
                            "session_id": f"thread-{index}",
                            "generation": generation,
                        }
                    }
                ),
            )
        )
    connection = studio_db.get_connection()
    try:
        studio_db.enqueue_project_hook_session_end_outbox(connection, records)
        connection.commit()
    finally:
        connection.close()
    for record_id, _snapshot in records:
        assert studio_db.mark_project_hook_session_end_outbox_failed(
            record_id,
            "poison",
            max_attempts = 1,
        )

    connection = studio_db.get_connection()
    try:
        unresolved = connection.execute(
            """
            SELECT COUNT(*)
            FROM project_hook_session_end_outbox
            WHERE consumed_at IS NULL AND dead_letter_at IS NOT NULL
            """
        ).fetchone()[0]
    finally:
        connection.close()
    assert unresolved == 257
    assert studio_db.project_hook_session_end_pending_for("project", "thread-0")


def test_a_clear_does_not_reap_an_image_registered_while_it_was_running(tmp_path, monkeypatch):
    """The reap is global; the delete it accompanies is not.

    Between the transaction committing and the reap there is archive and sandbox cleanup
    that can run for seconds. A chat created in that window survives the delete, so
    wiping the whole registry afterwards took ITS thumbnails and left its cards 404ing
    out of thumbnail_bytes. Independent of the replay case: this one is a first clear.

    The snapshot is taken before the slow work, so an id registered during it is not the
    clear's to reap.
    """
    from core.inference import search_images
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    monkeypatch.setattr(search_images, "_registry", {})
    monkeypatch.setattr(search_images, "_cleared_unservable", set())
    monkeypatch.setattr(search_images, "_cache_dir", lambda: tmp_path / "thumbs")
    (tmp_path / "thumbs").mkdir(parents = True, exist_ok = True)

    old = search_images.register_images(
        [
            {
                "title": "before",
                "image": "https://img.example.com/a.jpg",
                "thumbnail": "https://tse1.mm.bing.net/th?id=a",
                "url": "https://example.com/a",
                "source": "Bing",
            }
        ]
    )
    assert old, "fixture must register: the whole test turns on this id being reapable"
    old_id = old[0]["id"]

    late: dict[str, str] = {}

    async def remove_sandboxes(_thread_ids, _delete_files):
        # Stands in for the concurrent client: another tab registers an image for a chat
        # created after the transaction, while this slow cleanup is still running.
        entries = search_images.register_images(
            [
                {
                    "title": "during",
                    "image": "https://img.example.com/b.jpg",
                    "thumbnail": "https://tse1.mm.bing.net/th?id=b",
                    "url": "https://example.com/b",
                    "source": "Bing",
                }
            ]
        )
        late["id"] = entries[0]["id"]
        return 0, []

    monkeypatch.setattr(chat_history, "_remove_sandboxes", remove_sandboxes)
    monkeypatch.setattr(chat_history, "_cancel_active_generations", lambda _ids: None)
    monkeypatch.setattr(chat_history, "_cancel_research_runs", lambda _request, _ids: None)
    monkeypatch.setattr(
        chat_history, "_remove_conversation_archives", lambda _ids, cutoff = None: None
    )
    request = SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace()))

    studio_db.upsert_chat_thread(_clear_thread_row("before-clear"))
    asyncio.run(
        chat_history.clear_history(
            request,
            chat_history.ChatClearRequest(ids = [], operationId = "clear-operation-race"),
            current_subject = "test-user",
        )
    )

    assert late.get("id"), "the stand-in never registered, so this asserts nothing"
    assert search_images.lookup_image(late["id"]) is not None, (
        "an image registered while the clear was running belongs to a chat the clear "
        "kept, so reaping it 404s that chat's cards"
    )
    assert search_images.lookup_image(old_id) is None, (
        "the clear still has to reap what it was responsible for"
    )


def test_replayed_clear_keeps_the_thumbnails_of_a_chat_it_did_not_delete(tmp_path, monkeypatch):
    """A retry under a recorded operationId replays, so it must not reap the global cache.

    The frontend retries DELETE /chat once under the SAME operationId after its 30s
    abort, and Starlette does not cancel the first handler when the client hangs up, so
    the retry lands behind a transaction that already committed. That transaction
    deliberately leaves chats created since alone -- but the thumbnail registry is
    global, so reaping it again took the images of a chat this call is not deleting and
    left its cards 404ing.
    """
    from core.inference import search_images
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    reaped: list[str] = []
    monkeypatch.setattr(search_images, "clear_cache", lambda only_ids = None: reaped.append("reaped"))

    async def remove_sandboxes(_thread_ids, _delete_files):
        return 0, []

    monkeypatch.setattr(chat_history, "_remove_sandboxes", remove_sandboxes)
    monkeypatch.setattr(chat_history, "_cancel_active_generations", lambda _ids: None)
    monkeypatch.setattr(chat_history, "_cancel_research_runs", lambda _request, _ids: None)
    monkeypatch.setattr(
        chat_history, "_remove_conversation_archives", lambda _ids, cutoff = None: None
    )
    request = SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace()))

    def clear():
        return asyncio.run(
            chat_history.clear_history(
                request,
                chat_history.ChatClearRequest(ids = [], operationId = "clear-operation-retry"),
                current_subject = "test-user",
            )
        )

    studio_db.upsert_chat_thread(_clear_thread_row("before-clear"))
    assert clear()["deletedThreadIds"] == ["before-clear"]
    assert reaped == ["reaped"], "the first clear has to reap: the thumbnails say what was searched"

    # The image-bearing chat the delayed retry must not touch.
    studio_db.upsert_chat_thread(_clear_thread_row("after-clear"))

    replay = clear()
    assert replay["deletedThreadIds"] == ["before-clear"]
    assert studio_db.get_chat_thread("after-clear") is not None
    assert reaped == ["reaped"], "the replay reaped a surviving chat's thumbnails"


def test_the_replay_bit_comes_from_the_clear_transaction(monkeypatch, tmp_path):
    """Two concurrent retries of one operationId: exactly one of them performed the clear.

    Establishing `replayed` with a read taken before the transaction is a guess. Both
    requests carrying the same operationId see the same unrecorded ledger, so both
    conclude they cleared; BEGIN IMMEDIATE then serialises them and the loser silently
    replays while still believing otherwise. It would go on to reap the thumbnail
    registry -- which is global, and so is not covered by the ids the transaction
    deliberately kept -- taking the images of chats created since the winner committed.
    This is the retry the operationId exists to make safe: the frontend reissues the
    same id after its 30s abort, and Starlette does not cancel the handler the client
    hung up on, so both really do run at once.
    """
    from core.inference import search_images
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    reaps: list[str] = []
    reap_lock = threading.Lock()

    def record_reap(only_ids = None):
        with reap_lock:
            reaps.append("reaped")

    monkeypatch.setattr(search_images, "clear_cache", record_reap)

    async def remove_sandboxes(_thread_ids, _delete_files):
        return 0, []

    monkeypatch.setattr(chat_history, "_remove_sandboxes", remove_sandboxes)
    monkeypatch.setattr(chat_history, "_cancel_active_generations", lambda _ids: None)
    monkeypatch.setattr(chat_history, "_cancel_research_runs", lambda _request, _ids: None)
    monkeypatch.setattr(
        chat_history, "_remove_conversation_archives", lambda _ids, cutoff = None: None
    )
    request = SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace()))

    studio_db.upsert_chat_thread(_clear_thread_row("before-clear"))

    # Released together, so both are past the point the old code decided `replayed` at
    # before either transaction commits -- which is the whole race.
    start = threading.Barrier(2)
    failures: list[BaseException] = []

    def clear():
        try:
            start.wait(timeout = 10)
            asyncio.run(
                chat_history.clear_history(
                    request,
                    chat_history.ChatClearRequest(ids = [], operationId = "clear-operation-concurrent"),
                    current_subject = "test-user",
                )
            )
        except BaseException as exc:  # noqa: BLE001 -- re-raised on the main thread
            failures.append(exc)

    threads = [threading.Thread(target = clear) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout = 30)
    if failures:
        raise failures[0]

    assert reaps == ["reaped"], (
        "only the request that actually performed the clear may reap the global "
        f"thumbnail registry; got {len(reaps)} reaps"
    )


def test_clear_history_does_not_read_the_replay_ledger_outside_the_transaction():
    """The structural half of the race above, which no scheduling can hide.

    `replayed` has to be whatever the transaction did, so it is returned by the call
    that does the clear. A separate ledger read reintroduces the window even if the
    threads in the test above happen to serialise.
    """
    source = inspect.getsource(chat_history.clear_history)
    assert "clear_chat_history_with_replay_status" in source
    assert "chat_clear_operation_is_recorded" not in source, (
        "a pre-transaction ledger read cannot tell a replay from a concurrent clear"
    )


def test_a_chat_created_in_the_gap_after_the_clear_keeps_its_images(monkeypatch, tmp_path):
    """The snapshot has to be taken at the clear boundary, not one await later.

    `await run_in_threadpool(...)` is a yield point. With the clear and the snapshot in
    separate calls, the event loop can run another request in between: a chat created there
    survives the transaction (the clear only deletes what it saw), but its images register
    before the snapshot, so the reap that follows takes them and its cards 404 out of
    thumbnail_bytes. One threadpool call for both removes that gap.

    It does not make the two atomic -- another worker THREAD can still land between the
    commit and the read, and closing that would mean holding the image registry's lock across
    the whole transaction, stalling every search in the process for the length of a clear.
    This pins the gap that was worth removing.

    The interleave is forced rather than raced: `run_in_threadpool` is wrapped so the other
    tab registers its image immediately after the FIRST hop returns, which is exactly the
    window in question.
    """
    from core.inference import search_images
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    monkeypatch.setattr(search_images, "_registry", {})
    monkeypatch.setattr(search_images, "_cache_dir", lambda: tmp_path / "thumbs")
    (tmp_path / "thumbs").mkdir(parents = True, exist_ok = True)

    reaped: list = []
    monkeypatch.setattr(search_images, "clear_cache", lambda only_ids = None: reaped.append(only_ids))

    async def remove_sandboxes(_thread_ids, _delete_files):
        return 0, []

    monkeypatch.setattr(chat_history, "_remove_sandboxes", remove_sandboxes)
    monkeypatch.setattr(chat_history, "_cancel_active_generations", lambda _ids: None)
    monkeypatch.setattr(chat_history, "_cancel_research_runs", lambda _request, _ids: None)
    monkeypatch.setattr(
        chat_history, "_remove_conversation_archives", lambda _ids, cutoff = None: None
    )

    # The other tab's image, registered in the gap. Straight into the registry: this is about
    # WHEN the id becomes visible to the snapshot, not about how it got there.
    late_image_id = "beefbeefbeef"
    hops = {"n": 0}
    # The route imports it inside the handler, so the patch has to land on the module it
    # imports FROM, not on routes.chat_history.
    import starlette.concurrency

    real_run_in_threadpool = starlette.concurrency.run_in_threadpool

    async def interleaving_run_in_threadpool(func, *args, **kwargs):
        result = await real_run_in_threadpool(func, *args, **kwargs)
        hops["n"] += 1
        if hops["n"] == 1:
            search_images._registry[late_image_id] = {
                "thumbnail": "https://example.invalid/x.jpg",
                "source": "https://example.invalid/",
                "created": 0.0,
                "policy": None,
            }
        return result

    monkeypatch.setattr(starlette.concurrency, "run_in_threadpool", interleaving_run_in_threadpool)
    request = SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace()))

    studio_db.upsert_chat_thread(_clear_thread_row("before-clear"))
    asyncio.run(
        chat_history.clear_history(
            request,
            chat_history.ChatClearRequest(ids = [], operationId = "clear-operation-gap"),
            current_subject = "test-user",
        )
    )

    assert reaped, "the clear still has to reap what it was responsible for"
    snapshot = reaped[0]
    assert snapshot is not None, "a real clear reaps a bounded set, not everything"
    assert late_image_id not in snapshot, (
        "an image registered after the clear committed belongs to a chat the clear kept, "
        "so the reap must not be allowed to take it"
    )


def test_the_clear_and_its_image_snapshot_share_one_threadpool_hop():
    """The structural half of the race above, which no test scheduling can hide."""
    source = inspect.getsource(chat_history.clear_history)
    assert source.count("run_in_threadpool(_clear_rows)") == 1
    assert "run_in_threadpool(snapshot_and_fence_registrations)" not in source, (
        "a second hop for the snapshot reopens the gap the first one closed"
    )
    body = source.split("def _clear_rows(", 1)[1].split("\n    # The clear reports", 1)[0]
    assert "snapshot_and_fence_registrations()" in body, (
        "the snapshot belongs inside the clear's hop, and it carries the registration fence"
    )


def test_a_replay_finishes_a_reap_the_original_clear_died_before_running(monkeypatch, tmp_path):
    """A crash between the clear's commit and its thumbnail reap must not lose the reap.

    The reap runs after the transaction, behind seconds of archive and sandbox cleanup. Killed
    in that window the operation is already recorded, so the retry the frontend sends replays
    -- and a replay deliberately reaps nothing, because the chats created since the original
    clear are not its to take. The thumbnails of every deleted chat then stay on disk for good,
    saying what was searched for, which is the worse of the two failures this path weighs.

    The ledger now carries the original clear's own snapshot and whether the reap finished, so
    the replay can complete exactly that set. The crash is simulated by making the first reap
    raise, which is the same state a SIGKILL leaves behind: committed, recorded, unreaped.
    """
    from core.inference import search_images
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    monkeypatch.setattr(search_images, "_registry", {})
    monkeypatch.setattr(search_images, "_cache_dir", lambda: tmp_path / "thumbs")
    (tmp_path / "thumbs").mkdir(parents = True, exist_ok = True)

    doomed_image_id = "aaaabbbbcccc"
    search_images._registry[doomed_image_id] = {
        "thumbnail": "https://example.invalid/x.jpg",
        "source": "https://example.invalid/",
        "created": 0.0,
        "policy": None,
    }

    reaps: list = []

    def reap(only_ids = None):
        reaps.append(only_ids)
        if len(reaps) == 1:
            raise RuntimeError("process died before the reap finished")

    monkeypatch.setattr(search_images, "clear_cache", reap)

    async def remove_sandboxes(_thread_ids, _delete_files):
        return 0, []

    monkeypatch.setattr(chat_history, "_remove_sandboxes", remove_sandboxes)
    monkeypatch.setattr(chat_history, "_cancel_active_generations", lambda _ids: None)
    monkeypatch.setattr(chat_history, "_cancel_research_runs", lambda _request, _ids: None)
    monkeypatch.setattr(
        chat_history, "_remove_conversation_archives", lambda _ids, cutoff = None: None
    )
    request = SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace()))

    def clear():
        return asyncio.run(
            chat_history.clear_history(
                request,
                chat_history.ChatClearRequest(ids = [], operationId = "clear-operation-crash"),
                current_subject = "test-user",
            )
        )

    studio_db.upsert_chat_thread(_clear_thread_row("before-clear"))
    with pytest.raises(RuntimeError):
        clear()
    assert reaps == [{doomed_image_id}], "the first attempt got as far as its own reap"

    # A chat started after the crash, whose images the replay must NOT take.
    studio_db.upsert_chat_thread(_clear_thread_row("after-crash"))
    later_image_id = "ddddeeeeffff"
    search_images._registry[later_image_id] = {
        "thumbnail": "https://example.invalid/y.jpg",
        "source": "https://example.invalid/",
        "created": 0.0,
        "policy": None,
    }

    clear()
    assert len(reaps) == 2, "the replay has to finish the reap the crash interrupted"
    finished = reaps[1]
    assert finished == {doomed_image_id}, (
        "bounded to the original clear's own snapshot, so a chat created since keeps its images"
    )
    assert later_image_id not in finished

    # And a further retry has nothing left to do.
    clear()
    assert len(reaps) == 2, "the finished reap must be recorded, not repeated on every retry"


def test_a_plain_replay_with_nothing_outstanding_still_reaps_nothing(monkeypatch, tmp_path):
    """The ordinary retry. The first attempt completed its reap, so the replay must stay out
    of the global registry entirely -- that is what the replay branch is for."""
    from core.inference import search_images
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    monkeypatch.setattr(search_images, "_registry", {})
    monkeypatch.setattr(search_images, "_cache_dir", lambda: tmp_path / "thumbs")
    (tmp_path / "thumbs").mkdir(parents = True, exist_ok = True)

    reaps: list = []
    monkeypatch.setattr(search_images, "clear_cache", lambda only_ids = None: reaps.append(only_ids))

    async def remove_sandboxes(_thread_ids, _delete_files):
        return 0, []

    monkeypatch.setattr(chat_history, "_remove_sandboxes", remove_sandboxes)
    monkeypatch.setattr(chat_history, "_cancel_active_generations", lambda _ids: None)
    monkeypatch.setattr(chat_history, "_cancel_research_runs", lambda _request, _ids: None)
    monkeypatch.setattr(
        chat_history, "_remove_conversation_archives", lambda _ids, cutoff = None: None
    )
    request = SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace()))

    def clear():
        return asyncio.run(
            chat_history.clear_history(
                request,
                chat_history.ChatClearRequest(ids = [], operationId = "clear-operation-plain"),
                current_subject = "test-user",
            )
        )

    studio_db.upsert_chat_thread(_clear_thread_row("before-clear"))
    clear()
    assert len(reaps) == 1
    clear()
    assert len(reaps) == 1, "a replay behind a completed reap must not touch the registry"
