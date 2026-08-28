# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Focused execution coverage for reviewed project lifecycle hooks."""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.agent_workspace import hook_runtime, supervisor


def _handler(command: str, **extra) -> dict:
    return {"type": "command", "command": command, **extra}


def _write_hooks(root: Path, hooks: dict[str, list[dict]]) -> str:
    path = root / ".codex" / "hooks.json"
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(
        json.dumps({"hooks": hooks}, separators = (",", ":")),
        encoding = "utf-8",
    )
    return hook_runtime.discover_project_hooks(root)["contentHash"]


def _group(*handlers: dict, matcher: str | None = None) -> dict:
    group = {"hooks": list(handlers)}
    if matcher is not None:
        group["matcher"] = matcher
    return group


def _process(
    output: str = "",
    *,
    status: str = "passed",
    exit_code: int | None = 0,
    stderr: str = "",
    truncated: bool = False,
) -> supervisor.ProjectProcessResult:
    return supervisor.ProjectProcessResult(
        status,
        exit_code,
        output,
        len(output.encode()),
        truncated,
        stderr = stderr,
        stderr_bytes = len(stderr.encode()),
    )


@pytest.fixture(autouse = True)
def _reset_runtime_state():
    with hook_runtime._BACKGROUND_LOCK:
        for runs in hook_runtime._BACKGROUND.values():
            for run in runs.values():
                run.cancel_event.set()
        for runs in hook_runtime._SYNCHRONOUS.values():
            for run in runs.values():
                run.cancel_event.set()
        hook_runtime._BACKGROUND.clear()
        hook_runtime._SYNCHRONOUS.clear()
        hook_runtime._SESSION_WORK.clear()
        hook_runtime._SESSION_TURNS.clear()
        hook_runtime._BACKGROUND_COUNTS.clear()
        hook_runtime._COMPLETED.clear()
        hook_runtime._ENDED_SESSIONS.clear()
        hook_runtime._ENDING_SESSIONS.clear()
        hook_runtime._CAPTURING_SESSIONS.clear()
        hook_runtime._SESSION_END_EXECUTING.clear()
        hook_runtime._STARTED_SESSIONS.clear()
        hook_runtime._UNESTABLISHED_SESSIONS.clear()
        hook_runtime._STARTING_SESSIONS.clear()
        hook_runtime._ACTIVE_SESSIONS.clear()
        hook_runtime._SESSION_OWNERS.clear()
        hook_runtime._DEFERRED_SESSION_ENDS.clear()
        hook_runtime._DEFERRED_SESSION_ABORTS.clear()
        hook_runtime._DEFERRED_END_WAKE.clear()
        hook_runtime._GLOBAL_HOOK_WORK = 0
        hook_runtime._GLOBAL_SESSION_END_WORK = 0
        hook_runtime._COMPLETED_SEQUENCE = 0
    yield
    with hook_runtime._BACKGROUND_LOCK:
        for runs in hook_runtime._BACKGROUND.values():
            for run in runs.values():
                run.cancel_event.set()
        hook_runtime._COMPLETED.clear()
        hook_runtime._SYNCHRONOUS.clear()
        hook_runtime._SESSION_WORK.clear()
        hook_runtime._SESSION_TURNS.clear()
        hook_runtime._BACKGROUND_COUNTS.clear()
        hook_runtime._ENDED_SESSIONS.clear()
        hook_runtime._ENDING_SESSIONS.clear()
        hook_runtime._CAPTURING_SESSIONS.clear()
        hook_runtime._SESSION_END_EXECUTING.clear()
        hook_runtime._STARTED_SESSIONS.clear()
        hook_runtime._UNESTABLISHED_SESSIONS.clear()
        hook_runtime._STARTING_SESSIONS.clear()
        hook_runtime._ACTIVE_SESSIONS.clear()
        hook_runtime._SESSION_OWNERS.clear()
        hook_runtime._DEFERRED_SESSION_ENDS.clear()
        hook_runtime._DEFERRED_SESSION_ABORTS.clear()
        hook_runtime._DEFERRED_END_WAKE.clear()
        hook_runtime._GLOBAL_HOOK_WORK = 0
        hook_runtime._GLOBAL_SESSION_END_WORK = 0
        hook_runtime._COMPLETED_SEQUENCE = 0


def _bind_runtime(
    monkeypatch,
    root: Path,
    content_hash: str,
    *,
    disabled: tuple[str, ...] = (),
    revision: int = 4,
    workspace_revision: int = 7,
):
    metadata = root.stat()
    workspace = SimpleNamespace(
        project_id = "project",
        root = root,
        kind = "managed",
        device_id = metadata.st_dev,
        file_id = metadata.st_ino,
        revision = workspace_revision,
    )

    @contextmanager
    def access(project_id, **_kwargs):
        assert project_id == "project"
        yield workspace

    state = {
        "projectId": "project",
        "contentHash": content_hash,
        "trusted": True,
        "disabledHandlerIds": list(disabled),
        "revision": revision,
    }
    monkeypatch.setattr(hook_runtime, "project_workspace_access", access)
    monkeypatch.setattr(
        hook_runtime,
        "get_chat_project",
        lambda project_id: {"id": project_id, "archived": False},
    )
    monkeypatch.setattr(
        hook_runtime,
        "get_chat_thread",
        lambda session_id: {
            "id": session_id,
            "projectId": "project",
            "archived": False,
        },
    )
    monkeypatch.setattr(
        hook_runtime,
        "project_hook_session_end_pending_for",
        lambda _project_id, _session_id: False,
    )
    monkeypatch.setattr(
        hook_runtime,
        "get_project_hook_trust_state",
        lambda project_id, current_hash, **kwargs: dict(state),
    )
    return workspace, state


def test_controlling_hook_discovery_is_bounded_by_the_event_deadline(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    content_hash = _write_hooks(
        root,
        {"UserPromptSubmit": [_group(_handler("prompt-policy"))]},
    )
    _bind_runtime(monkeypatch, root, content_hash)
    entered = threading.Event()
    release = threading.Event()
    original_discovery = hook_runtime.discover_project_hooks

    def blocked_discovery(*args, **kwargs):
        entered.set()
        release.wait(2)
        return original_discovery(*args, **kwargs)

    monkeypatch.setattr(hook_runtime, "discover_project_hooks", blocked_discovery)
    monkeypatch.setattr(hook_runtime, "MAX_HOOK_EVENT_PREPARATION_SECONDS", 0.03)

    started = time.monotonic()
    try:
        outcome = hook_runtime.run_project_hook_event(
            "project",
            "UserPromptSubmit",
            {"prompt": "hello"},
            session_id = "thread-1",
        )
    finally:
        release.set()

    assert entered.is_set()
    assert time.monotonic() - started < 0.5
    assert len(outcome.runs) == 1
    assert outcome.runs[0].handler_id == "UserPromptSubmit:preparation"
    assert outcome.runs[0].status == "timed_out"
    failure = hook_runtime.project_hook_control_failure(outcome)
    assert failure is not None
    assert "deadline" in failure


def test_one_absolute_hook_deadline_covers_access_discovery_revalidation_and_spawn(
    tmp_path, monkeypatch
):
    root = tmp_path / "repository"
    root.mkdir()
    content_hash = _write_hooks(
        root,
        {"UserPromptSubmit": [_group(_handler("prompt-policy"))]},
    )
    workspace, _state = _bind_runtime(monkeypatch, root, content_hash)
    observed = {"access": [], "discovery": [], "spawn": []}
    original_discovery = hook_runtime.discover_project_hooks

    @contextmanager
    def access(
        project_id,
        *,
        cancel_event = None,
        deadline = None,
    ):
        assert project_id == "project"
        assert cancel_event is None
        observed["access"].append(deadline)
        yield workspace

    def discovery(*args, **kwargs):
        observed["discovery"].append(kwargs.get("deadline"))
        return original_discovery(*args, **kwargs)

    def run_process(_project_id, _argv, **kwargs):
        observed["spawn"].append(kwargs["absolute_deadline"])
        kwargs["before_spawn"](workspace)
        return _process()

    monkeypatch.setattr(hook_runtime, "project_workspace_access", access)
    monkeypatch.setattr(hook_runtime, "discover_project_hooks", discovery)
    monkeypatch.setattr(supervisor, "_run_project_process", run_process)

    result = hook_runtime.run_project_hook_event(
        "project",
        "UserPromptSubmit",
        {"prompt": "hello"},
        session_id = "thread",
    )

    assert result.runs[0].status == "passed"
    assert len(observed["access"]) == 1
    assert len(observed["discovery"]) == 2
    assert len(observed["spawn"]) == 1
    deadline = observed["access"][0]
    assert deadline is not None
    assert observed["discovery"] == [deadline, deadline]
    assert observed["spawn"] == [deadline]


def test_hook_deadline_includes_database_scope_admission_without_late_token(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    content_hash = _write_hooks(
        root,
        {"UserPromptSubmit": [_group(_handler("prompt-policy"))]},
    )
    _bind_runtime(monkeypatch, root, content_hash)
    entered = threading.Event()
    release = threading.Event()

    def blocked_project(project_id):
        entered.set()
        release.wait(2)
        return {"id": project_id, "archived": False}

    monkeypatch.setattr(hook_runtime, "get_chat_project", blocked_project)
    monkeypatch.setattr(hook_runtime, "MAX_HOOK_EVENT_PREPARATION_SECONDS", 0.03)

    started = time.monotonic()
    try:
        outcome = hook_runtime.run_project_hook_event(
            "project",
            "UserPromptSubmit",
            {"prompt": "hello"},
            session_id = "thread",
        )
    finally:
        release.set()

    assert entered.is_set()
    assert time.monotonic() - started < 0.5
    assert outcome.runs[0].handler_id == "UserPromptSubmit:preparation"
    assert outcome.runs[0].status == "timed_out"
    assert ("project", "thread") not in hook_runtime._ACTIVE_SESSIONS


def test_database_scope_wake_rechecks_deadline_before_creating_token(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    content_hash = _write_hooks(root, {})
    _bind_runtime(monkeypatch, root, content_hash)
    published = threading.Event()

    class DelayedSuccessfulWake:
        def set(self):
            published.set()

        def wait(self, timeout):
            signalled = published.wait(timeout)
            if signalled:
                time.sleep(0.04)
            return signalled

    monkeypatch.setattr(
        supervisor,
        "_new_preparation_event",
        lambda: DelayedSuccessfulWake(),
    )

    with pytest.raises(TimeoutError, match = "exceeded its deadline"):
        hook_runtime._admit_project_hook_session(
            "project",
            "thread",
            None,
            create = True,
            deadline = time.monotonic() + 0.02,
        )

    assert published.is_set()
    assert ("project", "thread") not in hook_runtime._ACTIVE_SESSIONS


def test_hook_deadline_includes_admission_fence_wait_without_late_token(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    content_hash = _write_hooks(
        root,
        {"UserPromptSubmit": [_group(_handler("prompt-policy"))]},
    )
    _bind_runtime(monkeypatch, root, content_hash)
    fence_held = threading.Event()
    release = threading.Event()

    def hold_admission_fence():
        with hook_runtime.project_hook_admission_fence():
            fence_held.set()
            assert release.wait(2)

    owner = threading.Thread(target = hold_admission_fence)
    owner.start()
    assert fence_held.wait(1)
    monkeypatch.setattr(hook_runtime, "MAX_HOOK_EVENT_PREPARATION_SECONDS", 0.03)

    started = time.monotonic()
    try:
        outcome = hook_runtime.run_project_hook_event(
            "project",
            "UserPromptSubmit",
            {"prompt": "hello"},
            session_id = "thread",
        )
    finally:
        release.set()
        owner.join(1)

    assert not owner.is_alive()
    assert time.monotonic() - started < 0.5
    assert outcome.runs[0].handler_id == "UserPromptSubmit:preparation"
    assert outcome.runs[0].status == "timed_out"
    assert ("project", "thread") not in hook_runtime._ACTIVE_SESSIONS


def test_failed_mixed_session_start_cancels_and_quiesces_async_handlers(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    content_hash = _write_hooks(
        root,
        {
            "SessionStart": [
                _group(
                    _handler("background", **{"async": True}),
                    _handler("controlling"),
                )
            ],
            "SessionEnd": [_group(_handler("must-not-run"))],
        },
    )
    _bind_runtime(monkeypatch, root, content_hash)
    background_started = threading.Event()
    side_effects = []
    executed = []

    def run(invocation, *, cancel_event, **_kwargs):
        executed.append(invocation.handler_id)
        if invocation.handler_id == "SessionStart:0:0":
            background_started.set()
            if cancel_event.wait(1):
                return _process(status = "cancelled", exit_code = None)
            side_effects.append("late startup side effect")
            return _process()
        assert background_started.wait(1)
        return _process(status = "failed", exit_code = 1, stderr = "startup failed")

    monkeypatch.setattr(supervisor, "_run_trusted_project_hook_process", run)

    result = hook_runtime.ensure_project_hook_session(
        "project",
        session_id = "thread",
    )
    token = result.session_token

    assert hook_runtime.project_hook_control_failure(result) == "startup failed"
    assert token is not None
    assert side_effects == []
    assert executed == ["SessionStart:0:0", "SessionStart:0:1"]
    with hook_runtime._BACKGROUND_LOCK:
        assert token not in hook_runtime._BACKGROUND
        assert token not in hook_runtime._SYNCHRONOUS
        assert token not in hook_runtime._STARTED_SESSIONS
        assert token not in hook_runtime._UNESTABLISHED_SESSIONS
        assert token not in hook_runtime._ENDING_SESSIONS
        assert ("project", "thread") not in hook_runtime._ACTIVE_SESSIONS


def test_failed_session_start_uses_deferred_quiescence_without_session_end(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    content_hash = _write_hooks(
        root,
        {
            "SessionStart": [
                _group(
                    _handler("background", **{"async": True}),
                    _handler("controlling"),
                )
            ],
            "SessionEnd": [_group(_handler("must-not-run"))],
        },
    )
    _bind_runtime(monkeypatch, root, content_hash)
    background_started = threading.Event()
    release = threading.Event()
    executed = []

    def run(invocation, **_kwargs):
        executed.append(invocation.handler_id)
        if invocation.handler_id == "SessionStart:0:0":
            background_started.set()
            release.wait(2)
            return _process(status = "cancelled", exit_code = None)
        assert background_started.wait(1)
        return _process(status = "failed", exit_code = 1, stderr = "startup failed")

    monkeypatch.setattr(supervisor, "_run_trusted_project_hook_process", run)
    monkeypatch.setattr(hook_runtime, "SESSION_END_DRAIN_TIMEOUT_SECONDS", 0.03)

    started = time.monotonic()
    result = hook_runtime.ensure_project_hook_session("project", session_id = "thread")
    elapsed = time.monotonic() - started
    token = result.session_token

    assert token is not None
    assert elapsed < 0.5
    with hook_runtime._BACKGROUND_LOCK:
        assert hook_runtime._ACTIVE_SESSIONS[("project", "thread")] == token
        assert token in hook_runtime._ENDING_SESSIONS
        assert token in hook_runtime._DEFERRED_SESSION_ABORTS
        assert token not in hook_runtime._DEFERRED_SESSION_ENDS
    release.set()
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        with hook_runtime._BACKGROUND_LOCK:
            if ("project", "thread") not in hook_runtime._ACTIVE_SESSIONS:
                break
        time.sleep(0.01)
    with hook_runtime._BACKGROUND_LOCK:
        assert ("project", "thread") not in hook_runtime._ACTIVE_SESSIONS
        assert token not in hook_runtime._BACKGROUND
        assert token not in hook_runtime._ENDING_SESSIONS
        assert token not in hook_runtime._DEFERRED_SESSION_ABORTS
    assert all(not handler_id.startswith("SessionEnd:") for handler_id in executed)


def test_runs_enabled_handlers_with_authoritative_bounded_stdin(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    content_hash = _write_hooks(
        root,
        {
            "PreToolUse": [
                _group(
                    _handler("first"),
                    _handler("disabled"),
                    matcher = "Bash",
                )
            ]
        },
    )
    workspace, _state = _bind_runtime(
        monkeypatch,
        root,
        content_hash,
        disabled = ("PreToolUse:0:1",),
    )
    observed = []

    def run(invocation, **kwargs):
        hook_runtime._revalidate_hook_invocation(invocation, workspace)
        argv, event_input = hook_runtime._hook_invocation_process_spec(invocation)
        observed.append((argv, json.loads(event_input), kwargs))
        return _process()

    monkeypatch.setattr(supervisor, "_run_trusted_project_hook_process", run)

    outcome = hook_runtime.run_project_hook_event(
        "project",
        "PreToolUse",
        {
            "turn_id": "turn-1",
            "tool_name": "Bash",
            "tool_use_id": "call-1",
            "tool_input": {"command": "pwd"},
        },
        session_id = "thread-1",
        model = "local-model",
        permission_mode = "default",
    )

    assert [run.handler_id for run in outcome.runs] == ["PreToolUse:0:0"]
    assert observed[0][0] == ("/bin/sh", "-c", "first")
    payload = observed[0][1]
    assert payload["cwd"] == str(root)
    assert payload["hook_event_name"] == "PreToolUse"
    assert payload["session_id"] == "thread-1"
    assert payload["tool_input"] == {"command": "pwd"}
    assert observed[0][2]["timeout_seconds"] == 600.0
    assert observed[0][2]["output_limit_bytes"] == hook_runtime.MAX_HOOK_OUTPUT_BYTES


def test_revalidates_source_trust_workspace_and_handler_before_each_spawn(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    content_hash = _write_hooks(
        root,
        {"PreToolUse": [_group(_handler("first"), matcher = "Bash")]},
    )
    workspace, state = _bind_runtime(monkeypatch, root, content_hash)
    attempts = 0

    def run(invocation, **kwargs):
        nonlocal attempts
        attempts += 1
        state["revision"] += 1
        hook_runtime._revalidate_hook_invocation(invocation, workspace)
        pytest.fail("stale hook authority reached process creation")

    monkeypatch.setattr(supervisor, "_run_trusted_project_hook_process", run)

    outcome = hook_runtime.run_project_hook_event(
        "project",
        "PreToolUse",
        {"tool_name": "Bash", "tool_input": {"command": "pwd"}},
        session_id = "thread-1",
    )

    assert attempts == 1
    assert outcome.runs[0].status == "failed"
    assert "trust changed" in outcome.errors[0].lower()


def test_aggregates_block_rewrite_and_context_in_declaration_order(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    content_hash = _write_hooks(
        root,
        {
            "PreToolUse": [
                _group(
                    _handler("rewrite-one", additionalContextLimit = 32),
                    _handler("block"),
                    _handler("rewrite-two"),
                    matcher = "Bash",
                )
            ]
        },
    )
    workspace, _state = _bind_runtime(monkeypatch, root, content_hash)
    outputs = {
        "rewrite-one": {
            "systemMessage": "first warning",
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
                "updatedInput": {"command": "printf first"},
                "additionalContext": "first context",
            },
        },
        "block": {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "blocked by policy",
            }
        },
        "rewrite-two": {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
                "updatedInput": {"command": "printf second"},
                "additionalContext": "second context",
            }
        },
    }

    def run(invocation, **kwargs):
        hook_runtime._revalidate_hook_invocation(invocation, workspace)
        argv, _event_input = hook_runtime._hook_invocation_process_spec(invocation)
        return _process(json.dumps(outputs[argv[-1]]))

    monkeypatch.setattr(supervisor, "_run_trusted_project_hook_process", run)

    outcome = hook_runtime.run_project_hook_event(
        "project",
        "PreToolUse",
        {"tool_name": "Bash", "tool_input": {"command": "pwd"}},
        session_id = "thread-1",
    )

    assert [run.handler_id for run in outcome.runs] == [
        "PreToolUse:0:0",
        "PreToolUse:0:1",
        "PreToolUse:0:2",
    ]
    assert outcome.blocked is True
    assert outcome.reason == "blocked by policy"
    assert outcome.updated_input == {"command": "printf second"}
    assert outcome.additional_context == ("first context", "second context")
    assert outcome.system_messages == ("first warning",)


def test_permission_request_rejects_updated_input(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    content_hash = _write_hooks(
        root,
        {"PermissionRequest": [_group(_handler("permission"), matcher = "Bash")]},
    )
    workspace, _state = _bind_runtime(monkeypatch, root, content_hash)

    def run(invocation, **kwargs):
        hook_runtime._revalidate_hook_invocation(invocation, workspace)
        return _process(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PermissionRequest",
                        "decision": {
                            "behavior": "allow",
                            "updatedInput": {"command": "printf reviewed"},
                        },
                    }
                }
            )
        )

    monkeypatch.setattr(supervisor, "_run_trusted_project_hook_process", run)
    outcome = hook_runtime.run_project_hook_event(
        "project",
        "PermissionRequest",
        {"tool_name": "Bash", "tool_input": {"command": "printf original"}},
        session_id = "thread-1",
    )

    assert outcome.blocked is False
    assert outcome.permission_decision is None
    assert outcome.updated_input is None
    assert "reserved decision field" in outcome.errors[0]


@pytest.mark.parametrize(
    "process, status, blocked",
    [
        (_process(status = "timed_out", exit_code = None), "timed_out", False),
        (_process(status = "cancelled", exit_code = None), "cancelled", False),
        (_process(stderr = "denied", status = "failed", exit_code = 2), "blocked", True),
        (_process("{" + "x" * 10, truncated = True), "failed", False),
    ],
)
def test_process_failures_are_bounded_and_do_not_crash_the_turn(
    tmp_path, monkeypatch, process, status, blocked
):
    root = tmp_path / "repository"
    root.mkdir()
    content_hash = _write_hooks(
        root,
        {"UserPromptSubmit": [_group(_handler("check"))]},
    )
    workspace, _state = _bind_runtime(monkeypatch, root, content_hash)

    def run(invocation, **kwargs):
        hook_runtime._revalidate_hook_invocation(invocation, workspace)
        return process

    monkeypatch.setattr(supervisor, "_run_trusted_project_hook_process", run)
    outcome = hook_runtime.run_project_hook_event(
        "project",
        "UserPromptSubmit",
        {"prompt": "hello"},
        session_id = "thread-1",
    )

    assert outcome.runs[0].status == status
    assert outcome.blocked is blocked


def test_async_hooks_are_bounded_informational_and_cancelled_at_session_end(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    content_hash = _write_hooks(
        root,
        {
            "PostToolUse": [_group(_handler("background", **{"async": True}), matcher = "Bash")],
            "SessionEnd": [_group(_handler("session-end"), matcher = "other")],
        },
    )
    workspace, _state = _bind_runtime(monkeypatch, root, content_hash)
    started = threading.Event()
    cancelled = threading.Event()
    session_end_executed = threading.Event()

    def run(invocation, **kwargs):
        hook_runtime._revalidate_hook_invocation(invocation, workspace)
        argv, _event_input = hook_runtime._hook_invocation_process_spec(invocation)
        if argv[-1] == "background":
            started.set()
            kwargs["cancel_event"].wait(2)
            cancelled.set()
            return _process(
                json.dumps({"decision": "block", "reason": "too late"}),
                status = "cancelled",
                exit_code = None,
            )
        session_end_executed.set()
        return _process()

    monkeypatch.setattr(supervisor, "_run_trusted_project_hook_process", run)
    scheduled = hook_runtime.run_project_hook_event(
        "project",
        "PostToolUse",
        {"tool_name": "Bash", "tool_response": "ok"},
        session_id = "thread-1",
    )
    assert scheduled.runs[0].status == "scheduled"
    assert started.wait(1)

    ended = hook_runtime.end_project_hook_session(
        "project",
        session_id = "thread-1",
    )

    assert cancelled.wait(1)
    assert session_end_executed.is_set()
    assert ended.blocked is False
    assert [run.handler_id for run in ended.runs] == ["SessionEnd:0:0"]
    assert ended.runs[0].status == "passed"
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        with hook_runtime._BACKGROUND_LOCK:
            if not hook_runtime._BACKGROUND.get(scheduled.session_token):
                break
        time.sleep(0.01)
    with hook_runtime._BACKGROUND_LOCK:
        assert scheduled.session_token not in hook_runtime._COMPLETED


def test_completed_async_results_are_bounded(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    content_hash = _write_hooks(
        root,
        {
            "PostToolUse": [
                _group(
                    *(
                        _handler(f"background-{index}", **{"async": True})
                        for index in range(hook_runtime.MAX_COMPLETED_HOOKS + 12)
                    ),
                    matcher = "Bash",
                )
            ]
        },
    )
    workspace, _state = _bind_runtime(monkeypatch, root, content_hash)

    def run(invocation, **kwargs):
        hook_runtime._revalidate_hook_invocation(invocation, workspace)
        return _process("context")

    monkeypatch.setattr(supervisor, "_run_trusted_project_hook_process", run)
    scheduled = hook_runtime.run_project_hook_event(
        "project",
        "PostToolUse",
        {"tool_name": "Bash", "tool_response": "ok"},
        session_id = "thread-1",
    )
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        with hook_runtime._BACKGROUND_LOCK:
            if not hook_runtime._BACKGROUND.get(scheduled.session_token):
                break
        time.sleep(0.01)
    with hook_runtime._BACKGROUND_LOCK:
        assert len(hook_runtime._COMPLETED.get(scheduled.session_token, ())) <= (
            hook_runtime.MAX_COMPLETED_HOOKS
        )


def test_private_supervisor_entrypoint_binds_revalidation_and_stdin(monkeypatch):
    invocation = hook_runtime._HookInvocation(
        project_id = "project",
        event = "UserPromptSubmit",
        event_input_json = b"{}",
        handler_id = "UserPromptSubmit:0:0",
        handler_json = b'{"command":"check","type":"command"}',
        content_hash = "a" * 64,
        workspace_identity = (1, 2),
        workspace_revision = 3,
        trust_revision = 4,
    )
    observed = {}

    def run(project_id, command, **kwargs):
        observed.update(project_id = project_id, command = command, kwargs = kwargs)
        return _process()

    monkeypatch.setattr(supervisor, "_run_project_process", run)
    monkeypatch.setattr(hook_runtime, "_revalidate_hook_invocation", lambda *_args: None)

    result = supervisor._run_trusted_project_hook_process(
        invocation,
        timeout_seconds = 3,
        output_limit_bytes = 1024,
        cancel_event = None,
    )

    assert result.status == "passed"
    assert observed["project_id"] == "project"
    assert observed["command"] == ("/bin/sh", "-c", "check")
    assert observed["kwargs"]["input_bytes"] == b"{}"
    assert observed["kwargs"]["separate_stderr"] is True
    assert callable(observed["kwargs"]["before_spawn"])


def test_captured_session_end_binds_stable_delivery_execution_fence(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    token = hook_runtime.HookSessionToken("project", "thread", "generation")
    delivery_id = "generation:SessionEnd"
    invocation = hook_runtime._HookInvocation(
        project_id = "project",
        event = "SessionEnd",
        event_input_json = json.dumps({"delivery_id": delivery_id}).encode(),
        handler_id = "SessionEnd:0:0",
        handler_json = b'{"command":"finalize","type":"command"}',
        content_hash = "a" * 64,
        workspace_identity = (root.stat().st_dev, root.stat().st_ino),
        workspace_revision = 0,
        trust_revision = 4,
        session_token = token,
    )
    workspace = hook_runtime.ProjectWorkspace(
        project_id = "project",
        root = root,
        kind = "managed",
        device_id = root.stat().st_dev,
        file_id = root.stat().st_ino,
    )
    snapshot = hook_runtime.HookSessionEndSnapshot(
        token = token,
        invocations = (invocation,),
        handlers = ({},),
        workspace = workspace,
        _seal = hook_runtime._END_SNAPSHOT_SEAL,
    )
    observed = {}

    def run(project_id, command, **kwargs):
        observed.update(project_id = project_id, command = command, kwargs = kwargs)
        return _process()

    monkeypatch.setattr(supervisor, "_run_project_process", run)
    monkeypatch.setattr(hook_runtime, "_revalidate_hook_invocation", lambda *_args, **_kwargs: None)

    result = supervisor._run_trusted_project_hook_process(
        invocation,
        timeout_seconds = 3,
        output_limit_bytes = 1024,
        cancel_event = None,
        end_snapshot = snapshot,
    )

    assert result.status == "passed"
    assert observed["kwargs"]["execution_fence_id"] == json.dumps(
        [delivery_id, invocation.handler_id],
        separators = (",", ":"),
    )


def test_session_end_sibling_handlers_have_independent_stable_execution_fences(
    tmp_path, monkeypatch
):
    root = tmp_path / "repository"
    root.mkdir()
    token = hook_runtime.HookSessionToken("project", "thread", "generation")
    delivery_id = "generation:SessionEnd"
    workspace = hook_runtime.ProjectWorkspace(
        project_id = "project",
        root = root,
        kind = "managed",
        device_id = root.stat().st_dev,
        file_id = root.stat().st_ino,
    )
    handlers = tuple(_handler(f"finalize-{index}", timeout = 2) for index in range(2))
    invocations = tuple(
        hook_runtime._HookInvocation(
            project_id = "project",
            event = "SessionEnd",
            event_input_json = json.dumps({"delivery_id": delivery_id}).encode(),
            handler_id = f"SessionEnd:0:{index}",
            handler_json = json.dumps(
                handler,
                separators = (",", ":"),
                sort_keys = True,
            ).encode(),
            content_hash = "a" * 64,
            workspace_identity = (root.stat().st_dev, root.stat().st_ino),
            workspace_revision = 0,
            trust_revision = 4,
            session_token = token,
            deadline_monotonic = time.monotonic() + 2,
        )
        for index, handler in enumerate(handlers)
    )
    snapshot = hook_runtime.HookSessionEndSnapshot(
        token = token,
        invocations = invocations,
        handlers = handlers,
        workspace = workspace,
        _seal = hook_runtime._END_SNAPSHOT_SEAL,
    )
    observed = []
    observed_lock = threading.Lock()
    siblings_started = threading.Barrier(2)

    def run(_project_id, _command, **kwargs):
        with observed_lock:
            observed.append(
                (
                    kwargs["execution_fence_id"],
                    json.loads(kwargs["input_bytes"])["delivery_id"],
                )
            )
        siblings_started.wait(timeout = 1)
        return _process()

    monkeypatch.setattr(supervisor, "_run_project_process", run)
    with hook_runtime._BACKGROUND_LOCK:
        hook_runtime._ACTIVE_SESSIONS[(token.project_id, token.session_id)] = token

    result = hook_runtime.end_project_hook_session(
        token.project_id,
        session_id = token.session_id,
        session_token = token,
        end_snapshot = snapshot,
        deadline_monotonic = time.monotonic() + 2,
    )

    assert [(run.handler_id, run.status) for run in result.runs] == [
        ("SessionEnd:0:0", "passed"),
        ("SessionEnd:0:1", "passed"),
    ]
    assert sorted(observed) == sorted(
        [
            (
                json.dumps([delivery_id, "SessionEnd:0:0"], separators = (",", ":")),
                delivery_id,
            ),
            (
                json.dumps([delivery_id, "SessionEnd:0:1"], separators = (",", ":")),
                delivery_id,
            ),
        ]
    )
    assert all(
        json.loads(invocation.event_input_json)["delivery_id"] == delivery_id
        for invocation in invocations
    )


def test_stop_hook_runs_only_after_clean_completion(monkeypatch):
    observed = []

    def run(_project_id, event, payload, **kwargs):
        observed.append((event, payload, kwargs))
        return hook_runtime.HookEventResult(event = event)

    monkeypatch.setattr(hook_runtime, "run_project_hook_event", run)

    @hook_runtime.project_stop_hook
    def loop(
        *,
        session_id,
        thread_id,
        cancel_event,
        failure = False,
    ):
        yield "started"
        if failure:
            raise RuntimeError("failed")

    assert list(
        loop(
            session_id = "project-project",
            thread_id = "clean",
            cancel_event = threading.Event(),
        )
    ) == ["started"]

    with pytest.raises(RuntimeError, match = "failed"):
        list(
            loop(
                session_id = "project-project",
                thread_id = "error",
                cancel_event = threading.Event(),
                failure = True,
            )
        )

    cancelled = threading.Event()
    cancelled.set()
    assert list(
        loop(
            session_id = "project-project",
            thread_id = "cancelled",
            cancel_event = cancelled,
        )
    ) == ["started"]

    interrupted = loop(
        session_id = "project-project",
        thread_id = "interrupted",
        cancel_event = threading.Event(),
    )
    assert next(interrupted) == "started"
    interrupted.close()

    assert [item[0] for item in observed] == ["Stop"]
    assert observed[0][1]["stop_hook_active"] is False
    assert observed[0][1]["last_assistant_message"] == "started"
    assert "terminal_reason" not in observed[0][1]


def test_stop_hook_bounds_oversized_candidate_without_losing_clean_response(monkeypatch):
    observed = []
    candidate = '\x01"\\\n' * (hook_runtime.MAX_HOOK_EVENT_BYTES // 2)

    def run(_project_id, event, payload, **_kwargs):
        observed.append((event, payload))
        return hook_runtime.HookEventResult(event = event)

    monkeypatch.setattr(hook_runtime, "run_project_hook_event", run)

    @hook_runtime.project_stop_hook
    def loop(*, session_id, thread_id, cancel_event):
        yield candidate

    streamed = list(
        loop(
            session_id = "project-project",
            thread_id = "oversized",
            cancel_event = threading.Event(),
        )
    )

    assert streamed == [candidate]
    assert len(observed) == 1
    event, payload = observed[0]
    assert event == "Stop"
    bounded = payload["last_assistant_message"]
    assert candidate.startswith(bounded)
    serialized = json.dumps(
        bounded,
        ensure_ascii = False,
        separators = (",", ":"),
    ).encode("utf-8")
    assert len(serialized) <= hook_runtime.MAX_STOP_CANDIDATE_BYTES
    assert len(bounded) < len(candidate)
    next_character = candidate[: len(bounded) + 1]
    assert (
        len(
            json.dumps(
                next_character,
                ensure_ascii = False,
                separators = (",", ":"),
            ).encode("utf-8")
        )
        > hook_runtime.MAX_STOP_CANDIDATE_BYTES
    )
    assert len(hook_runtime._event_json(payload)) <= hook_runtime.MAX_HOOK_EVENT_BYTES


def test_stop_turn_uses_real_identity_and_tracks_continuation_depth(monkeypatch):
    observed = []

    def run(project_id, event, payload, **kwargs):
        observed.append((project_id, event, payload, kwargs))
        return hook_runtime.HookEventResult(
            event = event,
            continuation_reason = "continue",
        )

    monkeypatch.setattr(hook_runtime, "run_project_hook_event", run)
    turn = hook_runtime.ProjectHookTurn(
        project_id = "project",
        session_id = "session",
        turn_id = "turn-real",
        model = "model",
        permission_mode = "acceptEdits",
        cancel_event = threading.Event(),
    )
    first = turn.stop(last_assistant_message = "answer")
    second = turn.stop(last_assistant_message = "continued answer")

    assert first.continuation_reason == "continue"
    assert second.continuation_reason == "continue"
    assert len(observed) == 2
    assert observed[0][2]["turn_id"] == "turn-real"
    assert observed[0][2]["last_assistant_message"] == "answer"
    assert observed[0][2]["stop_hook_active"] is False
    assert observed[1][2]["stop_hook_active"] is True
    assert observed[0][3]["permission_mode"] == "acceptEdits"


def test_stop_hook_allows_exactly_two_internal_continuations(monkeypatch):
    observed = []
    candidates = iter(("first", "second", "third"))

    def run(_project_id, event, payload, **_kwargs):
        observed.append((event, payload))
        return hook_runtime.HookEventResult(
            event = event,
            continuation_reason = "keep going",
            continuation_reasons = ("keep going",),
            continuation_fragments = (("review", "keep going"),),
        )

    monkeypatch.setattr(hook_runtime, "run_project_hook_event", run)

    @hook_runtime.project_stop_hook
    def loop(
        *,
        messages,
        session_id,
        thread_id,
        cancel_event,
        continuation_state = None,
    ):
        conversation = list(messages)
        if continuation_state is not None:
            continuation_state["messages"] = conversation
        yield {"type": "content", "text": next(candidates)}

    output = list(
        loop(
            messages = [{"role": "user", "content": "start"}],
            session_id = "project-project",
            thread_id = "thread",
            cancel_event = threading.Event(),
        )
    )

    assert [item["text"] for item in output] == ["first", "firstsecond", "firstsecondthird"]
    assert [payload["stop_hook_active"] for _, payload in observed] == [False, True, True]
    assert len(observed) == hook_runtime.MAX_STOP_CONTINUATIONS + 1


def test_stop_continuation_rechecks_cancellation_before_dispatch(monkeypatch):
    cancel_event = threading.Event()
    samples = []

    def run(_project_id, event, _payload, **_kwargs):
        cancel_event.set()
        return hook_runtime.HookEventResult(
            event = event,
            continuation_reason = "continue",
            continuation_reasons = ("continue",),
            continuation_fragments = (("review", "continue"),),
        )

    monkeypatch.setattr(hook_runtime, "run_project_hook_event", run)

    @hook_runtime.project_stop_hook
    def loop(
        *,
        messages,
        session_id,
        thread_id,
        cancel_event,
        continuation_state = None,
    ):
        samples.append(list(messages))
        if continuation_state is not None:
            continuation_state["messages"] = list(messages)
        yield {"type": "content", "text": "answer"}

    assert list(
        loop(
            messages = [{"role": "user", "content": "start"}],
            session_id = "project-project",
            thread_id = "thread",
            cancel_event = cancel_event,
        )
    ) == [{"type": "content", "text": "answer"}]
    assert len(samples) == 1


def test_anthropic_continuation_emits_block_boundary(monkeypatch):
    calls = 0

    def run(_project_id, event, _payload, **_kwargs):
        nonlocal calls
        calls += 1
        return hook_runtime.HookEventResult(
            event = event,
            continuation_reason = "continue" if calls == 1 else None,
            continuation_reasons = ("continue",) if calls == 1 else (),
            continuation_fragments = (("review", "continue"),) if calls == 1 else (),
        )

    monkeypatch.setattr(hook_runtime, "run_project_hook_event", run)
    turn = hook_runtime.ProjectHookTurn(
        project_id = "project",
        session_id = "thread",
        turn_id = "turn",
        model = "model",
        permission_mode = "default",
        cancel_event = threading.Event(),
        transport = "anthropic",
    )
    answers = iter(("first", "second"))

    @hook_runtime.project_stop_hook
    def loop(
        *,
        messages,
        session_id,
        thread_id,
        cancel_event,
        continuation_state = None,
    ):
        if continuation_state is not None:
            continuation_state["messages"] = list(messages)
        yield {"type": "content", "text": next(answers)}

    with hook_runtime.activate_project_hook_turn(turn):
        output = list(
            loop(
                messages = [{"role": "user", "content": "start"}],
                session_id = "project-project",
                thread_id = "thread",
                cancel_event = turn.cancel_event,
            )
        )

    assert output == [
        {"type": "content", "text": "first"},
        {"type": "hook_continuation_boundary"},
        {"type": "content", "text": "second"},
    ]


def test_continue_false_precedes_stop_continuation():
    outcome = hook_runtime._aggregate(
        "Stop",
        [
            hook_runtime.HookHandlerResult(
                "first",
                "passed",
                continuation_requested = True,
                reason = "continue",
            ),
            hook_runtime.HookHandlerResult(
                "second",
                "passed",
                stop_requested = True,
                reason = "stop",
            ),
        ],
    )

    assert outcome.stop_requested is True
    assert outcome.continuation_reason is None


@pytest.mark.parametrize(
    ("tool_name", "valid", "invalid"),
    [
        ("Bash", {"command": "printf ok"}, {"patch": "printf ok"}),
        ("apply_patch", {"patch": "*** Begin Patch\n*** End Patch"}, {"command": "patch"}),
        (
            "edit_file",
            {
                "path": "file.txt",
                "edits": [{"old_string": "before", "new_string": "after"}],
            },
            {"path": "file.txt", "patch": "after"},
        ),
    ],
)
def test_pretool_rewrites_use_exact_tool_input_shapes(tool_name, valid, invalid):
    invocation = hook_runtime._HookInvocation(
        project_id = "project",
        event = "PreToolUse",
        event_input_json = json.dumps({"tool_name": tool_name}).encode(),
        handler_id = "handler",
        handler_json = b"{}",
        content_hash = "hash",
        workspace_identity = (1, 2),
        workspace_revision = 3,
        trust_revision = 4,
    )

    assert hook_runtime._strict_updated_input(invocation, valid) == valid
    with pytest.raises(hook_runtime.AgentWorkspaceError, match = "updatedInput"):
        hook_runtime._strict_updated_input(invocation, invalid)


def test_stop_exit_two_and_block_preserve_ordered_reasons():
    def invocation(handler_id):
        return hook_runtime._HookInvocation(
            project_id = "project",
            event = "Stop",
            event_input_json = b"{}",
            handler_id = handler_id,
            handler_json = b"{}",
            content_hash = "hash",
            workspace_identity = (1, 2),
            workspace_revision = 3,
            trust_revision = 4,
        )

    first = hook_runtime._parse_hook_output(
        invocation("first"),
        {},
        _process(exit_code = 2, stderr = "first reason"),
    )
    second = hook_runtime._parse_hook_output(
        invocation("second"),
        {},
        _process(output = json.dumps({"decision": "block", "reason": "second reason"})),
    )
    outcome = hook_runtime._aggregate("Stop", [first, second])

    assert outcome.continuation_reasons == ("first reason", "second reason")
    assert outcome.continuation_fragments == (
        ("first", "first reason"),
        ("second", "second reason"),
    )


@pytest.mark.parametrize(
    ("process", "message"),
    [
        (_process(exit_code = 2), "without a continuation reason"),
        (_process(output = json.dumps({"decision": "block"})), "without a continuation reason"),
    ],
)
def test_empty_stop_block_warns_and_finalizes(process, message):
    invocation = hook_runtime._HookInvocation(
        project_id = "project",
        event = "Stop",
        event_input_json = b"{}",
        handler_id = "handler",
        handler_json = b"{}",
        content_hash = "hash",
        workspace_identity = (1, 2),
        workspace_revision = 3,
        trust_revision = 4,
    )

    result = hook_runtime._parse_hook_output(invocation, {}, process)

    assert result.continuation_requested is False
    assert message in (result.error or "")


def test_session_generation_rejects_cross_project_and_old_id_reuse(monkeypatch):
    monkeypatch.setattr(
        hook_runtime,
        "get_chat_project",
        lambda project_id: {"id": project_id, "archived": False},
    )
    monkeypatch.setattr(
        hook_runtime,
        "get_chat_thread",
        lambda session_id: {"id": session_id, "projectId": "one", "archived": False},
    )
    first = hook_runtime._resolve_session_token("one", "same", None, create = True)
    assert first is not None
    with pytest.raises(hook_runtime.AgentWorkspaceError, match = "no longer active"):
        hook_runtime._resolve_session_token("two", "same", first, create = True)

    with hook_runtime._BACKGROUND_LOCK:
        hook_runtime._ENDED_SESSIONS.add(first)
        hook_runtime._ACTIVE_SESSIONS.pop(("one", "same"))
    second = hook_runtime._resolve_session_token("one", "same", None, create = True)
    assert second is not None and second != first
    ended = hook_runtime.end_project_hook_session(
        "one",
        session_id = "same",
        session_token = first,
    )
    assert ended.session_token == first
    assert hook_runtime.snapshot_project_hook_session("one", "same") == second


def test_feedback_has_one_aggregate_budget_and_redacts_secrets():
    large = "x" * hook_runtime.MAX_HOOK_AGGREGATE_BYTES
    outcome = hook_runtime._aggregate(
        "UserPromptSubmit",
        [
            hook_runtime.HookHandlerResult(
                "one", "passed", additional_context = "api_key=top-secret"
            ),
            hook_runtime.HookHandlerResult(
                "two",
                "failed",
                error = "password=also-secret",
                system_message = large,
            ),
        ],
    )
    combined = "".join(
        (*outcome.additional_context, *outcome.system_messages, *outcome.errors)
    ).encode()
    assert len(combined) <= hook_runtime.MAX_HOOK_AGGREGATE_BYTES
    assert b"top-secret" not in combined
    assert b"also-secret" not in combined
    assert b"[REDACTED]" in combined


def test_feedback_redacts_multiline_and_multiple_quoted_secrets():
    secret = (
        'before {"token": "line one\nline two", '
        '"client_secret": "second-value"}\n'
        "Authorization: Bearer bearer-value\n"
        "after api_key='third\nvalue' and password = plain-value"
    )
    result = hook_runtime._aggregate(
        "UserPromptSubmit",
        [
            hook_runtime.HookHandlerResult(
                "one",
                "blocked",
                blocked = True,
                reason = secret,
                additional_context = secret,
                error = secret,
            )
        ],
    )
    exposed = "\n".join(
        value
        for value in (
            result.reason,
            *result.additional_context,
            *result.errors,
        )
        if value
    )
    for leaked in (
        "line one",
        "line two",
        "second-value",
        "bearer-value",
        "third",
        "plain-value",
    ):
        assert leaked not in exposed
    assert exposed.count("[REDACTED]") >= 5


def test_feedback_redaction_handles_json_escapes_and_preserves_adjacent_prose():
    secret_json = json.dumps(
        {
            "token": 'alpha "quoted" \\ path\nnext',
            "nested": {"client-secret": "beta"},
            "safe": "keep this",
        }
    )
    prose = (
        r"""prefix token="gamma \"quoted\" \\ path" middle """
        "password='delta\nvalue' suffix stays"
    )

    redacted_json = hook_runtime.redact_project_hook_feedback(secret_json)
    redacted_prose = hook_runtime.redact_project_hook_feedback(prose)

    for leaked in ("alpha", "quoted", "path", "next", "beta"):
        assert leaked not in redacted_json
    assert "keep this" in redacted_json
    for leaked in ("gamma", "delta", "value"):
        assert leaked not in redacted_prose
    assert "prefix" in redacted_prose
    assert "middle" in redacted_prose
    assert "suffix stays" in redacted_prose


def test_feedback_redacts_nested_access_refresh_session_private_and_auth_tokens():
    payload = {
        "safe": "visible",
        "nested": [
            {
                "access_token": "access-value",
                "refresh-token": "refresh-value",
                "session_secret": "session-value\nsecond-line",
            },
            {
                "private_key": 'pk-value "quoted-value" \\ path-value',
                "authToken": "auth-value",
                "client_token": "client-value",
            },
        ],
    }

    redacted = hook_runtime.redact_project_hook_feedback(json.dumps(payload, ensure_ascii = False))

    for leaked in (
        "access-value",
        "refresh-value",
        "session-value",
        "second-line",
        "pk-value",
        "quoted-value",
        "path-value",
        "auth-value",
        "client-value",
    ):
        assert leaked not in redacted
    assert "visible" in redacted


def test_session_end_failure_always_removes_generation_state(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    _bind_runtime(monkeypatch, root, _write_hooks(root, {}))
    token = hook_runtime._resolve_session_token("project", "thread", None, create = True)
    with hook_runtime._BACKGROUND_LOCK:
        hook_runtime._STARTED_SESSIONS.add(token)
        hook_runtime._SESSION_OWNERS[token] = "gguf:one"

    monkeypatch.setattr(
        hook_runtime,
        "run_project_hook_event",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("end failed")),
    )

    with pytest.raises(RuntimeError, match = "end failed"):
        hook_runtime.end_project_hook_session(
            "project",
            session_id = "thread",
            session_token = token,
        )

    with hook_runtime._BACKGROUND_LOCK:
        assert ("project", "thread") not in hook_runtime._ACTIVE_SESSIONS
        assert token not in hook_runtime._STARTED_SESSIONS
        assert token not in hook_runtime._ENDING_SESSIONS
        assert token not in hook_runtime._ENDED_SESSIONS
        assert token not in hook_runtime._SESSION_OWNERS


def test_transaction_capture_fences_generation_and_reuses_delivery_id_on_retry(
    tmp_path, monkeypatch
):
    root = tmp_path / "repository"
    root.mkdir()
    content_hash = _write_hooks(
        root,
        {"SessionEnd": [_group(_handler("session-end"), matcher = "delete")]},
    )
    workspace, _state = _bind_runtime(monkeypatch, root, content_hash)
    executed = []

    def run(invocation, **kwargs):
        hook_runtime._revalidate_hook_invocation(
            invocation,
            workspace,
            captured_end = kwargs.get("end_snapshot") is not None,
        )
        executed.append((invocation.handler_id, json.loads(invocation.event_input_json)))
        return _process()

    monkeypatch.setattr(supervisor, "_run_trusted_project_hook_process", run)
    token = hook_runtime._resolve_session_token("project", "thread", None, create = True)
    with hook_runtime._BACKGROUND_LOCK:
        hook_runtime._STARTED_SESSIONS.add(token)

    outcome = []
    finished = threading.Event()

    def ensure_again():
        try:
            hook_runtime.ensure_project_hook_session("project", session_id = "thread")
        except Exception as exc:  # noqa: BLE001
            outcome.append(str(exc))
        finally:
            finished.set()

    with hook_runtime.capture_project_hook_session_ledgers(
        [{"project_id": "project", "session_id": "thread", "model": "model"}],
        reason = "delete",
    ) as captured:
        thread = threading.Thread(target = ensure_again)
        thread.start()
        time.sleep(0.02)
        assert not finished.is_set()
        assert token in hook_runtime._ENDING_SESSIONS
        record_id, snapshot_json = hook_runtime.serialize_project_hook_session_end_snapshot(
            captured[0]["session_end_snapshot"]
        )
        replay_snapshot = hook_runtime._deserialize_project_hook_session_end_snapshot(snapshot_json)
        assert record_id == f"{token.generation}:SessionEnd"
        assert (
            json.loads(replay_snapshot.invocations[0].event_input_json)["delivery_id"] == record_id
        )
    thread.join(1)
    assert finished.is_set()
    assert outcome == ["Project hook session generation is ending."]

    ended = hook_runtime.end_project_hook_session(
        "project",
        session_id = "thread",
        session_token = token,
        end_snapshot = captured[0]["session_end_snapshot"],
        reason = "delete",
    )
    assert ended.session_token == token
    assert ended.runs[0].status == "passed"
    assert len(executed) == 1
    assert executed[0][0] == "SessionEnd:0:0"
    assert executed[0][1]["delivery_id"] == f"{token.generation}:SessionEnd"
    replay = hook_runtime.end_project_hook_session(
        "project",
        session_id = "thread",
        session_token = token,
        end_snapshot = captured[0]["session_end_snapshot"],
        reason = "delete",
    )
    assert replay.runs[0].status == "passed"
    assert len(executed) == 2
    assert executed[1][1]["delivery_id"] == executed[0][1]["delivery_id"]
    second = hook_runtime._resolve_session_token("project", "thread", None, create = True)
    assert second != token


def test_transaction_capture_rollback_preserves_live_hook_tool_and_turn_work(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    content_hash = _write_hooks(root, {})
    _bind_runtime(monkeypatch, root, content_hash)
    token = hook_runtime._resolve_session_token("project", "thread", None, create = True)
    hook_cancel = threading.Event()
    tool_cancel = threading.Event()
    turn_cancel = threading.Event()
    with hook_runtime._BACKGROUND_LOCK:
        hook_runtime._BACKGROUND[token] = {"hook": SimpleNamespace(cancel_event = hook_cancel)}
        hook_runtime._SESSION_WORK[token] = {1: tool_cancel}
        hook_runtime._SESSION_TURNS[token] = {2: turn_cancel}

    with pytest.raises(RuntimeError, match = "database commit failed"):
        with hook_runtime.capture_project_hook_session_ledgers(
            [{"project_id": "project", "session_id": "thread", "model": "model"}],
            reason = "archive",
        ):
            assert token in hook_runtime._ENDING_SESSIONS
            assert not hook_cancel.is_set()
            assert not tool_cancel.is_set()
            assert not turn_cancel.is_set()
            raise RuntimeError("database commit failed")

    assert hook_runtime._ACTIVE_SESSIONS[("project", "thread")] == token
    assert token not in hook_runtime._ENDING_SESSIONS
    assert not hook_cancel.is_set()
    assert not tool_cancel.is_set()
    assert not turn_cancel.is_set()


def test_transaction_capture_signals_live_work_only_after_successful_commit(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    content_hash = _write_hooks(root, {})
    _bind_runtime(monkeypatch, root, content_hash)
    token = hook_runtime._resolve_session_token("project", "thread", None, create = True)
    hook_cancel = threading.Event()
    tool_cancel = threading.Event()
    turn_cancel = threading.Event()
    with hook_runtime._BACKGROUND_LOCK:
        hook_runtime._BACKGROUND[token] = {"hook": SimpleNamespace(cancel_event = hook_cancel)}
        hook_runtime._SESSION_WORK[token] = {1: tool_cancel}
        hook_runtime._SESSION_TURNS[token] = {2: turn_cancel}

    with hook_runtime.capture_project_hook_session_ledgers(
        [{"project_id": "project", "session_id": "thread", "model": "model"}],
        reason = "archive",
    ):
        assert token in hook_runtime._ENDING_SESSIONS
        assert not hook_cancel.is_set()
        assert not tool_cancel.is_set()
        assert not turn_cancel.is_set()

    assert hook_cancel.is_set()
    assert tool_cancel.is_set()
    assert turn_cancel.is_set()


def test_successful_session_end_ignores_prior_async_failure_for_delivery_status(
    tmp_path, monkeypatch
):
    root = tmp_path / "repository"
    root.mkdir()
    content_hash = _write_hooks(
        root,
        {"SessionEnd": [_group(_handler("session-end"), matcher = "delete")]},
    )
    workspace, _state = _bind_runtime(monkeypatch, root, content_hash)
    token = hook_runtime._resolve_session_token("project", "thread", None, create = True)
    monkeypatch.setattr(
        supervisor,
        "_run_trusted_project_hook_process",
        lambda invocation, **kwargs: (
            hook_runtime._revalidate_hook_invocation(
                invocation,
                workspace,
                captured_end = kwargs.get("end_snapshot") is not None,
            ),
            _process(),
        )[1],
    )
    with hook_runtime.capture_project_hook_session_ledgers(
        [{"project_id": "project", "session_id": "thread", "model": "model"}],
        reason = "delete",
    ) as captured:
        snapshot = captured[0]["session_end_snapshot"]

    def quiesce(exact_token, *, deadline):
        del deadline
        with hook_runtime._BACKGROUND_LOCK:
            hook_runtime._COMPLETED[exact_token] = [
                (
                    1,
                    hook_runtime.HookHandlerResult(
                        handler_id = "earlier-async",
                        status = "failed",
                        error = "earlier failure",
                        asynchronous = True,
                    ),
                )
            ]
        return True, ()

    monkeypatch.setattr(hook_runtime, "_wait_for_project_hook_session_quiescence", quiesce)

    ended = hook_runtime.end_project_hook_session(
        "project",
        session_id = "thread",
        session_token = token,
        end_snapshot = snapshot,
        reason = "delete",
    )

    assert ended.errors == ()
    assert [(run.handler_id, run.status) for run in ended.runs] == [("SessionEnd:0:0", "passed")]


def test_session_end_delivery_ownership_loss_prevents_hook_spawn(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    content_hash = _write_hooks(
        root,
        {"SessionEnd": [_group(_handler("session-end"), matcher = "delete")]},
    )
    _bind_runtime(monkeypatch, root, content_hash)
    token = hook_runtime._resolve_session_token("project", "thread", None, create = True)
    with hook_runtime.capture_project_hook_session_ledgers(
        [{"project_id": "project", "session_id": "thread", "model": "model"}],
        reason = "delete",
    ) as captured:
        snapshot = captured[0]["session_end_snapshot"]
    spawned = []
    monkeypatch.setattr(
        supervisor,
        "_run_trusted_project_hook_process",
        lambda *_args, **_kwargs: spawned.append(True),
    )
    ownership_lost = threading.Event()
    ownership_lost.set()

    result = hook_runtime.end_project_hook_session(
        "project",
        session_id = "thread",
        session_token = token,
        end_snapshot = snapshot,
        reason = "delete",
        delivery_cancel_event = ownership_lost,
    )

    assert result.errors == ("Project SessionEnd delivery ownership was lost.",)
    assert spawned == []


def test_durable_shutdown_persists_nonquiescent_generation_for_restart(tmp_path, monkeypatch):
    from storage import studio_db

    root = tmp_path / "repository"
    root.mkdir()
    content_hash = _write_hooks(root, {})
    _bind_runtime(monkeypatch, root, content_hash)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio-home"))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "studio-home" / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    token = hook_runtime._resolve_session_token("project", "thread", None, create = True)
    monkeypatch.setattr(
        hook_runtime,
        "_wait_for_project_hook_session_quiescence",
        lambda *_args, **_kwargs: (False, ("model turn",)),
    )

    results = hook_runtime.end_all_project_hook_sessions(
        reason = "shutdown",
        durable = True,
    )

    assert results and "incomplete" in results[0].errors[0]
    connection = studio_db.get_connection()
    try:
        row = connection.execute(
            """
            SELECT id, consumed_at, attempt_count, snapshot_json
            FROM project_hook_session_end_outbox
            """
        ).fetchone()
    finally:
        connection.close()
    assert row["id"] == f"{token.generation}:SessionEnd"
    assert row["consumed_at"] is None
    assert row["attempt_count"] == 1
    assert json.loads(row["snapshot_json"])["token"]["generation"] == token.generation
    assert token in hook_runtime._ENDING_SESSIONS
    assert token not in hook_runtime._DEFERRED_SESSION_ENDS


@pytest.mark.parametrize("reason", ["archive", "other"], ids = ["archive", "folder-switch"])
def test_empty_session_end_snapshot_retries_quiescence_then_allows_readmission(
    tmp_path, monkeypatch, reason
):
    from storage import studio_db

    root = tmp_path / "repository"
    root.mkdir()
    content_hash = _write_hooks(root, {})
    _bind_runtime(monkeypatch, root, content_hash)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio-home"))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "studio-home" / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    monkeypatch.setattr(hook_runtime, "SESSION_END_DRAIN_TIMEOUT_SECONDS", 0.03)
    token = hook_runtime._resolve_session_token("project", "thread", None, create = True)
    entered = threading.Event()
    release = threading.Event()

    def hold_work():
        with hook_runtime.project_hook_session_work(token):
            entered.set()
            release.wait(2)

    worker = threading.Thread(target = hold_work)
    worker.start()
    assert entered.wait(1)
    connection = studio_db.get_connection()
    try:
        with hook_runtime.project_hook_admission_fence():
            connection.execute("BEGIN IMMEDIATE")
            with hook_runtime.capture_project_hook_session_ledgers(
                [{"project_id": "project", "session_id": "thread", "model": "model"}],
                reason = reason,
            ) as captured:
                snapshot = captured[0]["session_end_snapshot"]
                assert snapshot.invocations == ()
                record_id, snapshot_json = hook_runtime.serialize_project_hook_session_end_snapshot(
                    snapshot
                )
                studio_db.enqueue_project_hook_session_end_outbox(
                    connection,
                    [(record_id, snapshot_json, "project")],
                )
                connection.commit()
    finally:
        connection.close()

    first = hook_runtime.end_project_hook_session(
        "project",
        session_id = "thread",
        session_token = token,
        end_snapshot = snapshot,
        reason = reason,
    )
    assert first.errors and "incomplete" in first.errors[0]
    assert token in hook_runtime._ENDING_SESSIONS

    release.set()
    worker.join(1)
    recovered = hook_runtime.recover_pending_project_hook_session_ends()

    assert len(recovered) == 1
    assert recovered[0].errors == ()
    assert recovered[0].runs == ()
    with hook_runtime._BACKGROUND_LOCK:
        assert token not in hook_runtime._ENDING_SESSIONS
        assert token not in hook_runtime._SESSION_WORK
        assert ("project", "thread") not in hook_runtime._ACTIVE_SESSIONS
    replacement = hook_runtime._resolve_session_token("project", "thread", None, create = True)
    assert replacement != token


def test_durable_shutdown_keeps_good_session_when_bad_token_is_first(tmp_path, monkeypatch):
    from storage import studio_db

    root = tmp_path / "repository"
    root.mkdir()
    content_hash = _write_hooks(root, {})
    workspace, _state = _bind_runtime(monkeypatch, root, content_hash)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio-home"))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "studio-home" / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    good = hook_runtime._resolve_session_token("project", "thread-good", None, create = True)
    bad = hook_runtime._new_session_token("bad-project", "thread-bad")
    with hook_runtime._BACKGROUND_LOCK:
        hook_runtime._ACTIVE_SESSIONS.pop((good.project_id, good.session_id))
        hook_runtime._ACTIVE_SESSIONS[(bad.project_id, bad.session_id)] = bad
        hook_runtime._ACTIVE_SESSIONS[(good.project_id, good.session_id)] = good

    @contextmanager
    def access(project_id, **_kwargs):
        if project_id == "bad-project":
            raise hook_runtime.AgentWorkspaceError("bad capture sentinel")
        yield workspace

    monkeypatch.setattr(hook_runtime, "project_workspace_access", access)

    persisted, failures = hook_runtime._persist_all_project_hook_session_ends(reason = "shutdown")

    assert persisted == 1
    assert len(failures) == 1
    assert failures[0].session_token == bad
    assert failures[0].errors == ("Project SessionEnd durability failed: bad capture sentinel",)
    connection = studio_db.get_connection()
    try:
        rows = connection.execute(
            "SELECT id, consumed_at FROM project_hook_session_end_outbox"
        ).fetchall()
    finally:
        connection.close()
    assert [(row["id"], row["consumed_at"]) for row in rows] == [
        (f"{good.generation}:SessionEnd", None)
    ]
    assert good in hook_runtime._ENDING_SESSIONS
    assert bad not in hook_runtime._ENDING_SESSIONS


def test_failed_outbox_recovery_fences_same_thread_start_until_consumed(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    _bind_runtime(monkeypatch, root, _write_hooks(root, {}))
    pending = {"value": True}
    monkeypatch.setattr(
        hook_runtime,
        "project_hook_session_end_pending_for",
        lambda project_id, session_id: (
            pending["value"] and project_id == "project" and session_id == "thread"
        ),
    )

    with pytest.raises(hook_runtime.AgentWorkspaceError, match = "prior SessionEnd"):
        hook_runtime.ensure_project_hook_session("project", session_id = "thread")
    assert hook_runtime.snapshot_project_hook_session("project", "thread") is None

    pending["value"] = False
    started = hook_runtime.ensure_project_hook_session("project", session_id = "thread")
    assert started.session_token is not None


def test_thread_project_handoff_delivers_old_end_before_new_start(monkeypatch):
    from routes import chat_history

    entered_end = threading.Event()
    release_end = threading.Event()
    end_finished = threading.Event()
    started_new = threading.Event()
    timeline = []

    monkeypatch.setattr(chat_history, "get_chat_project", lambda _project_id: {"id": "project-b"})
    monkeypatch.setattr(
        chat_history,
        "upsert_chat_thread",
        lambda _thread, **_kwargs: {
            "id": "thread",
            "hookSessions": [{"project_id": "project-a", "session_id": "thread"}],
        },
    )
    monkeypatch.setattr(chat_history, "thread_from_row", lambda row: row)

    def end_old(_records, *, reason):
        assert reason == "other"
        timeline.append("A SessionEnd")
        entered_end.set()
        assert release_end.wait(1)
        end_finished.set()

    monkeypatch.setattr(chat_history, "_end_project_hook_sessions", end_old)
    monkeypatch.setattr(
        hook_runtime,
        "get_chat_project",
        lambda project_id: {"id": project_id, "archived": False},
    )
    monkeypatch.setattr(
        hook_runtime,
        "get_chat_thread",
        lambda session_id: {
            "id": session_id,
            "projectId": "project-b",
            "archived": False,
        },
    )
    monkeypatch.setattr(
        hook_runtime,
        "project_hook_session_end_pending_for",
        lambda _project_id, _session_id: not end_finished.is_set(),
    )

    payload = chat_history.ChatThread(
        id = "thread",
        modelType = "base",
        projectId = "project-b",
        createdAt = 1,
    )

    move = threading.Thread(target = lambda: chat_history.save_thread(payload, "subject"))

    def start_new():
        while True:
            try:
                hook_runtime._resolve_session_token("project-b", "thread", None, create = True)
                break
            except hook_runtime.AgentWorkspaceError as exc:
                assert "prior SessionEnd" in str(exc)
                release_end.wait(0.01)
        timeline.append("B SessionStart")
        started_new.set()

    move.start()
    assert entered_end.wait(1)
    start = threading.Thread(target = start_new)
    start.start()
    time.sleep(0.02)
    assert not started_new.is_set()
    release_end.set()
    move.join(1)
    start.join(1)

    assert not move.is_alive()
    assert not start.is_alive()
    assert timeline == ["A SessionEnd", "B SessionStart"]


def test_project_capture_includes_active_synthetic_generations(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    content_hash = _write_hooks(root, {})
    _bind_runtime(monkeypatch, root, content_hash)
    stored = hook_runtime._resolve_session_token("project", "stored", None, create = True)
    synthetic = hook_runtime._resolve_session_token(
        "project", "server-owned-synthetic", None, create = True
    )
    original_thread = hook_runtime.get_chat_thread
    monkeypatch.setattr(
        hook_runtime,
        "get_chat_thread",
        lambda session_id: (
            {"id": session_id, "projectId": "other", "archived": False}
            if session_id == "unrelated"
            else original_thread(session_id)
        ),
    )
    unrelated = hook_runtime._resolve_session_token("other", "unrelated", None, create = True)

    with hook_runtime.capture_project_hook_session_ledgers(
        [{"project_id": "project", "session_id": "stored", "model": "model"}],
        project_ids = ("project",),
        reason = "delete",
    ) as captured:
        assert {entry["session_token"] for entry in captured} == {stored, synthetic}
        assert all(entry["session_token"] in hook_runtime._ENDING_SESSIONS for entry in captured)
        assert unrelated not in hook_runtime._ENDING_SESSIONS


def test_session_end_uses_reserved_finalizer_when_normal_work_ignores_cancel(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    content_hash = _write_hooks(
        root,
        {
            "UserPromptSubmit": [_group(_handler("blocked-worker"))],
            "SessionEnd": [_group(_handler("session-end"), matcher = "other")],
        },
    )
    workspace, _state = _bind_runtime(monkeypatch, root, content_hash)
    worker_started = threading.Event()
    worker_release = threading.Event()
    end_executed = threading.Event()

    def run(invocation, **_kwargs):
        hook_runtime._revalidate_hook_invocation(invocation, workspace)
        if invocation.event == "UserPromptSubmit":
            worker_started.set()
            worker_release.wait(2)
        else:
            end_executed.set()
        return _process()

    monkeypatch.setattr(supervisor, "_run_trusted_project_hook_process", run)
    monkeypatch.setattr(hook_runtime, "SESSION_END_DRAIN_TIMEOUT_SECONDS", 0.01)
    prompt_result = []

    def prompt():
        prompt_result.append(
            hook_runtime.run_project_hook_event(
                "project",
                "UserPromptSubmit",
                {"prompt": "hello"},
                session_id = "thread",
            )
        )

    thread = threading.Thread(target = prompt)
    thread.start()
    assert worker_started.wait(1)
    started_at = time.monotonic()
    ended = hook_runtime.end_project_hook_session("project", session_id = "thread")
    elapsed = time.monotonic() - started_at

    assert elapsed < 1
    assert not end_executed.is_set()
    assert ended.runs == ()
    assert "incomplete" in ended.errors[0]
    worker_release.set()
    thread.join(1)
    assert prompt_result
    deadline = time.monotonic() + 1
    while not end_executed.is_set() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert end_executed.is_set()
    assert hook_runtime.snapshot_project_hook_session("project", "thread") is None


def test_owner_switch_waits_for_old_cancellation_insensitive_work(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    _bind_runtime(monkeypatch, root, _write_hooks(root, {}))
    monkeypatch.setattr(hook_runtime, "SESSION_END_DRAIN_TIMEOUT_SECONDS", 0.01)
    token = hook_runtime._resolve_session_token("project", "thread", None, create = True)
    with hook_runtime._BACKGROUND_LOCK:
        hook_runtime._STARTED_SESSIONS.add(token)
        hook_runtime._SESSION_OWNERS[token] = "runtime-a"

    work_started = threading.Event()
    release_work = threading.Event()

    def old_work():
        with hook_runtime.project_hook_session_work(token):
            work_started.set()
            release_work.wait(1)

    worker = threading.Thread(target = old_work)
    worker.start()
    assert work_started.wait(1)
    next_turn = hook_runtime.ProjectHookTurn(
        project_id = "project",
        session_id = "thread",
        turn_id = "turn-b",
        model = "model-b",
        permission_mode = "default",
        cancel_event = threading.Event(),
        runtime_owner = "runtime-b",
    )

    with hook_runtime.activate_project_hook_turn(next_turn):
        with pytest.raises(hook_runtime.AgentWorkspaceError, match = "incomplete"):
            hook_runtime.ensure_project_hook_session(
                "project",
                session_id = "thread",
                model = "model-b",
            )

    with hook_runtime._BACKGROUND_LOCK:
        assert hook_runtime._ACTIVE_SESSIONS[("project", "thread")] == token
        assert hook_runtime._SESSION_OWNERS[token] == "runtime-a"
        assert token in hook_runtime._ENDING_SESSIONS
    release_work.set()
    worker.join(1)
    deadline = time.monotonic() + 1
    while (
        hook_runtime.snapshot_project_hook_session("project", "thread") is not None
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    assert hook_runtime.snapshot_project_hook_session("project", "thread") is None


def test_session_end_reserved_capacity_is_independent(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    content_hash = _write_hooks(
        root,
        {"SessionEnd": [_group(_handler("session-end"), matcher = "other")]},
    )
    workspace, _state = _bind_runtime(monkeypatch, root, content_hash)
    executed = threading.Event()

    def run(invocation, **_kwargs):
        hook_runtime._revalidate_hook_invocation(invocation, workspace)
        executed.set()
        return _process()

    monkeypatch.setattr(supervisor, "_run_trusted_project_hook_process", run)
    token = hook_runtime._resolve_session_token("project", "thread", None, create = True)
    with hook_runtime._BACKGROUND_LOCK:
        hook_runtime._STARTED_SESSIONS.add(token)
        hook_runtime._GLOBAL_HOOK_WORK = hook_runtime.MAX_GLOBAL_HOOK_WORK

    ended = hook_runtime.end_project_hook_session(
        "project",
        session_id = "thread",
        session_token = token,
    )
    assert executed.is_set()
    assert ended.runs[0].status == "passed"


def test_session_end_capacity_wait_obeys_absolute_deadline(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    content_hash = _write_hooks(
        root,
        {
            "SessionEnd": [
                _group(
                    _handler("first", timeout = 1),
                    _handler("must-not-run", timeout = 1),
                    matcher = "other",
                )
            ]
        },
    )
    _bind_runtime(monkeypatch, root, content_hash)
    first_started = threading.Event()
    release_first = threading.Event()
    executed = []

    def run(invocation, **_kwargs):
        executed.append(invocation.handler_id)
        first_started.set()
        release_first.wait(1)
        return _process()

    monkeypatch.setattr(supervisor, "_run_trusted_project_hook_process", run)
    monkeypatch.setattr(hook_runtime, "MAX_GLOBAL_SESSION_END_WORK", 1)
    token = hook_runtime._resolve_session_token("project", "thread", None, create = True)
    with hook_runtime._BACKGROUND_LOCK:
        hook_runtime._STARTED_SESSIONS.add(token)

    started = time.monotonic()
    try:
        outcome = hook_runtime.end_project_hook_session(
            "project",
            session_id = "thread",
            session_token = token,
            deadline_monotonic = time.monotonic() + 0.04,
        )
    finally:
        release_first.set()

    assert first_started.is_set()
    assert time.monotonic() - started < 0.5
    assert [run.status for run in outcome.runs] == ["timed_out", "timed_out"]
    assert executed == ["SessionEnd:0:0"]


def test_session_end_ninth_handler_gets_timeout_when_worker_actually_starts(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    content_hash = _write_hooks(
        root,
        {
            "SessionEnd": [
                _group(
                    *(_handler(f"handler-{index}", timeout = 1) for index in range(9)),
                    matcher = "other",
                )
            ]
        },
    )
    _bind_runtime(monkeypatch, root, content_hash)
    budgets = {}

    def run(invocation, **_kwargs):
        index = int(invocation.handler_id.rsplit(":", 1)[1])
        budgets[index] = invocation.deadline_monotonic - time.monotonic()
        time.sleep(0.7 if index < 8 else 0.4)
        if time.monotonic() >= invocation.deadline_monotonic:
            return _process(status = "timed_out", exit_code = None)
        return _process()

    monkeypatch.setattr(supervisor, "_run_trusted_project_hook_process", run)
    token = hook_runtime._resolve_session_token("project", "thread", None, create = True)
    with hook_runtime._BACKGROUND_LOCK:
        hook_runtime._STARTED_SESSIONS.add(token)

    outcome = hook_runtime.end_project_hook_session(
        "project",
        session_id = "thread",
        session_token = token,
    )

    assert len(outcome.runs) == 9
    assert {result.status for result in outcome.runs} == {"passed"}
    assert budgets.keys() == set(range(9))
    assert min(budgets.values()) > 0.9


def test_session_end_runs_more_handlers_than_global_capacity_concurrently(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    content_hash = _write_hooks(
        root,
        {
            "SessionEnd": [
                _group(
                    *(_handler(f"handler-{index}") for index in range(33)),
                    matcher = "other",
                )
            ]
        },
    )
    _bind_runtime(monkeypatch, root, content_hash)
    siblings_started = threading.Barrier(2)
    lock = threading.Lock()
    executed = []
    budgets = []
    active = 0
    peak = 0

    def run(invocation, **_kwargs):
        nonlocal active, peak
        with lock:
            executed.append(invocation.handler_id)
            budgets.append(invocation.deadline_monotonic - time.monotonic())
            active += 1
            peak = max(peak, active)
        try:
            if invocation.handler_id in {"SessionEnd:0:0", "SessionEnd:0:1"}:
                siblings_started.wait(1)
            time.sleep(0.28)
            if time.monotonic() >= invocation.deadline_monotonic:
                return _process(status = "timed_out", exit_code = None)
            return _process()
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(supervisor, "_run_trusted_project_hook_process", run)
    token = hook_runtime._resolve_session_token("project", "thread", None, create = True)
    with hook_runtime._BACKGROUND_LOCK:
        hook_runtime._STARTED_SESSIONS.add(token)

    outcome = hook_runtime.end_project_hook_session(
        "project",
        session_id = "thread",
        session_token = token,
    )

    assert len(outcome.runs) == 33
    assert {result.status for result in outcome.runs} == {"passed"}
    assert set(executed) == {f"SessionEnd:0:{index}" for index in range(33)}
    assert peak > 1
    assert min(budgets) > 0.9


def test_session_end_outbox_retry_consumes_more_handlers_than_global_capacity(
    tmp_path, monkeypatch
):
    from core.inference import tools
    from storage import studio_db

    root = tmp_path / "repository"
    root.mkdir()
    content_hash = _write_hooks(
        root,
        {
            "SessionEnd": [
                _group(
                    *(_handler(f"handler-{index}") for index in range(33)),
                    matcher = "delete",
                )
            ]
        },
    )
    _bind_runtime(monkeypatch, root, content_hash)
    executed = []
    budgets = []

    def run(invocation, **_kwargs):
        executed.append(invocation.handler_id)
        budgets.append(invocation.deadline_monotonic - time.monotonic())
        time.sleep(0.28)
        if time.monotonic() >= invocation.deadline_monotonic:
            return _process(status = "timed_out", exit_code = None)
        return _process()

    monkeypatch.setattr(supervisor, "_run_trusted_project_hook_process", run)
    token = hook_runtime._resolve_session_token("project", "thread", None, create = True)
    with hook_runtime._BACKGROUND_LOCK:
        hook_runtime._STARTED_SESSIONS.add(token)
    with hook_runtime.capture_project_hook_session_ledgers(
        [{"project_id": "project", "session_id": "thread", "model": "model"}],
        reason = "delete",
    ) as captured:
        record_id, snapshot_json = hook_runtime.serialize_project_hook_session_end_snapshot(
            captured[0]["session_end_snapshot"]
        )

    records = [
        {
            "id": record_id,
            "snapshot_json": snapshot_json,
            "attempt_count": 1,
        }
    ]
    consumed = []
    collected = []

    def claim(_claim_owner, *, limit):
        assert limit == 1
        return [records.pop(0)] if records else []

    @contextmanager
    def heartbeat(_record_id, _claim_owner):
        yield threading.Event()

    monkeypatch.setattr(studio_db, "claim_pending_project_hook_session_end_outbox", claim)
    monkeypatch.setattr(studio_db, "project_hook_session_end_claim_heartbeat", heartbeat)
    monkeypatch.setattr(
        studio_db,
        "mark_project_hook_session_end_outbox_consumed",
        lambda delivery_id, **_kwargs: consumed.append(delivery_id) or True,
    )
    monkeypatch.setattr(
        studio_db,
        "mark_project_hook_session_end_outbox_failed",
        lambda *_args, **_kwargs: pytest.fail("valid SessionEnd delivery was retried"),
    )
    monkeypatch.setattr(
        tools,
        "collect_orphaned_project_workspaces",
        lambda: collected.append("released"),
    )

    recovered = hook_runtime.recover_pending_project_hook_session_ends(limit = 1)

    assert len(recovered) == 1
    assert len(recovered[0].runs) == 33
    assert {result.status for result in recovered[0].runs} == {"passed"}
    assert set(executed) == {f"SessionEnd:0:{index}" for index in range(33)}
    assert min(budgets) > 0.9
    assert consumed == [record_id]
    assert collected == ["released"]


def test_session_end_ignored_timeout_returns_explicit_failure(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    content_hash = _write_hooks(
        root,
        {"SessionEnd": [_group(_handler("session-end", timeout = 1), matcher = "other")]},
    )
    _bind_runtime(monkeypatch, root, content_hash)
    started = threading.Event()
    release = threading.Event()

    def run(_invocation, **_kwargs):
        started.set()
        release.wait(3)
        return _process()

    monkeypatch.setattr(supervisor, "_run_trusted_project_hook_process", run)
    monkeypatch.setattr(hook_runtime, "SESSION_END_FUTURE_GRACE_SECONDS", 0.01)
    token = hook_runtime._resolve_session_token("project", "thread", None, create = True)
    with hook_runtime._BACKGROUND_LOCK:
        hook_runtime._STARTED_SESSIONS.add(token)

    started_at = time.monotonic()
    ended = hook_runtime.end_project_hook_session(
        "project",
        session_id = "thread",
        session_token = token,
    )
    elapsed = time.monotonic() - started_at
    release.set()

    assert started.is_set()
    assert elapsed < 2
    assert ended.runs[0].status == "timed_out"
    assert ended.runs[0].error == "Project SessionEnd hook did not finish within its timeout."


def test_global_background_capacity_rejects_deterministically(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    content_hash = _write_hooks(
        root,
        {"PostToolUse": [_group(_handler("background", **{"async": True}))]},
    )
    _bind_runtime(monkeypatch, root, content_hash)
    started = threading.Event()
    release = threading.Event()

    def run(_invocation, **_kwargs):
        started.set()
        release.wait(1)
        return _process()

    monkeypatch.setattr(supervisor, "_run_trusted_project_hook_process", run)
    monkeypatch.setattr(hook_runtime, "MAX_GLOBAL_HOOK_WORK", 1)
    first = hook_runtime.run_project_hook_event(
        "project",
        "PostToolUse",
        {"tool_name": "Bash"},
        session_id = "one",
    )
    assert first.runs[0].status == "scheduled"
    assert started.wait(1)
    second = hook_runtime.run_project_hook_event(
        "project",
        "PostToolUse",
        {"tool_name": "Bash"},
        session_id = "two",
    )
    assert second.runs[0].status == "failed"
    assert second.runs[0].error == "Project hook process capacity is full."
    release.set()


def test_runtime_owner_end_is_scoped(monkeypatch):
    monkeypatch.setattr(
        hook_runtime,
        "get_chat_project",
        lambda project_id: {"id": project_id, "archived": False},
    )
    monkeypatch.setattr(
        hook_runtime,
        "get_chat_thread",
        lambda session_id: {"id": session_id, "projectId": "project", "archived": False},
    )
    first = hook_runtime._resolve_session_token("project", "first", None, create = True)
    second = hook_runtime._resolve_session_token("project", "second", None, create = True)
    hook_runtime.bind_project_hook_session_owner(first, "gguf:one")
    hook_runtime.bind_project_hook_session_owner(second, "external:one")
    ended = []

    def end(project_id, **kwargs):
        ended.append((project_id, kwargs["session_token"]))
        return hook_runtime.HookEventResult(event = "SessionEnd")

    monkeypatch.setattr(hook_runtime, "end_project_hook_session", end)
    hook_runtime.end_project_hook_sessions_for_owner("gguf:one", reason = "idle")
    assert ended == [("project", first)]


def test_unsupported_discovery_noops_only_when_config_is_absent(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    workspace, _state = _bind_runtime(monkeypatch, root, "unused")
    monkeypatch.setattr(hook_runtime, "secure_project_hook_discovery_supported", lambda: False)

    outcome = hook_runtime.run_project_hook_event(
        "project",
        "UserPromptSubmit",
        {"prompt": "hello"},
        session_id = "thread",
    )
    assert outcome.runs == ()

    (root / ".codex").mkdir()
    (root / ".codex" / "hooks.json").write_text("{}", encoding = "utf-8")
    monkeypatch.setattr(
        hook_runtime,
        "discover_project_hooks",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            hook_runtime.AgentWorkspaceError(
                "Project hook discovery is unavailable without secure directory traversal."
            )
        ),
    )
    failed = hook_runtime.run_project_hook_event(
        "project",
        "UserPromptSubmit",
        {"prompt": "hello"},
        session_id = "thread",
    )
    assert failed.runs[0].status == "failed"
    assert "secure directory traversal" in hook_runtime.project_hook_control_failure(failed)


def test_project_command_capability_revalidates_policy_immediately_before_spawn(
    tmp_path, monkeypatch
):
    root = tmp_path / "repository"
    root.mkdir()
    workspace = SimpleNamespace(
        project_id = "project",
        root = root,
        device_id = root.stat().st_dev,
        file_id = root.stat().st_ino,
        revision = 5,
    )
    argv = ("/bin/sh", "-c", "printf ok")
    proof = {
        "projectId": "project",
        "command": "printf ok",
        "argv": list(argv),
        "policyHash": "reviewed",
        "workspaceIdentity": [workspace.device_id, workspace.file_id],
        "workspaceRevision": 5,
        "approved": True,
    }
    capability = supervisor._bind_project_command_capability("project", "printf ok", argv, proof)

    @contextmanager
    def access(project_id, **_kwargs):
        assert project_id == "project"
        yield workspace

    from core.agent_workspace import rules

    monkeypatch.setattr(supervisor.common, "project_workspace_access", access)
    monkeypatch.setattr(supervisor, "spawn_on_lifetime_thread", lambda callback: callback())
    monkeypatch.setattr(rules, "secure_command_rule_traversal_supported", lambda: True)
    monkeypatch.setattr(rules, "discover_project_command_rules", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        rules,
        "evaluate_terminal_command_rules",
        lambda *_args, **_kwargs: {"policyHash": "changed", "decision": "allow"},
    )
    spawned = []
    monkeypatch.setattr(
        supervisor.subprocess,
        "Popen",
        lambda *args, **kwargs: spawned.append((args, kwargs)),
    )

    with pytest.raises(hook_runtime.AgentWorkspaceError, match = "policy changed"):
        supervisor._spawn_authorized_project_host_command(
            capability,
            {"cwd": str(root)},
        )
    assert spawned == []


def test_session_end_outbox_recovery_drains_beyond_one_page_fairly(monkeypatch):
    from storage import studio_db

    records = {
        f"generation-{index}:SessionEnd": {
            "id": f"generation-{index}:SessionEnd",
            "snapshot_json": str(index),
            "attempt_count": 0,
        }
        for index in range(300)
    }
    consumed = []

    def pending(_claim_owner, *, limit):
        return list(records.values())[:limit]

    def deserialize(raw):
        token = hook_runtime.HookSessionToken("project", f"thread-{raw}", f"generation-{raw}")
        return SimpleNamespace(token = token)

    def mark_consumed(record_id, **_kwargs):
        consumed.append(record_id)
        records.pop(record_id, None)

    monkeypatch.setattr(studio_db, "claim_pending_project_hook_session_end_outbox", pending)
    monkeypatch.setattr(studio_db, "mark_project_hook_session_end_outbox_consumed", mark_consumed)
    monkeypatch.setattr(
        studio_db,
        "mark_project_hook_session_end_outbox_failed",
        lambda record_id, _error, **_kwargs: records.pop(record_id, None),
    )
    monkeypatch.setattr(hook_runtime, "_deserialize_project_hook_session_end_snapshot", deserialize)
    monkeypatch.setattr(
        hook_runtime,
        "end_project_hook_session",
        lambda *_args, **_kwargs: hook_runtime.HookEventResult(event = "SessionEnd"),
    )

    recovered = hook_runtime.recover_pending_project_hook_session_ends()

    assert len(recovered) == 300
    assert len(consumed) == 300
    assert len(set(consumed)) == 300
    assert records == {}


def test_recovery_consumption_releases_managed_workspace_cleanup(monkeypatch):
    from core.inference import tools
    from storage import studio_db

    records = [
        {
            "id": "generation:SessionEnd",
            "snapshot_json": "snapshot",
            "attempt_count": 0,
        }
    ]
    token = hook_runtime.HookSessionToken("project", "thread", "generation")
    collected = []

    def claim(_claim_owner, *, limit):
        del limit
        return [records.pop(0)] if records else []

    monkeypatch.setattr(studio_db, "claim_pending_project_hook_session_end_outbox", claim)
    monkeypatch.setattr(
        studio_db,
        "mark_project_hook_session_end_outbox_consumed",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        studio_db,
        "mark_project_hook_session_end_outbox_failed",
        lambda *_args, **_kwargs: pytest.fail("successful finalizer was retried"),
    )
    monkeypatch.setattr(
        hook_runtime,
        "_deserialize_project_hook_session_end_snapshot",
        lambda _raw: SimpleNamespace(token = token),
    )
    monkeypatch.setattr(
        hook_runtime,
        "end_project_hook_session",
        lambda *_args, **_kwargs: hook_runtime.HookEventResult(event = "SessionEnd"),
    )
    monkeypatch.setattr(
        tools,
        "collect_orphaned_project_workspaces",
        lambda: collected.append("released"),
    )

    recovered = hook_runtime.recover_pending_project_hook_session_ends()

    assert len(recovered) == 1
    assert collected == ["released"]


def test_session_end_outbox_poison_record_does_not_starve_later_delivery(monkeypatch):
    from storage import studio_db

    records = {
        "poison:SessionEnd": {
            "id": "poison:SessionEnd",
            "snapshot_json": "poison",
            "attempt_count": 0,
        },
        "healthy:SessionEnd": {
            "id": "healthy:SessionEnd",
            "snapshot_json": "healthy",
            "attempt_count": 0,
        },
    }
    failed = []
    consumed = []

    def pending(_claim_owner, *, limit):
        return list(records.values())[:limit]

    def deserialize(raw):
        if raw == "poison":
            raise hook_runtime.AgentWorkspaceError("invalid snapshot")
        return SimpleNamespace(token = hook_runtime.HookSessionToken("project", "thread", "healthy"))

    def mark_failed(record_id, error, **_kwargs):
        failed.append((record_id, error))
        records.pop(record_id, None)

    def mark_consumed(record_id, **_kwargs):
        consumed.append(record_id)
        records.pop(record_id, None)

    monkeypatch.setattr(studio_db, "claim_pending_project_hook_session_end_outbox", pending)
    monkeypatch.setattr(studio_db, "mark_project_hook_session_end_outbox_failed", mark_failed)
    monkeypatch.setattr(studio_db, "mark_project_hook_session_end_outbox_consumed", mark_consumed)
    monkeypatch.setattr(hook_runtime, "_deserialize_project_hook_session_end_snapshot", deserialize)
    monkeypatch.setattr(
        hook_runtime,
        "end_project_hook_session",
        lambda *_args, **_kwargs: hook_runtime.HookEventResult(event = "SessionEnd"),
    )

    recovered = hook_runtime.recover_pending_project_hook_session_ends()

    assert len(recovered) == 2
    assert failed[0][0] == "poison:SessionEnd"
    assert consumed == ["healthy:SessionEnd"]
