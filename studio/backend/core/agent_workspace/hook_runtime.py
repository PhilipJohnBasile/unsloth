# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Execute reviewed project hooks inside the supervised workspace boundary."""

from __future__ import annotations

import concurrent.futures
import asyncio
import contextvars
import functools
import html
import inspect
import json
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace as dataclass_replace
from pathlib import Path
from typing import Any, Optional

from storage.project_hook_trust_db import (
    ProjectHookTrustStateError,
    get_project_hook_trust_state,
)
from storage.studio_db import (
    get_chat_project,
    get_chat_thread,
    project_hook_session_end_pending_for,
)

from .common import AgentWorkspaceError, ProjectWorkspace, project_workspace_access
from .execution import ProjectExecutionUnavailable
from .hooks import (
    HOOK_EVENTS,
    MAX_HOOK_TIMEOUT_SECONDS,
    discover_project_hooks,
    matching_project_hooks,
    secure_project_hook_discovery_supported,
)
from . import supervisor


MAX_HOOK_EVENT_BYTES = 256 * 1024
MAX_HOOK_OUTPUT_BYTES = 256 * 1024
MAX_HOOK_TEXT_BYTES = 256 * 1024
MAX_BACKGROUND_HOOKS = 8
MAX_BACKGROUND_QUEUE = 64
MAX_COMPLETED_HOOKS = 64
MAX_GLOBAL_HOOK_WORK = 256
MAX_GLOBAL_SESSION_END_WORK = 32
MAX_GLOBAL_COMPLETED_HOOKS = 512
MAX_HOOK_AGGREGATE_BYTES = 128 * 1024
MAX_HOOK_EVENT_PREPARATION_SECONDS = float(MAX_HOOK_TIMEOUT_SECONDS)
MAX_SESSION_END_SNAPSHOT_BYTES = 4 * 1024 * 1024
MAX_STOP_CONTINUATIONS = 2
MAX_STOP_PROMPT_BYTES = 10_000
MAX_STOP_CANDIDATE_BYTES = MAX_HOOK_EVENT_BYTES - 4096
SESSION_END_DRAIN_TIMEOUT_SECONDS = 5.0
SESSION_END_FUTURE_GRACE_SECONDS = 0.5
_PROJECT_SESSION_PREFIX = "project-"
_PERMISSION_MODES = frozenset({"default", "acceptEdits", "plan", "dontAsk", "bypassPermissions"})
_PLAIN_CONTEXT_EVENTS = frozenset({"SessionStart", "UserPromptSubmit", "SubagentStart"})
_CONTEXT_EVENTS = frozenset(
    {
        "SessionStart",
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "SubagentStart",
    }
)
_CONTINUE_EVENTS = frozenset(
    {
        "SessionStart",
        "UserPromptSubmit",
        "PostToolUse",
        "PreCompact",
        "PostCompact",
        "SubagentStop",
        "Stop",
    }
)


@dataclass(frozen = True)
class HookHandlerResult:
    handler_id: str
    status: str
    exit_code: Optional[int] = None
    blocked: bool = False
    reason: Optional[str] = None
    updated_input: Optional[dict[str, Any]] = None
    permission_decision: Optional[str] = None
    additional_context: Optional[str] = None
    system_message: Optional[str] = None
    status_message: Optional[str] = None
    error: Optional[str] = None
    continuation_requested: bool = False
    stop_requested: bool = False
    asynchronous: bool = False


@dataclass(frozen = True)
class HookEventResult:
    event: str
    runs: tuple[HookHandlerResult, ...] = ()
    blocked: bool = False
    reason: Optional[str] = None
    updated_input: Optional[dict[str, Any]] = None
    permission_decision: Optional[str] = None
    additional_context: tuple[str, ...] = ()
    system_messages: tuple[str, ...] = ()
    status_messages: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    continuation_reason: Optional[str] = None
    continuation_reasons: tuple[str, ...] = ()
    continuation_fragments: tuple[tuple[str, str], ...] = ()
    stop_requested: bool = False
    session_token: Optional["HookSessionToken"] = None


@dataclass(frozen = True)
class HookSessionToken:
    """Unforgeable-in-process generation identity for one project hook session."""

    project_id: str
    session_id: str
    generation: str
    synthetic: bool = False


@dataclass
class ProjectHookTurn:
    """Server-owned identity and Stop ledger for one model turn."""

    project_id: str
    session_id: str
    turn_id: str
    model: str
    permission_mode: str
    cancel_event: threading.Event
    transport: str = "chat"
    synthetic_session: bool = False
    runtime_owner: Optional[str] = None
    session_token: Optional[HookSessionToken] = None
    _stop_lock: threading.Lock = field(default_factory = threading.Lock, repr = False)
    _stop_calls: int = field(default = 0, repr = False)
    _stop_running: bool = field(default = False, repr = False)
    _assistant_candidates: list[str] = field(default_factory = list, repr = False)
    _lifetime_marker: Optional[int] = field(default = None, repr = False)

    def bind_session(self, result: HookEventResult) -> None:
        if result.session_token is not None:
            if self.session_token is not None and self.session_token != result.session_token:
                self.close()
            self.session_token = result.session_token
            _register_project_hook_turn(self)

    def close(self) -> None:
        _unregister_project_hook_turn(self)

    def stop(self, *, last_assistant_message: str) -> HookEventResult:
        """Run Stop for one clean, settled answer candidate."""
        if self.cancel_event.is_set():
            return HookEventResult(event = "Stop", session_token = self.session_token)
        with self._stop_lock:
            if self._stop_running:
                return HookEventResult(
                    event = "Stop",
                    errors = ("Nested Project Stop hook execution was ignored.",),
                    session_token = self.session_token,
                )
            stop_hook_active = self._stop_calls > 0
            self._stop_calls += 1
            self._assistant_candidates.append(_bounded_stop_candidate(last_assistant_message))
            self._stop_running = True
        try:
            result = run_project_hook_event(
                self.project_id,
                "Stop",
                {
                    "turn_id": self.turn_id,
                    "stop_hook_active": stop_hook_active,
                    "last_assistant_message": self._assistant_candidates[-1],
                },
                session_id = self.session_id,
                model = self.model,
                permission_mode = self.permission_mode,
                cancel_event = self.cancel_event,
                session_token = self.session_token,
            )
        except Exception as exc:  # noqa: BLE001 - Stop remains advisory
            result = HookEventResult(
                event = "Stop",
                errors = (_redact_feedback(f"Project Stop hook failed: {exc}"),),
                session_token = self.session_token,
            )
        finally:
            with self._stop_lock:
                self._stop_running = False
        return result

    @property
    def stop_calls(self) -> int:
        with self._stop_lock:
            return self._stop_calls

    @property
    def assistant_candidates(self) -> tuple[str, ...]:
        with self._stop_lock:
            return tuple(self._assistant_candidates)


@dataclass(frozen = True)
class ProjectHookPrompt:
    """Typed internal prompt synthesized from one blocking Stop handler."""

    hook_run_id: str
    text: str

    def as_message(self) -> dict[str, str]:
        run_id = html.escape(self.hook_run_id, quote = True)
        content = html.escape(self.text, quote = False)
        return {
            "role": "user",
            "content": f'<hook_prompt hook_run_id="{run_id}">{content}</hook_prompt>',
        }


@dataclass(frozen = True)
class _HookInvocation:
    project_id: str
    event: str
    event_input_json: bytes
    handler_id: str
    handler_json: bytes
    content_hash: str
    workspace_identity: tuple[int, int]
    workspace_revision: int
    trust_revision: int
    session_token: Optional[HookSessionToken] = None
    deadline_monotonic: float = field(default = float("inf"), compare = False)


@dataclass
class _BackgroundRun:
    future: Optional[concurrent.futures.Future]
    cancel_event: threading.Event
    started: threading.Event = field(default_factory = threading.Event)
    deadline_monotonic: float = float("inf")


@dataclass
class _SessionStartRun:
    done: threading.Event = field(default_factory = threading.Event)
    result: Optional[HookEventResult] = None
    error: Optional[BaseException] = None


@dataclass(frozen = True)
class HookSessionEndSnapshot:
    """Pre-delete authority needed to deliver one exact SessionEnd event."""

    token: HookSessionToken
    invocations: tuple[_HookInvocation, ...]
    handlers: tuple[dict[str, Any], ...]
    workspace: Any
    established: bool = True
    _seal: Any = field(default = None, repr = False, compare = False)


_SYNC_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers = MAX_BACKGROUND_HOOKS,
    thread_name_prefix = "unsloth-project-hook",
)
_BACKGROUND_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers = MAX_BACKGROUND_HOOKS,
    thread_name_prefix = "unsloth-project-hook-async",
)
_SESSION_END_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers = MAX_BACKGROUND_HOOKS,
    thread_name_prefix = "unsloth-project-hook-finalizer",
)
_ADMISSION_FENCE = threading.RLock()
_BACKGROUND_LOCK = threading.Lock()
_SESSION_STATE_CHANGED = threading.Condition(_BACKGROUND_LOCK)
_BACKGROUND: dict[HookSessionToken, dict[concurrent.futures.Future, _BackgroundRun]] = {}
_SYNCHRONOUS: dict[HookSessionToken, dict[concurrent.futures.Future, _BackgroundRun]] = {}
_SESSION_WORK: dict[HookSessionToken, dict[int, threading.Event]] = {}
_SESSION_TURNS: dict[HookSessionToken, dict[int, threading.Event]] = {}
_BACKGROUND_COUNTS: dict[HookSessionToken, int] = {}
_COMPLETED: dict[HookSessionToken, list[tuple[int, HookHandlerResult]]] = {}
_COMPLETED_SEQUENCE = 0
_GLOBAL_HOOK_WORK = 0
_GLOBAL_SESSION_END_WORK = 0
_ENDED_SESSIONS: set[HookSessionToken] = set()
_ENDING_SESSIONS: set[HookSessionToken] = set()
_CAPTURING_SESSIONS: set[HookSessionToken] = set()
_SESSION_END_EXECUTING: set[HookSessionToken] = set()
_STARTED_SESSIONS: set[HookSessionToken] = set()
_UNESTABLISHED_SESSIONS: set[HookSessionToken] = set()
_STARTING_SESSIONS: dict[HookSessionToken, _SessionStartRun] = {}
_ACTIVE_SESSIONS: dict[tuple[str, str], HookSessionToken] = {}
_SESSION_OWNERS: dict[HookSessionToken, str] = {}
_DEFERRED_SESSION_ENDS: dict[
    HookSessionToken, tuple[dict[str, Any], Optional[HookSessionEndSnapshot]]
] = {}
_DEFERRED_SESSION_ABORTS: set[HookSessionToken] = set()
_DEFERRED_END_WAKE = threading.Event()
_DEFERRED_END_OWNER_LOCK = threading.Lock()
_DEFERRED_END_OWNER: Optional[threading.Thread] = None
_END_SNAPSHOT_SEAL = object()
_CURRENT_TURN: contextvars.ContextVar[Optional[ProjectHookTurn]] = contextvars.ContextVar(
    "project_hook_turn",
    default = None,
)


_QUOTED_FEEDBACK_SECRET = re.compile(
    r"""((?:["']?(?:(?:access|refresh|session|private|client|api|auth)[_-]?"""
    r"""(?:token|secret|key)|authorization|password|secret|token)["']?)\s*[=:]\s*)"""
    r"""(?P<quote>["'])(?P<value>(?:\\.|(?!(?P=quote))[\s\S])*)(?P=quote)""",
    re.IGNORECASE | re.DOTALL,
)
_FEEDBACK_REDACTIONS = (
    re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[^\s,;}]+"),
    re.compile(r"(?i)(\bbearer\s+)[A-Za-z0-9._~+/=-]+"),
    re.compile(
        r"""(?i)((?:["']?(?:(?:access|refresh|session|private|client|api|auth)[_-]?"""
        r"""(?:token|secret|key)|authorization|password|secret|token)["']?)"""
        r"""\s*[=:]\s*)[^\s,"';}]+"""
    ),
)


def project_id_from_session_id(session_id: Optional[str]) -> Optional[str]:
    if not isinstance(session_id, str) or not session_id.startswith(_PROJECT_SESSION_PREFIX):
        return None
    project_id = session_id[len(_PROJECT_SESSION_PREFIX) :]
    return project_id or None


def canonical_permission_mode(value: Optional[str], *, bypass_permissions: bool = False) -> str:
    """Translate Unsloth request modes to the stable Codex hook wire values."""
    if bypass_permissions or value == "full":
        return "bypassPermissions"
    translated = {
        None: "default",
        "": "default",
        "ask": "default",
        "auto": "acceptEdits",
        "off": "dontAsk",
        "default": "default",
        "acceptEdits": "acceptEdits",
        "plan": "plan",
        "dontAsk": "dontAsk",
        "bypassPermissions": "bypassPermissions",
    }.get(value)
    return translated if translated in _PERMISSION_MODES else "default"


def _new_session_token(
    project_id: str,
    session_id: str,
    *,
    synthetic: bool = False,
) -> HookSessionToken:
    return HookSessionToken(project_id, session_id, uuid.uuid4().hex, synthetic)


@contextmanager
def _bounded_project_hook_admission(
    *, cancel_event: Optional[threading.Event], deadline: Optional[float], label: str
):
    """Acquire the lifecycle fence without outliving the caller's one deadline."""
    acquired = False
    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError(f"{label} was cancelled.")
            now = time.monotonic()
            if deadline is not None and deadline != float("inf") and now >= deadline:
                raise TimeoutError(f"{label} exceeded its deadline.")
            if cancel_event is None and (deadline is None or deadline == float("inf")):
                _ADMISSION_FENCE.acquire()
                acquired = True
            else:
                remaining = (
                    float("inf")
                    if deadline is None or deadline == float("inf")
                    else max(0.0, deadline - now)
                )
                acquired = _ADMISSION_FENCE.acquire(timeout = min(0.05, remaining))
            if acquired:
                # Acquiring the lock is another scheduler wake. Recheck the
                # caller's authority before exposing the protected state.
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError(f"{label} was cancelled.")
                if (
                    deadline is not None
                    and deadline != float("inf")
                    and time.monotonic() >= deadline
                ):
                    raise TimeoutError(f"{label} exceeded its deadline.")
                break
        yield
    finally:
        if acquired:
            _ADMISSION_FENCE.release()


def _require_active_project_hook_scope(
    project_id: str,
    session_id: str,
    *,
    allow_synthetic: bool = False,
) -> None:
    """Reject hook admission outside an active persisted project/thread scope."""
    if project_hook_session_end_pending_for(project_id, session_id):
        raise AgentWorkspaceError(
            "Project hook session is waiting for a prior SessionEnd delivery."
        )
    project = get_chat_project(project_id)
    if project is None or bool(project.get("archived")):
        raise AgentWorkspaceError("Project hook project is missing or archived.")
    thread = get_chat_thread(session_id)
    if thread is None:
        if allow_synthetic:
            return
        raise AgentWorkspaceError("Project hook session is not a persisted project thread.")
    if bool(thread.get("archived")) or str(thread.get("projectId") or "") != project_id:
        raise AgentWorkspaceError("Project hook session is not an active member of this project.")


def _bounded_active_project_hook_scope(
    project_id: str,
    session_id: str,
    *,
    allow_synthetic: bool,
    cancel_event: Optional[threading.Event],
    deadline: Optional[float],
) -> None:
    """Bound database-backed scope admission without letting a late owner create a token."""

    def check() -> None:
        _require_active_project_hook_scope(
            project_id,
            session_id,
            allow_synthetic = allow_synthetic,
        )

    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError("Project hook session admission was cancelled.")
    if deadline is None or deadline == float("inf"):
        check()
        return
    if time.monotonic() >= deadline:
        raise TimeoutError("Project hook session admission exceeded its deadline.")
    supervisor._bounded_preparation(
        check,
        cancel_event = cancel_event,
        deadline = deadline,
        label = "hook session admission",
    )


def _resolve_session_token(
    project_id: str,
    session_id: str,
    supplied: Optional[HookSessionToken],
    *,
    create: bool,
    allow_ended: bool = False,
    create_synthetic: bool = False,
    scope_checked: bool = False,
    cancel_event: Optional[threading.Event] = None,
    deadline: Optional[float] = None,
) -> Optional[HookSessionToken]:
    key = (project_id, session_id)
    with _bounded_project_hook_admission(
        cancel_event = cancel_event,
        deadline = deadline,
        label = "Project hook session admission",
    ):
        with _SESSION_STATE_CHANGED:
            current = _ACTIVE_SESSIONS.get(key)
            if supplied is not None:
                if (
                    supplied.project_id != project_id
                    or supplied.session_id != session_id
                    or current != supplied
                    or (supplied in _ENDING_SESSIONS | _ENDED_SESSIONS and not allow_ended)
                ):
                    raise AgentWorkspaceError(
                        "Project hook session generation is no longer active."
                    )
                return supplied
            if current is not None:
                if current in _ENDING_SESSIONS | _ENDED_SESSIONS:
                    if create:
                        raise AgentWorkspaceError("Project hook session generation is ending.")
                    return None
                return current
            if not create:
                return None
        if not scope_checked:
            _bounded_active_project_hook_scope(
                project_id,
                session_id,
                allow_synthetic = create_synthetic,
                cancel_event = cancel_event,
                deadline = deadline,
            )
        with _SESSION_STATE_CHANGED:
            current = _ACTIVE_SESSIONS.get(key)
            if current is not None:
                if current in _ENDING_SESSIONS | _ENDED_SESSIONS:
                    raise AgentWorkspaceError("Project hook session generation is ending.")
                return current
            # Database publication and lock acquisition can both wake before
            # the caller is scheduled again. Never commit a fresh generation
            # after cancellation or the one absolute admission deadline.
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Project hook session admission was cancelled.")
            if deadline is not None and deadline != float("inf") and time.monotonic() >= deadline:
                raise TimeoutError("Project hook session admission exceeded its deadline.")
            token = _new_session_token(project_id, session_id, synthetic = create_synthetic)
            _ACTIVE_SESSIONS[key] = token
            return token


def _admit_project_hook_session(
    project_id: str,
    session_id: str,
    supplied: Optional[HookSessionToken],
    *,
    create: bool,
    allow_ended: bool = False,
    create_synthetic: bool = False,
    cancel_event: Optional[threading.Event] = None,
    deadline: Optional[float] = None,
    mark_unestablished: bool = False,
) -> Optional[HookSessionToken]:
    """Order the bounded database check before the in-memory generation transition."""
    with _bounded_project_hook_admission(
        cancel_event = cancel_event,
        deadline = deadline,
        label = "Project hook session admission",
    ):
        _bounded_active_project_hook_scope(
            project_id,
            session_id,
            allow_synthetic = create_synthetic,
            cancel_event = cancel_event,
            deadline = deadline,
        )
        token = _resolve_session_token(
            project_id,
            session_id,
            supplied,
            create = create,
            allow_ended = allow_ended,
            create_synthetic = create_synthetic,
            scope_checked = True,
        )
        if mark_unestablished and token is not None:
            with _SESSION_STATE_CHANGED:
                if token not in _STARTED_SESSIONS:
                    _UNESTABLISHED_SESSIONS.add(token)
        return token


def snapshot_project_hook_session(project_id: str, session_id: str) -> Optional[HookSessionToken]:
    """Capture the exact active generation for a deletion transaction."""
    return _resolve_session_token(project_id, session_id, None, create = False)


@contextmanager
def capture_project_hook_session_ledgers(
    records: list[dict[str, str]],
    *,
    session_ids: tuple[str, ...] = (),
    project_ids: tuple[str, ...] = (),
    include_all: bool = False,
    reason: str = "clear",
    workspace_snapshots: Optional[dict[str, ProjectWorkspace]] = None,
    deadline_monotonic: Optional[float] = None,
):
    """Fence exact generations before the caller's destructive DB commit."""
    event_started = time.monotonic()
    deadline = min(
        event_started + 3.0,
        deadline_monotonic if deadline_monotonic is not None else float("inf"),
    )
    with _bounded_project_hook_admission(
        cancel_event = None,
        deadline = deadline,
        label = "Project SessionEnd capture",
    ):
        with _SESSION_STATE_CHANGED:
            captured = []
            seen: set[HookSessionToken] = set()
            for record in records:
                project_id = str(record.get("project_id") or "")
                session_id = str(record.get("session_id") or "")
                token = _ACTIVE_SESSIONS.get((project_id, session_id))
                if token is None or token in seen:
                    continue
                seen.add(token)
                captured.append(
                    {
                        **record,
                        "session_token": token,
                        "session_established": token not in _UNESTABLISHED_SESSIONS,
                    }
                )
            requested = set(session_ids)
            requested_projects = set(project_ids)
            for token in _ACTIVE_SESSIONS.values():
                if (
                    not include_all
                    and token.session_id not in requested
                    and token.project_id not in requested_projects
                ) or token in seen:
                    continue
                seen.add(token)
                captured.append(
                    {
                        "project_id": token.project_id,
                        "session_id": token.session_id,
                        "model": "",
                        "session_token": token,
                        "session_established": token not in _UNESTABLISHED_SESSIONS,
                    }
                )
            for record in captured:
                token = record["session_token"]
                _ENDING_SESSIONS.add(token)
                _CAPTURING_SESSIONS.add(token)
            _SESSION_STATE_CHANGED.notify_all()
        try:
            for record in captured:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Project SessionEnd capture exceeded its deadline.")
                token = record["session_token"]

                def prepare_snapshot() -> HookSessionEndSnapshot:
                    captured_workspace = (workspace_snapshots or {}).get(token.project_id)
                    workspace_context = (
                        nullcontext(captured_workspace)
                        if captured_workspace is not None
                        else project_workspace_access(token.project_id, deadline = deadline)
                    )
                    with workspace_context as workspace:
                        payload = {
                            "reason": reason,
                            "delivery_id": f"{token.generation}:SessionEnd",
                            "session_id": token.session_id,
                            "transcript_path": None,
                            "cwd": str(workspace.root),
                            "hook_event_name": "SessionEnd",
                            "model": _bounded_text(
                                str(record.get("model") or ""),
                                label = "Project hook model",
                                maximum = 512,
                            ),
                        }
                        invocations, handlers = _trusted_invocations(
                            token.project_id,
                            "SessionEnd",
                            payload,
                            session_token = token,
                            workspace_authority = workspace,
                            event_started_monotonic = event_started,
                            deadline_monotonic = deadline,
                        )
                        return HookSessionEndSnapshot(
                            token = token,
                            invocations = tuple(invocations),
                            handlers = tuple(dict(handler) for handler in handlers),
                            workspace = workspace,
                            established = bool(record["session_established"]),
                            _seal = _END_SNAPSHOT_SEAL,
                        )

                record["session_end_snapshot"] = supervisor._bounded_preparation(
                    prepare_snapshot,
                    cancel_event = None,
                    deadline = deadline,
                    label = "SessionEnd capture",
                )
            yield captured
            # The exact generation is fenced before the destructive SQL starts,
            # but cancellation is a post-commit side effect. Every storage
            # ledger commits inside this context. Delaying signals until its
            # successful exit lets SQL and commit failures roll back without
            # killing a model turn whose persisted lifecycle never changed.
            with _SESSION_STATE_CHANGED:
                runs = []
                work_events = []
                turn_events = []
                for record in captured:
                    token = record["session_token"]
                    _CAPTURING_SESSIONS.discard(token)
                    runs.extend(_BACKGROUND.get(token, {}).values())
                    runs.extend(_SYNCHRONOUS.get(token, {}).values())
                    work_events.extend(_SESSION_WORK.get(token, {}).values())
                    turn_events.extend(_SESSION_TURNS.get(token, {}).values())
                _SESSION_STATE_CHANGED.notify_all()
            for run in runs:
                run.cancel_event.set()
            for work_event in work_events:
                work_event.set()
            for turn_event in turn_events:
                turn_event.set()
        except BaseException:
            with _SESSION_STATE_CHANGED:
                for record in captured:
                    token = record["session_token"]
                    _CAPTURING_SESSIONS.discard(token)
                    if _ACTIVE_SESSIONS.get((token.project_id, token.session_id)) == token:
                        _ENDING_SESSIONS.discard(token)
                _SESSION_STATE_CHANGED.notify_all()
            raise


@contextmanager
def project_hook_admission_fence():
    """Order destructive DB transactions before new project hook generations."""
    with _ADMISSION_FENCE:
        yield


def serialize_project_hook_session_end_snapshot(
    snapshot: HookSessionEndSnapshot,
) -> tuple[str, str]:
    """Serialize one sealed pre-delete SessionEnd capability for the DB outbox."""
    if not isinstance(snapshot, HookSessionEndSnapshot) or snapshot._seal is not _END_SNAPSHOT_SEAL:
        raise AgentWorkspaceError("Project hook end snapshot authority is invalid.")
    workspace = snapshot.workspace
    if not isinstance(workspace, ProjectWorkspace) and not all(
        hasattr(workspace, field_name)
        for field_name in ("project_id", "root", "kind", "device_id", "file_id", "revision")
    ):
        raise AgentWorkspaceError("Project hook end workspace snapshot is invalid.")
    document = {
        "version": 1,
        "established": snapshot.established,
        "token": {
            "project_id": snapshot.token.project_id,
            "session_id": snapshot.token.session_id,
            "generation": snapshot.token.generation,
            "synthetic": snapshot.token.synthetic,
        },
        "workspace": {
            "project_id": str(workspace.project_id),
            "root": str(workspace.root),
            "kind": str(workspace.kind),
            "device_id": int(workspace.device_id),
            "file_id": int(workspace.file_id),
            "revision": int(workspace.revision),
        },
        "invocations": [
            {
                "project_id": invocation.project_id,
                "event": invocation.event,
                "event_input_json": invocation.event_input_json.decode("utf-8"),
                "handler_id": invocation.handler_id,
                "handler_json": invocation.handler_json.decode("utf-8"),
                "content_hash": invocation.content_hash,
                "workspace_identity": list(invocation.workspace_identity),
                "workspace_revision": invocation.workspace_revision,
                "trust_revision": invocation.trust_revision,
            }
            for invocation in snapshot.invocations
        ],
        "handlers": [dict(handler) for handler in snapshot.handlers],
    }
    encoded = json.dumps(
        document,
        ensure_ascii = False,
        separators = (",", ":"),
        sort_keys = True,
    )
    if len(encoded.encode("utf-8")) > MAX_SESSION_END_SNAPSHOT_BYTES:
        raise AgentWorkspaceError("Project hook end snapshot exceeds the durable size limit.")
    return f"{snapshot.token.generation}:SessionEnd", encoded


def _deserialize_project_hook_session_end_snapshot(raw: str) -> HookSessionEndSnapshot:
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_SESSION_END_SNAPSHOT_BYTES:
        raise AgentWorkspaceError("Project hook end outbox snapshot is invalid.")
    try:
        document = json.loads(raw)
        token_data = document["token"]
        workspace_data = document["workspace"]
        invocation_data = document["invocations"]
        handlers = document["handlers"]
    except (KeyError, TypeError, ValueError) as exc:
        raise AgentWorkspaceError("Project hook end outbox snapshot is invalid.") from exc
    established = document.get("established", True)
    if (
        document.get("version") != 1
        or not isinstance(invocation_data, list)
        or not isinstance(established, bool)
    ):
        raise AgentWorkspaceError("Project hook end outbox snapshot version is unsupported.")
    if not isinstance(handlers, list) or len(handlers) != len(invocation_data):
        raise AgentWorkspaceError("Project hook end outbox handler snapshot is invalid.")
    try:
        token = HookSessionToken(
            project_id = str(token_data["project_id"]),
            session_id = str(token_data["session_id"]),
            generation = str(token_data["generation"]),
            synthetic = bool(token_data.get("synthetic")),
        )
        workspace = ProjectWorkspace(
            project_id = str(workspace_data["project_id"]),
            root = Path(str(workspace_data["root"])),
            kind = str(workspace_data["kind"]),
            device_id = int(workspace_data["device_id"]),
            file_id = int(workspace_data["file_id"]),
            revision = int(workspace_data["revision"]),
        )
        invocations = []
        for data, handler in zip(invocation_data, handlers):
            if not isinstance(data, dict) or not isinstance(handler, dict):
                raise TypeError
            timeout_seconds = float(handler["timeout"])
            event_name = str(data["event"])
            identity = data["workspace_identity"]
            if not isinstance(identity, list) or len(identity) != 2:
                raise TypeError
            invocations.append(
                _HookInvocation(
                    project_id = str(data["project_id"]),
                    event = event_name,
                    event_input_json = str(data["event_input_json"]).encode("utf-8"),
                    handler_id = str(data["handler_id"]),
                    handler_json = str(data["handler_json"]).encode("utf-8"),
                    content_hash = str(data["content_hash"]),
                    workspace_identity = (int(identity[0]), int(identity[1])),
                    workspace_revision = int(data["workspace_revision"]),
                    trust_revision = int(data["trust_revision"]),
                    session_token = token,
                    deadline_monotonic = time.monotonic()
                    + (
                        MAX_HOOK_EVENT_PREPARATION_SECONDS
                        if event_name == "SessionEnd"
                        else timeout_seconds
                    ),
                )
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise AgentWorkspaceError("Project hook end outbox snapshot is invalid.") from exc
    if (
        not token.project_id
        or not token.session_id
        or not token.generation
        or workspace.project_id != token.project_id
        or any(
            invocation.project_id != token.project_id or invocation.event != "SessionEnd"
            for invocation in invocations
        )
    ):
        raise AgentWorkspaceError("Project hook end outbox authority does not match.")
    return HookSessionEndSnapshot(
        token = token,
        invocations = tuple(invocations),
        handlers = tuple(dict(handler) for handler in handlers),
        workspace = workspace,
        established = established,
        _seal = _END_SNAPSHOT_SEAL,
    )


def new_project_hook_turn(
    project_id: str,
    session_id: str,
    *,
    model: str,
    permission_mode: str,
    cancel_event: threading.Event,
    transport: str = "chat",
    synthetic_session: bool = False,
) -> ProjectHookTurn:
    return ProjectHookTurn(
        project_id = project_id,
        session_id = session_id,
        turn_id = f"turn_{uuid.uuid4().hex}",
        model = model,
        permission_mode = canonical_permission_mode(permission_mode),
        cancel_event = cancel_event,
        transport = transport,
        synthetic_session = synthetic_session,
    )


def current_project_hook_turn() -> Optional[ProjectHookTurn]:
    return _CURRENT_TURN.get()


def _register_project_hook_turn(turn: ProjectHookTurn) -> None:
    token = turn.session_token
    if token is None:
        return
    with _SESSION_STATE_CHANGED:
        if turn._lifetime_marker is not None:
            return
        current = _ACTIVE_SESSIONS.get((token.project_id, token.session_id))
        if current != token or token in _ENDING_SESSIONS | _ENDED_SESSIONS:
            turn.cancel_event.set()
            raise AgentWorkspaceError("Project hook session generation is no longer active.")
        marker = id(turn) ^ time.monotonic_ns()
        turn._lifetime_marker = marker
        _SESSION_TURNS.setdefault(token, {})[marker] = turn.cancel_event
        _SESSION_STATE_CHANGED.notify_all()


def _unregister_project_hook_turn(turn: ProjectHookTurn) -> None:
    token = turn.session_token
    marker = turn._lifetime_marker
    if token is None or marker is None:
        return
    with _SESSION_STATE_CHANGED:
        turns = _SESSION_TURNS.get(token)
        if turns is not None:
            turns.pop(marker, None)
            if not turns:
                _SESSION_TURNS.pop(token, None)
        turn._lifetime_marker = None
        _SESSION_STATE_CHANGED.notify_all()


@contextmanager
def activate_project_hook_turn(turn: Optional[ProjectHookTurn]):
    if turn is None:
        yield
        return
    token = _CURRENT_TURN.set(turn)
    try:
        yield
    finally:
        _CURRENT_TURN.reset(token)


def _redact_feedback(value: str) -> str:
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, RecursionError):
        parsed = None
    else:
        secret_keys = {
            "accesskey",
            "accesssecret",
            "accesstoken",
            "apikey",
            "apisecret",
            "apitoken",
            "authkey",
            "authsecret",
            "authtoken",
            "clientsecret",
            "clientkey",
            "clienttoken",
            "password",
            "privatekey",
            "privatesecret",
            "privatetoken",
            "refreshkey",
            "refreshsecret",
            "refreshtoken",
            "secret",
            "sessionkey",
            "sessionsecret",
            "sessiontoken",
            "token",
            "authorization",
        }

        def redact_json(item: Any) -> Any:
            if isinstance(item, dict):
                return {
                    key: (
                        "[REDACTED]"
                        if str(key).lower().replace("_", "").replace("-", "") in secret_keys
                        else redact_json(child)
                    )
                    for key, child in item.items()
                }
            if isinstance(item, list):
                return [redact_json(child) for child in item]
            return item

        if isinstance(parsed, (dict, list)):
            return json.dumps(redact_json(parsed), ensure_ascii = False, separators = (",", ":"))
    redacted = _QUOTED_FEEDBACK_SECRET.sub(
        lambda match: f"{match.group(1)}{match.group('quote')}[REDACTED]{match.group('quote')}",
        value,
    )
    for pattern in _FEEDBACK_REDACTIONS:
        redacted = pattern.sub(r"\1[REDACTED]", redacted)
    return redacted


def redact_project_hook_feedback(value: Any) -> str:
    """Bound and redact one hook-derived client/model-visible string."""
    return _redact_feedback(
        _bounded_text(
            value,
            label = "Project hook feedback",
            maximum = MAX_HOOK_TEXT_BYTES,
        )
    )


def _bounded_feedback(values: Any) -> tuple[str, ...]:
    accepted: list[str] = []
    used = 0
    for raw in values:
        if not isinstance(raw, str) or not raw:
            continue
        value = _redact_feedback(raw)
        encoded = value.encode("utf-8", errors = "replace")
        remaining = MAX_HOOK_AGGREGATE_BYTES - used
        if remaining <= 0:
            break
        if len(encoded) > remaining:
            value = encoded[:remaining].decode("utf-8", errors = "ignore")
            encoded = value.encode("utf-8")
        if value and value not in accepted:
            accepted.append(value)
            used += len(encoded)
    return tuple(accepted)


def _bounded_feedback_groups(*groups: Any) -> tuple[tuple[str, ...], ...]:
    accepted_groups: list[tuple[str, ...]] = []
    used = 0
    seen: set[str] = set()
    for values in groups:
        accepted = []
        for raw in values:
            if not isinstance(raw, str) or not raw:
                continue
            value = _redact_feedback(raw)
            if value in seen:
                continue
            encoded = value.encode("utf-8", errors = "replace")
            remaining = MAX_HOOK_AGGREGATE_BYTES - used
            if remaining <= 0:
                break
            if len(encoded) > remaining:
                value = encoded[:remaining].decode("utf-8", errors = "ignore")
                encoded = value.encode("utf-8")
            if value:
                accepted.append(value)
                seen.add(value)
                used += len(encoded)
        accepted_groups.append(tuple(accepted))
    return tuple(accepted_groups)


def hook_model_feedback(result: HookEventResult) -> tuple[str, ...]:
    """Return bounded, redacted feedback safe to add at a model boundary."""
    messages = [*result.additional_context]
    messages.extend(f"Project hook warning: {value}" for value in result.system_messages)
    messages.extend(f"Project hook status: {value}" for value in result.status_messages)
    messages.extend(f"Project hook failure: {value}" for value in result.errors)
    return _bounded_feedback(messages)


def _bounded_text(
    value: Any,
    *,
    label: str,
    maximum: int = MAX_HOOK_TEXT_BYTES,
) -> str:
    if not isinstance(value, str):
        raise AgentWorkspaceError(f"{label} must be a string.")
    if "\x00" in value:
        raise AgentWorkspaceError(f"{label} cannot contain NUL characters.")
    encoded = value.encode("utf-8", errors = "strict")
    if len(encoded) > maximum:
        raise AgentWorkspaceError(f"{label} exceeds the supported size limit.")
    return value


def _bounded_stop_candidate(value: Any) -> str:
    """Keep advisory Stop input bounded without rejecting a completed answer."""
    if not isinstance(value, str):
        return ""
    candidate = value.encode("utf-8", errors = "replace").decode("utf-8")

    def serialized_size(length: int) -> int:
        return len(
            json.dumps(
                candidate[:length],
                ensure_ascii = False,
                separators = (",", ":"),
            ).encode("utf-8")
        )

    if serialized_size(len(candidate)) <= MAX_STOP_CANDIDATE_BYTES:
        return candidate
    lower = 0
    upper = len(candidate)
    while lower < upper:
        middle = (lower + upper + 1) // 2
        if serialized_size(middle) <= MAX_STOP_CANDIDATE_BYTES:
            lower = middle
        else:
            upper = middle - 1
    return candidate[:lower]


def _event_json(value: dict[str, Any]) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii = False,
            separators = (",", ":"),
            sort_keys = True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise AgentWorkspaceError("Project hook input must be bounded JSON data.") from exc
    if len(encoded) > MAX_HOOK_EVENT_BYTES:
        raise AgentWorkspaceError("Project hook input exceeds the supported size limit.")
    return encoded


def _handler_json(handler: dict[str, Any]) -> bytes:
    return json.dumps(
        handler,
        ensure_ascii = False,
        separators = (",", ":"),
        sort_keys = True,
    ).encode("utf-8")


def _hook_invocation_process_spec(invocation: _HookInvocation) -> tuple[tuple[str, ...], bytes]:
    """Derive the only argv and stdin authorized by a reviewed invocation."""
    if not isinstance(invocation, _HookInvocation):
        raise AgentWorkspaceError("Project hook invocation authority is invalid.")
    try:
        handler = json.loads(invocation.handler_json)
    except (TypeError, ValueError) as exc:
        raise AgentWorkspaceError("Project hook handler authority is invalid.") from exc
    if not isinstance(handler, dict):
        raise AgentWorkspaceError("Project hook handler authority is invalid.")
    command = handler.get("commandWindows") if os.name == "nt" else handler.get("command")
    if not isinstance(command, str) or not command or "\x00" in command:
        raise AgentWorkspaceError("Project hook command is unavailable.")
    argv = ("cmd.exe", "/d", "/s", "/c", command) if os.name == "nt" else ("/bin/sh", "-c", command)
    return argv, invocation.event_input_json


def _workspace_identity(workspace: Any) -> tuple[int, int]:
    try:
        return int(workspace.device_id), int(workspace.file_id)
    except (TypeError, ValueError) as exc:
        raise AgentWorkspaceError("Project hook workspace identity is invalid.") from exc


def _trusted_invocations(
    project_id: str,
    event: str,
    payload: dict[str, Any],
    *,
    session_token: Optional[HookSessionToken] = None,
    workspace_authority: Optional[ProjectWorkspace] = None,
    cancel_event: Optional[threading.Event] = None,
    event_started_monotonic: Optional[float] = None,
    deadline_monotonic: Optional[float] = None,
) -> tuple[list[_HookInvocation], list[dict[str, Any]]]:
    event_started = (
        time.monotonic() if event_started_monotonic is None else float(event_started_monotonic)
    )

    def check_preparation() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Project hook preparation was cancelled.")
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise TimeoutError("Project hook preparation timed out.")

    check_preparation()
    workspace_context = (
        nullcontext(workspace_authority)
        if workspace_authority is not None
        else project_workspace_access(
            project_id,
            cancel_event = cancel_event,
            deadline = deadline_monotonic,
        )
    )
    with workspace_context as workspace:
        check_preparation()
        identity = _workspace_identity(workspace)
        if not secure_project_hook_discovery_supported():
            candidate = os.path.join(str(workspace.root), ".codex", "hooks.json")
            try:
                os.lstat(candidate)
            except FileNotFoundError:
                return [], []
            except OSError as exc:
                raise AgentWorkspaceError(
                    "Project hooks cannot be checked safely on this platform."
                ) from exc
        config = discover_project_hooks(
            workspace.root,
            expected_identity = identity,
            cancel_event = cancel_event,
            deadline = deadline_monotonic,
        )
        check_preparation()
        if not config.get("exists"):
            return [], []
        state = get_project_hook_trust_state(
            project_id,
            config.get("contentHash"),
            workspace_identity = identity,
            workspace_revision = int(workspace.revision),
        )
        check_preparation()
        if not state["trusted"]:
            return [], []
        disabled = set(state["disabledHandlerIds"])
        handlers = [
            handler
            for handler in matching_project_hooks(config, event, payload)
            if handler["id"] not in disabled
        ]
        event_input_json = _event_json(payload)
        invocations = [
            _HookInvocation(
                project_id = project_id,
                event = event,
                event_input_json = event_input_json,
                handler_id = handler["id"],
                handler_json = _handler_json(handler),
                content_hash = config["contentHash"],
                workspace_identity = identity,
                workspace_revision = int(workspace.revision),
                trust_revision = int(state["revision"]),
                session_token = session_token,
                deadline_monotonic = (
                    (
                        deadline_monotonic
                        if deadline_monotonic is not None
                        else event_started + MAX_HOOK_EVENT_PREPARATION_SECONDS
                    )
                    if event == "SessionEnd"
                    else min(
                        event_started + float(handler["timeout"]),
                        deadline_monotonic if deadline_monotonic is not None else float("inf"),
                    )
                ),
            )
            for handler in handlers
        ]
        return invocations, handlers


def _revalidate_hook_invocation(
    invocation: _HookInvocation,
    workspace: Any,
    *,
    captured_end: bool = False,
) -> None:
    """Fail closed if any reviewed authority changed before process creation."""
    if not captured_end:
        token = invocation.session_token
        if token is None:
            raise AgentWorkspaceError("Project hook session authority is missing.")
        _admit_project_hook_session(
            invocation.project_id,
            token.session_id,
            token,
            create = False,
            allow_ended = invocation.event == "SessionEnd",
            create_synthetic = token.synthetic,
            deadline = invocation.deadline_monotonic,
        )
    identity = _workspace_identity(workspace)
    if (
        identity != invocation.workspace_identity
        or int(workspace.revision) != invocation.workspace_revision
    ):
        raise AgentWorkspaceError("Project hook workspace changed before execution.")
    config = discover_project_hooks(
        workspace.root,
        expected_identity = identity,
        deadline = invocation.deadline_monotonic,
    )
    if config.get("contentHash") != invocation.content_hash:
        raise AgentWorkspaceError("Project hook source changed before execution.")
    if not captured_end:
        state = get_project_hook_trust_state(
            invocation.project_id,
            config.get("contentHash"),
            workspace_identity = identity,
            workspace_revision = int(workspace.revision),
        )
        if not state["trusted"] or int(state["revision"]) != invocation.trust_revision:
            raise AgentWorkspaceError("Project hook trust changed before execution.")
        if invocation.handler_id in set(state["disabledHandlerIds"]):
            raise AgentWorkspaceError("Project hook handler was disabled before execution.")
    try:
        payload = json.loads(invocation.event_input_json)
    except (TypeError, ValueError) as exc:
        raise AgentWorkspaceError("Project hook input authority is invalid.") from exc
    handlers = matching_project_hooks(config, invocation.event, payload)
    selected = next(
        (handler for handler in handlers if handler.get("id") == invocation.handler_id),
        None,
    )
    if selected is None or _handler_json(selected) != invocation.handler_json:
        raise AgentWorkspaceError("Project hook handler changed before execution.")


def _context_limit_bytes(handler: dict[str, Any]) -> int:
    requested = int(handler.get("additionalContextLimit") or 0)
    if requested == 0:
        return MAX_HOOK_TEXT_BYTES
    return min(MAX_HOOK_TEXT_BYTES, requested * 4)


def _optional_output_text(value: Any, *, label: str, maximum: int) -> Optional[str]:
    if value is None:
        return None
    return _bounded_text(value, label = label, maximum = maximum)


def _output_error(
    handler_id: str,
    message: str,
    *,
    exit_code: Optional[int] = None,
):
    return HookHandlerResult(
        handler_id = handler_id,
        status = "failed",
        exit_code = exit_code,
        error = _redact_feedback(message),
    )


def _strict_updated_input(invocation: _HookInvocation, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AgentWorkspaceError("PreToolUse updatedInput must be an object.")
    try:
        event_input = json.loads(invocation.event_input_json)
    except (TypeError, ValueError) as exc:
        raise AgentWorkspaceError("Project hook input authority is invalid.") from exc
    tool_name = event_input.get("tool_name") if isinstance(event_input, dict) else None
    if tool_name == "Bash":
        if set(value) != {"command"} or not isinstance(value.get("command"), str):
            raise AgentWorkspaceError(
                f"{tool_name} updatedInput must contain only a string command field."
            )
        _bounded_text(value["command"], label = f"{tool_name} updated command")
    elif tool_name == "apply_patch":
        if set(value) != {"patch"} or not isinstance(value.get("patch"), str):
            raise AgentWorkspaceError(
                "apply_patch updatedInput must contain only a string patch field."
            )
        _bounded_text(value["patch"], label = "apply_patch updated patch")
    elif tool_name == "edit_file":
        if set(value) != {"path", "edits"} or not isinstance(value.get("path"), str):
            raise AgentWorkspaceError(
                "edit_file updatedInput must contain only path and edits fields."
            )
        edits = value.get("edits")
        if not isinstance(edits, list) or not edits:
            raise AgentWorkspaceError("edit_file updatedInput edits must be a non-empty array.")
        for edit in edits:
            if not isinstance(edit, dict) or not {"old_string", "new_string"}.issubset(edit):
                raise AgentWorkspaceError("edit_file updatedInput contains an invalid edit.")
            if set(edit) - {"old_string", "new_string", "replace_all"}:
                raise AgentWorkspaceError("edit_file updatedInput contains an unknown edit field.")
            if not isinstance(edit["old_string"], str) or not isinstance(edit["new_string"], str):
                raise AgentWorkspaceError("edit_file replacement values must be strings.")
            if "replace_all" in edit and not isinstance(edit["replace_all"], bool):
                raise AgentWorkspaceError("edit_file replace_all must be a boolean.")
    _event_json(value)
    return value


def _parse_hook_output(
    invocation: _HookInvocation, handler: dict[str, Any], process: supervisor.ProjectProcessResult
) -> HookHandlerResult:
    handler_id = invocation.handler_id
    if process.status == "cancelled":
        return HookHandlerResult(handler_id, "cancelled")
    if process.status == "timed_out":
        return HookHandlerResult(handler_id, "timed_out")
    if process.output_truncated or process.stderr_truncated:
        return _output_error(
            handler_id,
            "Project hook output exceeded the supported size limit.",
            exit_code = process.exit_code,
        )
    stderr = process.stderr.strip()
    if process.exit_code == 2:
        if invocation.event == "Stop":
            reason = _optional_output_text(
                stderr,
                label = "Project Stop hook continuation reason",
                maximum = MAX_HOOK_TEXT_BYTES,
            )
            if not reason:
                return _output_error(
                    handler_id,
                    "Project Stop hook exited with code 2 without a continuation reason.",
                    exit_code = process.exit_code,
                )
            return HookHandlerResult(
                handler_id,
                "passed",
                exit_code = 2,
                reason = _redact_feedback(reason),
                continuation_requested = True,
            )
        reason = _optional_output_text(
            stderr or "Project hook blocked the operation.",
            label = "Project hook blocking reason",
            maximum = MAX_HOOK_TEXT_BYTES,
        )
        return HookHandlerResult(
            handler_id,
            "blocked",
            exit_code = 2,
            blocked = True,
            reason = _redact_feedback(reason) if reason else reason,
        )
    if process.status != "passed" or process.exit_code not in {0, None}:
        error = stderr or f"Project hook exited with status {process.exit_code}."
        return _output_error(handler_id, error, exit_code = process.exit_code)

    raw = process.output.strip()
    if not raw:
        return HookHandlerResult(handler_id, "passed", exit_code = process.exit_code)
    if not raw.startswith("{"):
        if invocation.event not in _PLAIN_CONTEXT_EVENTS:
            return _output_error(
                handler_id,
                f"{invocation.event} hook output must be a JSON object.",
                exit_code = process.exit_code,
            )
        context = _bounded_text(
            process.output,
            label = "Project hook additional context",
            maximum = _context_limit_bytes(handler),
        )
        return HookHandlerResult(
            handler_id,
            "passed",
            exit_code = process.exit_code,
            additional_context = context,
        )
    try:
        document = json.loads(raw)
    except (TypeError, ValueError, RecursionError) as exc:
        return _output_error(
            handler_id,
            f"Project hook returned invalid JSON: {exc}.",
            exit_code = process.exit_code,
        )
    if not isinstance(document, dict):
        return _output_error(
            handler_id,
            "Project hook output must be a JSON object.",
            exit_code = process.exit_code,
        )
    if document.get("suppressOutput") not in {None, False}:
        return _output_error(
            handler_id,
            f"{invocation.event} does not support suppressOutput.",
            exit_code = process.exit_code,
        )

    system_message = _optional_output_text(
        document.get("systemMessage"),
        label = "Project hook system message",
        maximum = MAX_HOOK_TEXT_BYTES,
    )
    blocked = False
    reason = None
    updated_input = None
    permission_decision = None
    additional_context = None
    continuation_requested = False
    stop_requested = False
    decision = document.get("decision")
    if decision is not None and decision != "block":
        return _output_error(
            handler_id,
            f"{invocation.event} hook returned an unsupported decision.",
            exit_code = process.exit_code,
        )
    if decision == "block":
        continuation_requested = invocation.event in {"Stop", "SubagentStop"}
        blocked = not continuation_requested
        raw_reason = document.get("reason")
        reason = _optional_output_text(
            raw_reason
            if continuation_requested
            else raw_reason or "Project hook blocked the operation.",
            label = "Project hook blocking reason",
            maximum = MAX_HOOK_TEXT_BYTES,
        )
        if continuation_requested and not reason:
            return _output_error(
                handler_id,
                f"{invocation.event} hook blocked without a continuation reason.",
                exit_code = process.exit_code,
            )
    if "continue" in document:
        if invocation.event not in _CONTINUE_EVENTS or not isinstance(document["continue"], bool):
            return _output_error(
                handler_id,
                f"{invocation.event} hook returned unsupported continue behavior.",
                exit_code = process.exit_code,
            )
        if document["continue"] is False:
            stop_requested = True
            blocked = invocation.event not in {"Stop", "SubagentStop"}
            reason = _optional_output_text(
                document.get("stopReason") or reason or "Project hook stopped the operation.",
                label = "Project hook stop reason",
                maximum = MAX_HOOK_TEXT_BYTES,
            )
    elif "stopReason" in document:
        return _output_error(
            handler_id,
            "Project hook returned stopReason without continue.",
            exit_code = process.exit_code,
        )

    specific = document.get("hookSpecificOutput")
    if specific is not None:
        if not isinstance(specific, dict) or specific.get("hookEventName") != invocation.event:
            return _output_error(
                handler_id,
                "Project hook-specific output names the wrong event.",
                exit_code = process.exit_code,
            )
        additional_context = _optional_output_text(
            specific.get("additionalContext"),
            label = "Project hook additional context",
            maximum = _context_limit_bytes(handler),
        )
        if invocation.event == "PreToolUse":
            permission = specific.get("permissionDecision")
            if permission not in {None, "allow", "deny"}:
                return _output_error(
                    handler_id,
                    "PreToolUse hook returned an unsupported permission decision.",
                    exit_code = process.exit_code,
                )
            if permission == "deny":
                blocked = True
                reason = _optional_output_text(
                    specific.get("permissionDecisionReason")
                    or reason
                    or "Project hook denied the tool call.",
                    label = "Project hook denial reason",
                    maximum = MAX_HOOK_TEXT_BYTES,
                )
            permission_decision = permission
            if "updatedInput" in specific:
                if permission != "allow":
                    return _output_error(
                        handler_id,
                        "PreToolUse updatedInput requires an allow decision and an object.",
                        exit_code = process.exit_code,
                    )
                try:
                    updated_input = _strict_updated_input(invocation, specific["updatedInput"])
                except AgentWorkspaceError as exc:
                    return _output_error(handler_id, str(exc), exit_code = process.exit_code)
        elif invocation.event == "PermissionRequest":
            permission = specific.get("decision")
            if not isinstance(permission, dict):
                return _output_error(
                    handler_id,
                    "PermissionRequest hook decision must be an object.",
                    exit_code = process.exit_code,
                )
            behavior = permission.get("behavior")
            if behavior not in {"allow", "deny"}:
                return _output_error(
                    handler_id,
                    "PermissionRequest hook returned an unsupported behavior.",
                    exit_code = process.exit_code,
                )
            permission_decision = behavior
            if behavior == "deny":
                blocked = True
                reason = _optional_output_text(
                    permission.get("message") or reason or "Project hook denied permission.",
                    label = "Project hook denial reason",
                    maximum = MAX_HOOK_TEXT_BYTES,
                )
            forbidden_permission_fields = {
                "updatedInput",
                "updatedPermissions",
                "interrupt",
            }.intersection(permission)
            if forbidden_permission_fields:
                return _output_error(
                    handler_id,
                    "PermissionRequest returned a reserved decision field.",
                    exit_code = process.exit_code,
                )
        elif "permissionDecision" in specific or "updatedInput" in specific:
            return _output_error(
                handler_id,
                f"{invocation.event} hook returned PreToolUse-only fields.",
                exit_code = process.exit_code,
            )
    if additional_context is not None and invocation.event not in _CONTEXT_EVENTS:
        return _output_error(
            handler_id,
            f"{invocation.event} does not support additionalContext.",
            exit_code = process.exit_code,
        )
    return HookHandlerResult(
        handler_id,
        "blocked" if blocked else "passed",
        exit_code = process.exit_code,
        blocked = blocked,
        reason = _redact_feedback(reason) if reason else reason,
        updated_input = updated_input,
        permission_decision = permission_decision,
        additional_context = additional_context,
        system_message = system_message,
        status_message = handler.get("statusMessage"),
        continuation_requested = continuation_requested,
        stop_requested = stop_requested,
    )


def _run_invocation(
    invocation: _HookInvocation,
    handler: dict[str, Any],
    cancel_event: Optional[threading.Event],
    end_snapshot: Optional[HookSessionEndSnapshot] = None,
) -> HookHandlerResult:
    try:
        process = supervisor._run_trusted_project_hook_process(
            invocation,
            timeout_seconds = float(handler["timeout"]),
            output_limit_bytes = MAX_HOOK_OUTPUT_BYTES,
            cancel_event = cancel_event,
            end_snapshot = end_snapshot,
        )
        return _parse_hook_output(invocation, handler, process)
    except (AgentWorkspaceError, ProjectExecutionUnavailable, ProjectHookTrustStateError) as exc:
        return _output_error(invocation.handler_id, str(exc))
    except Exception as exc:  # noqa: BLE001 - a hook failure must not crash the agent loop
        return _output_error(invocation.handler_id, f"Project hook execution failed: {exc}")


def _informational(result: HookHandlerResult) -> HookHandlerResult:
    return HookHandlerResult(
        handler_id = result.handler_id,
        status = result.status,
        exit_code = result.exit_code,
        additional_context = result.additional_context,
        system_message = result.system_message,
        status_message = result.status_message,
        error = result.error,
        asynchronous = True,
    )


def _background_done(session_token: HookSessionToken, future: concurrent.futures.Future) -> None:
    global _GLOBAL_HOOK_WORK, _COMPLETED_SEQUENCE
    try:
        result = _informational(future.result())
    except Exception as exc:  # noqa: BLE001
        result = HookHandlerResult(
            "unknown",
            "failed",
            error = _redact_feedback(str(exc)),
            asynchronous = True,
        )
    with _SESSION_STATE_CHANGED:
        ended = (
            session_token in _ENDED_SESSIONS
            or (session_token in _ENDING_SESSIONS and session_token not in _CAPTURING_SESSIONS)
            or _ACTIVE_SESSIONS.get((session_token.project_id, session_token.session_id))
            != session_token
        )
        runs = _BACKGROUND.get(session_token)
        if runs is not None:
            runs.pop(future, None)
            if not runs:
                _BACKGROUND.pop(session_token, None)
        _GLOBAL_HOOK_WORK = max(0, _GLOBAL_HOOK_WORK - 1)
        if not ended:
            completed = _COMPLETED.setdefault(session_token, [])
            _COMPLETED_SEQUENCE += 1
            completed.append((_COMPLETED_SEQUENCE, result))
            del completed[:-MAX_COMPLETED_HOOKS]
            while sum(len(values) for values in _COMPLETED.values()) > MAX_GLOBAL_COMPLETED_HOOKS:
                oldest_token = min(
                    (candidate for candidate, values in _COMPLETED.items() if values),
                    key = lambda candidate: _COMPLETED[candidate][0][0],
                )
                _COMPLETED[oldest_token].pop(0)
                if not _COMPLETED[oldest_token]:
                    _COMPLETED.pop(oldest_token, None)
        count = _BACKGROUND_COUNTS.get(session_token, 0) - 1
        if count > 0:
            _BACKGROUND_COUNTS[session_token] = count
        else:
            _BACKGROUND_COUNTS.pop(session_token, None)
        _SESSION_STATE_CHANGED.notify_all()


def _schedule_background(
    session_token: HookSessionToken, invocation: _HookInvocation, handler: dict[str, Any]
) -> HookHandlerResult:
    global _GLOBAL_HOOK_WORK
    cancel_event = threading.Event()
    with _SESSION_STATE_CHANGED:
        if session_token in _ENDING_SESSIONS | _ENDED_SESSIONS:
            return HookHandlerResult(
                invocation.handler_id,
                "cancelled",
                asynchronous = True,
            )
        if _BACKGROUND_COUNTS.get(session_token, 0) >= MAX_BACKGROUND_HOOKS + MAX_BACKGROUND_QUEUE:
            return dataclass_replace(
                _output_error(
                    invocation.handler_id,
                    "Project hook background queue is full for this session.",
                ),
                asynchronous = True,
            )
        if _GLOBAL_HOOK_WORK >= MAX_GLOBAL_HOOK_WORK:
            return dataclass_replace(
                _output_error(
                    invocation.handler_id,
                    "Project hook process capacity is full.",
                ),
                asynchronous = True,
            )
        _BACKGROUND_COUNTS[session_token] = _BACKGROUND_COUNTS.get(session_token, 0) + 1
        _GLOBAL_HOOK_WORK += 1
        try:
            future = _BACKGROUND_EXECUTOR.submit(
                _run_invocation,
                invocation,
                handler,
                cancel_event,
            )
        except Exception as exc:
            count = _BACKGROUND_COUNTS.get(session_token, 0) - 1
            if count > 0:
                _BACKGROUND_COUNTS[session_token] = count
            else:
                _BACKGROUND_COUNTS.pop(session_token, None)
            _GLOBAL_HOOK_WORK = max(0, _GLOBAL_HOOK_WORK - 1)
            return HookHandlerResult(
                invocation.handler_id,
                "failed",
                error = _redact_feedback(f"Project async hook could not be scheduled: {exc}"),
                asynchronous = True,
            )
        _BACKGROUND.setdefault(session_token, {})[future] = _BackgroundRun(future, cancel_event)
    future.add_done_callback(lambda completed: _background_done(session_token, completed))
    return HookHandlerResult(
        invocation.handler_id,
        "scheduled",
        status_message = handler.get("statusMessage"),
        asynchronous = True,
    )


def _drain_completed(session_token: HookSessionToken) -> list[HookHandlerResult]:
    with _BACKGROUND_LOCK:
        return [result for _sequence, result in _COMPLETED.pop(session_token, [])]


def _synchronous_done(session_token: HookSessionToken, future: concurrent.futures.Future) -> None:
    global _GLOBAL_HOOK_WORK
    with _SESSION_STATE_CHANGED:
        runs = _SYNCHRONOUS.get(session_token)
        if runs is not None:
            runs.pop(future, None)
            if not runs:
                _SYNCHRONOUS.pop(session_token, None)
        _GLOBAL_HOOK_WORK = max(0, _GLOBAL_HOOK_WORK - 1)
        _SESSION_STATE_CHANGED.notify_all()


def _session_end_done(session_token: HookSessionToken, future: concurrent.futures.Future) -> None:
    global _GLOBAL_SESSION_END_WORK
    with _SESSION_STATE_CHANGED:
        runs = _SYNCHRONOUS.get(session_token)
        if runs is not None:
            runs.pop(future, None)
            if not runs:
                _SYNCHRONOUS.pop(session_token, None)
        _GLOBAL_SESSION_END_WORK = max(0, _GLOBAL_SESSION_END_WORK - 1)
        _SESSION_STATE_CHANGED.notify_all()


def _run_scheduled_invocation(
    invocation: _HookInvocation,
    handler: dict[str, Any],
    cancel_event: threading.Event,
    end_snapshot: Optional[HookSessionEndSnapshot],
    scheduled: _BackgroundRun,
) -> HookHandlerResult:
    deadline = invocation.deadline_monotonic
    if invocation.event == "SessionEnd":
        deadline = min(
            invocation.deadline_monotonic,
            time.monotonic() + float(handler["timeout"]),
        )
        invocation = dataclass_replace(
            invocation,
            deadline_monotonic = deadline,
        )
    scheduled.deadline_monotonic = deadline
    scheduled.started.set()
    return _run_invocation(invocation, handler, cancel_event, end_snapshot)


def _schedule_synchronous(
    session_token: HookSessionToken,
    invocation: _HookInvocation,
    handler: dict[str, Any],
    cancel_event: Optional[threading.Event],
    *,
    allow_ending: bool,
    end_snapshot: Optional[HookSessionEndSnapshot] = None,
) -> _BackgroundRun | HookHandlerResult:
    global _GLOBAL_HOOK_WORK, _GLOBAL_SESSION_END_WORK
    effective_cancel = cancel_event if cancel_event is not None else threading.Event()
    with _SESSION_STATE_CHANGED:
        while True:
            finalizer_authorized = (
                allow_ending
                and session_token in _ENDING_SESSIONS
                and session_token in _SESSION_END_EXECUTING
            )
            if (
                session_token in _ENDED_SESSIONS
                or (session_token in _ENDING_SESSIONS and not finalizer_authorized)
                or (allow_ending and not finalizer_authorized)
            ):
                return HookHandlerResult(invocation.handler_id, "cancelled")
            if not finalizer_authorized or (_GLOBAL_SESSION_END_WORK < MAX_GLOBAL_SESSION_END_WORK):
                break
            if effective_cancel.is_set():
                return HookHandlerResult(invocation.handler_id, "cancelled")
            remaining = invocation.deadline_monotonic - time.monotonic()
            if remaining <= 0:
                return HookHandlerResult(
                    invocation.handler_id,
                    "timed_out",
                    error = "Project SessionEnd hook did not finish within its timeout.",
                )
            _SESSION_STATE_CHANGED.wait(min(0.05, remaining))
        if finalizer_authorized:
            if effective_cancel.is_set():
                return HookHandlerResult(invocation.handler_id, "cancelled")
            if time.monotonic() >= invocation.deadline_monotonic:
                return HookHandlerResult(
                    invocation.handler_id,
                    "timed_out",
                    error = "Project SessionEnd hook did not finish within its timeout.",
                )
        if not finalizer_authorized and _GLOBAL_HOOK_WORK >= MAX_GLOBAL_HOOK_WORK:
            return _output_error(invocation.handler_id, "Project hook process capacity is full.")
        if finalizer_authorized:
            _GLOBAL_SESSION_END_WORK += 1
            executor = _SESSION_END_EXECUTOR
        else:
            _GLOBAL_HOOK_WORK += 1
            executor = _SYNC_EXECUTOR
        scheduled = _BackgroundRun(None, effective_cancel)
        try:
            future = executor.submit(
                _run_scheduled_invocation,
                invocation,
                handler,
                effective_cancel,
                end_snapshot,
                scheduled,
            )
        except Exception:
            if finalizer_authorized:
                _GLOBAL_SESSION_END_WORK = max(0, _GLOBAL_SESSION_END_WORK - 1)
            else:
                _GLOBAL_HOOK_WORK = max(0, _GLOBAL_HOOK_WORK - 1)
            raise
        scheduled.future = future
        _SYNCHRONOUS.setdefault(session_token, {})[future] = scheduled
    done_callback = _session_end_done if finalizer_authorized else _synchronous_done
    future.add_done_callback(lambda completed: done_callback(session_token, completed))
    return scheduled


@contextmanager
def project_hook_session_work(
    session_token: HookSessionToken, cancel_event: Optional[threading.Event] = None
):
    """Fence one side-effecting tool operation to the exact hook generation."""
    if not isinstance(session_token, HookSessionToken):
        raise AgentWorkspaceError("Project hook session generation is invalid.")
    effective_cancel = cancel_event if cancel_event is not None else threading.Event()
    marker = id(effective_cancel) ^ id(threading.current_thread()) ^ time.monotonic_ns()
    with _SESSION_STATE_CHANGED:
        current = _ACTIVE_SESSIONS.get((session_token.project_id, session_token.session_id))
        if current != session_token or session_token in _ENDING_SESSIONS | _ENDED_SESSIONS:
            raise AgentWorkspaceError("Project hook session generation is no longer active.")
        _SESSION_WORK.setdefault(session_token, {})[marker] = effective_cancel
        _SESSION_STATE_CHANGED.notify_all()
    try:
        if effective_cancel.is_set():
            raise AgentWorkspaceError("Project hook session work was cancelled.")
        yield effective_cancel
    finally:
        with _SESSION_STATE_CHANGED:
            work = _SESSION_WORK.get(session_token)
            if work is not None:
                work.pop(marker, None)
                if not work:
                    _SESSION_WORK.pop(session_token, None)
            _SESSION_STATE_CHANGED.notify_all()


def _aggregate(
    event: str,
    results: list[HookHandlerResult],
    *,
    session_token: Optional[HookSessionToken] = None,
) -> HookEventResult:
    results = [
        dataclass_replace(
            result,
            reason = _redact_feedback(result.reason) if result.reason else result.reason,
        )
        for result in results
    ]
    blocked_results = [result for result in results if result.blocked]
    updated = [result.updated_input for result in results if result.updated_input is not None]
    permissions = [result.permission_decision for result in results]
    permission = "deny" if "deny" in permissions else "allow" if "allow" in permissions else None
    context, messages, statuses, errors = _bounded_feedback_groups(
        (result.additional_context for result in results if result.additional_context is not None),
        (result.system_message for result in results if result.system_message is not None),
        (result.status_message for result in results if result.status_message is not None),
        (result.error for result in results if result.error is not None),
    )
    stop_requested = any(result.stop_requested for result in results)
    continuation_fragments: list[tuple[str, str]] = []
    continuation_used = 0
    if not stop_requested:
        for result in results:
            if not result.continuation_requested or not result.reason:
                continue
            reason = _redact_feedback(result.reason)
            encoded = reason.encode("utf-8", errors = "replace")
            remaining = MAX_STOP_PROMPT_BYTES - continuation_used
            if remaining <= 0:
                break
            if len(encoded) > remaining:
                reason = encoded[:remaining].decode("utf-8", errors = "ignore")
                encoded = reason.encode("utf-8")
            if reason:
                continuation_fragments.append((result.handler_id, reason))
                continuation_used += len(encoded)
    continuation_reasons = tuple(reason for _, reason in continuation_fragments)
    return HookEventResult(
        event = event,
        runs = tuple(results),
        blocked = bool(blocked_results),
        reason = next((result.reason for result in blocked_results if result.reason), None),
        # Matching hooks see the same original input. Declaration order makes the
        # last accepted rewrite deterministic when more than one returns one.
        updated_input = updated[-1] if updated else None,
        permission_decision = permission,
        additional_context = context,
        system_messages = messages,
        status_messages = statuses,
        errors = errors,
        continuation_reason = continuation_reasons[0] if continuation_reasons else None,
        continuation_reasons = continuation_reasons,
        continuation_fragments = tuple(continuation_fragments),
        stop_requested = stop_requested,
        session_token = session_token,
    )


def project_hook_continuation_prompts(result: HookEventResult) -> tuple[ProjectHookPrompt, ...]:
    """Return ordered typed prompts for the valid Stop continuations in a result."""
    fragments = result.continuation_fragments
    if not fragments and result.continuation_reasons:
        fragments = tuple(
            (f"stop_{index}", reason) for index, reason in enumerate(result.continuation_reasons)
        )
    if not fragments and result.continuation_reason:
        fragments = (("stop_0", result.continuation_reason),)
    return tuple(ProjectHookPrompt(handler_id, reason) for handler_id, reason in fragments)


def project_hook_control_failure(result: HookEventResult) -> Optional[str]:
    """Return a bounded failure from a synchronous controlling hook, if any.

    Async declarations are informational by contract. Their delayed failure may
    be surfaced at a later model-safe point, but it must never deny or delay the
    operation whose event scheduled them.
    """
    failed = next(
        (
            run
            for run in result.runs
            if not run.asynchronous and run.status in {"failed", "timed_out", "cancelled"}
        ),
        None,
    )
    if failed is None:
        return None
    return _redact_feedback(
        failed.error or failed.reason or f"Project {result.event} hook did not finish."
    )


def run_project_hook_event(
    project_id: str,
    event: str,
    event_input: Optional[dict[str, Any]] = None,
    *,
    session_id: str,
    model: str = "",
    permission_mode: str = "default",
    cancel_event: Optional[threading.Event] = None,
    session_token: Optional[HookSessionToken] = None,
    end_snapshot: Optional[HookSessionEndSnapshot] = None,
    event_started_monotonic: Optional[float] = None,
    deadline_monotonic: Optional[float] = None,
) -> HookEventResult:
    """Run all active handlers and aggregate results in declaration order."""
    event_started = (
        time.monotonic() if event_started_monotonic is None else float(event_started_monotonic)
    )
    preparation_limit = MAX_HOOK_EVENT_PREPARATION_SECONDS
    event_deadline = min(
        event_started + preparation_limit,
        deadline_monotonic if deadline_monotonic is not None else float("inf"),
    )
    if event not in HOOK_EVENTS:
        raise AgentWorkspaceError("Project hook event is invalid.")
    if not isinstance(session_id, str) or not session_id:
        raise AgentWorkspaceError("Project hook session id is invalid.")
    if end_snapshot is None:
        try:
            token = _admit_project_hook_session(
                project_id,
                session_id,
                session_token,
                create = True,
                allow_ended = event == "SessionEnd",
                create_synthetic = bool(session_token is not None and session_token.synthetic),
                cancel_event = cancel_event,
                deadline = event_deadline,
            )
        except InterruptedError as exc:
            return _aggregate(
                event,
                [
                    HookHandlerResult(
                        f"{event}:preparation",
                        "cancelled",
                        error = _redact_feedback(str(exc)),
                    )
                ],
                session_token = session_token,
            )
        except TimeoutError as exc:
            return _aggregate(
                event,
                [
                    HookHandlerResult(
                        f"{event}:preparation",
                        "timed_out",
                        error = _redact_feedback(str(exc)),
                    )
                ],
                session_token = session_token,
            )
        except ProjectExecutionUnavailable as exc:
            return _aggregate(
                event,
                [
                    HookHandlerResult(
                        f"{event}:preparation",
                        "failed",
                        error = _redact_feedback(str(exc)),
                    )
                ],
                session_token = session_token,
            )
        if token is None:
            raise AgentWorkspaceError("Project hook session generation is unavailable.")
    else:
        token = end_snapshot.token
        if (
            event != "SessionEnd"
            or (session_token is not None and session_token != token)
            or token.project_id != project_id
            or token.session_id != session_id
            or end_snapshot._seal is not _END_SNAPSHOT_SEAL
        ):
            raise AgentWorkspaceError("Project hook end snapshot does not match this session.")
    if end_snapshot is not None:
        invocations = list(end_snapshot.invocations)
        handlers = list(end_snapshot.handlers)
    else:
        supplied = dict(event_input or {})

        def prepare_event():
            with project_workspace_access(
                project_id,
                cancel_event = cancel_event,
                deadline = event_deadline,
            ) as workspace:
                payload = {
                    **supplied,
                    "session_id": session_id,
                    "transcript_path": None,
                    "cwd": str(workspace.root),
                    "hook_event_name": event,
                    "model": _bounded_text(model, label = "Project hook model", maximum = 512),
                }
                if event == "SessionEnd":
                    payload["delivery_id"] = f"{token.generation}:SessionEnd"
                if event in {
                    "SessionStart",
                    "PreToolUse",
                    "PermissionRequest",
                    "PostToolUse",
                    "UserPromptSubmit",
                    "SubagentStart",
                    "SubagentStop",
                    "Stop",
                }:
                    payload["permission_mode"] = _bounded_text(
                        canonical_permission_mode(permission_mode),
                        label = "Project hook permission mode",
                        maximum = 64,
                    )
                return _trusted_invocations(
                    project_id,
                    event,
                    payload,
                    session_token = token,
                    workspace_authority = workspace,
                    cancel_event = cancel_event,
                    event_started_monotonic = event_started,
                    deadline_monotonic = event_deadline,
                )

        preparation_error = None
        try:
            invocations, handlers = supervisor._bounded_preparation(
                prepare_event,
                cancel_event = cancel_event,
                deadline = event_deadline,
                label = "hook event preparation",
            )
        except InterruptedError as exc:
            preparation_error = HookHandlerResult(
                f"{event}:preparation",
                "cancelled",
                error = _redact_feedback(str(exc)),
            )
        except TimeoutError as exc:
            preparation_error = HookHandlerResult(
                f"{event}:preparation",
                "timed_out",
                error = _redact_feedback(str(exc)),
            )
        except (
            AgentWorkspaceError,
            ProjectExecutionUnavailable,
            ProjectHookTrustStateError,
        ) as exc:
            preparation_error = HookHandlerResult(
                f"{event}:preparation",
                "failed",
                error = _redact_feedback(str(exc)),
            )
        if preparation_error is not None:
            return _aggregate(event, [preparation_error], session_token = token)
    if event == "SessionEnd":
        # A SessionEnd handler's configured timeout starts when an executor
        # worker actually begins it. Before then the shared delivery deadline
        # bounds admission and queueing, so valid sibling fanout cannot consume
        # every handler's budget while waiting behind the first worker wave.
        invocations = [
            dataclass_replace(invocation, deadline_monotonic = event_deadline)
            for invocation in invocations
        ]
    completed_before_event = _drain_completed(token)
    # SessionEnd delivery status describes only the sealed finalizer handlers.
    # Earlier async informational feedback has already had a model-safe delivery
    # opportunity and must not make a successful durable finalizer retry forever.
    prior = [] if event == "SessionEnd" else completed_before_event
    synchronous: list[tuple[int, _BackgroundRun, dict[str, Any], str]] = []
    results: list[Optional[HookHandlerResult]] = [None] * len(invocations)
    for index, (invocation, handler) in enumerate(zip(invocations, handlers)):
        if handler.get("async") and event != "SessionEnd":
            results[index] = _schedule_background(token, invocation, handler)
            continue
        scheduled = _schedule_synchronous(
            token,
            invocation,
            handler,
            cancel_event,
            allow_ending = event == "SessionEnd",
            end_snapshot = end_snapshot,
        )
        if isinstance(scheduled, HookHandlerResult):
            results[index] = scheduled
        else:
            synchronous.append((index, scheduled, handler, invocation.handler_id))
    for index, scheduled, handler, handler_id in synchronous:
        future = scheduled.future
        if future is None:
            results[index] = HookHandlerResult(
                handler_id = handler_id,
                status = "failed",
                error = "Project hook process was not scheduled.",
            )
            continue
        try:
            invocation = invocations[index]
            remaining = max(0.0, invocation.deadline_monotonic - time.monotonic())
            if event == "SessionEnd" and not scheduled.started.wait(remaining):
                raise concurrent.futures.TimeoutError
            if event == "SessionEnd":
                remaining = max(0.0, scheduled.deadline_monotonic - time.monotonic())
            results[index] = future.result(timeout = remaining)
        except concurrent.futures.TimeoutError:
            scheduled.cancel_event.set()
            results[index] = HookHandlerResult(
                handler_id = handler_id,
                status = "timed_out",
                error = _redact_feedback(
                    (
                        "Project SessionEnd hook did not finish within its timeout."
                        if event == "SessionEnd"
                        else "Project hook did not finish within its end-to-end timeout."
                    )
                ),
            )
    return _aggregate(
        event,
        [*prior, *(result for result in results if result is not None)],
        session_token = token,
    )


def ensure_project_hook_session(
    project_id: str,
    *,
    session_id: str,
    model: str = "",
    permission_mode: str = "default",
    cancel_event: Optional[threading.Event] = None,
    source: str = "startup",
    synthetic_session: bool = False,
) -> HookEventResult:
    startup_started = time.monotonic()
    startup_deadline = startup_started + MAX_HOOK_EVENT_PREPARATION_SECONDS
    try:
        token = _admit_project_hook_session(
            project_id,
            session_id,
            None,
            create = True,
            create_synthetic = synthetic_session,
            cancel_event = cancel_event,
            deadline = startup_deadline,
            mark_unestablished = True,
        )
    except InterruptedError as exc:
        return _aggregate(
            "SessionStart",
            [
                HookHandlerResult(
                    "SessionStart:preparation",
                    "cancelled",
                    error = _redact_feedback(str(exc)),
                )
            ],
        )
    except TimeoutError as exc:
        return _aggregate(
            "SessionStart",
            [
                HookHandlerResult(
                    "SessionStart:preparation",
                    "timed_out",
                    error = _redact_feedback(str(exc)),
                )
            ],
        )
    except ProjectExecutionUnavailable as exc:
        return _aggregate(
            "SessionStart",
            [
                HookHandlerResult(
                    "SessionStart:preparation",
                    "failed",
                    error = _redact_feedback(str(exc)),
                )
            ],
        )
    if token is None:
        raise AgentWorkspaceError("Project hook session generation is unavailable.")
    turn = current_project_hook_turn()
    runtime_owner = (
        turn.runtime_owner
        if turn is not None and turn.project_id == project_id and turn.session_id == session_id
        else None
    )
    if runtime_owner:
        with _BACKGROUND_LOCK:
            prior_owner = _SESSION_OWNERS.get(token)
        if prior_owner is not None and prior_owner != runtime_owner:
            # A generation's async feedback and Stop ledger belong to one model
            # runtime. Crossing runtimes ends that generation first. An
            # overlapping old turn is cancelled and fenced by SessionEnd.
            ended = end_project_hook_session(
                project_id,
                session_id = session_id,
                model = model,
                permission_mode = permission_mode,
                reason = "other",
                session_token = token,
                deadline_monotonic = startup_deadline,
            )
            with _BACKGROUND_LOCK:
                old_generation_fenced = (
                    _ACTIVE_SESSIONS.get((project_id, session_id)) == token
                    or token in _ENDING_SESSIONS
                )
            if old_generation_fenced:
                detail = (
                    ended.errors[0]
                    if ended.errors
                    else ("The previous project hook runtime is still ending.")
                )
                raise AgentWorkspaceError(detail)
            token = _admit_project_hook_session(
                project_id,
                session_id,
                None,
                create = True,
                create_synthetic = synthetic_session,
                cancel_event = cancel_event,
                deadline = startup_deadline,
                mark_unestablished = True,
            )
            if token is None:
                raise AgentWorkspaceError("Project hook session generation is unavailable.")
        bind_project_hook_session_owner(token, runtime_owner)
    start_run: Optional[_SessionStartRun] = None
    start_leader = False
    with _BACKGROUND_LOCK:
        if token not in _STARTED_SESSIONS:
            start_run = _STARTING_SESSIONS.get(token)
            if start_run is None:
                start_run = _SessionStartRun()
                _STARTING_SESSIONS[token] = start_run
                start_leader = True
        if token in _STARTED_SESSIONS:
            resolved_source = "resume" if source == "auto" else None
        else:
            resolved_source = "startup" if source == "auto" else source
    if start_run is not None and not start_leader:
        while not start_run.done.wait(0.05):
            if cancel_event is not None and cancel_event.is_set():
                raise AgentWorkspaceError("Project hook session startup was cancelled.")
        if start_run.error is not None:
            raise start_run.error
        if start_run.result is None:
            raise AgentWorkspaceError("Project hook session startup did not complete.")
        return start_run.result
    if resolved_source is None:
        with _BACKGROUND_LOCK:
            completed = [result for _sequence, result in _COMPLETED.pop(token, [])]
        return _aggregate("SessionStart", completed, session_token = token)
    try:
        result = run_project_hook_event(
            project_id,
            "SessionStart",
            {"source": resolved_source},
            session_id = session_id,
            model = model,
            permission_mode = permission_mode,
            cancel_event = cancel_event,
            session_token = token,
            event_started_monotonic = startup_started,
            deadline_monotonic = startup_deadline,
        )
        startup_failed = (
            result.blocked
            or result.stop_requested
            or project_hook_control_failure(result) is not None
        )
        if startup_failed:
            _abort_unestablished_project_hook_session(
                token,
                deadline = min(
                    startup_deadline,
                    time.monotonic() + SESSION_END_DRAIN_TIMEOUT_SECONDS,
                ),
            )
        if start_leader and start_run is not None:
            with _BACKGROUND_LOCK:
                current = _ACTIVE_SESSIONS.get((project_id, session_id))
                if (
                    current == token
                    and token not in _ENDING_SESSIONS | _ENDED_SESSIONS
                    and not startup_failed
                ):
                    _STARTED_SESSIONS.add(token)
                    _UNESTABLISHED_SESSIONS.discard(token)
                elif not startup_failed:
                    raise AgentWorkspaceError("Project hook session ended during startup.")
                start_run.result = result
                _STARTING_SESSIONS.pop(token, None)
                start_run.done.set()
        return result
    except BaseException as exc:
        _abort_unestablished_project_hook_session(
            token,
            deadline = min(
                startup_deadline,
                time.monotonic() + SESSION_END_DRAIN_TIMEOUT_SECONDS,
            ),
        )
        if start_leader and start_run is not None:
            with _BACKGROUND_LOCK:
                start_run.error = exc
                _STARTING_SESSIONS.pop(token, None)
                _STARTED_SESSIONS.discard(token)
                start_run.done.set()
        raise


def _signal_project_hook_session_cancellation(token: HookSessionToken) -> None:
    with _SESSION_STATE_CHANGED:
        _ENDING_SESSIONS.add(token)
        runs = [
            *_BACKGROUND.get(token, {}).values(),
            *_SYNCHRONOUS.get(token, {}).values(),
        ]
        work_events = list(_SESSION_WORK.get(token, {}).values())
        turn_events = list(_SESSION_TURNS.get(token, {}).values())
        _SESSION_STATE_CHANGED.notify_all()
    for run in runs:
        run.cancel_event.set()
    for work_event in work_events:
        work_event.set()
    for turn_event in turn_events:
        turn_event.set()


def _retire_project_hook_session_state(token: HookSessionToken) -> None:
    """Retire one quiescent generation without synthesizing lifecycle events."""
    with _SESSION_STATE_CHANGED:
        if (
            _BACKGROUND.get(token)
            or _SYNCHRONOUS.get(token)
            or _SESSION_WORK.get(token)
            or _SESSION_TURNS.get(token)
        ):
            raise AgentWorkspaceError("Project hook session is not quiescent.")
        _COMPLETED.pop(token, None)
        _STARTED_SESSIONS.discard(token)
        _UNESTABLISHED_SESSIONS.discard(token)
        _SESSION_OWNERS.pop(token, None)
        if _ACTIVE_SESSIONS.get((token.project_id, token.session_id)) == token:
            _ACTIVE_SESSIONS.pop((token.project_id, token.session_id), None)
        _ENDING_SESSIONS.discard(token)
        _ENDED_SESSIONS.discard(token)
        _CAPTURING_SESSIONS.discard(token)
        _DEFERRED_SESSION_ABORTS.discard(token)
        _DEFERRED_SESSION_ENDS.pop(token, None)
        _SESSION_STATE_CHANGED.notify_all()


def _wait_for_project_hook_session_quiescence(
    token: HookSessionToken, *, deadline: float
) -> tuple[bool, tuple[str, ...]]:
    """Wait for registered model, hook, and side-effecting tool work to leave."""
    with _SESSION_STATE_CHANGED:
        while True:
            # A completed Future is not quiescent until its callback has removed
            # the ledger and released global capacity. Waiting for the ledgers to
            # empty makes startup failure cleanup observable and order independent.
            active_hook_work = bool(_BACKGROUND.get(token) or _SYNCHRONOUS.get(token))
            active_tool_work = bool(_SESSION_WORK.get(token))
            active_turns = bool(_SESSION_TURNS.get(token))
            if not active_hook_work and not active_tool_work and not active_turns:
                return True, ()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                pending = []
                if active_turns:
                    pending.append("model turn")
                if active_tool_work:
                    pending.append("tool work")
                if active_hook_work:
                    pending.append("hook work")
                return False, tuple(pending)
            _SESSION_STATE_CHANGED.wait(min(0.05, remaining))


def _abort_unestablished_project_hook_session(token: HookSessionToken, *, deadline: float) -> bool:
    """Cancel startup work and retire only after every scheduled handler quiesces."""
    _signal_project_hook_session_cancellation(token)
    drained, _pending = _wait_for_project_hook_session_quiescence(token, deadline = deadline)
    if drained:
        _retire_project_hook_session_state(token)
        return True
    _queue_deferred_session_abort(token)
    return False


def _deferred_session_end_loop() -> None:
    while True:
        _DEFERRED_END_WAKE.wait(0.1)
        _DEFERRED_END_WAKE.clear()
        with _BACKGROUND_LOCK:
            pending = list(_DEFERRED_SESSION_ENDS.items())
            pending_aborts = tuple(_DEFERRED_SESSION_ABORTS)
        for token in pending_aborts:
            drained, _pending_kinds = _wait_for_project_hook_session_quiescence(
                token,
                deadline = time.monotonic(),
            )
            if not drained:
                continue
            try:
                _retire_project_hook_session_state(token)
            except AgentWorkspaceError:
                continue
        for token, (arguments, snapshot) in pending:
            drained, _pending_kinds = _wait_for_project_hook_session_quiescence(
                token,
                deadline = time.monotonic(),
            )
            if not drained:
                continue
            with _BACKGROUND_LOCK:
                current = _DEFERRED_SESSION_ENDS.pop(token, None)
            if current is None:
                continue
            try:
                end_project_hook_session(
                    token.project_id,
                    session_id = token.session_id,
                    session_token = token,
                    end_snapshot = snapshot,
                    **arguments,
                )
            except BaseException:
                # A durable destructive lifecycle also has the SQLite outbox.
                # Request-scoped and runtime teardown paths retain the fenced
                # generation here for another bounded retry.
                with _BACKGROUND_LOCK:
                    if token in _ENDING_SESSIONS:
                        _DEFERRED_SESSION_ENDS.setdefault(token, (arguments, snapshot))
                _DEFERRED_END_WAKE.set()


def _ensure_deferred_session_owner() -> None:
    global _DEFERRED_END_OWNER
    with _DEFERRED_END_OWNER_LOCK:
        if _DEFERRED_END_OWNER is None or not _DEFERRED_END_OWNER.is_alive():
            _DEFERRED_END_OWNER = threading.Thread(
                target = _deferred_session_end_loop,
                name = "unsloth-project-hook-end-reaper",
                daemon = True,
            )
            _DEFERRED_END_OWNER.start()


def _queue_deferred_session_abort(token: HookSessionToken) -> None:
    with _BACKGROUND_LOCK:
        _DEFERRED_SESSION_ENDS.pop(token, None)
        _DEFERRED_SESSION_ABORTS.add(token)
    _ensure_deferred_session_owner()
    _DEFERRED_END_WAKE.set()


def _queue_deferred_session_end(
    token: HookSessionToken,
    *,
    model: str,
    permission_mode: str,
    reason: str,
    end_snapshot: Optional[HookSessionEndSnapshot],
) -> None:
    with _BACKGROUND_LOCK:
        _DEFERRED_SESSION_ABORTS.discard(token)
        _DEFERRED_SESSION_ENDS.setdefault(
            token,
            (
                {
                    "model": model,
                    "permission_mode": permission_mode,
                    "reason": reason,
                },
                end_snapshot,
            ),
        )
    _ensure_deferred_session_owner()
    _DEFERRED_END_WAKE.set()


def end_project_hook_session(
    project_id: str,
    *,
    session_id: str,
    model: str = "",
    permission_mode: str = "default",
    reason: str = "other",
    session_token: Optional[HookSessionToken] = None,
    end_snapshot: Optional[HookSessionEndSnapshot] = None,
    delivery_cancel_event: Optional[threading.Event] = None,
    deadline_monotonic: Optional[float] = None,
) -> HookEventResult:
    if end_snapshot is not None:
        token = end_snapshot.token
        if (
            end_snapshot._seal is not _END_SNAPSHOT_SEAL
            or token.project_id != project_id
            or token.session_id != session_id
            or (session_token is not None and session_token != token)
        ):
            raise AgentWorkspaceError("Project hook end snapshot does not match this session.")
    else:
        if session_token is not None:
            with _BACKGROUND_LOCK:
                if (
                    session_token in _ENDED_SESSIONS
                    or session_token in _SESSION_END_EXECUTING
                    or _ACTIVE_SESSIONS.get((project_id, session_id)) != session_token
                ):
                    return HookEventResult(event = "SessionEnd", session_token = session_token)
        token = _resolve_session_token(
            project_id,
            session_id,
            session_token,
            create = False,
            allow_ended = True,
        )
        if token is None:
            return HookEventResult(event = "SessionEnd")
    with _SESSION_STATE_CHANGED:
        if token in _SESSION_END_EXECUTING:
            return HookEventResult(
                event = "SessionEnd",
                errors = ("Project SessionEnd delivery is already running.",),
                session_token = token,
            )
        active_exact_generation = _ACTIVE_SESSIONS.get((project_id, session_id)) == token
        if end_snapshot is None and not active_exact_generation:
            return HookEventResult(event = "SessionEnd", session_token = token)
        established = (
            end_snapshot.established
            if end_snapshot is not None
            else token not in _UNESTABLISHED_SESSIONS
        )
        # The exact token gates the reserved finalizer executor. A detached
        # durable replay may coexist with a newer generation under the same
        # thread id without fencing that newer token.
        _ENDING_SESSIONS.add(token)
        _SESSION_END_EXECUTING.add(token)
        runs = [
            *_BACKGROUND.get(token, {}).values(),
            *_SYNCHRONOUS.get(token, {}).values(),
        ]
        work_events = list(_SESSION_WORK.get(token, {}).values())
        turn_events = list(_SESSION_TURNS.get(token, {}).values())
        _COMPLETED.pop(token, None)
        _SESSION_STATE_CHANGED.notify_all()
    for run in runs:
        run.cancel_event.set()
    for work_event in work_events:
        work_event.set()
    for turn_event in turn_events:
        turn_event.set()
    quiesced = False
    try:
        drained, pending_kinds = _wait_for_project_hook_session_quiescence(
            token,
            deadline = min(
                time.monotonic() + SESSION_END_DRAIN_TIMEOUT_SECONDS,
                deadline_monotonic if deadline_monotonic is not None else float("inf"),
            ),
        )
        if not drained:
            detail = ", ".join(pending_kinds) or "session work"
            if end_snapshot is None and established:
                # Request-scoped and runtime teardown need an in-process retry.
                # Durable destructive lifecycles are retried only through their
                # claimed outbox record so a detached reaper cannot execute the
                # external side effect without owning its delivery lease.
                _queue_deferred_session_end(
                    token,
                    model = model,
                    permission_mode = permission_mode,
                    reason = reason,
                    end_snapshot = None,
                )
            elif end_snapshot is None:
                _queue_deferred_session_abort(token)
            return HookEventResult(
                event = "SessionEnd",
                errors = (
                    _redact_feedback(
                        f"Project SessionEnd was incomplete because {detail} did not stop."
                    ),
                ),
                session_token = token,
            )
        quiesced = True
        if delivery_cancel_event is not None and delivery_cancel_event.is_set():
            return HookEventResult(
                event = "SessionEnd",
                errors = ("Project SessionEnd delivery ownership was lost.",),
                session_token = token,
            )
        if not established:
            return HookEventResult(event = "SessionEnd", session_token = token)
        return run_project_hook_event(
            project_id,
            "SessionEnd",
            {"reason": reason},
            session_id = session_id,
            model = model,
            permission_mode = permission_mode,
            cancel_event = delivery_cancel_event,
            session_token = token,
            end_snapshot = end_snapshot,
            deadline_monotonic = deadline_monotonic,
        )
    finally:
        with _SESSION_STATE_CHANGED:
            _SESSION_END_EXECUTING.discard(token)
            if not quiesced:
                # Keep the exact generation fenced and retain every outstanding
                # work ledger. A later durable retry or lifecycle reaper may run
                # SessionEnd only after those operations actually quiesce.
                _ENDING_SESSIONS.add(token)
            else:
                if active_exact_generation:
                    _ENDED_SESSIONS.add(token)
                _COMPLETED.pop(token, None)
                _STARTED_SESSIONS.discard(token)
                _UNESTABLISHED_SESSIONS.discard(token)
                _SESSION_OWNERS.pop(token, None)
                if _ACTIVE_SESSIONS.get((project_id, session_id)) == token:
                    _ACTIVE_SESSIONS.pop((project_id, session_id), None)
                start_run = _STARTING_SESSIONS.pop(token, None)
                if start_run is not None and not start_run.done.is_set():
                    start_run.error = AgentWorkspaceError(
                        "Project hook session ended during startup."
                    )
                    start_run.done.set()
                _SESSION_TURNS.pop(token, None)
                _SESSION_WORK.pop(token, None)
                if active_exact_generation:
                    _ENDED_SESSIONS.discard(token)
                _ENDING_SESSIONS.discard(token)
                _CAPTURING_SESSIONS.discard(token)
                _DEFERRED_SESSION_ABORTS.discard(token)
                _DEFERRED_SESSION_ENDS.pop(token, None)
            _SESSION_STATE_CHANGED.notify_all()


def recover_pending_project_hook_session_ends(
    *, limit: int = 4096, stop_event: Optional[threading.Event] = None
) -> tuple[HookEventResult, ...]:
    """Replay the durable at-least-once SessionEnd outbox fairly and boundedly."""
    from storage.studio_db import (  # noqa: PLC0415
        claim_pending_project_hook_session_end_outbox,
        mark_project_hook_session_end_outbox_consumed,
        mark_project_hook_session_end_outbox_failed,
        project_hook_session_end_claim_heartbeat,
    )

    results = []
    remaining = max(1, min(int(limit), 16_384))
    claim_owner = f"recovery:{uuid.uuid4().hex}"
    while remaining > 0:
        if stop_event is not None and stop_event.is_set():
            break
        records = claim_pending_project_hook_session_end_outbox(
            claim_owner,
            # Claim only the delivery that is about to be heartbeated. Claiming
            # a batch lets later leases expire behind one slow finalizer and
            # permits a competing recovery worker to reclaim them.
            limit = 1,
        )
        if not records:
            break
        for record in records:
            if stop_event is not None and stop_event.is_set():
                break
            remaining -= 1
            record_id = record["id"]
            try:
                with project_hook_session_end_claim_heartbeat(
                    record_id, claim_owner
                ) as ownership_lost:
                    snapshot = _deserialize_project_hook_session_end_snapshot(
                        record["snapshot_json"]
                    )
                    token = snapshot.token
                    if record_id != f"{token.generation}:SessionEnd":
                        raise AgentWorkspaceError(
                            "Project hook end outbox id does not match its snapshot."
                        )
                    if ownership_lost.is_set():
                        raise AgentWorkspaceError("Project SessionEnd delivery ownership was lost.")
                    result = end_project_hook_session(
                        token.project_id,
                        session_id = token.session_id,
                        session_token = token,
                        end_snapshot = snapshot,
                        delivery_cancel_event = ownership_lost,
                    )
                results.append(result)
                failed = (
                    ownership_lost.is_set()
                    or bool(result.errors)
                    or any(
                        run.status in {"failed", "timed_out", "cancelled"} for run in result.runs
                    )
                )
                if failed:
                    detail = (
                        "; ".join(
                            (*result.errors, *(run.error or run.status for run in result.runs))
                        )
                        or "Project SessionEnd delivery ownership was lost."
                    )
                    mark_project_hook_session_end_outbox_failed(
                        record_id,
                        detail,
                        claim_owner = claim_owner,
                    )
                else:
                    consumed = mark_project_hook_session_end_outbox_consumed(
                        record_id,
                        claim_owner = claim_owner,
                    )
                    if consumed:
                        from core.inference.tools import (  # noqa: PLC0415
                            collect_orphaned_project_workspaces,
                        )
                        try:
                            collect_orphaned_project_workspaces()
                        except Exception:
                            # Consumption released the sealed workspace
                            # authority. The durable orphan record keeps a
                            # failed file cleanup retryable on the next sweep.
                            pass
            except Exception as exc:  # noqa: BLE001 - retain for bounded retry
                error = _redact_feedback(f"Project SessionEnd outbox recovery failed: {exc}")
                mark_project_hook_session_end_outbox_failed(
                    record_id,
                    error,
                    claim_owner = claim_owner,
                )
                results.append(
                    HookEventResult(
                        event = "SessionEnd",
                        errors = (error,),
                    )
                )
            if remaining <= 0:
                break
        if stop_event is not None and stop_event.is_set():
            break
    return tuple(results)


def _persist_all_project_hook_session_ends(
    *, reason: str
) -> tuple[int, tuple[HookEventResult, ...]]:
    """Seal live generations independently into the at-least-once outbox."""
    from storage.studio_db import (  # noqa: PLC0415
        enqueue_project_hook_session_end_outbox,
        get_connection,
    )

    persisted = 0
    failures = []
    # A malformed or unavailable project must not roll back durable finalizers
    # already sealed for unrelated live sessions during process shutdown.
    with _ADMISSION_FENCE:
        with _BACKGROUND_LOCK:
            tokens = tuple(dict.fromkeys(_ACTIVE_SESSIONS.values()))
        if not tokens:
            return 0, ()
        for token in tokens:
            connection = get_connection()
            try:
                connection.execute("BEGIN IMMEDIATE")
                with capture_project_hook_session_ledgers(
                    [
                        {
                            "project_id": token.project_id,
                            "session_id": token.session_id,
                            "model": "",
                        }
                    ],
                    reason = reason,
                ) as captured:
                    durable = []
                    for record in captured:
                        snapshot = record.get("session_end_snapshot")
                        if snapshot is not None:
                            durable.append(serialize_project_hook_session_end_snapshot(snapshot))
                    enqueue_project_hook_session_end_outbox(connection, durable)
                    connection.commit()
                persisted += len(durable)
            except BaseException as exc:  # noqa: BLE001 - preserve every other session
                connection.rollback()
                failures.append(
                    HookEventResult(
                        event = "SessionEnd",
                        errors = (_redact_feedback(f"Project SessionEnd durability failed: {exc}"),),
                        session_token = token,
                    )
                )
            finally:
                connection.close()
    return persisted, tuple(failures)


def end_all_project_hook_sessions(
    *, reason: str, durable: bool = False
) -> tuple[HookEventResult, ...]:
    """End the exact generations active at a process or model lifecycle boundary."""
    if durable:
        persisted, persistence_failures = _persist_all_project_hook_session_ends(reason = reason)
        if persisted:
            # A quiescent handler normally completes before shutdown continues.
            # A cancellation-insensitive generation remains fenced in the
            # durable outbox and is retried at the next startup.
            return (*persistence_failures, *recover_pending_project_hook_session_ends())
        return persistence_failures

    with _BACKGROUND_LOCK:
        tokens = tuple(dict.fromkeys(_ACTIVE_SESSIONS.values()))
    results = []
    for token in tokens:
        try:
            results.append(
                end_project_hook_session(
                    token.project_id,
                    session_id = token.session_id,
                    reason = reason,
                    session_token = token,
                )
            )
        except Exception as exc:  # noqa: BLE001 - lifecycle cleanup remains best effort
            results.append(
                HookEventResult(
                    event = "SessionEnd",
                    errors = (_redact_feedback(f"Project SessionEnd hook failed: {exc}"),),
                    session_token = token,
                )
            )
    return tuple(results)


def bind_project_hook_session_owner(
    session_token: Optional[HookSessionToken], runtime_owner: str
) -> None:
    """Associate an active generation with the runtime that is serving it."""
    if session_token is None or not isinstance(runtime_owner, str) or not runtime_owner:
        return
    with _BACKGROUND_LOCK:
        current = _ACTIVE_SESSIONS.get((session_token.project_id, session_token.session_id))
        if current != session_token or session_token in _ENDING_SESSIONS | _ENDED_SESSIONS:
            raise AgentWorkspaceError("Project hook session generation is no longer active.")
        owner = _SESSION_OWNERS.get(session_token)
        if owner is not None and owner != runtime_owner:
            raise AgentWorkspaceError(
                "Project hook session is already bound to another model runtime."
            )
        _SESSION_OWNERS[session_token] = runtime_owner


def end_project_hook_sessions_for_owner(
    runtime_owner: str, *, reason: str
) -> tuple[HookEventResult, ...]:
    """End only generations served by one concrete model runtime."""
    with _BACKGROUND_LOCK:
        tokens = tuple(token for token, owner in _SESSION_OWNERS.items() if owner == runtime_owner)
    results = []
    for token in tokens:
        try:
            results.append(
                end_project_hook_session(
                    token.project_id,
                    session_id = token.session_id,
                    reason = reason,
                    session_token = token,
                )
            )
        except Exception as exc:  # noqa: BLE001 - lifecycle cleanup remains best effort
            results.append(
                HookEventResult(
                    event = "SessionEnd",
                    errors = (_redact_feedback(f"Project SessionEnd hook failed: {exc}"),),
                    session_token = token,
                )
            )
    return tuple(results)


def project_stop_hook(function):
    """Run Stop after each clean tool-loop candidate and continue internally."""
    signature = inspect.signature(function)

    def turn_for(bound: inspect.BoundArguments) -> Optional[ProjectHookTurn]:
        arguments = bound.arguments
        run = arguments.get("run")
        policy = arguments.get("policy")
        owner = arguments.get("self")
        project_session_id = arguments.get("session_id") or getattr(run, "session_id", None)
        thread_id = (
            arguments.get("hook_session_id")
            or arguments.get("thread_id")
            or getattr(run, "thread_id", None)
        )
        project_id = project_id_from_session_id(project_session_id)
        if project_id is None or not thread_id:
            return None
        current = current_project_hook_turn()
        if (
            current is not None
            and current.project_id == project_id
            and current.session_id == thread_id
        ):
            return current
        bypass = bool(arguments.get("bypass_permissions")) or bool(
            getattr(policy, "bypass_permissions", False)
        )
        permission_mode = arguments.get("permission_mode") or getattr(
            policy, "permission_mode", None
        )
        if bypass:
            permission_mode = "bypassPermissions"
        model = (
            getattr(run, "model", None)
            or getattr(owner, "_model_identifier", None)
            or getattr(owner, "active_model_name", None)
            or ""
        )
        cancel_event = arguments.get("cancel_event")
        if not isinstance(cancel_event, threading.Event):
            cancel_event = threading.Event()
        return ProjectHookTurn(
            project_id = project_id,
            session_id = thread_id,
            turn_id = str(arguments.get("hook_turn_id") or f"turn_{uuid.uuid4().hex}"),
            model = str(model),
            permission_mode = canonical_permission_mode(
                str(permission_mode or "default"),
                bypass_permissions = bypass,
            ),
            cancel_event = cancel_event,
        )

    def assistant_text(item: Any) -> tuple[str, bool]:
        if isinstance(item, dict):
            if item.get("type") in {"content", "text"} and isinstance(item.get("text"), str):
                return item["text"], True
            return "", False
        if not isinstance(item, str):
            return "", False
        candidate = item.strip()
        if not candidate.startswith("data:"):
            return item, True
        try:
            payload = json.loads(candidate[5:].strip())
        except (TypeError, ValueError):
            return "", False
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list) or not choices:
            return "", False
        delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
        value = delta.get("content") if isinstance(delta, dict) else None
        return (value, False) if isinstance(value, str) else ("", False)

    def aborts_candidate(item: Any) -> bool:
        if isinstance(item, dict):
            return bool(
                item.get("type") == "error"
                or item.get("hook_blocked")
                or item.get("hook_stopped_after_compaction")
            )
        if not isinstance(item, str):
            return False
        stripped = item.strip()
        if not stripped.startswith("data:"):
            return False
        try:
            payload = json.loads(stripped[5:].strip())
        except (TypeError, ValueError):
            return False
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list):
            return bool(isinstance(payload, dict) and payload.get("error"))
        return any(
            isinstance(choice, dict)
            and (
                choice.get("finish_reason") == "tool_calls"
                or bool(isinstance(choice.get("delta"), dict) and choice["delta"].get("tool_calls"))
            )
            for choice in choices
        )

    def validate_single_choice(item: Any, turn: Optional[ProjectHookTurn]) -> None:
        if turn is None:
            return
        document = item if isinstance(item, dict) else None
        if isinstance(item, str):
            stripped = item.strip()
            if not stripped.startswith("data:") or stripped[5:].strip() == "[DONE]":
                return
            try:
                document = json.loads(stripped[5:].strip())
            except (TypeError, ValueError):
                return
        choices = document.get("choices") if isinstance(document, dict) else None
        if isinstance(choices, list) and len(choices) > 1:
            raise AgentWorkspaceError(
                "Project hook transports support exactly one provider choice."
            )

    def continuation_messages(
        state: dict[str, Any], text: str, result: HookEventResult
    ) -> Optional[list[dict[str, Any]]]:
        prompts = project_hook_continuation_prompts(result)
        messages = state.get("messages")
        if not prompts or not isinstance(messages, list):
            return None
        continued = [dict(message) for message in messages if isinstance(message, dict)]
        if text:
            trailing = continued[-1] if continued else None
            if not (
                isinstance(trailing, dict)
                and trailing.get("role") == "assistant"
                and trailing.get("content") == text
            ):
                continued.append({"role": "assistant", "content": text})
        continued.append(
            {
                "role": "user",
                "content": "\n".join(prompt.as_message()["content"] for prompt in prompts),
            }
        )
        state.setdefault("hook_prompts", []).extend(prompts)
        return continued

    def prepare_continuation(
        bound: inspect.BoundArguments, state: dict[str, Any], messages: list[dict[str, Any]]
    ) -> None:
        if "run" in bound.arguments:
            run = bound.arguments["run"]
            bound.arguments["run"] = dataclass_replace(
                run,
                messages = messages,
                continue_final_message = False,
            )
        elif "messages" in bound.arguments:
            bound.arguments["messages"] = messages
            if "system_prompt" in bound.arguments:
                bound.arguments["system_prompt"] = ""
            if "continue_final_message" in bound.arguments:
                bound.arguments["continue_final_message"] = False
        elif isinstance(bound.arguments.get("gen_kwargs"), dict):
            gen_kwargs = dict(bound.arguments["gen_kwargs"])
            gen_kwargs["messages"] = messages
            gen_kwargs["system_prompt"] = ""
            gen_kwargs["continue_final_message"] = False
            bound.arguments["gen_kwargs"] = gen_kwargs
        state["messages"] = messages

    def clean_candidate(
        turn: Optional[ProjectHookTurn],
        bound: inspect.BoundArguments,
        state: dict[str, Any],
        text: str,
        continuation_count: int,
    ) -> bool:
        if turn is None or turn.cancel_event.is_set():
            return False
        result = turn.stop(last_assistant_message = text)
        if result.stop_requested or not project_hook_continuation_prompts(result):
            return False
        if continuation_count >= MAX_STOP_CONTINUATIONS or turn.cancel_event.is_set():
            return False
        messages = continuation_messages(state, text, result)
        if messages is None:
            return False
        prepare_continuation(bound, state, messages)
        return not turn.cancel_event.is_set()

    def prefixed_item(item: Any, prefix: str, cumulative: bool) -> Any:
        if not prefix or not cumulative:
            return item
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            return {**item, "text": prefix + item["text"]}
        if isinstance(item, str):
            return prefix + item
        return item

    def fold_stats(aggregate: dict[str, Any], incoming: Any, *, key: str) -> None:
        if not isinstance(incoming, dict):
            return
        from core.inference.orchestrator import _summed_tool_loop_stats  # noqa: PLC0415
        aggregate[key] = _summed_tool_loop_stats(aggregate.get(key), incoming)

    def capture_terminal_item(item: Any, aggregate: dict[str, Any]) -> tuple[bool, Any]:
        if isinstance(item, dict) and item.get("type") == "_project_hook_terminal":
            terminal_protocol = item.get("protocol")
            if terminal_protocol in {"dict", "sse"}:
                aggregate["terminal_verified_protocol"] = terminal_protocol
            return True, item
        if isinstance(item, dict) and item.get("type") == "metadata":
            terminal_protocol = item.get("_project_hook_terminal_complete")
            if terminal_protocol in {"dict", "sse"}:
                aggregate["terminal_verified_protocol"] = terminal_protocol
            item = {
                key: value
                for key, value in item.items()
                if key != "_project_hook_terminal_complete"
            }
            fold_stats(
                aggregate,
                {"usage": item.get("usage") or {}, "timings": item.get("timings") or {}},
                key = "terminal_stats",
            )
            aggregate["dict_template"] = item
            return True, item
        if not isinstance(item, str):
            return False, item
        stripped = item.strip()
        if not stripped.startswith("data:"):
            return False, item
        if stripped[5:].strip() == "[DONE]":
            aggregate["done_item"] = item
            return True, item
        try:
            payload = json.loads(stripped[5:].strip())
        except (TypeError, ValueError):
            return False, item
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if isinstance(choices, list) and any(
            isinstance(choice, dict) and choice.get("finish_reason") is not None
            for choice in choices
        ):
            aggregate["finish_item"] = item
            return True, item
        if not (
            isinstance(payload, dict)
            and payload.get("choices") == []
            and isinstance(payload.get("usage"), dict)
        ):
            return False, item
        fold_stats(aggregate, {"usage": payload["usage"]}, key = "terminal_stats")
        aggregate["sse_template"] = payload
        return True, item

    def final_terminal_items(aggregate: dict[str, Any]) -> tuple[Any, ...]:
        items = []
        if aggregate.get("finish_item") is not None:
            items.append(aggregate["finish_item"])
        if "dict_template" in aggregate:
            template = aggregate["dict_template"]
            stats = aggregate.get("terminal_stats") or {}
            items.append(
                {
                    **template,
                    "usage": stats.get("usage"),
                    "timings": stats.get("timings"),
                }
            )
        elif "sse_template" in aggregate:
            stats = aggregate.get("terminal_stats") or {}
            payload = {**aggregate["sse_template"], "usage": stats.get("usage")}
            items.append(f"data: {json.dumps(payload, separators = (',', ':'))}")
        if aggregate.get("done_item") is not None:
            items.append(aggregate["done_item"])
        elif aggregate.get("terminal_verified_protocol") == "sse":
            items.append("data: [DONE]")
        return tuple(items)

    def required_terminal_protocol(bound: inspect.BoundArguments) -> Optional[str]:
        # External transports speak OpenAI SSE. The two llama.cpp generators
        # consume that SSE internally and expose typed events instead. Both
        # contracts need a positive terminal proof before Stop may run.
        if "transport" in bound.arguments:
            return "sse"
        owner = bound.arguments.get("self")
        if owner is not None and function.__name__ in {
            "generate_chat_completion",
            "generate_chat_completion_with_tools",
        }:
            return "dict"
        return None

    def terminal_is_complete(bound: inspect.BoundArguments, aggregate: dict[str, Any]) -> bool:
        protocol = required_terminal_protocol(bound)
        if protocol is None:
            return True
        if protocol == "sse":
            done = aggregate.get("done_item") is not None or (
                aggregate.get("terminal_verified_protocol") == "sse"
            )
            return bool(aggregate.get("finish_item") is not None and done)
        return bool(
            aggregate.get("terminal_verified_protocol") == "dict" and "dict_template" in aggregate
        )

    def capture_stats(bound: inspect.BoundArguments, aggregate: dict[str, Any]) -> None:
        holder = bound.arguments.get("stats_holder")
        if not isinstance(holder, dict) or not isinstance(holder.get("stats"), dict):
            return
        fold_stats(aggregate, holder["stats"], key = "holder_stats")
        holder.pop("stats", None)

    def restore_stats(bound: inspect.BoundArguments, aggregate: dict[str, Any]) -> None:
        holder = bound.arguments.get("stats_holder")
        if isinstance(holder, dict) and isinstance(aggregate.get("holder_stats"), dict):
            holder["stats"] = aggregate["holder_stats"]

    if inspect.isasyncgenfunction(function):

        @functools.wraps(function)
        async def async_wrapper(*args, **kwargs):
            bound = signature.bind_partial(*args, **kwargs)
            turn = turn_for(bound)
            continuation_count = 0
            display_prefix = ""
            aggregate: dict[str, Any] = {}
            state: dict[str, Any] = {}
            if "continuation_state" in signature.parameters:
                bound.arguments["continuation_state"] = state
            while True:
                if continuation_count and turn is not None and turn.cancel_event.is_set():
                    return
                aggregate.pop("finish_item", None)
                aggregate.pop("done_item", None)
                aggregate.pop("terminal_verified_protocol", None)
                text_parts: list[str] = []
                last_snapshot = ""
                candidate_aborted = False
                async for item in function(*bound.args, **bound.kwargs):
                    if (
                        isinstance(item, dict)
                        and item.get("type") == "_project_hook_candidate_boundary"
                    ):
                        text_parts.clear()
                        last_snapshot = ""
                        candidate_aborted = False
                        aggregate.pop("finish_item", None)
                        aggregate.pop("done_item", None)
                        aggregate.pop("terminal_verified_protocol", None)
                        continue
                    validate_single_choice(item, turn)
                    candidate_aborted = candidate_aborted or aborts_candidate(item)
                    text, cumulative = assistant_text(item)
                    if text:
                        if cumulative:
                            last_snapshot = text
                        else:
                            text_parts.append(text)
                    terminal, _ = capture_terminal_item(item, aggregate)
                    if not terminal:
                        yield prefixed_item(
                            item,
                            ""
                            if turn is not None and turn.transport == "anthropic"
                            else display_prefix,
                            cumulative,
                        )
                capture_stats(bound, aggregate)
                candidate = last_snapshot or "".join(text_parts)
                if candidate_aborted:
                    restore_stats(bound, aggregate)
                    for terminal in final_terminal_items(aggregate):
                        yield terminal
                    return
                if turn is not None and not terminal_is_complete(bound, aggregate):
                    raise RuntimeError(
                        "Provider stream ended without a complete terminal sequence."
                    )
                should_continue = await asyncio.to_thread(
                    clean_candidate,
                    turn,
                    bound,
                    state,
                    candidate,
                    continuation_count,
                )
                if not should_continue:
                    restore_stats(bound, aggregate)
                    for terminal in final_terminal_items(aggregate):
                        yield terminal
                    return
                if turn is not None and turn.transport == "anthropic":
                    yield {"type": "hook_continuation_boundary"}
                display_prefix += candidate
                continuation_count += 1

        return async_wrapper

    @functools.wraps(function)
    def sync_wrapper(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        turn = turn_for(bound)
        continuation_count = 0
        display_prefix = ""
        aggregate: dict[str, Any] = {}
        state: dict[str, Any] = {}
        if "continuation_state" in signature.parameters:
            bound.arguments["continuation_state"] = state
        while True:
            if continuation_count and turn is not None and turn.cancel_event.is_set():
                return
            aggregate.pop("finish_item", None)
            aggregate.pop("done_item", None)
            aggregate.pop("terminal_verified_protocol", None)
            text_parts: list[str] = []
            last_snapshot = ""
            candidate_aborted = False
            for item in function(*bound.args, **bound.kwargs):
                if (
                    isinstance(item, dict)
                    and item.get("type") == "_project_hook_candidate_boundary"
                ):
                    text_parts.clear()
                    last_snapshot = ""
                    candidate_aborted = False
                    aggregate.pop("finish_item", None)
                    aggregate.pop("done_item", None)
                    aggregate.pop("terminal_verified_protocol", None)
                    continue
                validate_single_choice(item, turn)
                candidate_aborted = candidate_aborted or aborts_candidate(item)
                text, cumulative = assistant_text(item)
                if text:
                    if cumulative:
                        last_snapshot = text
                    else:
                        text_parts.append(text)
                terminal, _ = capture_terminal_item(item, aggregate)
                if not terminal:
                    yield prefixed_item(
                        item,
                        ""
                        if turn is not None and turn.transport == "anthropic"
                        else display_prefix,
                        cumulative,
                    )
            capture_stats(bound, aggregate)
            candidate = last_snapshot or "".join(text_parts)
            if candidate_aborted:
                restore_stats(bound, aggregate)
                for terminal in final_terminal_items(aggregate):
                    yield terminal
                return
            if turn is not None and not terminal_is_complete(bound, aggregate):
                raise RuntimeError("Provider stream ended without a complete terminal sequence.")
            if not clean_candidate(turn, bound, state, candidate, continuation_count):
                restore_stats(bound, aggregate)
                for terminal in final_terminal_items(aggregate):
                    yield terminal
                return
            if turn is not None and turn.transport == "anthropic":
                yield {"type": "hook_continuation_boundary"}
            display_prefix += candidate
            continuation_count += 1

    return sync_wrapper


__all__ = [
    "HookEventResult",
    "HookHandlerResult",
    "HookSessionToken",
    "MAX_STOP_CONTINUATIONS",
    "MAX_STOP_PROMPT_BYTES",
    "ProjectHookPrompt",
    "ProjectHookTurn",
    "activate_project_hook_turn",
    "bind_project_hook_session_owner",
    "canonical_permission_mode",
    "capture_project_hook_session_ledgers",
    "current_project_hook_turn",
    "end_all_project_hook_sessions",
    "end_project_hook_sessions_for_owner",
    "end_project_hook_session",
    "ensure_project_hook_session",
    "hook_model_feedback",
    "new_project_hook_turn",
    "project_id_from_session_id",
    "project_hook_control_failure",
    "project_hook_continuation_prompts",
    "project_hook_admission_fence",
    "project_hook_session_work",
    "project_stop_hook",
    "recover_pending_project_hook_session_ends",
    "run_project_hook_event",
    "serialize_project_hook_session_end_snapshot",
    "snapshot_project_hook_session",
]
