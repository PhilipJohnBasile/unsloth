# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Lifecycle integration coverage for project hook consumers."""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

from core.agent_workspace import hook_runtime
from core.inference import llama_cpp
from core.inference import tools as inference_tools
from core.inference.anthropic_compat import AnthropicStreamEmitter
from core.inference.studio_tool_loop import ToolLoopPolicy, ToolLoopRun
from models.inference import ChatCompletionRequest, ChatMessage, ResponsesRequest
from routes import chat_history, inference


def _event(
    name: str,
    *,
    blocked: bool = False,
    reason: str | None = None,
    updated_input: dict | None = None,
    context: tuple[str, ...] = (),
) -> hook_runtime.HookEventResult:
    return hook_runtime.HookEventResult(
        event = name,
        blocked = blocked,
        reason = reason,
        updated_input = updated_input,
        additional_context = context,
    )


def _install_responses_stream_attempts(monkeypatch, attempts):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        index = len(calls)
        calls.append(json.loads(request.content))
        frames = attempts[index]
        content = "".join(
            "data: [DONE]\n\n" if frame == "[DONE]" else f"data: {json.dumps(frame)}\n\n"
            for frame in frames
        )
        return httpx.Response(
            200,
            content = content.encode(),
            headers = {"content-type": "text/event-stream"},
        )

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def client(*_args, **kwargs):
        return real_async_client(
            transport = transport,
            timeout = kwargs.get("timeout", 600),
        )

    monkeypatch.setattr(inference.httpx, "AsyncClient", client)
    monkeypatch.setattr(
        inference,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(
            is_loaded = True,
            is_vision = False,
            context_length = 4096,
            base_url = "http://llama.test",
            supports_reasoning = False,
            reasoning_always_on = False,
            model_identifier = "model",
            _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
        ),
    )
    return calls


def test_anthropic_passthrough_runs_two_real_stop_continuations(monkeypatch):
    stop_flags = []

    def run(_project_id, event, payload, **_kwargs):
        stop_flags.append(payload["stop_hook_active"])
        return hook_runtime.HookEventResult(
            event = event,
            continuation_reason = "continue",
            continuation_reasons = ("continue",),
            continuation_fragments = ((f"hook-{len(stop_flags)}", "continue"),),
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
    body = {
        "messages": [{"role": "user", "content": "start"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    posted = []

    async def post(payload):
        posted.append(payload)
        answer = "second" if len(posted) == 1 else "third"
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 1},
        }

    class Request:
        async def is_disconnected(self):
            return False

    request = Request()

    async def exercise():
        with hook_runtime.activate_project_hook_turn(turn):
            return await inference._anthropic_passthrough_stop_continuations(
                first_data = {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "first"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"completion_tokens": 1},
                },
                body = body,
                post = post,
                request = request,
                cancel_event = turn.cancel_event,
            )

    attempts = asyncio.run(exercise())

    assert [inference._anthropic_passthrough_candidate(item)[0] for item in attempts] == [
        "first",
        "second",
        "third",
    ]
    assert stop_flags == [False, True, True]
    assert len(posted) == hook_runtime.MAX_STOP_CONTINUATIONS
    assert posted[0]["stream"] is False
    assert "stream_options" not in posted[0]
    assert posted[0]["messages"][-1]["content"].startswith("<hook_prompt hook_run_id=")
    assert body == {
        "messages": [{"role": "user", "content": "start"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }


@pytest.mark.parametrize("attempt", ["initial", "continuation"])
def test_responses_stream_rejects_multichoice_provider_frames(monkeypatch, attempt):
    valid_attempt = [
        {"choices": [{"index": 0, "delta": {"content": "first"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        "[DONE]",
    ]
    ambiguous_attempt = [
        {
            "choices": [
                {"index": 0, "delta": {"content": "unreviewed-a"}},
                {"index": 1, "delta": {"content": "unreviewed-b"}},
            ]
        },
        "[DONE]",
    ]
    calls = _install_responses_stream_attempts(
        monkeypatch,
        [ambiguous_attempt] if attempt == "initial" else [valid_attempt, ambiguous_attempt],
    )
    stop_calls = []

    def run(_project_id, event, _payload, **_kwargs):
        stop_calls.append(event)
        return hook_runtime.HookEventResult(
            event = event,
            continuation_reason = "continue",
            continuation_reasons = ("continue",),
            continuation_fragments = (("review", "continue"),),
        )

    monkeypatch.setattr(hook_runtime, "run_project_hook_event", run)
    turn = hook_runtime.ProjectHookTurn(
        project_id = "project",
        session_id = "thread",
        turn_id = "turn",
        model = "model",
        permission_mode = "default",
        cancel_event = threading.Event(),
        transport = "responses",
    )
    payload = ResponsesRequest(model = "model", input = "start", stream = True)
    messages = [ChatMessage(role = "user", content = "start")]

    class Request:
        state = SimpleNamespace()

        async def is_disconnected(self):
            return False

    async def exercise():
        with hook_runtime.activate_project_hook_turn(turn):
            response = await inference._responses_stream(payload, messages, Request())
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
            return "".join(chunks)

    output = asyncio.run(exercise())

    assert "response.failed" in output
    assert "unreviewed-a" not in output
    assert "unreviewed-b" not in output
    assert len(calls) == (1 if attempt == "initial" else 2)
    assert stop_calls == ([] if attempt == "initial" else ["Stop"])


def test_responses_disconnect_before_continuation_headers_stops_without_terminal(monkeypatch):
    calls = _install_responses_stream_attempts(
        monkeypatch,
        [
            [
                {"choices": [{"index": 0, "delta": {"content": "first"}}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                "[DONE]",
            ]
        ],
    )
    stop_calls = []

    def run(_project_id, event, payload, **_kwargs):
        stop_calls.append(payload["stop_hook_active"])
        return hook_runtime.HookEventResult(
            event = event,
            continuation_reason = "continue",
            continuation_reasons = ("continue",),
            continuation_fragments = (("review", "continue"),),
        )

    original_send = inference._send_stream_with_preheader_cancel
    send_attempts = 0

    async def disconnect_second_send(*args, **kwargs):
        nonlocal send_attempts
        send_attempts += 1
        if send_attempts == 2:
            return None
        return await original_send(*args, **kwargs)

    monkeypatch.setattr(hook_runtime, "run_project_hook_event", run)
    monkeypatch.setattr(
        inference,
        "_send_stream_with_preheader_cancel",
        disconnect_second_send,
    )
    turn = hook_runtime.ProjectHookTurn(
        project_id = "project",
        session_id = "thread",
        turn_id = "turn",
        model = "model",
        permission_mode = "default",
        cancel_event = threading.Event(),
        transport = "responses",
    )
    payload = ResponsesRequest(model = "model", input = "start", stream = True)
    messages = [ChatMessage(role = "user", content = "start")]

    class Request:
        state = SimpleNamespace()

        async def is_disconnected(self):
            return False

    async def exercise():
        chunks = []
        with hook_runtime.activate_project_hook_turn(turn):
            response = await inference._responses_stream(payload, messages, Request())
            with pytest.raises(asyncio.CancelledError):
                async for chunk in response.body_iterator:
                    chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        return "".join(chunks)

    output = asyncio.run(exercise())

    assert len(calls) == 1
    assert send_attempts == 2
    assert stop_calls == [False]
    assert "response.completed" not in output


def test_responses_real_preheader_disconnect_sets_request_cancellation_before_continuation(
    monkeypatch,
):
    calls = _install_responses_stream_attempts(
        monkeypatch,
        [
            [
                {"choices": [{"index": 0, "delta": {"content": "first"}}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                "[DONE]",
            ]
        ],
    )
    stop_calls = []

    class Request:
        def __init__(self):
            self.state = SimpleNamespace()
            self.disconnected = False

        async def is_disconnected(self):
            return self.disconnected

    request = Request()
    cancel_event = inference._chat_cancel_event(request)

    def run(_project_id, event, payload, **_kwargs):
        stop_calls.append(payload["stop_hook_active"])
        request.disconnected = True
        return hook_runtime.HookEventResult(
            event = event,
            continuation_reason = "continue",
            continuation_reasons = ("continue",),
            continuation_fragments = (("review", "continue"),),
        )

    monkeypatch.setattr(hook_runtime, "run_project_hook_event", run)
    turn = hook_runtime.ProjectHookTurn(
        project_id = "project",
        session_id = "thread",
        turn_id = "turn",
        model = "model",
        permission_mode = "default",
        cancel_event = cancel_event,
        transport = "responses",
    )
    payload = ResponsesRequest(model = "model", input = "start", stream = True)
    messages = [ChatMessage(role = "user", content = "start")]

    async def exercise():
        chunks = []
        with hook_runtime.activate_project_hook_turn(turn):
            response = await inference._responses_stream(payload, messages, request)
            async for chunk in response.body_iterator:
                chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        return "".join(chunks)

    output = asyncio.run(exercise())

    assert len(calls) == 1
    assert stop_calls == [False]
    assert cancel_event.is_set()
    assert "response.completed" not in output


def test_request_cancellation_identity_survives_all_hook_boundaries():
    request = type("Request", (), {"state": type("State", (), {})()})()
    first = inference._chat_cancel_event(request)
    first.set()

    assert inference._chat_cancel_event(request) is first
    assert request.state.generation_cancel_event.is_set()


def test_tool_hooks_rewrite_block_and_wrap_model_context(monkeypatch):
    calls = []
    executions = []

    def run(_project_id, event, payload, **kwargs):
        calls.append((event, payload, kwargs))
        if event == "PreToolUse":
            return _event(
                event,
                updated_input = {"path": "rewritten.txt"},
                context = ("before context",),
            )
        return _event(event, context = ("after context",))

    monkeypatch.setattr(hook_runtime, "run_project_hook_event", run)
    monkeypatch.setattr(
        inference_tools,
        "_execute_tool_without_project_hooks",
        lambda name, arguments, **kwargs: executions.append((name, arguments, kwargs)) or "body",
    )

    result = inference_tools.execute_tool(
        "read_file",
        {"path": "original.txt"},
        session_id = "project-project",
        thread_id = "thread-1",
        tool_use_id = "call-1",
    )

    assert executions[0][1] == {"path": "rewritten.txt"}
    assert [call[0] for call in calls] == ["PreToolUse", "PostToolUse"]
    assert calls[0][1]["tool_use_id"] == "call-1"
    assert calls[1][1]["tool_response"] == "body"
    assert result == (
        "body\n\n[Project hook context]\nafter context\n\n[Project hook context]\nbefore context"
    )

    monkeypatch.setattr(
        hook_runtime,
        "run_project_hook_event",
        lambda *_args, **_kwargs: _event("PreToolUse", blocked = True, reason = "denied"),
    )
    executions.clear()
    blocked = inference_tools.execute_tool(
        "read_file",
        {"path": "original.txt"},
        session_id = "project-project",
        thread_id = "thread-1",
    )
    assert executions == []
    assert blocked == "Error: tool call blocked by project hook: denied"


def test_prepare_tool_hook_preserves_valid_empty_object_rewrite(monkeypatch):
    monkeypatch.setattr(
        hook_runtime,
        "run_project_hook_event",
        lambda *_args, **_kwargs: _event("PreToolUse", updated_input = {}),
    )

    arguments, result, tool_use_id = inference_tools.prepare_project_tool_hook(
        "web_search",
        {"query": "discard me"},
        session_id = "project-project",
        hook_session_id = "thread",
        hook_turn_id = "turn",
        tool_use_id = "call-empty",
        permission_mode = "default",
        bypass_permissions = False,
    )

    assert arguments == {}
    assert result.updated_input == {}
    assert tool_use_id == "call-empty"


def test_permission_hook_runs_before_resolution_and_fails_closed(monkeypatch):
    observed = []

    def allow(project_id, event, payload, **kwargs):
        observed.append((project_id, event, payload, kwargs))
        return hook_runtime.HookEventResult(
            event = event,
            permission_decision = "allow",
        )

    monkeypatch.setattr(hook_runtime, "run_project_hook_event", allow)
    result = inference_tools.run_project_permission_hook(
        "terminal",
        {"command": "printf original"},
        session_id = "project-project",
        thread_id = "thread-1",
        tool_use_id = "call-1",
        permission_mode = "ask",
        bypass_permissions = False,
    )
    assert result.permission_decision == "allow"
    assert result.updated_input is None
    assert observed[0][1] == "PermissionRequest"
    assert observed[0][2]["tool_name"] == "Bash"
    assert observed[0][2]["tool_use_id"] == "call-1"

    monkeypatch.setattr(
        hook_runtime,
        "run_project_hook_event",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("trust changed")),
    )
    failed = inference_tools.run_project_permission_hook(
        "terminal",
        {"command": "printf original"},
        session_id = "project-project",
        thread_id = "thread-1",
        tool_use_id = "call-1",
        permission_mode = "ask",
        bypass_permissions = False,
    )
    assert failed.blocked is True
    assert "trust changed" in failed.reason


def test_prompt_hooks_start_once_inject_context_and_block(monkeypatch):
    calls = []

    def start(project_id, **kwargs):
        calls.append(("SessionStart", project_id, kwargs))
        return _event("SessionStart", context = ("session context",))

    def run(project_id, event, payload, **kwargs):
        calls.append((event, project_id, payload, kwargs))
        return _event(event, context = ("prompt context",))

    monkeypatch.setattr(hook_runtime, "ensure_project_hook_session", start)
    monkeypatch.setattr(hook_runtime, "run_project_hook_event", run)
    messages = [
        ChatMessage(role = "system", content = "caller"),
        ChatMessage(role = "user", content = "hello"),
    ]

    resolved = inference._with_project_prompt_hook_messages(
        messages,
        project_session_id = "project-project",
        thread_id = "thread-1",
        model = "model",
        bypass_permissions = False,
    )

    assert [message.role for message in resolved] == ["system", "system", "user"]
    assert resolved[1].content == "session context\n\nprompt context"
    assert calls[1][2]["prompt"] == "hello"
    assert calls[1][3]["session_id"] == "thread-1"

    monkeypatch.setattr(
        hook_runtime,
        "ensure_project_hook_session",
        lambda *_args, **_kwargs: _event(
            "SessionStart",
            blocked = True,
            reason = "startup denied",
        ),
    )
    with pytest.raises(HTTPException) as captured:
        inference._with_project_prompt_hook_messages(
            messages,
            project_session_id = "project-project",
            thread_id = "thread-1",
            model = "model",
            bypass_permissions = False,
        )
    assert captured.value.status_code == 409
    assert captured.value.detail["error"]["code"] == "project_hook_blocked"


def test_compaction_hooks_bracket_fit_and_resume_session(monkeypatch):
    calls = []
    messages = [{"role": "user", "content": "before"}]
    fitted = [{"role": "user", "content": "after"}]
    truncation = {"fits": True, "dropped_messages": 2}
    monkeypatch.setattr(
        llama_cpp,
        "_fit_context_without_project_hooks",
        lambda _messages, **_kwargs: (fitted, truncation),
    )

    def run(_project_id, event, payload, **kwargs):
        calls.append((event, payload, kwargs))
        return _event(event)

    monkeypatch.setattr(hook_runtime, "run_project_hook_event", run)
    context = {
        "project_session_id": "project-project",
        "thread_id": "thread-1",
        "model": "model",
        "permission_mode": "default",
        "cancel_event": None,
    }
    result, detail = llama_cpp._fit_context(messages, project_hook_context = context)
    assert result is fitted
    assert [call[0] for call in calls] == ["PreCompact"]
    result, detail = llama_cpp._finish_project_compact_hooks(result, detail)
    assert result is fitted
    assert [call[0] for call in calls] == ["PreCompact", "PostCompact", "SessionStart"]
    assert calls[0][1]["trigger"] == "auto"
    assert calls[0][1]["custom_instructions"] == ""
    assert calls[2][1]["source"] == "compact"

    monkeypatch.setattr(
        hook_runtime,
        "run_project_hook_event",
        lambda *_args, **_kwargs: _event("PreCompact", blocked = True, reason = "keep it"),
    )
    result, detail = llama_cpp._fit_context(messages, project_hook_context = context)
    assert result is messages
    assert detail["hook_blocked"] is True
    assert detail["hook_reason"] == "keep it"
    assert detail["dropped_messages"] == 0

    def stop_after(_project_id, event, payload, **kwargs):
        if event == "PostCompact":
            return hook_runtime.HookEventResult(
                event = event,
                blocked = True,
                stop_requested = True,
                reason = "stop after compact",
            )
        return _event(event)

    monkeypatch.setattr(hook_runtime, "run_project_hook_event", stop_after)
    result, detail = llama_cpp._fit_context(messages, project_hook_context = context)
    result, detail = llama_cpp._finish_project_compact_hooks(result, detail)
    assert result is fitted
    assert detail["hook_stopped_after_compaction"] is True
    assert detail["hook_reason"] == "stop after compact"


def test_compact_session_start_control_survives_feedback_count_failure(monkeypatch):
    messages = [{"role": "user", "content": "compacted"}]
    truncation = {
        "fits": True,
        "dropped_messages": 2,
        "_project_hook_state": {
            "project_id": "project",
            "turn_id": "turn",
            "common": {
                "session_id": "thread",
                "model": "model",
                "permission_mode": "default",
                "cancel_event": None,
                "session_token": None,
            },
            "before": _event("PreCompact", context = ("pre feedback",)),
            "feedback_fit": {
                "context_length": 128,
                "max_tokens": 32,
                "count_tokens": lambda _messages: (_ for _ in ()).throw(
                    RuntimeError("tokenizer unavailable")
                ),
            },
        },
    }

    def run(_project_id, event, _payload, **_kwargs):
        if event == "SessionStart":
            return _event(
                event,
                blocked = True,
                reason = "compact resume denied",
                context = ("resume feedback",),
            )
        return _event(event, context = ("post feedback",))

    monkeypatch.setattr(hook_runtime, "run_project_hook_event", run)

    result, detail = llama_cpp._finish_project_compact_hooks(messages, truncation)

    assert result is messages
    assert detail["hook_stopped_after_compaction"] is True
    assert detail["hook_reason"] == "compact resume denied"
    assert detail["hook_feedback"][:3] == ["pre feedback", "post feedback", "resume feedback"]
    assert "tokenizer unavailable" not in detail["hook_feedback"][-1]


def test_session_end_cleanup_is_best_effort(monkeypatch):
    calls = []

    def end(project_id, **kwargs):
        calls.append((project_id, kwargs))
        if kwargs["session_id"] == "thread-2":
            raise RuntimeError("closed")
        return _event("SessionEnd")

    monkeypatch.setattr(hook_runtime, "end_project_hook_session", end)
    chat_history._end_project_hook_sessions(
        [
            {"project_id": "one", "session_id": "thread-1", "model": "m1"},
            {"project_id": "two", "session_id": "thread-2", "model": "m2"},
        ]
    )

    assert [call[1]["session_id"] for call in calls] == ["thread-1", "thread-2"]
    assert all(call[1]["reason"] == "clear" for call in calls)


def test_anthropic_stop_continuation_uses_distinct_blocks_and_one_terminal():
    emitter = AnthropicStreamEmitter(parse_think = False)
    lines = emitter.start("msg", "model")
    lines.extend(emitter.feed({"type": "content", "text": "first"}))
    lines.extend(emitter.feed({"type": "hook_continuation_boundary"}))
    lines.extend(emitter.feed({"type": "content", "text": "second"}))
    lines.extend(emitter.finish())

    payloads = [json.loads(line.split("data: ", 1)[1]) for line in lines if "data: " in line]
    starts = [payload for payload in payloads if payload["type"] == "content_block_start"]
    assert [payload["index"] for payload in starts] == [0, 1]
    assert sum(payload["type"] == "message_stop" for payload in payloads) == 1


@pytest.mark.parametrize("stream", [False, True])
def test_external_project_route_honors_stop_continuation(monkeypatch, stream):
    stop_flags = []

    def run(_project_id, event, payload, **_kwargs):
        assert event == "Stop"
        stop_flags.append(payload["stop_hook_active"])
        if len(stop_flags) <= 2:
            reason = f"repair {len(stop_flags)}"
            return hook_runtime.HookEventResult(
                event = event,
                blocked = True,
                continuation_reason = reason,
                continuation_reasons = (reason,),
                continuation_fragments = ((f"hook-{len(stop_flags)}", reason),),
            )
        return hook_runtime.HookEventResult(event = event)

    @hook_runtime.project_stop_hook
    async def fake_loop(
        transport,
        *,
        run,
        policy: object,
        cancel_event: threading.Event,
        continuation_state = None,
    ):
        del transport, policy, cancel_event
        if continuation_state is not None:
            continuation_state["messages"] = [dict(message) for message in run.messages]
        continuation_count = sum(
            "<hook_prompt " in str(message.get("content") or "")
            for message in run.messages
            if isinstance(message, dict)
        )
        answer = "ABC"[continuation_count]
        yield "data: " + json.dumps(
            {
                "id": "chatcmpl-route",
                "model": "provider-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": answer},
                        "finish_reason": None,
                    }
                ],
            }
        )
        yield 'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
        yield 'data: {"choices":[],"usage":{"total_tokens":1}}'
        yield "data: [DONE]"

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def close(self):
            pass

    monkeypatch.setattr(hook_runtime, "run_project_hook_event", run)
    monkeypatch.setattr(inference, "stream_with_project_stop_passthrough", fake_loop)
    monkeypatch.setattr(inference, "ExternalProviderClient", Client)
    monkeypatch.setattr(
        inference.providers_db,
        "get_provider",
        lambda provider_id: {
            "id": provider_id,
            "provider_type": "openai",
            "base_url": "https://api.openai.com/v1",
            "display_name": "test",
            "is_enabled": True,
        },
    )
    monkeypatch.setattr(
        inference,
        "resolve_provider_api_key_or_400",
        lambda *_args, **_kwargs: "secret",
    )
    payload = ChatCompletionRequest(
        model = "default",
        external_model = "provider-model",
        provider_id = "provider-1",
        session_id = "project-project",
        thread_id = "thread-1",
        messages = [ChatMessage(role = "user", content = "start")],
        stream = stream,
    )
    request = SimpleNamespace(
        headers = {},
        state = SimpleNamespace(skip_api_monitor = True),
        url = SimpleNamespace(path = "/v1/chat/completions"),
        method = "POST",
        is_disconnected = lambda: asyncio.sleep(0, result = False),
    )
    turn = hook_runtime.ProjectHookTurn(
        project_id = "project",
        session_id = "thread-1",
        turn_id = "turn-1",
        model = "provider-model",
        permission_mode = "default",
        cancel_event = threading.Event(),
    )

    async def exercise():
        with hook_runtime.activate_project_hook_turn(turn):
            response = await inference._proxy_to_external_provider(
                payload,
                request,
                current_subject = "test",
            )
            if not stream:
                return json.loads(response.body)
            return "".join([chunk async for chunk in response.body_iterator])

    result = asyncio.run(exercise())

    assert stop_flags == [False, True, True]
    if stream:
        assert all(answer in result for answer in "ABC")
        assert result.count("data: [DONE]") == 1
    else:
        assert result["choices"][0]["message"]["content"] == "ABC"
        assert result["usage"]["total_tokens"] == 3


@pytest.mark.parametrize("stream", [False, True])
def test_external_project_route_continue_false_finalizes(monkeypatch, stream):
    stop_calls = []

    def run(_project_id, event, payload, **_kwargs):
        stop_calls.append(payload["last_assistant_message"])
        return hook_runtime.HookEventResult(event = event, stop_requested = True)

    @hook_runtime.project_stop_hook
    async def fake_loop(
        transport,
        *,
        run,
        policy: object,
        cancel_event: threading.Event,
        continuation_state = None,
    ):
        del transport, policy, cancel_event
        if continuation_state is not None:
            continuation_state["messages"] = [dict(message) for message in run.messages]
        yield 'data: {"choices":[{"index":0,"delta":{"content":"final"},"finish_reason":null}]}'
        yield 'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
        yield "data: [DONE]"

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def close(self):
            pass

    monkeypatch.setattr(hook_runtime, "run_project_hook_event", run)
    monkeypatch.setattr(inference, "stream_with_project_stop_passthrough", fake_loop)
    monkeypatch.setattr(inference, "ExternalProviderClient", Client)
    monkeypatch.setattr(
        inference.providers_db,
        "get_provider",
        lambda provider_id: {
            "id": provider_id,
            "provider_type": "openai",
            "base_url": "https://api.openai.com/v1",
            "display_name": "test",
            "is_enabled": True,
        },
    )
    monkeypatch.setattr(
        inference,
        "resolve_provider_api_key_or_400",
        lambda *_args, **_kwargs: "secret",
    )
    payload = ChatCompletionRequest(
        model = "default",
        external_model = "provider-model",
        provider_id = "provider-1",
        session_id = "project-project",
        thread_id = "thread-1",
        messages = [ChatMessage(role = "user", content = "start")],
        stream = stream,
    )
    request = SimpleNamespace(
        headers = {},
        state = SimpleNamespace(skip_api_monitor = True),
        url = SimpleNamespace(path = "/v1/chat/completions"),
        method = "POST",
        is_disconnected = lambda: asyncio.sleep(0, result = False),
    )
    turn = hook_runtime.ProjectHookTurn(
        project_id = "project",
        session_id = "thread-1",
        turn_id = "turn-1",
        model = "provider-model",
        permission_mode = "default",
        cancel_event = threading.Event(),
    )

    async def exercise():
        with hook_runtime.activate_project_hook_turn(turn):
            response = await inference._proxy_to_external_provider(
                payload,
                request,
                current_subject = "test",
            )
            if not stream:
                return json.loads(response.body)
            return "".join([chunk async for chunk in response.body_iterator])

    result = asyncio.run(exercise())

    assert stop_calls == ["final"]
    if stream:
        assert result.count("data: [DONE]") == 1
    else:
        assert result["choices"][0]["message"]["content"] == "final"


@pytest.mark.parametrize(
    "frames",
    [
        (),
        ("data: [DONE]",),
        ('data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',),
    ],
)
def test_external_project_stop_rejects_incomplete_terminal_sequences(monkeypatch, frames):
    stop_calls = []

    def run(_project_id, event, _payload, **_kwargs):
        stop_calls.append(event)
        return hook_runtime.HookEventResult(event = event)

    class Transport:
        def stream(self, **_kwargs):
            async def iterator():
                for frame in frames:
                    yield frame

            return iterator()

    turn = hook_runtime.ProjectHookTurn(
        project_id = "project",
        session_id = "thread",
        turn_id = "turn",
        model = "model",
        permission_mode = "default",
        cancel_event = threading.Event(),
    )
    monkeypatch.setattr(hook_runtime, "run_project_hook_event", run)

    async def exercise():
        with hook_runtime.activate_project_hook_turn(turn):
            return [
                item
                async for item in inference.stream_with_project_stop_passthrough(
                    Transport(),
                    run = ToolLoopRun(
                        messages = [{"role": "user", "content": "start"}],
                        session_id = "project-project",
                        thread_id = "thread",
                        model = "model",
                    ),
                    policy = ToolLoopPolicy(
                        tools = [],
                        max_calls = 0,
                        timeout = 1,
                        permission_mode = "default",
                        confirm_calls = False,
                        bypass_permissions = False,
                        rag_scope = None,
                    ),
                    cancel_event = turn.cancel_event,
                )
            ]

    with pytest.raises(RuntimeError, match = "complete terminal sequence"):
        asyncio.run(exercise())
    assert stop_calls == []


def test_nonstream_project_turn_cleanup_runs_when_asgi_body_send_fails(monkeypatch):
    ended = []
    turn = hook_runtime.ProjectHookTurn(
        project_id = "project",
        session_id = "anonymous",
        turn_id = "turn",
        model = "model",
        permission_mode = "default",
        cancel_event = threading.Event(),
        synthetic_session = True,
        session_token = hook_runtime.HookSessionToken("project", "anonymous", "generation", True),
    )

    def end(project_id, **kwargs):
        ended.append((project_id, kwargs["session_token"]))
        return hook_runtime.HookEventResult(event = "SessionEnd")

    monkeypatch.setattr(hook_runtime, "end_project_hook_session", end)
    response = inference._attach_project_hook_nonstream_turn(
        inference.JSONResponse({"ok": True}), turn
    )
    sent = []

    async def send(message):
        sent.append(message["type"])
        if message["type"] == "http.response.body":
            raise OSError("client disconnected")

    async def receive():
        return {"type": "http.disconnect"}

    with pytest.raises(OSError, match = "client disconnected"):
        asyncio.run(response({"type": "http"}, receive, send))

    assert sent == ["http.response.start", "http.response.body"]
    assert ended == [("project", turn.session_token)]


@pytest.mark.parametrize("body_started", [False, True])
@pytest.mark.parametrize("synthetic", [False, True])
def test_stream_project_turn_cleanup_runs_on_cancelled_send(monkeypatch, body_started, synthetic):
    closed = []
    ended = []
    token = hook_runtime.HookSessionToken(
        "project", "anonymous" if synthetic else "thread", "generation", synthetic
    )
    turn = hook_runtime.ProjectHookTurn(
        project_id = "project",
        session_id = token.session_id,
        turn_id = "turn",
        model = "model",
        permission_mode = "default",
        cancel_event = threading.Event(),
        synthetic_session = synthetic,
        session_token = token,
    )

    monkeypatch.setattr(
        hook_runtime,
        "_unregister_project_hook_turn",
        lambda value: closed.append(value),
    )

    def end(project_id, **kwargs):
        ended.append((project_id, kwargs["session_token"]))
        return hook_runtime.HookEventResult(event = "SessionEnd")

    monkeypatch.setattr(hook_runtime, "end_project_hook_session", end)

    async def body():
        yield b"chunk"

    response = inference._SameTaskStreamingResponse(body())
    inference._attach_project_hook_turn(response, turn)
    sent = []

    async def send(message):
        sent.append(message["type"])
        if (not body_started and message["type"] == "http.response.start") or (
            body_started and message["type"] == "http.response.body"
        ):
            raise asyncio.CancelledError()

    async def receive():
        return {"type": "http.disconnect"}

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(response({"type": "http"}, receive, send))

    assert closed == [turn]
    assert ended == ([("project", token)] if synthetic else [])


def test_stock_stream_project_turn_cleanup_runs_on_preheader_send_failure(monkeypatch):
    closed = []
    ended = []
    token = hook_runtime.HookSessionToken("project", "anonymous", "generation", True)
    turn = hook_runtime.ProjectHookTurn(
        project_id = "project",
        session_id = "anonymous",
        turn_id = "turn",
        model = "model",
        permission_mode = "default",
        cancel_event = threading.Event(),
        synthetic_session = True,
        session_token = token,
    )
    monkeypatch.setattr(
        hook_runtime,
        "_unregister_project_hook_turn",
        lambda value: closed.append(value),
    )
    monkeypatch.setattr(
        hook_runtime,
        "end_project_hook_session",
        lambda project_id, **kwargs: (
            ended.append((project_id, kwargs["session_token"])),
            hook_runtime.HookEventResult(event = "SessionEnd"),
        )[1],
    )

    async def body():
        yield b"chunk"

    response = inference._attach_project_hook_turn(
        inference.StreamingResponse(
            body(),
            status_code = 206,
            headers = {"X-Test-Header": "preserved"},
        ),
        turn,
    )
    assert isinstance(response, inference._SameTaskStreamingResponse)
    assert response.status_code == 206
    assert response.headers["x-test-header"] == "preserved"

    async def send(message):
        if message["type"] == "http.response.start":
            raise OSError("client disconnected before body start")

    async def receive():
        return {"type": "http.disconnect"}

    with pytest.raises(inference.ClientDisconnect):
        asyncio.run(response({"type": "http"}, receive, send))

    assert closed == [turn]
    assert ended == [("project", token)]


def test_wrapped_stock_stream_preheader_failure_releases_all_lifecycle_owners_once(monkeypatch):
    events = []
    token = hook_runtime.HookSessionToken("project", "anonymous", "generation", True)
    turn = hook_runtime.ProjectHookTurn(
        project_id = "project",
        session_id = "anonymous",
        turn_id = "turn",
        model = "model",
        permission_mode = "default",
        cancel_event = threading.Event(),
        synthetic_session = True,
        session_token = token,
    )
    monkeypatch.setattr(
        hook_runtime,
        "_unregister_project_hook_turn",
        lambda value: events.append(("turn", value)),
    )
    monkeypatch.setattr(
        hook_runtime,
        "end_project_hook_session",
        lambda project_id, **kwargs: (
            events.append(("session_end", project_id, kwargs["session_token"])),
            hook_runtime.HookEventResult(event = "SessionEnd"),
        )[1],
    )

    class Lease:
        async def release(self):
            events.append(("lease",))

    async def acquire(_session_id):
        return Lease()

    async def body():
        events.append(("body_started",))
        yield b"chunk"

    async def openai_chat_completions(_payload, *, request):
        assert request is not None
        return inference.StreamingResponse(body(), status_code = 206)

    monkeypatch.setattr(
        inference,
        "_project_hook_turn_for_request",
        lambda _payload, _request: turn,
    )
    monkeypatch.setattr(
        inference.ProjectWorkspaceRequestLease,
        "acquire",
        acquire,
    )
    wrapped = inference._hold_project_workspace_for_request(openai_chat_completions)

    async def exercise():
        response = await wrapped(
            SimpleNamespace(session_id = "project-project"),
            request = SimpleNamespace(),
        )
        assert isinstance(response, inference._SameTaskStreamingResponse)

        async def send(message):
            if message["type"] == "http.response.start":
                raise OSError("client disconnected before body start")

        async def receive():
            return {"type": "http.disconnect"}

        with pytest.raises(inference.ClientDisconnect):
            await response({"type": "http"}, receive, send)

    asyncio.run(exercise())

    assert ("body_started",) not in events
    assert events.count(("turn", turn)) == 1
    assert events.count(("session_end", "project", token)) == 1
    assert events.count(("lease",)) == 1


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
def test_prompt_hook_disconnect_watcher_stops_on_normal_cleanup(monkeypatch, protocol):
    cancel_event = threading.Event()
    watcher_started = asyncio.Event()
    watcher_stopped = asyncio.Event()
    seen_stop_signals = []

    async def watch(_request, shared_cancel_event, stop_signal):
        assert shared_cancel_event is cancel_event
        assert stop_signal is not cancel_event
        seen_stop_signals.append(stop_signal)
        watcher_started.set()
        while not stop_signal.is_set():
            await asyncio.sleep(0)
        watcher_stopped.set()

    monkeypatch.setattr(inference, "_chat_cancel_event", lambda _request: cancel_event)
    monkeypatch.setattr(inference, "_await_disconnect_then_cancel", watch)

    async def exercise():
        if protocol == "openai":
            monkeypatch.setattr(
                inference,
                "_with_project_prompt_hook_messages",
                lambda messages, **_kwargs: messages,
            )
            result = await inference._with_project_prompt_hook_messages_async(
                [{"role": "user", "content": "hello"}],
                project_session_id = "project-project",
                thread_id = "thread",
                model = "model",
                bypass_permissions = False,
                request = SimpleNamespace(),
            )
            assert result == [{"role": "user", "content": "hello"}]
        else:
            monkeypatch.setattr(
                inference,
                "_with_anthropic_project_prompt_hooks",
                lambda system, **_kwargs: system,
            )
            result = await inference._with_anthropic_project_prompt_hooks_async(
                "system",
                messages = [{"role": "user", "content": "hello"}],
                project_session_id = "project-project",
                thread_id = "thread",
                model = "model",
                bypass_permissions = False,
                request = SimpleNamespace(),
            )
            assert result == "system"

        assert watcher_started.is_set()
        assert watcher_stopped.is_set()
        assert len(seen_stop_signals) == 1
        assert seen_stop_signals[0].is_set()
        assert not cancel_event.is_set()

    asyncio.run(exercise())


def test_project_hook_recovery_worker_stops_in_a_bounded_finally():
    import main

    events = []
    stop_event = threading.Event()

    class Worker:
        def start(self):
            events.append(("start",))

        def join(self, timeout):
            events.append(("join", timeout))

        def is_alive(self):
            return True

    class Log:
        def warning(self, message, *args):
            events.append(("warning", message, args))

    async def exercise():
        with pytest.raises(RuntimeError, match = "startup owner failed"):
            async with main._project_hook_recovery_worker(Worker(), stop_event, Log()):
                assert events == [("start",)]
                raise RuntimeError("startup owner failed")

    asyncio.run(exercise())

    assert stop_event.is_set()
    assert ("join", main._PROJECT_HOOK_RECOVERY_JOIN_TIMEOUT_S) in events
    assert any(event[0] == "warning" and "exceeded" in event[1] for event in events)


def test_stacked_stream_cancellation_closes_every_owned_lifecycle(monkeypatch):
    events = []
    token = hook_runtime.HookSessionToken("project", "anonymous", "generation", True)
    turn = hook_runtime.ProjectHookTurn(
        project_id = "project",
        session_id = "anonymous",
        turn_id = "turn",
        model = "model",
        permission_mode = "default",
        cancel_event = threading.Event(),
        synthetic_session = True,
        session_token = token,
    )
    monkeypatch.setattr(
        hook_runtime,
        "_unregister_project_hook_turn",
        lambda value: events.append(("turn", value)),
    )
    monkeypatch.setattr(
        hook_runtime,
        "end_project_hook_session",
        lambda project_id, **kwargs: (
            events.append(("session_end", project_id, kwargs["session_token"])),
            hook_runtime.HookEventResult(event = "SessionEnd"),
        )[1],
    )

    async def upstream():
        try:
            try:
                yield b"chunk"
            finally:
                events.append(("upstream",))
        finally:
            events.append(("tracker",))

    class Lease:
        async def release(self):
            events.append(("lease",))

    async def acquire(_session_id):
        return Lease()

    async def openai_chat_completions(_payload, *, request):
        assert request is not None
        return inference._SameTaskStreamingResponse(upstream())

    monkeypatch.setattr(
        inference,
        "_project_hook_turn_for_request",
        lambda _payload, _request: turn,
    )
    monkeypatch.setattr(
        inference.ProjectWorkspaceRequestLease,
        "acquire",
        acquire,
    )
    wrapped = inference._hold_project_workspace_for_request(openai_chat_completions)

    async def build_response():
        return await wrapped(
            SimpleNamespace(session_id = "project-project"),
            request = SimpleNamespace(),
        )

    response = asyncio.run(build_response())
    assert isinstance(response, inference._SameTaskStreamingResponse)

    async def send(message):
        if message["type"] == "http.response.body":
            raise asyncio.CancelledError()

    async def receive():
        return {"type": "http.disconnect"}

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(response({"type": "http"}, receive, send))

    assert ("upstream",) in events
    assert ("tracker",) in events
    assert ("turn", turn) in events
    assert ("session_end", "project", token) in events
    assert ("lease",) in events


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_external_project_nonstream_preserves_unicode_separators_for_stop(monkeypatch, separator):
    answer = f"left{separator}right"
    reviewed = []

    def run(_project_id, event, payload, **kwargs):
        reviewed.append((event, payload["last_assistant_message"], kwargs["model"]))
        return hook_runtime.HookEventResult(event = event)

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def stream_chat_completion(self, **_kwargs):
            yield "data: " + json.dumps(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": answer},
                            "finish_reason": None,
                        }
                    ]
                },
                ensure_ascii = False,
            )
            yield 'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
            yield "data: [DONE]"

        async def close(self):
            pass

    monkeypatch.setattr(hook_runtime, "run_project_hook_event", run)
    monkeypatch.setattr(inference, "ExternalProviderClient", Client)
    monkeypatch.setattr(
        inference.providers_db,
        "get_provider",
        lambda provider_id: {
            "id": provider_id,
            "provider_type": "openai",
            "base_url": "https://api.openai.com/v1",
            "display_name": "test",
            "is_enabled": True,
        },
    )
    monkeypatch.setattr(
        inference,
        "resolve_provider_api_key_or_400",
        lambda *_args, **_kwargs: "secret",
    )
    payload = ChatCompletionRequest(
        model = "default",
        external_model = "actual-model",
        provider_id = "provider-1",
        session_id = "project-project",
        thread_id = "thread-1",
        messages = [ChatMessage(role = "user", content = "start")],
        stream = False,
    )
    request = SimpleNamespace(
        headers = {},
        state = SimpleNamespace(skip_api_monitor = True),
        url = SimpleNamespace(path = "/v1/chat/completions"),
        method = "POST",
        is_disconnected = lambda: asyncio.sleep(0, result = False),
    )
    turn = hook_runtime.ProjectHookTurn(
        project_id = "project",
        session_id = "thread-1",
        turn_id = "turn-1",
        model = "actual-model",
        permission_mode = "default",
        cancel_event = threading.Event(),
    )

    async def exercise():
        with hook_runtime.activate_project_hook_turn(turn):
            response = await inference._proxy_to_external_provider(
                payload, request, current_subject = "test"
            )
            return json.loads(response.body)

    result = asyncio.run(exercise())

    assert result["choices"][0]["message"]["content"] == answer
    assert reviewed == [("Stop", answer, "actual-model")]


@pytest.mark.parametrize("stream", [False, True])
def test_external_project_prompt_hooks_use_dispatched_model(monkeypatch, stream):
    observed = []

    async def prompt(messages, **kwargs):
        observed.append(kwargs["model"])
        return messages

    async def guidance(messages, _session_id):
        return messages

    async def proxy(_payload, _request, _subject):
        return inference.JSONResponse({"ok": True})

    monkeypatch.setattr(inference, "_with_project_prompt_hook_messages_async", prompt)
    monkeypatch.setattr(inference, "_with_project_guidance_messages_async", guidance)
    monkeypatch.setattr(inference, "_proxy_to_external_provider", proxy)
    monkeypatch.setattr(inference, "_bind_current_project_hook_runtime", lambda _owner: None)
    monkeypatch.setattr(inference, "untrack_current_request", lambda *_args: None, raising = False)
    payload = ChatCompletionRequest(
        model = "default",
        external_model = "actual-model",
        provider_type = "openai",
        session_id = "project-project",
        thread_id = "thread-1",
        messages = [ChatMessage(role = "user", content = "start")],
        stream = stream,
    )
    request = SimpleNamespace(
        headers = {"authorization": "Bearer test"},
        state = SimpleNamespace(_project_hooks_resolved = False),
        scope = {},
        is_disconnected = lambda: asyncio.sleep(0, result = False),
    )
    monkeypatch.setattr(
        "auth.authentication.request_admitted_without_credential", lambda _request: False
    )
    monkeypatch.setattr(
        "core.inference.llama_keepwarm.untrack_current_request", lambda _scope: None
    )

    asyncio.run(
        inference.produce_openai_chat_completions(
            payload,
            request,
            "test",
            cancel_on_disconnect = False,
        )
    )

    assert observed == ["actual-model"]
