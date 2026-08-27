# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.responses import Response, StreamingResponse

from auth import authentication
from core import manual_compaction
from core.inference import llama_keepwarm
from models.inference import ChatCompletionRequest
from routes import inference
from storage import studio_db


def _reset_db(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)


def _message(message_id, parent_id, role, text):
    return {
        "id": message_id,
        "threadId": "thread-route",
        "parentId": parent_id,
        "role": role,
        "content": [{"type": "text", "text": text}],
        "createdAt": 10,
    }


def _prepare(tmp_path, monkeypatch):
    _reset_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(
        {
            "id": "thread-route",
            "title": "Compaction route test",
            "modelType": "base",
            "modelId": "model",
            "archived": False,
            "createdAt": 1,
        }
    )
    rows = [
        _message("u1", None, "user", "Explain the migration."),
        _message("a1", "u1", "assistant", "Use a staged rollout."),
        _message("compact-1", "a1", "user", "/compact"),
    ]
    for row in rows:
        studio_db.upsert_chat_message(row)
    messages = [
        {"role": "system", "content": "Project rules"},
        {"role": "user", "content": "Explain the migration."},
        {"role": "assistant", "content": "Use a staged rollout."},
        {"role": "user", "content": "/compact"},
    ]
    parsed = ChatCompletionRequest(model = "model", messages = messages)
    prepared = manual_compaction.prepare_manual_compaction(
        "thread-route",
        attempt_id = "attempt-route",
        command_message_id = "compact-1",
        expected_head_message_id = "compact-1",
        message_ids = ["u1", "a1", "compact-1"],
        request_messages = parsed.messages,
    )
    return prepared, messages


def _payload(
    prepared,
    messages,
    *,
    stream,
    provider = False,
):
    return ChatCompletionRequest(
        model = "model",
        external_model = "provider-model" if provider else None,
        provider_type = "openai" if provider else None,
        provider_base_url = "https://api.openai.com/v1" if provider else None,
        thread_id = "thread-route",
        messages = messages,
        stream = stream,
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


class _Request:
    def __init__(self):
        self.state = SimpleNamespace(skip_api_monitor = True)
        self.scope = {}
        self.url = SimpleNamespace(path = "/v1/chat/completions")
        self.method = "POST"

    async def is_disconnected(self):
        return False


def _stream_response(chunks):
    async def body():
        for chunk in chunks:
            yield chunk

    return StreamingResponse(body(), media_type = "text/event-stream")


def _sse(content = None, finish_reason = None):
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
            }
        )
        + "\n\n"
    )


def _json_response(text, finish_reason = "stop"):
    return Response(
        content = json.dumps(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": finish_reason,
                    }
                ]
            }
        ),
        media_type = "application/json",
    )


def _raw_json_response(value):
    content = value if isinstance(value, str) else json.dumps(value, allow_nan = True)
    return Response(content = content, media_type = "application/json")


async def _consume(response):
    return [chunk async for chunk in response.body_iterator]


def _allow_external_provider(monkeypatch):
    monkeypatch.setattr(authentication, "request_admitted_without_credential", lambda _r: False)
    monkeypatch.setattr(llama_keepwarm, "untrack_current_request", lambda _scope: None)


def _route_to_gguf_passthrough(monkeypatch, response):
    class Backend:
        is_loaded = True
        _is_audio = False
        model_identifier = "local/model"
        context_length = 4096
        supports_tools = True
        supports_tool_passthrough = True
        is_vision = False

    async def no_switch(*_args, **_kwargs):
        return None

    monkeypatch.setattr(inference, "_should_validate_before_switch", lambda: False)
    monkeypatch.setattr(inference, "_maybe_auto_switch_model", no_switch)
    monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: Backend())
    monkeypatch.setattr(inference, "_llama_public_model_id", lambda *_args: "local/model")
    monkeypatch.setattr(inference, "_fill_recommended_sampling_openai", lambda *_args: None)
    monkeypatch.setattr(inference, "_takes_tool_passthrough", lambda *_args: True)
    if isinstance(response, StreamingResponse):

        async def passthrough_stream(*_args, **_kwargs):
            return response

        monkeypatch.setattr(inference, "_openai_passthrough_stream", passthrough_stream)
    else:

        async def passthrough_nonstream(*_args, **_kwargs):
            return response

        monkeypatch.setattr(
            inference,
            "_openai_passthrough_non_streaming",
            passthrough_nonstream,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids = ["nonstream", "stream"])
async def test_external_provider_exit_records_server_output(tmp_path, monkeypatch, stream):
    prepared, messages = _prepare(tmp_path, monkeypatch)
    payload = _payload(prepared, messages, stream = stream, provider = True)
    payload.studio_tool_history = True
    payload.openai_code_exec_container_id = "request-openai-container"
    payload.anthropic_code_exec_container_id = "request-anthropic-container"
    payload.permission_mode = "full"
    payload.bypass_permissions = True
    payload.auto_heal_tool_calls = True
    payload.nudge_tool_calls = True
    payload.tool_call_timeout = 9999
    _allow_external_provider(monkeypatch)

    async def proxy(claimed, *_args):
        assert (
            claimed.messages[-1].content == manual_compaction.MANUAL_COMPACTION_HANDOFF_INSTRUCTION
        )
        assert claimed.studio_tool_history is None
        assert claimed.openai_code_exec_container_id is None
        assert claimed.anthropic_code_exec_container_id is None
        assert claimed.permission_mode == "off"
        assert claimed.bypass_permissions is False
        assert claimed.auto_heal_tool_calls is False
        assert claimed.nudge_tool_calls is False
        assert claimed.tool_call_timeout == 1
        assert claimed.provider_type == "openai"
        assert claimed.stream is stream
        if not stream:
            line = json.dumps(
                {
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "provider summary"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            )
            return _stream_response([line + "\n\n", "data: [DONE]\n\n"])
        return _stream_response(
            [_sse(content = "provider summary"), _sse(finish_reason = "stop"), "data: [DONE]\n\n"]
        )

    monkeypatch.setattr(inference, "_proxy_to_external_provider", proxy)
    response = await inference.produce_openai_chat_completions(
        payload,
        _Request(),
        "test-user",
        cancel_on_disconnect = True,
    )
    await _consume(response)

    stored = manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])
    assert stored["state"] == "running"
    assert stored["outputSummaryHash"] == manual_compaction.summary_hash("provider summary")
    assert stored["outputFinishReason"] == "stop"


@pytest.mark.asyncio
async def test_external_provider_accepts_combined_first_role_and_content_delta(
    tmp_path, monkeypatch
):
    prepared, messages = _prepare(tmp_path, monkeypatch)
    payload = _payload(prepared, messages, stream = True, provider = True)
    _allow_external_provider(monkeypatch)
    first = {
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "provider summary"},
                "finish_reason": None,
            }
        ]
    }

    async def proxy(*_args):
        return _stream_response(
            [
                "data: " + json.dumps(first) + "\n\n",
                _sse(finish_reason = "stop"),
                "data: [DONE]\n\n",
            ]
        )

    monkeypatch.setattr(inference, "_proxy_to_external_provider", proxy)
    response = await inference.produce_openai_chat_completions(
        payload,
        _Request(),
        "test-user",
        cancel_on_disconnect = True,
    )
    await _consume(response)

    stored = manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])
    assert stored["state"] == "running"
    assert stored["outputSummaryHash"] == manual_compaction.summary_hash("provider summary")


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids = ["nonstream", "stream"])
async def test_gguf_passthrough_exit_records_server_output(tmp_path, monkeypatch, stream):
    prepared, messages = _prepare(tmp_path, monkeypatch)
    payload = _payload(prepared, messages, stream = stream)
    response = (
        _stream_response(
            [_sse(content = "gguf summary"), _sse(finish_reason = "stop"), "data: [DONE]\n\n"]
        )
        if stream
        else _json_response("gguf summary")
    )
    _route_to_gguf_passthrough(monkeypatch, response)

    routed = await inference.produce_openai_chat_completions(
        payload,
        _Request(),
        "test-user",
        cancel_on_disconnect = True,
    )
    if stream:
        await _consume(routed)

    stored = manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])
    assert stored["state"] == "running"
    assert stored["outputSummaryHash"] == manual_compaction.summary_hash("gguf summary")
    assert stored["outputFinishReason"] == "stop"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chunks", "expected_state"),
    [
        (["data: not-json\n\n"], "failed"),
        (['data: {"error":{"message":"upstream failed"}}\n\n'], "failed"),
        ([_sse(content = "partial")], "failed"),
        ([_sse(content = "partial"), _sse(finish_reason = "stop")], "failed"),
        ([_sse(content = "partial"), _sse(finish_reason = "length"), "data: [DONE]\n\n"], "failed"),
        (
            [
                _sse(content = "partial"),
                _sse(finish_reason = "content_filter"),
                "data: [DONE]\n\n",
            ],
            "failed",
        ),
        ([_sse(content = "x" * (64 * 1024 + 1))], "failed"),
    ],
    ids = [
        "invalid-sse",
        "error-event",
        "missing-choice-terminal",
        "missing-done",
        "length",
        "content-filter",
        "oversize",
    ],
)
async def test_provider_stream_failures_terminalize_attempt(
    tmp_path, monkeypatch, chunks, expected_state
):
    prepared, messages = _prepare(tmp_path, monkeypatch)
    payload = _payload(prepared, messages, stream = True, provider = True)
    _allow_external_provider(monkeypatch)

    async def proxy(*_args):
        return _stream_response(chunks)

    monkeypatch.setattr(inference, "_proxy_to_external_provider", proxy)
    response = await inference.produce_openai_chat_completions(
        payload,
        _Request(),
        "test-user",
        cancel_on_disconnect = True,
    )
    visible = await _consume(response)
    assert len(visible) == 1
    assert "manual_compaction_failed" in visible[0]
    assert not any(raw in visible[0] for raw in chunks if raw != "data: [DONE]\n\n")

    stored = manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])
    assert stored["state"] == expected_state
    assert stored["outputSummaryHash"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        _json_response("partial", "length"),
        _json_response("partial", "content_filter"),
        _json_response("x" * (64 * 1024 + 1)),
        Response(content = "not-json", media_type = "application/json"),
    ],
    ids = ["length", "content-filter", "oversize", "invalid-json"],
)
async def test_gguf_passthrough_nonstream_failure_terminalizes_attempt(
    tmp_path, monkeypatch, response
):
    prepared, messages = _prepare(tmp_path, monkeypatch)
    payload = _payload(prepared, messages, stream = False)
    _route_to_gguf_passthrough(monkeypatch, response)

    visible = await inference.produce_openai_chat_completions(
        payload,
        _Request(),
        "test-user",
        cancel_on_disconnect = True,
    )
    assert visible.status_code == 502
    assert json.loads(visible.body)["error"]["code"] == "manual_compaction_failed"

    stored = manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])
    assert stored["state"] == "failed"
    assert stored["outputSummaryHash"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        _raw_json_response({"error": {"message": "failed"}, "choices": []}),
        _raw_json_response({"choices": []}),
        _raw_json_response(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "one"},
                        "finish_reason": "stop",
                    },
                    {
                        "index": 1,
                        "message": {"role": "assistant", "content": "two"},
                        "finish_reason": "stop",
                    },
                ]
            }
        ),
        _raw_json_response(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "summary"},
                        "finish_reason": "stop",
                    }
                ]
            }
        ),
        _raw_json_response(
            {
                "choices": [
                    {
                        "index": 1,
                        "message": {"role": "assistant", "content": "summary"},
                        "finish_reason": "stop",
                    }
                ]
            }
        ),
        _raw_json_response(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "user", "content": "summary"},
                        "finish_reason": "stop",
                    }
                ]
            }
        ),
        *[
            _raw_json_response(
                {
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "summary",
                                field: value,
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            )
            for field, value in (
                ("tool_calls", [{"id": "call"}]),
                ("function_call", {"name": "tool"}),
                ("refusal", "no"),
                ("audio", {"id": "audio"}),
                ("reasoning_content", "hidden"),
            )
        ],
        _raw_json_response(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "summary"}],
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        ),
        _raw_json_response(
            '{"choices":[{"index":0,"message":{"role":"assistant",'
            '"content":"summary"},"finish_reason":"stop"}],"usage":{"total_tokens":NaN}}'
        ),
    ],
    ids = [
        "error",
        "empty-choices",
        "multiple-choices",
        "missing-index",
        "wrong-index",
        "wrong-role",
        "tool-calls",
        "function-call",
        "refusal",
        "audio",
        "reasoning",
        "structured-content",
        "non-rfc-json",
    ],
)
async def test_nonstream_canonical_observer_rejects_unsupported_output(
    tmp_path, monkeypatch, response
):
    prepared, messages = _prepare(tmp_path, monkeypatch)
    payload = _payload(prepared, messages, stream = False)
    _route_to_gguf_passthrough(monkeypatch, response)

    visible = await inference.produce_openai_chat_completions(
        payload,
        _Request(),
        "test-user",
        cancel_on_disconnect = True,
    )
    assert visible.status_code == 502
    assert json.loads(visible.body)["error"]["code"] == "manual_compaction_failed"

    stored = manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])
    assert stored["state"] == "failed"
    assert stored["outputSummaryHash"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event",
    [
        {"error": {"message": "failed"}, "choices": []},
        {"choices": []},
        {"choices": [{"index": 0, "delta": {}}, {"index": 1, "delta": {}}]},
        {"choices": [{"delta": {"content": "summary"}}]},
        {"choices": [{"index": 1, "delta": {"content": "summary"}}]},
        {"choices": [{"index": 0, "delta": {"role": "user"}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"id": "call"}]}}]},
        {"choices": [{"index": 0, "delta": {"function_call": {"name": "tool"}}}]},
        {"choices": [{"index": 0, "delta": {"refusal": "no"}}]},
        {"choices": [{"index": 0, "delta": {"audio": {"id": "audio"}}}]},
        {"choices": [{"index": 0, "delta": {"content": {"text": "summary"}}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": None}]},
        {
            "created": float("nan"),
            "choices": [{"index": 0, "delta": {"content": "summary"}}],
        },
    ],
    ids = [
        "error",
        "empty-choices",
        "multiple-choices",
        "missing-index",
        "wrong-index",
        "wrong-role",
        "tool-calls",
        "function-call",
        "refusal",
        "audio",
        "structured-content",
        "empty-delta",
        "non-rfc-json",
    ],
)
async def test_stream_canonical_observer_rejects_unsupported_output(tmp_path, monkeypatch, event):
    prepared, messages = _prepare(tmp_path, monkeypatch)
    payload = _payload(prepared, messages, stream = True, provider = True)
    _allow_external_provider(monkeypatch)

    async def proxy(*_args):
        return _stream_response(["data: " + json.dumps(event) + "\n\n", "data: [DONE]\n\n"])

    monkeypatch.setattr(inference, "_proxy_to_external_provider", proxy)
    response = await inference.produce_openai_chat_completions(
        payload,
        _Request(),
        "test-user",
        cancel_on_disconnect = True,
    )
    visible = await _consume(response)
    assert len(visible) == 1
    assert "manual_compaction_failed" in visible[0]
    raw_event = "data: " + json.dumps(event) + "\n\n"
    assert raw_event not in visible[0]

    stored = manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])
    assert stored["state"] == "failed"
    assert stored["outputSummaryHash"] is None


@pytest.mark.asyncio
async def test_gguf_passthrough_nonstream_cancel_terminalizes_attempt(tmp_path, monkeypatch):
    prepared, messages = _prepare(tmp_path, monkeypatch)
    payload = _payload(prepared, messages, stream = False)
    _route_to_gguf_passthrough(monkeypatch, _json_response("unused"))

    async def cancelled_passthrough(*_args, **kwargs):
        kwargs["cancel_event"].set()
        return _json_response("partial")

    monkeypatch.setattr(
        inference,
        "_openai_passthrough_non_streaming",
        cancelled_passthrough,
    )
    visible = await inference.produce_openai_chat_completions(
        payload,
        _Request(),
        "test-user",
        cancel_on_disconnect = True,
    )
    assert visible.status_code == 499
    assert json.loads(visible.body)["error"]["code"] == "manual_compaction_cancelled"

    stored = manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])
    assert stored["state"] == "cancelled"
    assert stored["outputSummaryHash"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exception", "expected_state", "expected_reason"),
    [
        (RuntimeError("routing failed"), "failed", "inference_failed"),
        (asyncio.CancelledError(), "cancelled", "inference_cancelled"),
    ],
    ids = ["failure", "cancelled"],
)
async def test_pre_response_failure_terminalizes_and_blocks_retry(
    tmp_path, monkeypatch, exception, expected_state, expected_reason
):
    prepared, messages = _prepare(tmp_path, monkeypatch)
    payload = _payload(prepared, messages, stream = False, provider = True)
    _allow_external_provider(monkeypatch)

    async def proxy(*_args):
        raise exception

    monkeypatch.setattr(inference, "_proxy_to_external_provider", proxy)
    if isinstance(exception, asyncio.CancelledError):
        with pytest.raises(asyncio.CancelledError):
            await inference.produce_openai_chat_completions(
                payload,
                _Request(),
                "test-user",
                cancel_on_disconnect = True,
            )
    else:
        visible = await inference.produce_openai_chat_completions(
            payload,
            _Request(),
            "test-user",
            cancel_on_disconnect = True,
        )
        assert visible.status_code == 502
        assert json.loads(visible.body)["error"]["code"] == "manual_compaction_failed"

    stored = manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])
    assert stored["state"] == expected_state
    assert stored["terminalReason"] == expected_reason
    retry = _payload(prepared, messages, stream = False, provider = True)
    with pytest.raises(HTTPException, match = "no longer claimable"):
        await inference.produce_openai_chat_completions(
            retry,
            _Request(),
            "test-user",
            cancel_on_disconnect = True,
        )


@pytest.mark.asyncio
async def test_claimed_failure_never_persists_or_returns_exception_secrets(tmp_path, monkeypatch):
    prepared, messages = _prepare(tmp_path, monkeypatch)
    payload = _payload(prepared, messages, stream = False, provider = True)
    _allow_external_provider(monkeypatch)
    secret = "Bearer sk-secret-do-not-store"

    async def proxy(*_args):
        raise RuntimeError(f"provider rejected {secret}")

    monkeypatch.setattr(inference, "_proxy_to_external_provider", proxy)
    visible = await inference.produce_openai_chat_completions(
        payload,
        _Request(),
        "test-user",
        cancel_on_disconnect = True,
    )
    assert visible.status_code == 502
    assert json.loads(visible.body)["error"]["code"] == "manual_compaction_failed"
    assert secret not in visible.body.decode("utf-8")

    stored = manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])
    assert stored["terminalReason"] == "inference_failed"
    assert secret not in json.dumps(stored)
    conn = studio_db.get_connection()
    try:
        raw = conn.execute(
            "SELECT terminal_reason FROM manual_compactions WHERE attempt_id = ?",
            (prepared["attemptId"],),
        ).fetchone()["terminal_reason"]
    finally:
        conn.close()
    assert raw == "inference_failed"
    assert secret not in raw


@pytest.mark.asyncio
async def test_nonstream_provider_error_body_is_replaced_without_leaking_secrets(
    tmp_path, monkeypatch
):
    prepared, messages = _prepare(tmp_path, monkeypatch)
    payload = _payload(prepared, messages, stream = False)
    secret = "sk-provider-body-secret"
    upstream = _raw_json_response({"error": {"message": secret}, "choices": []})
    upstream.status_code = 401
    _route_to_gguf_passthrough(monkeypatch, upstream)

    visible = await inference.produce_openai_chat_completions(
        payload,
        _Request(),
        "test-user",
        cancel_on_disconnect = True,
    )

    body = visible.body.decode("utf-8")
    assert visible.status_code == 502
    assert "manual_compaction_failed" in body
    assert secret not in body


@pytest.mark.asyncio
async def test_stream_provider_error_event_is_replaced_without_leaking_secrets(
    tmp_path, monkeypatch
):
    prepared, messages = _prepare(tmp_path, monkeypatch)
    payload = _payload(prepared, messages, stream = True, provider = True)
    _allow_external_provider(monkeypatch)
    secret = "Bearer sk-provider-sse-secret"

    async def proxy(*_args):
        return _stream_response(["data: " + json.dumps({"error": {"message": secret}}) + "\n\n"])

    monkeypatch.setattr(inference, "_proxy_to_external_provider", proxy)
    response = await inference.produce_openai_chat_completions(
        payload,
        _Request(),
        "test-user",
        cancel_on_disconnect = True,
    )
    visible = "".join(await _consume(response))

    assert "manual_compaction_failed" in visible
    assert secret not in visible
    stored = manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])
    assert stored["state"] == "failed"


@pytest.mark.asyncio
async def test_stream_disconnect_cancels_attempt(tmp_path, monkeypatch):
    prepared, messages = _prepare(tmp_path, monkeypatch)
    payload = _payload(prepared, messages, stream = True, provider = True)
    _allow_external_provider(monkeypatch)

    async def proxy(*_args):
        async def body():
            yield _sse(content = "partial")
            raise asyncio.CancelledError()

        return StreamingResponse(body(), media_type = "text/event-stream")

    monkeypatch.setattr(inference, "_proxy_to_external_provider", proxy)
    response = await inference.produce_openai_chat_completions(
        payload,
        _Request(),
        "test-user",
        cancel_on_disconnect = True,
    )
    iterator = response.body_iterator
    with pytest.raises(asyncio.CancelledError):
        await iterator.__anext__()

    stored = manual_compaction.get_manual_compaction_attempt(prepared["attemptId"])
    assert stored["state"] == "cancelled"
    assert stored["outputSummaryHash"] is None
