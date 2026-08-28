# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Restart survival for the OpenAI /v1/chat/completions passthrough.

A crashed llama-server relaunches on a NEW ephemeral port. /v1/messages already
respawns and retries; this surface did not, so a harness on the OpenAI API kept
posting to the dead port and stayed broken until the user reloaded the model by
hand, while an Anthropic-API client on the same backend recovered itself.

Twin of test_anthropic_passthrough_respawn.py, same stubs and same cases.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

import routes.inference as inf_mod
from core.agent_workspace.hook_runtime import (
    HookEventResult,
    activate_project_hook_turn,
)
from models.inference import ChatCompletionRequest, ChatMessage
from routes.inference import (
    _is_lost_upstream_connection,
    _openai_passthrough_non_streaming_upstream,
    _openai_passthrough_stream_admitted,
    _passthrough_retry_url,
)

_DEAD = "http://127.0.0.1:57953"
_FRESH = "http://127.0.0.1:62933"


class _Backend:
    """Stub llama backend whose base_url moves to a new port once respawned."""

    def __init__(
        self,
        *,
        respawn_ok = True,
        mtp_handled = False,
        stays_dead = False,
    ):
        self.base_url = _DEAD
        self.context_length = 4096
        self.respawn_calls = 0
        self.mtp_calls = 0
        self._respawn_ok = respawn_ok
        self._mtp_handled = mtp_handled
        # Models a relaunch that reports success but is not actually serving.
        self._stays_dead = stays_dead

    def count_chat_tokens(self, *_args, **_kwargs):
        return 2

    def _request_reasoning_kwargs(self, *_args, **_kwargs):
        return None

    def _maybe_recover_from_mtp_crash(self, _exc):
        self.mtp_calls += 1
        return self._mtp_handled

    def _respawn_if_dead(self):
        self.respawn_calls += 1
        if not self._respawn_ok:
            return False
        if not self._stays_dead:
            self.base_url = _FRESH
        return True


class _Request:
    async def is_disconnected(self):
        return False


class _Lease:
    """Records that the slot came back. Repeat calls are fine: the real lease is
    idempotent, and the stream's nested handlers both release."""

    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


class _Tracker:
    def __exit__(self, *_exc):
        return False


class _FakeNonStreamingClient:
    def __init__(self):
        self.urls = []

    async def aclose(self):
        pass

    async def post(self, url, **_kwargs):
        self.urls.append(url)
        if url.startswith(_DEAD):
            raise httpx.ConnectError("connection refused")
        return httpx.Response(
            200,
            json = {
                "id": "chatcmpl-1",
                "choices": [
                    {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            },
        )


class _StopTurn:
    def __init__(self, results):
        self.cancel_event = threading.Event()
        self._results = list(results)
        self.candidates = []
        self.stop_calls = 0

    def stop(self, *, last_assistant_message):
        self.candidates.append(last_assistant_message)
        self.stop_calls += 1
        if self._results:
            return self._results.pop(0)
        return HookEventResult(event = "Stop")


def _continue(reason, handler_id):
    return HookEventResult(
        event = "Stop",
        blocked = True,
        continuation_reason = reason,
        continuation_reasons = (reason,),
        continuation_fragments = ((handler_id, reason),),
    )


def _completion(
    content,
    *,
    usage = None,
    finish_reason = "stop",
    tool_calls = None,
):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": f"chatcmpl-{content or 'tool'}",
        "object": "chat.completion",
        "created": 1,
        "model": "gguf",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage or {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


class _SequenceNonStreamingClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def aclose(self):
        pass

    async def post(self, url, **kwargs):
        self.requests.append((url, kwargs["json"]))
        return httpx.Response(200, json = self.responses.pop(0))


def _install_stream_transport(monkeypatch, calls):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if str(request.url).startswith(_DEAD):
            raise httpx.ConnectError("connection refused")
        content = (
            f"data: {json.dumps({'choices': [{'delta': {'content': 'hi'}}]})}\n\ndata: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            content = content.encode(),
            headers = {"content-type": "text/event-stream"},
        )

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def _client(*_args, **kwargs):
        return real_client(transport = transport, timeout = kwargs.get("timeout", 600))

    monkeypatch.setattr(inf_mod.httpx, "AsyncClient", _client)


def _payload():
    return ChatCompletionRequest(
        model = "default",
        messages = [ChatMessage(role = "user", content = "hi")],
    )


async def _run_non_streaming(backend):
    return await _openai_passthrough_non_streaming_upstream(
        backend,
        _payload(),
        "test-model",
        request = _Request(),
        cancel_event = threading.Event(),
    )


async def _run_stream(backend, lease = None):
    payload = _payload()
    payload.stream = True
    response = await _openai_passthrough_stream_admitted(
        _Request(),
        threading.Event(),
        backend,
        payload,
        "test-model",
        "chatcmpl-local",
        admission_lease = lease or _Lease(),
        tracker = _Tracker(),
    )
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode() if isinstance(chunk, (bytes, bytearray)) else chunk)
    return "".join(chunks)


# ── Helper ────────────────────────────────────────────────────


def test_the_retry_url_is_shared_with_the_anthropic_surface():
    """Both passthroughs post to the same upstream route, so one helper serves both.
    A rename that leaves this surface behind is the bug being fixed."""
    backend = _Backend()

    url = asyncio.run(_passthrough_retry_url(backend, httpx.ConnectError("x")))

    assert url == f"{_FRESH}/v1/chat/completions"
    assert backend.respawn_calls == 1


# ── Non-streaming ─────────────────────────────────────────────


def test_non_streaming_retries_against_the_new_port(monkeypatch):
    client = _FakeNonStreamingClient()
    monkeypatch.setattr(inf_mod, "_cancelable_nonstreaming_client", lambda: client)
    backend = _Backend()

    response = asyncio.run(_run_non_streaming(backend))

    assert response.status_code == 200
    assert backend.respawn_calls == 1
    assert client.urls == [f"{_DEAD}/v1/chat/completions", f"{_FRESH}/v1/chat/completions"]


def test_non_streaming_still_502s_when_the_server_stays_dead(monkeypatch):
    client = _FakeNonStreamingClient()
    monkeypatch.setattr(inf_mod, "_cancelable_nonstreaming_client", lambda: client)
    backend = _Backend(respawn_ok = False)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(_run_non_streaming(backend))

    assert exc.value.status_code == 502
    assert client.urls == [f"{_DEAD}/v1/chat/completions"]  # no blind retry


def test_non_streaming_does_not_retry_an_mtp_crash(monkeypatch):
    # An MTP+tensor crash schedules its own reload; retrying would race it.
    client = _FakeNonStreamingClient()
    monkeypatch.setattr(inf_mod, "_cancelable_nonstreaming_client", lambda: client)
    backend = _Backend(mtp_handled = True)

    with pytest.raises(HTTPException):
        asyncio.run(_run_non_streaming(backend))

    assert backend.respawn_calls == 0


def test_non_streaming_respawns_at_most_once(monkeypatch):
    """A relaunch that reports success but is not serving must end in a 502, not a
    loop that respawns the model on every attempt."""
    client = _FakeNonStreamingClient()
    monkeypatch.setattr(inf_mod, "_cancelable_nonstreaming_client", lambda: client)
    backend = _Backend(stays_dead = True)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(_run_non_streaming(backend))

    assert exc.value.status_code == 502
    assert backend.respawn_calls == 1
    assert client.urls == [f"{_DEAD}/v1/chat/completions"] * 2


def test_non_streaming_runs_two_real_stop_continuations(monkeypatch):
    client = _SequenceNonStreamingClient(
        [
            _completion("A", usage = {"total_tokens": 2}),
            _completion("B", usage = {"total_tokens": 3}),
            _completion("C", usage = {"total_tokens": 5}),
        ]
    )
    monkeypatch.setattr(inf_mod, "_cancelable_nonstreaming_client", lambda: client)
    turn = _StopTurn(
        [
            _continue("repair one", "first"),
            _continue("repair two", "second"),
            HookEventResult(event = "Stop"),
        ]
    )

    with activate_project_hook_turn(turn):
        response = asyncio.run(_run_non_streaming(_Backend()))

    data = json.loads(response.body)
    assert data["choices"][0]["message"]["content"] == "ABC"
    assert data["usage"]["total_tokens"] == 10
    assert turn.candidates == ["A", "B", "C"]
    assert len(client.requests) == 3
    first_continuation = client.requests[1][1]["messages"]
    second_continuation = client.requests[2][1]["messages"]
    assert first_continuation[-2] == {"role": "assistant", "content": "A"}
    assert "repair one" in first_continuation[-1]["content"]
    assert [message["role"] for message in second_continuation[-4:]] == [
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert "repair two" in second_continuation[-1]["content"]


def test_non_streaming_continue_false_finalizes_without_dispatch(monkeypatch):
    client = _SequenceNonStreamingClient([_completion("final")])
    monkeypatch.setattr(inf_mod, "_cancelable_nonstreaming_client", lambda: client)
    turn = _StopTurn([HookEventResult(event = "Stop", stop_requested = True)])

    with activate_project_hook_turn(turn):
        response = asyncio.run(_run_non_streaming(_Backend()))

    assert json.loads(response.body)["choices"][0]["message"]["content"] == "final"
    assert turn.candidates == ["final"]
    assert len(client.requests) == 1


def test_non_streaming_pending_tool_call_skips_stop(monkeypatch):
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "lookup", "arguments": "{}"},
    }
    client = _SequenceNonStreamingClient(
        [_completion(None, finish_reason = "tool_calls", tool_calls = [tool_call])]
    )
    monkeypatch.setattr(inf_mod, "_cancelable_nonstreaming_client", lambda: client)
    turn = _StopTurn([_continue("must not run", "bad")])

    with activate_project_hook_turn(turn):
        response = asyncio.run(_run_non_streaming(_Backend()))

    assert json.loads(response.body)["choices"][0]["finish_reason"] == "tool_calls"
    assert turn.candidates == []
    assert len(client.requests) == 1


@pytest.mark.parametrize(
    "response",
    [
        {"choices": []},
        {
            "choices": [
                {"message": {"content": "one"}, "finish_reason": "stop"},
                {"message": {"content": "two"}, "finish_reason": "stop"},
            ]
        },
        {"choices": [{"message": {"content": "unfinished"}, "finish_reason": None}]},
    ],
)
def test_non_streaming_project_stop_rejects_ambiguous_or_incomplete_json(monkeypatch, response):
    client = _SequenceNonStreamingClient([response])
    monkeypatch.setattr(inf_mod, "_cancelable_nonstreaming_client", lambda: client)
    turn = _StopTurn([_continue("must not run", "bad")])

    with activate_project_hook_turn(turn):
        with pytest.raises(RuntimeError):
            asyncio.run(_run_non_streaming(_Backend()))

    assert turn.candidates == []
    assert len(client.requests) == 1


def test_non_streaming_project_stop_rejects_unparseable_success(monkeypatch):
    class Client(_SequenceNonStreamingClient):
        async def post(self, url, **kwargs):
            self.requests.append((url, kwargs["json"]))
            return httpx.Response(200, content = b"not-json")

    client = Client([])
    monkeypatch.setattr(inf_mod, "_cancelable_nonstreaming_client", lambda: client)
    turn = _StopTurn([_continue("must not run", "bad")])

    with activate_project_hook_turn(turn):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(_run_non_streaming(_Backend()))

    assert exc.value.status_code == 502
    assert turn.candidates == []


# ── Streaming ─────────────────────────────────────────────────


def test_streaming_retries_against_the_new_port(monkeypatch):
    calls = []
    _install_stream_transport(monkeypatch, calls)
    backend = _Backend()

    blob = asyncio.run(_run_stream(backend))

    assert backend.respawn_calls == 1
    assert calls == [f"{_DEAD}/v1/chat/completions", f"{_FRESH}/v1/chat/completions"]
    # The retried stream really produced the turn, not just a clean-looking stop.
    assert "hi" in blob
    assert "[DONE]" in blob


def test_streaming_holds_terminal_and_runs_two_stop_continuations(monkeypatch):
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if body.get("stream"):
            content = "".join(
                [
                    'data: {"id":"chatcmpl-a","model":"gguf","created":1,'
                    '"choices":[{"index":0,"delta":{"content":"A"},'
                    '"finish_reason":null}]}\n\n',
                    'data: {"id":"chatcmpl-a","model":"gguf","created":1,'
                    '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
                    'data: {"id":"chatcmpl-a","model":"gguf","created":1,'
                    '"choices":[],"usage":{"total_tokens":2}}\n\n',
                    "data: [DONE]\n\n",
                ]
            )
            return httpx.Response(
                200,
                content = content.encode(),
                headers = {"content-type": "text/event-stream"},
            )
        content = "B" if len(requests) == 2 else "C"
        return httpx.Response(
            200,
            json = _completion(content, usage = {"total_tokens": len(requests) + 1}),
        )

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        inf_mod.httpx,
        "AsyncClient",
        lambda *_args, **kwargs: real_client(
            transport = transport,
            timeout = kwargs.get("timeout", 600),
        ),
    )
    turn = _StopTurn(
        [
            _continue("repair one", "first"),
            _continue("repair two", "second"),
            HookEventResult(event = "Stop"),
        ]
    )

    with activate_project_hook_turn(turn):
        blob = asyncio.run(_run_stream(_Backend()))

    assert turn.candidates == ["A", "B", "C"], blob
    assert len(requests) == 3
    assert blob.count("data: [DONE]") == 1
    assert blob.count('"finish_reason": "stop"') == 1
    assert '"content":"A"' in blob
    assert '"content": "B"' in blob
    assert '"content": "C"' in blob
    assert "repair one" in requests[1]["messages"][-1]["content"]
    assert "repair two" in requests[2]["messages"][-1]["content"]


@pytest.mark.parametrize("interrupt", ["cancel", "disconnect"])
def test_streaming_interrupts_a_slow_stop_continuation_post(monkeypatch, interrupt):
    class DisconnectableRequest(_Request):
        def __init__(self):
            self.disconnected = False

        async def is_disconnected(self):
            return self.disconnected

    class WorkspaceLease:
        def __init__(self):
            self.release_calls = 0

        async def release(self):
            self.release_calls += 1

    async def exercise():
        content = "".join(
            [
                'data: {"id":"chatcmpl-a","model":"gguf","created":1,'
                '"choices":[{"index":0,"delta":{"content":"A"},'
                '"finish_reason":null}]}\n\n',
                'data: {"id":"chatcmpl-a","model":"gguf","created":1,'
                '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
                "data: [DONE]\n\n",
            ]
        )
        transport = httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                content = content.encode(),
                headers = {"content-type": "text/event-stream"},
            )
        )
        real_client = httpx.AsyncClient(transport = transport, timeout = 600)

        class SlowContinuationClient:
            def __init__(self):
                self.started = asyncio.Event()
                self.closed = asyncio.Event()
                self.close_calls = 0

            async def post(self, _url, **_kwargs):
                self.started.set()
                await self.closed.wait()
                raise httpx.ReadError("continuation client closed")

            async def aclose(self):
                self.close_calls += 1
                self.closed.set()

        slow_client = SlowContinuationClient()
        clients = iter((real_client, slow_client))
        monkeypatch.setattr(inf_mod.httpx, "AsyncClient", lambda *_args, **_kwargs: next(clients))
        request = DisconnectableRequest()
        cancel_event = threading.Event()
        admission_lease = _Lease()
        workspace_lease = WorkspaceLease()
        payload = _payload()
        payload.stream = True
        turn = _StopTurn([_continue("repair", "stop-handler")])

        with activate_project_hook_turn(turn):
            response = await _openai_passthrough_stream_admitted(
                request,
                cancel_event,
                _Backend(),
                payload,
                "test-model",
                "chatcmpl-local",
                admission_lease = admission_lease,
                tracker = _Tracker(),
            )
            inf_mod._attach_project_workspace_lease(response, workspace_lease)

            async def consume_body():
                async for _chunk in response.body_iterator:
                    pass

            consume = asyncio.create_task(consume_body())
            await asyncio.wait_for(slow_client.started.wait(), timeout = 1)
            if interrupt == "cancel":
                cancel_event.set()
            else:
                request.disconnected = True
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(consume, timeout = 2)

        assert cancel_event.is_set()
        assert slow_client.close_calls >= 1
        assert admission_lease.released
        assert workspace_lease.release_calls == 1

    asyncio.run(exercise())


def test_streaming_continue_false_finalizes_once(monkeypatch):
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        content = "".join(
            [
                'data: {"choices":[{"index":0,"delta":{"content":"final"},'
                '"finish_reason":null}]}\n\n',
                'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
                "data: [DONE]\n\n",
            ]
        )
        return httpx.Response(
            200,
            content = content.encode(),
            headers = {"content-type": "text/event-stream"},
        )

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        inf_mod.httpx,
        "AsyncClient",
        lambda *_args, **kwargs: real_client(
            transport = transport,
            timeout = kwargs.get("timeout", 600),
        ),
    )
    turn = _StopTurn([HookEventResult(event = "Stop", stop_requested = True)])

    with activate_project_hook_turn(turn):
        blob = asyncio.run(_run_stream(_Backend()))

    assert turn.candidates == ["final"]
    assert len(requests) == 1
    assert blob.count("data: [DONE]") == 1
    assert blob.count('"finish_reason": "stop"') == 1


@pytest.mark.parametrize(
    "content",
    [
        "",
        "data: [DONE]\n\n",
        'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
        (
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            'data: {"choices":[],"usage":{"total_tokens":2}}\n\n'
        ),
        (
            'data: {"choices":['
            '{"index":0,"delta":{},"finish_reason":"stop"},'
            '{"index":1,"delta":{},"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n"
        ),
    ],
)
def test_streaming_project_stop_requires_finish_and_done(monkeypatch, content):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content = content.encode(),
            headers = {"content-type": "text/event-stream"},
        )

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        inf_mod.httpx,
        "AsyncClient",
        lambda *_args, **kwargs: real_client(
            transport = transport,
            timeout = kwargs.get("timeout", 600),
        ),
    )
    turn = _StopTurn([_continue("must not run", "bad")])

    with activate_project_hook_turn(turn):
        blob = asyncio.run(_run_stream(_Backend()))

    assert turn.candidates == []
    assert '"error"' in blob
    assert blob.count("data: [DONE]") == 1


def test_streaming_project_stop_rejects_finish_then_read_timeout(monkeypatch):
    class TimeoutStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield (
                b'data: {"choices":[{"index":0,"delta":{"content":"partial"},'
                b'"finish_reason":null}]}\n\n'
                b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            )
            raise httpx.ReadTimeout("missing DONE")

        async def aclose(self):
            return None

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream = TimeoutStream(),
            headers = {"content-type": "text/event-stream"},
        )

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        inf_mod.httpx,
        "AsyncClient",
        lambda *_args, **kwargs: real_client(
            transport = transport,
            timeout = kwargs.get("timeout", 600),
        ),
    )
    turn = _StopTurn([_continue("must not run", "bad")])

    with activate_project_hook_turn(turn):
        blob = asyncio.run(_run_stream(_Backend()))

    assert turn.candidates == []
    assert '"error"' in blob
    assert blob.count("data: [DONE]") == 1


def test_streaming_still_502s_when_the_server_stays_dead(monkeypatch):
    calls = []
    _install_stream_transport(monkeypatch, calls)
    backend = _Backend(respawn_ok = False)
    lease = _Lease()

    with pytest.raises(HTTPException) as exc:
        asyncio.run(_run_stream(backend, lease))

    assert exc.value.status_code == 502
    assert calls == [f"{_DEAD}/v1/chat/completions"]  # no blind retry
    assert lease.released, "the failed dispatch kept its admission slot"


def test_streaming_does_not_retry_an_mtp_crash(monkeypatch):
    calls = []
    _install_stream_transport(monkeypatch, calls)
    backend = _Backend(mtp_handled = True)

    with pytest.raises(HTTPException):
        asyncio.run(_run_stream(backend))

    assert backend.respawn_calls == 0
    assert calls == [f"{_DEAD}/v1/chat/completions"]


def test_streaming_respawns_at_most_once(monkeypatch):
    calls = []
    _install_stream_transport(monkeypatch, calls)
    backend = _Backend(stays_dead = True)
    lease = _Lease()

    with pytest.raises(HTTPException):
        asyncio.run(_run_stream(backend, lease))

    assert backend.respawn_calls == 1
    assert calls == [f"{_DEAD}/v1/chat/completions"] * 2
    assert lease.released, "the slot was leaked across the respawn retry"


def test_a_backend_without_respawn_hooks_is_untouched(monkeypatch):
    """Remote and external backends have no llama-server to relaunch."""
    client = _FakeNonStreamingClient()
    monkeypatch.setattr(inf_mod, "_cancelable_nonstreaming_client", lambda: client)
    backend = SimpleNamespace(
        base_url = _DEAD,
        context_length = 4096,
        count_chat_tokens = lambda *_a, **_k: 2,
        _request_reasoning_kwargs = lambda *_a, **_k: None,
    )

    with pytest.raises(HTTPException):
        asyncio.run(_run_non_streaming(backend))

    assert client.urls == [f"{_DEAD}/v1/chat/completions"]


# ── Only a lost connection may be replayed ────────────────────


@pytest.mark.parametrize(
    "exc, retryable",
    [
        (httpx.ConnectError("refused"), True),
        (httpx.ReadError("reset"), True),
        (httpx.WriteError("broken pipe"), True),
        (httpx.CloseError("close"), True),
        (httpx.RemoteProtocolError("Server disconnected without sending a response."), True),
        (httpx.ReadTimeout("slow"), False),
        (httpx.ConnectTimeout("slow connect"), False),
        (httpx.WriteTimeout("slow write"), False),
        (httpx.PoolTimeout("no free connection"), False),
    ],
)
def test_only_lost_connections_are_replayable(exc, retryable):
    """A timeout means the server is slow, not gone.

    ``httpx.RequestError`` also covers ``TimeoutException``, and a 20-minute
    generation on a live llama-server raises ``ReadTimeout``: replaying it
    resubmits a prompt the server is still decoding. Same split as
    ``_open_chat_stream_with_respawn_retry``. ``RemoteProtocolError`` is a
    sibling of ``NetworkError``, not a subclass, so it is named explicitly.
    """
    assert _is_lost_upstream_connection(exc) is retryable


class _TimingOutClient:
    """Healthy but slow: every post exceeds the first-token budget."""

    def __init__(self):
        self.urls = []

    async def aclose(self):
        pass

    async def post(self, url, **_kwargs):
        self.urls.append(url)
        raise httpx.ReadTimeout("the model did not produce a first token in time")


def test_non_streaming_does_not_replay_a_slow_generation(monkeypatch):
    client = _TimingOutClient()
    monkeypatch.setattr(inf_mod, "_cancelable_nonstreaming_client", lambda: client)
    # A live server: _respawn_if_dead reports _healthy, so a retry would go back
    # to the SAME port with the same prompt while the first copy is still decoding.
    backend = _Backend(stays_dead = True)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(_run_non_streaming(backend))

    assert exc.value.status_code == 502
    assert backend.respawn_calls == 0, "a timeout respawned a healthy llama-server"
    assert client.urls == [f"{_DEAD}/v1/chat/completions"], "the slow generation was replayed"


# ── The retry must follow the respawned server's new api key ──


class _RotatingKeyBackend(_Backend):
    """llama-server mints a fresh --api-key on every launch (UNSLOTH_DIRECT_STREAM)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self._api_key = "key-before-the-crash"

    @property
    def _auth_headers(self):
        return {"Authorization": f"Bearer {self._api_key}"}

    def _respawn_if_dead(self):
        started = super()._respawn_if_dead()
        if started:
            self._api_key = "key-after-the-respawn"
        return started


class _AuthRecordingClient:
    def __init__(self):
        self.sent = []

    async def aclose(self):
        pass

    async def post(self, url, **kwargs):
        self.sent.append((url, dict(kwargs.get("headers") or {})))
        if url.startswith(_DEAD):
            raise httpx.ConnectError("connection refused")
        return httpx.Response(
            200,
            json = {
                "id": "chatcmpl-1",
                "choices": [
                    {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                ],
            },
        )


def test_non_streaming_retry_uses_the_respawned_api_key(monkeypatch):
    client = _AuthRecordingClient()
    monkeypatch.setattr(inf_mod, "_cancelable_nonstreaming_client", lambda: client)
    backend = _RotatingKeyBackend()

    resp = asyncio.run(_run_non_streaming(backend))

    assert resp.status_code == 200
    assert client.sent[-1][1]["Authorization"] == "Bearer key-after-the-respawn", (
        "the retry presented the pre-crash key, which the new server 401s"
    )


def test_streaming_retry_uses_the_respawned_api_key(monkeypatch):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), request.headers.get("authorization")))
        if str(request.url).startswith(_DEAD):
            raise httpx.ConnectError("connection refused")
        content = (
            f"data: {json.dumps({'choices': [{'delta': {'content': 'hi'}}]})}\n\ndata: [DONE]\n\n"
        )
        return httpx.Response(
            200, content = content.encode(), headers = {"content-type": "text/event-stream"}
        )

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        inf_mod.httpx,
        "AsyncClient",
        lambda *_a, **kw: real_client(transport = transport, timeout = kw.get("timeout", 600)),
    )
    backend = _RotatingKeyBackend()

    blob = asyncio.run(_run_stream(backend))

    assert "[DONE]" in blob
    assert seen[-1][1] == "Bearer key-after-the-respawn"


# ── A crash after the pre-header status window ────────────────


class _SlowDeadTransport(httpx.AsyncBaseTransport):
    """The dead port takes longer than the 100 ms pre-header window to fail.

    That is the ordinary shape of a llama-server that dies while the request is
    queued or prefilling: dispatch is still pending when the status window
    closes, so the failure surfaces inside _stream, not in the pre-header
    handler.
    """

    def __init__(
        self,
        calls,
        delay = 0.3,
    ):
        self.calls = calls
        self.delay = delay

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(str(request.url))
        if str(request.url).startswith(_DEAD):
            await asyncio.sleep(self.delay)
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
        content = (
            f"data: {json.dumps({'choices': [{'delta': {'content': 'hi'}}]})}\n\ndata: [DONE]\n\n"
        )
        return httpx.Response(
            200, content = content.encode(), headers = {"content-type": "text/event-stream"}
        )


def test_streaming_retries_a_crash_that_lands_after_the_status_window(monkeypatch):
    calls = []
    transport = _SlowDeadTransport(calls)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        inf_mod.httpx,
        "AsyncClient",
        lambda *_a, **kw: real_client(transport = transport, timeout = kw.get("timeout", 600)),
    )
    backend = _Backend()

    blob = asyncio.run(_run_stream(backend))

    assert calls == [f"{_DEAD}/v1/chat/completions", f"{_FRESH}/v1/chat/completions"]
    assert backend.respawn_calls == 1
    assert "hi" in blob and "[DONE]" in blob
    # No SSE error chunk leaked to the client before the recovery.
    assert "Lost connection" not in blob


class _SlowTimeoutTransport(_SlowDeadTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(str(request.url))
        await asyncio.sleep(self.delay)
        raise httpx.ReadTimeout("the model did not produce a first token in time")


def test_streaming_does_not_replay_a_slow_generation_after_the_status_window(monkeypatch):
    calls = []
    transport = _SlowTimeoutTransport(calls)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        inf_mod.httpx,
        "AsyncClient",
        lambda *_a, **kw: real_client(transport = transport, timeout = kw.get("timeout", 600)),
    )
    backend = _Backend(stays_dead = True)

    blob = asyncio.run(_run_stream(backend))

    assert calls == [f"{_DEAD}/v1/chat/completions"], "the slow generation was replayed"
    assert backend.respawn_calls == 0
    assert "[DONE]" in blob


class _SlowRespawnBackend(_Backend):
    """A relaunch that takes real time, the way reloading a large GGUF does.

    Blocks until the consumer reports a keep-alive that arrived AFTER the reload
    began, so the stub records whether the downstream connection was still being
    fed while the model loaded.
    """

    def __init__(self):
        super().__init__()
        self.respawn_started = threading.Event()
        self.keepalive_during_respawn = threading.Event()
        self.fed_while_loading = False

    def _respawn_if_dead(self):
        self.respawn_calls += 1
        self.respawn_started.set()
        self.fed_while_loading = self.keepalive_during_respawn.wait(timeout = 5.0)
        self.base_url = _FRESH
        return True


def test_streaming_keeps_the_stream_alive_while_the_server_respawns(monkeypatch):
    """The reload is a full model load, minutes for a large GGUF. The response is
    already committed and this loop keeps it alive every five seconds, so going
    silent for the reload lets a proxy or client drop the stream before the
    recovered request is ever submitted."""
    monkeypatch.setattr(inf_mod, "_OPENAI_PASSTHROUGH_PENDING_RESPONSE_KEEPALIVE_S", 0.05)
    calls = []
    transport = _SlowDeadTransport(calls)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        inf_mod.httpx,
        "AsyncClient",
        lambda *_a, **kw: real_client(transport = transport, timeout = kw.get("timeout", 600)),
    )
    backend = _SlowRespawnBackend()

    async def _drive():
        response = await _openai_passthrough_stream_admitted(
            _Request(),
            threading.Event(),
            backend,
            _payload(),
            "test-model",
            "chatcmpl-local",
            admission_lease = _Lease(),
            tracker = _Tracker(),
        )
        chunks = []
        async for chunk in response.body_iterator:
            text = chunk.decode() if isinstance(chunk, (bytes, bytearray)) else chunk
            chunks.append(text)
            if (
                backend.respawn_started.is_set()
                and text == inf_mod._OPENAI_PASSTHROUGH_SSE_KEEPALIVE
            ):
                backend.keepalive_during_respawn.set()
        return "".join(chunks)

    blob = asyncio.run(_drive())

    assert backend.respawn_calls == 1
    assert backend.fed_while_loading, "the stream went silent for the whole reload"
    assert calls == [f"{_DEAD}/v1/chat/completions", f"{_FRESH}/v1/chat/completions"]
    assert "hi" in blob and "[DONE]" in blob
