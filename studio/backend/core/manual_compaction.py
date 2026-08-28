# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Durable, branch-pinned state for the manual ``/compact`` handoff.

The browser owns presentation and slash-command interception. This module owns the
security boundary: the exact stored branch, its revision and the summary that may
become active are all re-read under SQLite write transactions.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import re
import time
from typing import Any, Iterable


MAX_MANUAL_COMPACTION_MESSAGES = 10_000
MAX_MANUAL_COMPACTION_REQUEST_MESSAGES = 40_000
MAX_MANUAL_COMPACTION_SOURCE_BYTES = 16 * 1024 * 1024
MAX_MANUAL_COMPACTION_SUMMARY_BYTES = 64 * 1024
MAX_MANUAL_COMPACTION_SUMMARY_TOKENS = MAX_MANUAL_COMPACTION_SUMMARY_BYTES // 4
MAX_MANUAL_COMPACTION_ID_CHARS = 128
MANUAL_COMPACTION_LEASE_MS = 60 * 60 * 1000
MANUAL_COMPACTION_PENDING_TTL_MS = 60 * 60 * 1000
MANUAL_COMPACTION_TERMINAL_RETENTION_MS = 30 * 24 * 60 * 60 * 1000
MAX_MANUAL_COMPACTION_TERMINAL_ATTEMPTS_PER_THREAD = 32
MANUAL_COMPACTION_SCHEMA_VERSION = 1
MANUAL_COMPACTION_CLIENT_KEY = "manualCompactionClient"
MANUAL_COMPACTION_HANDOFF_INSTRUCTION = (
    "Create a durable handoff summary of the conversation above. Preserve the user's goals, "
    "decisions, constraints, relevant files and commands, observed errors, completed work, and "
    "unresolved next steps. Return only the summary. Do not use tools and do not continue the "
    "conversation."
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CLAIM_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_AUDIO_DATA_RE = re.compile(r"data:audio/[a-z0-9.+-]+;base64,[A-Za-z0-9+/=]+")
_SEARCH_IMAGE_ID_RE = re.compile(r"^[0-9a-f]{12}$")
_SEARCH_IMAGE_TOKEN_RE = re.compile(
    r"\n\n[ \t]*\[\[img:[0-9a-f]{12}\]\][ \t]*(?=\n\n|\n?$)"
    r"|\[\[img:[0-9a-f]{12}\]\]"
)
_FENCED_CODE_RE = re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~")
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_SERVER_BUILTIN_TOOL_NAMES = frozenset(
    {"web_search", "web_fetch", "code_execution", "image_generation"}
)
_SANDBOX_TOOL_NAMES = frozenset({"python", "terminal"})
_EMPTY_JSON = "[]"
_EMPTY_JSON_HASH = hashlib.sha256(_EMPTY_JSON.encode("utf-8")).hexdigest()
_INFERENCE_TERMINAL_REASONS = frozenset(
    {
        "inference_cancelled",
        "inference_failed",
        "request_rewrite_failed",
    }
)
_TERMINAL_REASONS = _INFERENCE_TERMINAL_REASONS | frozenset(
    {
        "cancelled",
        "finish_content_filter",
        "finish_function_call",
        "finish_length",
        "finish_tool_calls",
        "invalid_summary_output",
        "lease_expired",
        "migrated_cancelled",
        "migrated_duplicate_live_branch",
        "migrated_failed",
        "pending_expired",
        "prepare_invalidated",
        "replaced",
    }
)


class ManualCompactionError(RuntimeError):
    status_code = 400


class ManualCompactionNotFound(ManualCompactionError):
    status_code = 404


class ManualCompactionConflict(ManualCompactionError):
    status_code = 409


def _db():
    from storage import studio_db
    return studio_db


def _strict_utf8(
    value: str,
    *,
    label: str,
    max_bytes: int | None = None,
) -> bytes:
    if not isinstance(value, str):
        raise ManualCompactionError(f"{label} must be a string")
    try:
        encoded = value.encode("utf-8", errors = "strict")
    except UnicodeEncodeError as exc:
        raise ManualCompactionError(f"{label} must be valid UTF-8") from exc
    if max_bytes is not None and len(encoded) > max_bytes:
        raise ManualCompactionError(f"{label} is too large")
    return encoded


def _bounded_id(value: str, label: str) -> str:
    encoded = _strict_utf8(value, label = label, max_bytes = MAX_MANUAL_COMPACTION_ID_CHARS)
    if not encoded or not value.strip():
        raise ManualCompactionError(f"{label} must not be blank")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise ManualCompactionError(f"{label} contains control characters")
    return value


def _json_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none = True)
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


_ABSENT_JSON = object()


def _reject_non_rfc_json_constant(value: str) -> None:
    raise ValueError(f"Non-RFC JSON constant {value!r}")


def _strict_json_loads(value: str) -> Any:
    value.encode("utf-8", errors = "strict")
    return json.loads(value, parse_constant = _reject_non_rfc_json_constant)


def _strict_json_dumps(value: Any, *, sort_keys: bool = False) -> str:
    return json.dumps(
        value,
        ensure_ascii = False,
        allow_nan = False,
        sort_keys = sort_keys,
        separators = (",", ":"),
    )


def _claim_token_hash(value: Any) -> str:
    if not isinstance(value, str) or _CLAIM_ID_RE.fullmatch(value) is None:
        raise ManualCompactionError("claim id must be a 256-bit lowercase hexadecimal token")
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _stored_bounded_id(value: Any, label: str) -> str:
    try:
        return _bounded_id(value, label)
    except ManualCompactionError as exc:
        raise ManualCompactionConflict(f"Stored manual compaction {label} is invalid") from exc


_CLIENT_BASE_KEYS = frozenset(
    {
        "schemaVersion",
        "state",
        "threadId",
        "commandMessageId",
        "attemptId",
        "summaryMessageId",
        "attemptSequence",
        "expectedHeadMessageId",
    }
)
_CLIENT_ENVELOPE_KEYS = frozenset(
    {
        "attemptId",
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
    }
)


def _manual_compaction_client_record(message: dict[str, Any]) -> dict[str, Any] | None:
    metadata = message.get("metadata")
    if not isinstance(metadata, dict) or MANUAL_COMPACTION_CLIENT_KEY not in metadata:
        return None
    raw = metadata[MANUAL_COMPACTION_CLIENT_KEY]
    if not isinstance(raw, dict):
        raise ManualCompactionConflict("Stored manual compaction client metadata is invalid")
    state = raw.get("state")
    expected_keys = set(_CLIENT_BASE_KEYS)
    if state in ("prepared", "summary_saved"):
        expected_keys.update(("sourceHeadMessageId", "envelope"))
    if state == "summary_saved":
        expected_keys.add("summaryHash")
    if state not in ("bound", "prepared", "summary_saved") or set(raw) != expected_keys:
        raise ManualCompactionConflict("Stored manual compaction client metadata is invalid")
    if raw.get("schemaVersion") != MANUAL_COMPACTION_SCHEMA_VERSION:
        raise ManualCompactionConflict("Stored manual compaction client metadata is invalid")

    for key, label in (
        ("threadId", "thread id"),
        ("commandMessageId", "command message id"),
        ("attemptId", "attempt id"),
        ("summaryMessageId", "summary message id"),
        ("expectedHeadMessageId", "expected head message id"),
    ):
        _stored_bounded_id(raw.get(key), label)
    sequence = raw.get("attemptSequence")
    if (
        not isinstance(sequence, int)
        or isinstance(sequence, bool)
        or sequence < 1
        or sequence > 2**53 - 1
    ):
        raise ManualCompactionConflict("Stored manual compaction attempt sequence is invalid")
    if (
        raw["threadId"] != message.get("threadId")
        or raw["commandMessageId"] != message.get("id")
        or raw["expectedHeadMessageId"] != raw["commandMessageId"]
        or raw["summaryMessageId"] == raw["commandMessageId"]
    ):
        raise ManualCompactionConflict("Stored manual compaction client identity is inconsistent")

    if state == "bound":
        return dict(raw)

    source_head_message_id = _stored_bounded_id(
        raw.get("sourceHeadMessageId"), "source head message id"
    )
    if source_head_message_id == raw["commandMessageId"]:
        raise ManualCompactionConflict("Stored manual compaction source head is invalid")
    envelope = raw.get("envelope")
    if not isinstance(envelope, dict) or set(envelope) != _CLIENT_ENVELOPE_KEYS:
        raise ManualCompactionConflict("Stored manual compaction envelope is invalid")
    for key, label in (
        ("attemptId", "envelope attempt id"),
        ("threadId", "envelope thread id"),
        ("commandMessageId", "envelope command message id"),
        ("expectedHeadMessageId", "envelope expected head message id"),
    ):
        if _stored_bounded_id(envelope.get(key), label) != raw[key]:
            raise ManualCompactionConflict("Stored manual compaction envelope is inconsistent")
    for key in (
        "sourceHash",
        "requestHash",
        "projectInstructionDigest",
        "contextDigest",
    ):
        value = envelope.get(key)
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise ManualCompactionConflict("Stored manual compaction envelope is invalid")
    request_message_count = envelope.get("requestMessageCount")
    project_instruction_revision = envelope.get("projectInstructionRevision")
    revision = envelope.get("revision")
    if (
        not isinstance(request_message_count, int)
        or isinstance(request_message_count, bool)
        or not 2 <= request_message_count <= MAX_MANUAL_COMPACTION_REQUEST_MESSAGES
        or not isinstance(project_instruction_revision, int)
        or isinstance(project_instruction_revision, bool)
        or project_instruction_revision < 0
        or not isinstance(revision, int)
        or isinstance(revision, bool)
        or revision < 1
    ):
        raise ManualCompactionConflict("Stored manual compaction envelope is invalid")
    if state == "summary_saved":
        summary_hash = raw.get("summaryHash")
        if not isinstance(summary_hash, str) or _SHA256_RE.fullmatch(summary_hash) is None:
            raise ManualCompactionConflict("Stored manual compaction summary hash is invalid")
    return dict(raw)


def _client_record_identity(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        record[key]
        for key in (
            "threadId",
            "commandMessageId",
            "attemptId",
            "summaryMessageId",
            "attemptSequence",
            "expectedHeadMessageId",
        )
    )


def _reservation_identity(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        record[key]
        for key in (
            "attemptId",
            "threadId",
            "commandMessageId",
            "summaryMessageId",
            "attemptSequence",
        )
    )


def _reservation_from_row(row: Any) -> dict[str, Any]:
    data = dict(row)
    reservation = {
        "attemptId": _stored_bounded_id(data.get("attempt_id"), "reserved attempt id"),
        "threadId": _stored_bounded_id(data.get("thread_id"), "reserved thread id"),
        "commandMessageId": _stored_bounded_id(
            data.get("command_message_id"), "reserved command message id"
        ),
        "summaryMessageId": _stored_bounded_id(
            data.get("summary_message_id"), "reserved summary message id"
        ),
        "attemptSequence": data.get("attempt_sequence"),
    }
    sequence = reservation["attemptSequence"]
    if (
        not isinstance(sequence, int)
        or isinstance(sequence, bool)
        or not 1 <= sequence <= 2**53 - 1
    ):
        raise ManualCompactionConflict("Stored manual compaction reservation sequence is invalid")
    created_at = data.get("created_at")
    if not isinstance(created_at, int) or isinstance(created_at, bool) or created_at < 1:
        raise ManualCompactionConflict("Stored manual compaction reservation time is invalid")
    retired_at = data.get("retired_at")
    if retired_at is not None and (
        not isinstance(retired_at, int) or isinstance(retired_at, bool) or retired_at < 1
    ):
        raise ManualCompactionConflict(
            "Stored manual compaction reservation retirement time is invalid"
        )
    reservation["retiredAt"] = retired_at
    return reservation


def _reserve_manual_compaction_attempt(conn, record: dict[str, Any], *, created_at: int) -> None:
    """Reserve one global attempt identity under the bind writer transaction."""
    existing = conn.execute(
        "SELECT * FROM manual_compaction_attempt_reservations WHERE attempt_id = ?",
        (record["attemptId"],),
    ).fetchone()
    if existing is not None:
        stored_reservation = _reservation_from_row(existing)
        if stored_reservation["retiredAt"] is not None:
            raise ManualCompactionConflict("Attempt id is permanently retired")
        if _reservation_identity(stored_reservation) != _reservation_identity(record):
            raise ManualCompactionConflict("Attempt id is already reserved for another binding")
        return
    occupied_sequence = conn.execute(
        "SELECT * FROM manual_compaction_attempt_reservations "
        "WHERE thread_id = ? AND command_message_id = ? AND attempt_sequence = ?",
        (record["threadId"], record["commandMessageId"], record["attemptSequence"]),
    ).fetchone()
    if occupied_sequence is not None:
        raise ManualCompactionConflict("Attempt sequence is already reserved for this command")
    occupied_summary = conn.execute(
        "SELECT thread_id, command_message_id FROM manual_compaction_attempt_reservations "
        "WHERE summary_message_id = ? LIMIT 1",
        (record["summaryMessageId"],),
    ).fetchone()
    if occupied_summary is not None and (
        str(occupied_summary["thread_id"]) != record["threadId"]
        or str(occupied_summary["command_message_id"]) != record["commandMessageId"]
    ):
        raise ManualCompactionConflict("Summary message id is reserved for another command")
    conn.execute(
        "INSERT INTO manual_compaction_attempt_reservations "
        "(attempt_id, thread_id, command_message_id, summary_message_id, attempt_sequence, "
        "created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (*_reservation_identity(record), created_at),
    )


def _require_manual_compaction_attempt_reservation(conn, record: dict[str, Any]) -> None:
    row = conn.execute(
        "SELECT * FROM manual_compaction_attempt_reservations WHERE attempt_id = ?",
        (record["attemptId"],),
    ).fetchone()
    reservation = _reservation_from_row(row) if row is not None else None
    if (
        reservation is None
        or reservation["retiredAt"] is not None
        or _reservation_identity(reservation) != _reservation_identity(record)
    ):
        raise ManualCompactionConflict("Manual compaction attempt reservation is stale")


def _write_manual_compaction_client_record(
    conn, message: dict[str, Any], record: dict[str, Any]
) -> dict[str, Any]:
    metadata = message.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    metadata[MANUAL_COMPACTION_CLIENT_KEY] = record
    changed = conn.execute(
        "UPDATE chat_messages SET metadata_json = ? WHERE thread_id = ? AND id = ?",
        (
            _strict_json_dumps(metadata),
            message["threadId"],
            message["id"],
        ),
    ).rowcount
    if changed != 1:
        raise ManualCompactionConflict("The /compact command changed during its state transition")
    updated = dict(message)
    updated["metadata"] = metadata
    return updated


def _decode_stored_json(
    value: Any,
    *,
    label: str,
    absent: Any = _ABSENT_JSON,
) -> Any:
    if value is None:
        if absent is _ABSENT_JSON:
            raise ManualCompactionConflict(f"Stored {label} JSON is missing")
        return absent
    if not isinstance(value, str):
        raise ManualCompactionConflict(f"Stored {label} JSON is invalid")
    try:
        return _strict_json_loads(value)
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ManualCompactionConflict(f"Stored {label} JSON is invalid") from exc


def _message_dict(row: Any) -> dict[str, Any]:
    data = dict(row)
    content = _decode_stored_json(data.get("content_json"), label = "message content")
    attachments = _decode_stored_json(
        data.get("attachments_json"), label = "message attachments", absent = None
    )
    metadata = _decode_stored_json(data.get("metadata_json"), label = "message metadata", absent = None)
    if not isinstance(content, (str, list)):
        raise ManualCompactionConflict("Stored message content JSON has an invalid shape")
    if attachments is not None and not isinstance(attachments, list):
        raise ManualCompactionConflict("Stored message attachments JSON has an invalid shape")
    if metadata is not None and not isinstance(metadata, dict):
        raise ManualCompactionConflict("Stored message metadata JSON has an invalid shape")
    return {
        "id": str(data["id"]),
        "threadId": str(data["thread_id"]),
        "parentId": data.get("parent_id"),
        "role": str(data["role"]),
        "content": content,
        "attachments": attachments,
        "metadata": metadata,
        "createdAt": int(data["created_at"]),
    }


def _replay_metadata(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    custom = value.get("custom")
    if not isinstance(custom, dict):
        return None
    relevant = {
        key: custom[key]
        for key in ("anthropicRefusal", "openaiCodexReasoning", "incomplete")
        if key in custom
    }
    return {"custom": relevant} if relevant else None


def _canonical_source(messages: Iterable[dict[str, Any]]) -> bytes:
    canonical = [
        {
            "id": message["id"],
            "parentId": message.get("parentId"),
            "role": message["role"],
            "content": _json_value(message.get("content", [])),
            "attachments": _json_value(message.get("attachments")),
            "replayMetadata": _json_value(_replay_metadata(message.get("metadata"))),
        }
        for message in messages
    ]
    try:
        encoded = json.dumps(
            canonical,
            ensure_ascii = False,
            allow_nan = False,
            sort_keys = True,
            separators = (",", ":"),
        ).encode("utf-8", errors = "strict")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ManualCompactionError("Compaction source is not canonical UTF-8 JSON") from exc
    if len(encoded) > MAX_MANUAL_COMPACTION_SOURCE_BYTES:
        raise ManualCompactionError("Compaction source is too large")
    return encoded


def canonical_source_hash(messages: Iterable[dict[str, Any]]) -> str:
    return hashlib.sha256(_canonical_source(messages)).hexdigest()


def _canonical_messages(messages: Iterable[Any]) -> tuple[bytes, list[Any]]:
    request_messages = list(messages)
    if not 2 <= len(request_messages) <= MAX_MANUAL_COMPACTION_REQUEST_MESSAGES:
        raise ManualCompactionError(
            "Manual compaction request must contain "
            f"2 to {MAX_MANUAL_COMPACTION_REQUEST_MESSAGES} messages"
        )
    canonical = []
    for message in request_messages:
        if hasattr(message, "model_dump"):
            item = message.model_dump(exclude_none = True)
        elif isinstance(message, dict):
            item = {
                str(key): _json_value(value) for key, value in message.items() if value is not None
            }
        else:
            raise ManualCompactionError("Manual compaction messages must be objects")
        if item.get("role") == "assistant" and "content" not in item and not item.get("tool_calls"):
            item["content"] = ""
        canonical.append(item)
    try:
        encoded = json.dumps(
            canonical,
            ensure_ascii = False,
            allow_nan = False,
            sort_keys = True,
            separators = (",", ":"),
        ).encode("utf-8", errors = "strict")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ManualCompactionError(
            "Manual compaction request is not canonical UTF-8 JSON"
        ) from exc
    if len(encoded) > MAX_MANUAL_COMPACTION_SOURCE_BYTES:
        raise ManualCompactionError("Manual compaction request is too large")
    return encoded, request_messages


def _canonical_request(messages: Iterable[Any]) -> tuple[bytes, list[Any]]:
    from models.inference import ChatCompletionRequest

    raw_messages = [
        message.model_dump(exclude_none = True)
        if hasattr(message, "model_dump")
        else _json_value(message)
        for message in messages
    ]
    try:
        request_messages = ChatCompletionRequest(
            model = "manual-compaction-normalization",
            messages = raw_messages,
        ).messages
    except Exception as exc:
        raise ManualCompactionError(
            "Manual compaction request cannot be normalized as a chat transcript"
        ) from exc
    encoded, request_messages = _canonical_messages(request_messages)
    seen_branch = False
    for message in request_messages:
        role = _role(message)
        if role in ("system", "developer"):
            if seen_branch:
                raise ManualCompactionConflict(
                    "Manual compaction request has a system message inside the stored branch"
                )
        else:
            seen_branch = True
    return encoded, request_messages


def canonical_request_hash(messages: Iterable[Any]) -> str:
    encoded, _request_branch = _canonical_request(messages)
    return hashlib.sha256(encoded).hexdigest()


def _role(message: Any) -> Any:
    return message.get("role") if isinstance(message, dict) else getattr(message, "role", None)


def _request_content(message: Any) -> Any:
    return (
        message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    )


def _parts(value: Any, *, label: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        return [{"type": "text", "text": value}]
    if not isinstance(value, list) or not all(isinstance(part, dict) for part in value):
        raise ManualCompactionConflict(f"{label} cannot be mapped to the inference transcript")
    return value


def _attachment_parts(message: dict[str, Any]) -> list[dict[str, Any]]:
    attachments = message.get("attachments")
    if attachments is None:
        return []
    if not isinstance(attachments, list):
        raise ManualCompactionConflict(
            "Stored message attachments cannot be mapped to the inference transcript"
        )
    parts: list[dict[str, Any]] = []
    for attachment in attachments:
        if not isinstance(attachment, dict):
            raise ManualCompactionConflict(
                "Stored message attachments cannot be mapped to the inference transcript"
            )
        parts.extend(_parts(attachment.get("content"), label = "Stored attachment"))
    return parts


def _image_parts(parts: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    for part in parts:
        if part.get("type") != "image":
            continue
        source = part.get("image")
        if not isinstance(source, str) or not source:
            raise ManualCompactionConflict(
                "Stored image cannot be mapped to the inference transcript"
            )
        images.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": source
                    if source.startswith("data:")
                    else f"data:image/png;base64,{source}",
                    "detail": "auto",
                },
            }
        )
    return images


def _text_parts(parts: Iterable[dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    for part in parts:
        if part.get("type") != "text":
            continue
        text = part.get("text")
        if not isinstance(text, str):
            raise ManualCompactionConflict(
                "Stored text cannot be mapped to the inference transcript"
            )
        texts.append(text)
    return texts


def _replay_content(text: str, images: list[dict[str, Any]]) -> Any:
    return [{"type": "text", "text": text}, *images] if images else text


def _valid_stored_audio_part(part: dict[str, Any]) -> bool:
    audio = part.get("audio")
    if isinstance(audio, str):
        return bool(audio)
    return bool(
        isinstance(audio, dict)
        and isinstance(audio.get("data"), str)
        and audio.get("data")
        and isinstance(audio.get("format"), str)
        and audio.get("format")
    )


def _valid_stored_video_part(part: dict[str, Any]) -> bool:
    mime_type = part.get("mimeType")
    return bool(
        isinstance(part.get("data"), str)
        and part.get("data")
        and isinstance(mime_type, str)
        and re.match(r"^video/", mime_type, re.IGNORECASE)
        and ("filename" not in part or isinstance(part.get("filename"), str))
    )


def _valid_stored_source_part(part: dict[str, Any]) -> bool:
    metadata = part.get("metadata")
    return bool(
        set(part) <= {"type", "sourceType", "id", "url", "title", "parentId", "metadata"}
        and part.get("sourceType") == "url"
        and isinstance(part.get("id"), str)
        and part.get("id")
        and isinstance(part.get("url"), str)
        and part.get("url")
        and ("title" not in part or isinstance(part.get("title"), str))
        and ("parentId" not in part or isinstance(part.get("parentId"), str))
        and (metadata is None or isinstance(metadata, dict))
        and (
            not isinstance(metadata, dict)
            or (
                set(metadata) <= {"description"}
                and ("description" not in metadata or isinstance(metadata.get("description"), str))
            )
        )
    )


def _validate_user_replay_parts(parts: Iterable[dict[str, Any]]) -> None:
    for part in parts:
        part_type = part.get("type")
        if part_type in ("text", "image"):
            continue
        if part_type == "audio" and _valid_stored_audio_part(part):
            continue
        if part_type == "file" and _valid_stored_video_part(part):
            continue
        raise ManualCompactionConflict(
            "Stored user content cannot be mapped to the inference transcript"
        )


def _tool_builtin(part: dict[str, Any]) -> tuple[bool, bool]:
    args = part.get("args")
    args = args if isinstance(args, dict) else {}
    if str(part.get("toolName") or "").lower() not in _SERVER_BUILTIN_TOOL_NAMES:
        return False, False
    google = args.get("google")
    native = isinstance(google, dict) and isinstance(google.get("native_part"), dict)
    return bool(args.get("_server_tool") is True or native), native


def _tool_arguments(part: dict[str, Any]) -> str:
    args_text = part.get("argsText")
    if isinstance(args_text, str) and args_text:
        try:
            _strict_json_loads(args_text)
        except (ValueError, UnicodeEncodeError):
            pass
        else:
            return args_text
    try:
        return json.dumps(
            part.get("args") if part.get("args") is not None else {},
            ensure_ascii = False,
            allow_nan = False,
            separators = (",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ManualCompactionConflict(
            "Stored tool arguments cannot be mapped to the inference transcript"
        ) from exc


def _unwrapped_tool_result(result: Any, tool_name: str) -> str | None:
    if not isinstance(result, dict) or not isinstance(result.get("text"), str):
        return None
    web_images = result.get("webImages")
    if (
        isinstance(web_images, list)
        and web_images
        and all(_search_image_entry(image) for image in web_images)
    ):
        return _strip_search_image_tokens(result["text"])
    images = result.get("images")
    if not isinstance(images, list):
        return None
    files = result.get("files")
    sandbox_files = files is None or (
        isinstance(files, list)
        and all(isinstance(item, dict) and isinstance(item.get("name"), str) for item in files)
    )
    if (
        tool_name in _SANDBOX_TOOL_NAMES
        and isinstance(result.get("sessionId"), str)
        and sandbox_files
    ):
        return result["text"]
    if (
        "sessionId" not in result
        and images
        and all(
            isinstance(image, dict)
            and isinstance(image.get("data"), str)
            and isinstance(image.get("mimeType"), str)
            for image in images
        )
    ):
        return result["text"]
    return None


def _tool_result(result: Any, tool_name: str) -> str:
    if isinstance(result, str):
        return result if result else '{"result":""}'
    unwrapped = _unwrapped_tool_result(result, tool_name)
    if unwrapped is not None:
        return unwrapped if unwrapped else '{"result":""}'
    try:
        return json.dumps(
            result,
            ensure_ascii = False,
            allow_nan = False,
            separators = (",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ManualCompactionConflict(
            "Stored tool result cannot be mapped to the inference transcript"
        ) from exc


def _local_round_id(part: dict[str, Any]) -> int | None:
    provenance = part.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("source") != "local":
        return None
    value = provenance.get("round_id")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _flush_local_pair(part: dict[str, Any]) -> bool:
    provenance = part.get("provenance")
    return bool(
        isinstance(provenance, dict)
        and provenance.get("source") == "local"
        and not _tool_builtin(part)[0]
        and part.get("result") is not None
    )


def _tool_call(
    part: dict[str, Any], assistant_ordinal: int, call_ordinal: int
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    builtin, native = _tool_builtin(part)
    if builtin and not native:
        return None, None
    if not builtin and part.get("result") is None:
        return None, None
    call_id = part.get("toolCallId")
    tool_name = part.get("toolName")
    if not isinstance(tool_name, str):
        raise ManualCompactionConflict(
            "Stored tool call cannot be mapped to the inference transcript"
        )
    arguments = _tool_arguments(part)
    if not isinstance(call_id, str) or not call_id:
        from models.inference import stable_tool_call_id
        call_id = stable_tool_call_id(tool_name, arguments, assistant_ordinal, call_ordinal)
    call: dict[str, Any] = {
        "id": call_id,
        "type": "function",
        "function": {"name": tool_name, "arguments": arguments},
    }
    args = part.get("args")
    google = args.get("google") if isinstance(args, dict) else None
    if "extra_content" in part:
        call["extra_content"] = part["extra_content"]
    elif isinstance(google, dict):
        call["extra_content"] = {"google": google}
    result = None
    if not builtin:
        result = {
            "role": "tool",
            "content": _tool_result(part.get("result"), tool_name),
            "tool_call_id": call_id,
            **({"name": tool_name} if tool_name else {}),
        }
    return call, result


def _search_image_entry(value: Any) -> bool:
    return bool(
        isinstance(value, dict)
        and isinstance(value.get("id"), str)
        and _SEARCH_IMAGE_ID_RE.fullmatch(value["id"])
        and isinstance(value.get("title"), str)
        and isinstance(value.get("domain"), str)
        and isinstance(value.get("source"), str)
        and re.match(r"^https?://", value["source"], re.IGNORECASE)
        and ("subject" not in value or isinstance(value.get("subject"), str))
    )


def _code_regions(text: str) -> list[tuple[int, int]]:
    fenced = [(match.start(), match.end()) for match in _FENCED_CODE_RE.finditer(text)]
    inline: list[tuple[int, int]] = []
    fenced_index = 0
    for match in _INLINE_CODE_RE.finditer(text):
        start, end = match.span()
        while fenced_index < len(fenced) and fenced[fenced_index][1] <= start:
            fenced_index += 1
        block = fenced[fenced_index] if fenced_index < len(fenced) else None
        if block is None or not (start >= block[0] and end <= block[1]):
            inline.append((start, end))
    merged: list[tuple[int, int]] = []
    for start, end in sorted([*fenced, *inline]):
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _strip_search_image_tokens(text: str) -> str:
    if "[[img:" not in text:
        return text
    regions = _code_regions(text)

    def in_code_region(position: int) -> bool:
        low = 0
        high = len(regions) - 1
        while low <= high:
            middle = (low + high) // 2
            start, end = regions[middle]
            if position < start:
                high = middle - 1
            elif position >= end:
                low = middle + 1
            else:
                return True
        return False

    def replace(match: re.Match[str]) -> str:
        return match.group(0) if in_code_region(match.start()) else ""

    return _SEARCH_IMAGE_TOKEN_RE.sub(replace, text)


def _sanitize_assistant_text(text: str) -> str:
    return _AUDIO_DATA_RE.sub("[audio]", _strip_search_image_tokens(text))


def _codex_reasoning_ledger(custom: Any) -> tuple[dict[str, list[Any]], list[Any] | None]:
    value = custom.get("openaiCodexReasoning") if isinstance(custom, dict) else None
    if isinstance(value, list):
        return {}, list(value) if value else None
    if value is None:
        return {}, None
    if not isinstance(value, dict):
        raise ManualCompactionConflict(
            "Stored Codex reasoning cannot be mapped to the inference transcript"
        )
    raw_by_call = value.get("byToolCall")
    if raw_by_call is not None and not isinstance(raw_by_call, dict):
        raise ManualCompactionConflict(
            "Stored Codex reasoning cannot be mapped to the inference transcript"
        )
    by_call = {
        str(call_id): list(items)
        for call_id, items in (raw_by_call or {}).items()
        if isinstance(items, list) and items
    }
    final = value.get("final")
    return by_call, list(final) if isinstance(final, list) and final else None


def _add_assistant_extra(message: dict[str, Any], key: str, value: Any) -> None:
    extra = message.get("extra_content")
    extra = dict(extra) if isinstance(extra, dict) else {}
    extra[key] = value
    message["extra_content"] = extra


def _assistant_wire_messages(
    message: dict[str, Any], *, include_reasoning: bool, assistant_ordinal_start: int
) -> list[dict[str, Any]]:
    metadata = message.get("metadata")
    custom = metadata.get("custom") if isinstance(metadata, dict) else None
    if isinstance(custom, dict) and custom.get("anthropicRefusal") is True:
        return []
    reasoning_by_call, final_reasoning = _codex_reasoning_ledger(custom)
    parts = _parts(message.get("content"), label = "Stored assistant message")
    attachments = _attachment_parts(message)
    if any(part.get("type") == "text" for part in attachments):
        raise ManualCompactionConflict(
            "Stored assistant text attachment cannot be mapped to the inference transcript"
        )
    for part in [*parts, *attachments]:
        part_type = part.get("type")
        if part_type in ("text", "reasoning", "image", "tool-call"):
            continue
        if part_type == "source" and _valid_stored_source_part(part):
            continue
        raise ManualCompactionConflict(
            "Stored assistant content cannot be mapped to the inference transcript"
        )
    thought_signature = next(
        (
            part.get("_google_thought_signature")
            for part in reversed(parts)
            if part.get("type") == "text"
            and isinstance(part.get("_google_thought_signature"), str)
            and part.get("_google_thought_signature")
        ),
        None,
    )
    images = _image_parts([*parts, *attachments])
    images_pending = True
    wire: list[dict[str, Any]] = []
    pending_text: list[str] = []
    pending_reasoning: list[str] = []
    pending_calls: list[dict[str, Any]] = []
    pending_results: list[dict[str, Any]] = []
    pending_round: int | None = None
    assistant_flush_count = 0

    def flush(force: bool = False) -> None:
        nonlocal images_pending, pending_round, assistant_flush_count
        text = _sanitize_assistant_text("\n".join(pending_text))
        active_images = images if images_pending else []
        has_content = bool(text or active_images)
        has_calls = bool(pending_calls)
        reasoning = "\n".join(pending_reasoning)
        incomplete = _assistant_turn_ended_early(message)
        has_reasoning = bool(
            include_reasoning and reasoning and (has_content or has_calls or not incomplete)
        )
        if not force and not has_content and not has_calls and not has_reasoning:
            return
        assistant: dict[str, Any] = {
            "role": "assistant",
            "content": (
                _replay_content(text, active_images) if has_content else (None if has_calls else "")
            ),
        }
        if has_calls:
            assistant["tool_calls"] = list(pending_calls)
            for call in pending_calls:
                items = reasoning_by_call.get(str(call.get("id") or ""))
                if items:
                    _add_assistant_extra(assistant, "openai_codex_reasoning", items)
                    break
        if has_reasoning:
            assistant["reasoning_content"] = reasoning
        wire.append(assistant)
        assistant_flush_count += 1
        wire.extend(pending_results)
        pending_text.clear()
        pending_reasoning.clear()
        pending_calls.clear()
        pending_results.clear()
        pending_round = None
        images_pending = False

    for part in parts:
        part_type = part.get("type")
        if part_type == "reasoning":
            if pending_calls:
                flush()
            text = part.get("text")
            if not isinstance(text, str):
                raise ManualCompactionConflict(
                    "Stored reasoning cannot be mapped to the inference transcript"
                )
            pending_reasoning.append(text)
            continue
        if part_type == "text":
            if pending_calls:
                flush()
            text = part.get("text")
            if not isinstance(text, str):
                raise ManualCompactionConflict(
                    "Stored assistant text cannot be mapped to the inference transcript"
                )
            pending_text.append(text)
            continue
        if part_type == "image":
            continue
        if part_type == "source":
            continue
        call, result = _tool_call(
            part,
            assistant_ordinal_start + assistant_flush_count,
            len(pending_calls),
        )
        if call is None:
            continue
        round_id = _local_round_id(part)
        if pending_calls and pending_round is not None and round_id is not None:
            if pending_round != round_id:
                flush()
        if round_id is not None:
            pending_round = round_id
        flush_pair = round_id is None and _flush_local_pair(part)
        if flush_pair and pending_calls:
            flush()
        pending_calls.append(call)
        if result is not None:
            pending_results.append(result)
        if flush_pair:
            flush()
    flush(force = not wire)
    final_assistant = next(
        (message for message in reversed(wire) if message.get("role") == "assistant"),
        None,
    )
    if final_assistant is not None and thought_signature is not None:
        extra = final_assistant.get("extra_content")
        extra = dict(extra) if isinstance(extra, dict) else {}
        google = extra.get("google")
        google = dict(google) if isinstance(google, dict) else {}
        google["thought_signature"] = thought_signature
        extra["google"] = google
        final_assistant["extra_content"] = extra
    if final_assistant is not None and final_reasoning is not None:
        _add_assistant_extra(final_assistant, "openai_codex_reasoning", final_reasoning)
    return wire


def _is_anthropic_refusal(message: dict[str, Any]) -> bool:
    if message.get("role") != "assistant":
        return False
    metadata = message.get("metadata")
    custom = metadata.get("custom") if isinstance(metadata, dict) else None
    return isinstance(custom, dict) and custom.get("anthropicRefusal") is True


def _assistant_turn_carries_payload(message: dict[str, Any]) -> bool:
    parts = _parts(message.get("content"), label = "Stored assistant message")
    for part in parts:
        if part.get("type") != "text":
            continue
        text = part.get("text")
        if not isinstance(text, str):
            raise ManualCompactionConflict(
                "Stored assistant text cannot be mapped to the inference transcript"
            )
        if text.strip():
            return True
    attachments = message.get("attachments")
    attachment_parts = _attachment_parts(message)
    if _image_parts([*parts, *attachment_parts]):
        return True
    return isinstance(attachments, list) and bool(attachments)


def _assistant_turn_ended_early(message: dict[str, Any]) -> bool:
    metadata = message.get("metadata")
    custom = metadata.get("custom") if isinstance(metadata, dict) else None
    incomplete = custom.get("incomplete") if isinstance(custom, dict) else None
    return bool(
        isinstance(incomplete, dict)
        and incomplete.get("reason") in ("length", "cancelled", "interrupted")
    )


def _has_replay_content(content: Any) -> bool:
    return bool(content.strip()) if isinstance(content, str) else bool(content)


def _is_abandoned_assistant_turn(message: dict[str, Any], *, include_reasoning: bool) -> bool:
    if message.get("role") != "assistant":
        return False
    if _assistant_turn_carries_payload(message):
        return False
    parts = _parts(message.get("content"), label = "Stored assistant message")
    if not _assistant_turn_ended_early(message) and any(
        part.get("type") == "reasoning" for part in parts
    ):
        return False
    wire = _assistant_wire_messages(
        message,
        include_reasoning = include_reasoning,
        assistant_ordinal_start = 0,
    )
    if len(wire) != 1:
        return False
    only = wire[0]
    return bool(
        only.get("role") == "assistant"
        and not _has_replay_content(only.get("content"))
        and not only.get("tool_calls")
        and not only.get("reasoning_content")
    )


def _prune_stored_branch(
    branch: list[dict[str, Any]], *, include_reasoning: bool
) -> list[dict[str, Any]]:
    abandoned = [
        _is_abandoned_assistant_turn(message, include_reasoning = include_reasoning)
        for message in branch
    ]
    last_surviving = -1
    for index in range(len(branch) - 1, -1, -1):
        if not abandoned[index] and not _is_anthropic_refusal(branch[index]):
            last_surviving = index
            break

    surviving: list[dict[str, Any]] = []
    for index, message in enumerate(branch):
        refused = _is_anthropic_refusal(message)
        if refused or abandoned[index]:
            if (refused or index < last_surviving) and surviving:
                if surviving[-1].get("role") == "user":
                    surviving.pop()
            continue
        surviving.append(message)
    return surviving


def _stored_branch_wire_variants(branch: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    variants: list[list[dict[str, Any]]] = []
    for include_reasoning in (False, True):
        wire: list[dict[str, Any]] = []
        for message in _prune_stored_branch(branch, include_reasoning = include_reasoning):
            role = message.get("role")
            parts = _parts(message.get("content"), label = "Stored message")
            attachments = _attachment_parts(message)
            if role == "assistant":
                wire.extend(
                    _assistant_wire_messages(
                        message,
                        include_reasoning = include_reasoning,
                        assistant_ordinal_start = sum(
                            1 for candidate in wire if candidate.get("role") == "assistant"
                        ),
                    )
                )
                continue
            if role != "user":
                raise ManualCompactionConflict(
                    "Stored branch role cannot be mapped to the inference transcript"
                )
            _validate_user_replay_parts([*parts, *attachments])
            wire.append(
                {
                    "role": "user",
                    "content": _replay_content(
                        "\n".join(_text_parts([*parts, *attachments])),
                        _image_parts([*parts, *attachments]),
                    ),
                }
            )
        if wire not in variants:
            variants.append(wire)
    return variants


def _archive_payload(branch: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str, str]:
    if not branch:
        raise ManualCompactionConflict("Manual compaction archive branch is empty")
    command_id = branch[-1].get("id")
    pruned = _prune_stored_branch(branch, include_reasoning = True)
    payload = [
        {"role": message["role"], "content": _json_value(message.get("content", []))}
        for message in pruned
        if message.get("id") != command_id
    ]
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii = False,
            allow_nan = False,
            sort_keys = True,
            separators = (",", ":"),
        ).encode("utf-8", errors = "strict")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ManualCompactionConflict(
            "Stored archive payload cannot be represented safely"
        ) from exc
    return payload, encoded.decode("utf-8"), hashlib.sha256(encoded).hexdigest()


def _validate_request_covers_branch(
    request_messages: list[Any], branch: list[dict[str, Any]]
) -> None:
    prefix_length = 0
    for message in request_messages:
        if _role(message) not in ("system", "developer"):
            break
        prefix_length += 1
    actual_branch = request_messages[prefix_length:]
    actual_encoded, _ = _canonical_messages(actual_branch)
    expected_encodings = {
        _canonical_messages(candidate)[0] for candidate in _stored_branch_wire_variants(branch)
    }
    if actual_encoded not in expected_encodings:
        raise ManualCompactionConflict(
            "Manual compaction request does not match the exact stored branch"
        )
    if _content_text(_request_content(actual_branch[-1])) != "/compact":
        raise ManualCompactionConflict(
            "Manual compaction request must end with the literal /compact command"
        )


def _digest_json(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii = False,
            allow_nan = False,
            sort_keys = True,
            separators = (",", ":"),
        ).encode("utf-8", errors = "strict")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ManualCompactionError("Project context is not canonical UTF-8 JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _project_context_state(conn, thread_id: str) -> dict[str, Any]:
    row = conn.execute(
        "SELECT t.project_id, p.instructions, p.archived, p.updated_at "
        "FROM chat_threads AS t "
        "LEFT JOIN chat_projects AS p ON p.id = t.project_id WHERE t.id = ?",
        (thread_id,),
    ).fetchone()
    if row is None:
        raise ManualCompactionNotFound(f"Thread {thread_id} not found")
    instructions = str(row["instructions"] or "") if row["project_id"] is not None else ""
    project_state = {
        "projectId": str(row["project_id"]) if row["project_id"] is not None else None,
        "archived": bool(row["archived"]) if row["project_id"] is not None else False,
        "instructions": instructions,
    }
    revision = int(row["updated_at"] or 0) if row["project_id"] is not None else 0
    instruction_digest = _digest_json(project_state)
    return {
        "projectInstructionDigest": instruction_digest,
        "projectInstructionRevision": revision,
        "contextDigest": _digest_json(
            {
                "schemaVersion": 1,
                "projectInstructions": {
                    "digest": instruction_digest,
                    "revision": revision,
                },
            }
        ),
        "instructions": instructions,
        "active": row["project_id"] is not None and not bool(row["archived"]),
    }


def _validate_project_instructions(
    conn, thread_id: str, request_messages: list[Any]
) -> dict[str, Any]:
    state = _project_context_state(conn, thread_id)
    if not state["active"] or not state["instructions"].strip():
        return state
    expected = (
        "<project_instructions>\n" + state["instructions"].strip() + "\n</project_instructions>"
    )
    leading = [message for message in request_messages if _role(message) in ("system", "developer")]
    if not leading or not _content_text(_request_content(leading[0])).startswith(expected):
        raise ManualCompactionConflict(
            "Manual compaction request does not contain the stored project instructions"
        )
    return state


def _recheck_project_context(conn, thread_id: str, attempt: dict[str, Any]) -> None:
    current = _project_context_state(conn, thread_id)
    for key in (
        "projectInstructionDigest",
        "projectInstructionRevision",
        "contextDigest",
    ):
        if current[key] != attempt.get(key):
            raise ManualCompactionConflict("Project instructions changed after prepare")


def _content_text(content: Any) -> str:
    content = _json_value(content)
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") in ("text", "input_text", "output_text"):
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _literal_compact(message: dict[str, Any]) -> bool:
    return message.get("role") == "user" and _content_text(message.get("content")) == "/compact"


def _load_rows(conn, thread_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    thread = conn.execute("SELECT * FROM chat_threads WHERE id = ?", (thread_id,)).fetchone()
    if thread is None:
        raise ManualCompactionNotFound(f"Thread {thread_id} not found")
    rows = conn.execute(
        "SELECT * FROM chat_messages WHERE thread_id = ?",
        (thread_id,),
    ).fetchall()
    return dict(thread), {str(row["id"]): row for row in rows}


def _exact_branch(
    conn, thread_id: str, message_ids: list[str], expected_head_message_id: str
) -> list[dict[str, Any]]:
    if not 2 <= len(message_ids) <= MAX_MANUAL_COMPACTION_MESSAGES:
        raise ManualCompactionError(
            f"Compaction branch must contain 2 to {MAX_MANUAL_COMPACTION_MESSAGES} messages"
        )
    checked = [_bounded_id(value, "message id") for value in message_ids]
    if len(set(checked)) != len(checked):
        raise ManualCompactionConflict("Compaction branch contains a cycle or duplicate message")
    expected_head_message_id = _bounded_id(expected_head_message_id, "expected head message id")
    if checked[-1] != expected_head_message_id:
        raise ManualCompactionConflict("Expected head does not match the supplied branch")

    _thread, by_id = _load_rows(conn, thread_id)
    branch: list[dict[str, Any]] = []
    previous: str | None = None
    for index, message_id in enumerate(checked):
        row = by_id.get(message_id)
        if row is None:
            foreign = conn.execute(
                "SELECT thread_id FROM chat_messages WHERE id = ?",
                (message_id,),
            ).fetchone()
            if foreign is not None:
                raise ManualCompactionConflict("Compaction branch crosses thread boundaries")
            raise ManualCompactionConflict(f"Compaction branch message {message_id} is missing")
        message = _message_dict(row)
        parent_id = message.get("parentId")
        if (index == 0 and parent_id not in (None, "")) or (index > 0 and parent_id != previous):
            raise ManualCompactionConflict("Compaction branch is not the exact stored ancestry")
        branch.append(message)
        previous = message_id
    return branch


def _complete_command_branch(conn, thread_id: str, command_message_id: str) -> list[dict[str, Any]]:
    """Load and validate the complete same-thread root-to-command ancestry."""
    _thread, by_id = _load_rows(conn, thread_id)
    cursor: str | None = command_message_id
    reversed_branch: list[dict[str, Any]] = []
    seen: set[str] = set()
    while cursor is not None:
        if cursor in seen:
            raise ManualCompactionConflict("The /compact command ancestry contains a cycle")
        if len(reversed_branch) >= MAX_MANUAL_COMPACTION_MESSAGES:
            raise ManualCompactionError(
                f"Compaction branch must contain 2 to {MAX_MANUAL_COMPACTION_MESSAGES} messages"
            )
        seen.add(cursor)
        row = by_id.get(cursor)
        if row is None:
            if not reversed_branch:
                raise ManualCompactionNotFound("The /compact command was not found")
            foreign = conn.execute(
                "SELECT thread_id FROM chat_messages WHERE id = ?", (cursor,)
            ).fetchone()
            if foreign is not None:
                raise ManualCompactionConflict("The /compact command ancestry crosses threads")
            raise ManualCompactionConflict("The /compact command ancestry is incomplete")
        message = _message_dict(row)
        reversed_branch.append(message)
        parent_id = message.get("parentId")
        if parent_id in (None, ""):
            cursor = None
        elif not isinstance(parent_id, str):
            raise ManualCompactionConflict("The /compact command ancestry is invalid")
        else:
            cursor = parent_id
    branch = list(reversed(reversed_branch))
    if len(branch) < 2:
        raise ManualCompactionError("Compaction branch must contain at least 2 messages")
    return branch


def _require_branch_lifecycle_quiescent(conn, thread_id: str, branch: list[dict[str, Any]]) -> None:
    """Let an existing producer finish before its stored rows become immutable."""
    message_ids = {str(message["id"]) for message in branch}
    generation = conn.execute(
        "SELECT user_message_id, assistant_message_id FROM chat_generation_runs "
        "WHERE thread_id = ? AND status IN ('queued', 'running', 'cancelling')",
        (thread_id,),
    ).fetchall()
    if any(
        str(row["user_message_id"]) in message_ids
        or str(row["assistant_message_id"]) in message_ids
        for row in generation
    ):
        raise ManualCompactionConflict(
            "The /compact ancestry still has an active generation lifecycle"
        )
    research = conn.execute(
        "SELECT user_message_id, assistant_message_id FROM research_runs "
        "WHERE thread_id = ? AND status IN "
        "('planning', 'awaiting_approval', 'queued', 'running', 'paused', 'cancelling')",
        (thread_id,),
    ).fetchall()
    if any(
        str(row["user_message_id"]) in message_ids
        or (
            row["assistant_message_id"] is not None
            and str(row["assistant_message_id"]) in message_ids
        )
        for row in research
    ):
        raise ManualCompactionConflict(
            "The /compact ancestry still has an active research lifecycle"
        )


def _active_compaction_metadata(message: dict[str, Any]) -> dict[str, Any] | None:
    metadata = message.get("metadata")
    raw = metadata.get("manualCompaction") if isinstance(metadata, dict) else None
    if raw is None:
        return None
    if not isinstance(raw, dict) or raw.get("schemaVersion") != MANUAL_COMPACTION_SCHEMA_VERSION:
        raise ManualCompactionConflict("Stored manual compaction metadata is invalid")
    if raw.get("state") != "active" or raw.get("summaryMessageId") != message["id"]:
        raise ManualCompactionConflict("Stored manual compaction metadata is inconsistent")
    revision = raw.get("revision")
    source_hash = raw.get("sourceHash")
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        raise ManualCompactionConflict("Stored manual compaction revision is invalid")
    if not isinstance(source_hash, str) or _SHA256_RE.fullmatch(source_hash) is None:
        raise ManualCompactionConflict("Stored manual compaction hash is invalid")
    return raw


def _active_summary_start(conn, thread_id: str, source: list[dict[str, Any]]) -> tuple[int, int]:
    previous = 0
    latest_index = 0
    for index, message in enumerate(source):
        active = _active_compaction_metadata(message)
        if active is None:
            continue
        revision = int(active["revision"])
        if revision <= previous:
            raise ManualCompactionConflict("Stored manual compaction revisions are not monotonic")
        if active.get("threadId") != thread_id:
            raise ManualCompactionConflict("Stored manual compaction belongs to another thread")
        if index < 1 or source[index - 1]["id"] != active.get("commandMessageId"):
            raise ManualCompactionConflict("Stored manual compaction command ancestry is invalid")
        if message.get("parentId") != active.get("commandMessageId"):
            raise ManualCompactionConflict("Stored manual compaction summary ancestry is invalid")
        source_head = active.get("sourceHeadMessageId")
        audit_source = source[: index - 1]
        if not audit_source or audit_source[-1]["id"] != source_head:
            raise ManualCompactionConflict("Stored manual compaction source ancestry is invalid")
        if canonical_source_hash(audit_source) != active.get("sourceHash"):
            raise ManualCompactionConflict("Stored manual compaction source hash is invalid")
        _text, _content_json, stored_summary_hash = _strict_summary(message, active = True)
        if stored_summary_hash != active.get("summaryHash"):
            raise ManualCompactionConflict("Stored manual compaction summary hash is invalid")
        row = conn.execute(
            "SELECT * FROM manual_compactions WHERE attempt_id = ?",
            (active.get("attemptId"),),
        ).fetchone()
        if row is None:
            raise ManualCompactionConflict("Stored manual compaction row is missing")
        stored = _attempt_from_row(row)
        if (
            stored["state"] != "active"
            or stored["threadId"] != thread_id
            or stored["summaryMessageId"] != message["id"]
            or stored["revision"] != revision
            or stored["sourceHash"] != active.get("sourceHash")
        ):
            raise ManualCompactionConflict("Stored manual compaction row is inconsistent")
        previous = revision
        latest_index = index
    return latest_index, previous


def _next_branch_revision(conn, thread_id: str, source: list[dict[str, Any]]) -> int:
    _latest_index, previous = _active_summary_start(conn, thread_id, source)
    return previous + 1


def _effective_branch(
    conn, thread_id: str, source: list[dict[str, Any]], command: dict[str, Any]
) -> list[dict[str, Any]]:
    latest_index, revision = _active_summary_start(conn, thread_id, source)
    start = latest_index if revision else 0
    return [*source[start:], command]


def _attempt_from_row(row: Any) -> dict[str, Any]:
    data = dict(row)
    state = data.get("state")
    if state not in ("pending", "running", "active", "cancelled", "failed"):
        raise ManualCompactionConflict("Stored manual compaction state is invalid")
    command_hash = data.get("command_hash")
    command_hash_state = data.get("command_hash_state")
    valid_command_hash = (
        isinstance(command_hash, str) and _SHA256_RE.fullmatch(command_hash) is not None
    )
    if state in ("pending", "running") and not (
        valid_command_hash and command_hash_state == "prepared"
    ):
        raise ManualCompactionConflict("Stored manual compaction command hash is inconsistent")
    if state == "active" and not (valid_command_hash and command_hash_state == "summary_saved"):
        raise ManualCompactionConflict("Stored manual compaction command hash is inconsistent")
    if state in ("cancelled", "failed") and not (
        (command_hash is None and command_hash_state is None)
        or valid_command_hash
        and command_hash_state in ("prepared", "summary_saved")
    ):
        raise ManualCompactionConflict("Stored manual compaction command hash is inconsistent")
    claim_token_hash = data.get("claim_token_hash")
    if claim_token_hash is not None and (
        not isinstance(claim_token_hash, str) or _SHA256_RE.fullmatch(claim_token_hash) is None
    ):
        raise ManualCompactionConflict("Stored manual compaction claim ownership is invalid")
    if (state == "running") != (claim_token_hash is not None) and state not in (
        "cancelled",
        "failed",
    ):
        raise ManualCompactionConflict("Stored manual compaction claim ownership is inconsistent")
    source_message_ids = _decode_stored_json(
        data.get("source_message_ids_json"),
        label = "manual compaction source message ids",
    )
    effective_source_message_ids = _decode_stored_json(
        data.get("effective_source_message_ids_json"),
        label = "manual compaction effective source message ids",
    )
    for label, values in (
        ("source message ids", source_message_ids),
        ("effective source message ids", effective_source_message_ids),
    ):
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value for value in values
        ):
            raise ManualCompactionConflict(
                f"Stored manual compaction {label} JSON has an invalid shape"
            )
        if not values and state not in ("cancelled", "failed"):
            raise ManualCompactionConflict(
                f"Stored manual compaction {label} JSON has an invalid shape"
            )
    archive_payload = _decode_stored_json(
        data.get("archive_payload_json"),
        label = "manual compaction archive payload",
    )
    archive_payload_hash = data.get("archive_payload_hash")
    if not isinstance(archive_payload, list) or not all(
        isinstance(message, dict)
        and message.get("role") in ("user", "assistant")
        and "content" in message
        for message in archive_payload
    ):
        raise ManualCompactionConflict(
            "Stored manual compaction archive payload JSON has an invalid shape"
        )
    try:
        archive_encoded = json.dumps(
            archive_payload,
            ensure_ascii = False,
            allow_nan = False,
            sort_keys = True,
            separators = (",", ":"),
        ).encode("utf-8", errors = "strict")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ManualCompactionConflict(
            "Stored manual compaction archive payload JSON is invalid"
        ) from exc
    if (
        not isinstance(archive_payload_hash, str)
        or _SHA256_RE.fullmatch(archive_payload_hash) is None
        or hashlib.sha256(archive_encoded).hexdigest() != archive_payload_hash
    ):
        raise ManualCompactionConflict("Stored manual compaction archive payload hash is invalid")
    output_summary_hash = data.get("output_summary_hash")
    output_finish_reason = data.get("output_finish_reason")
    output_recorded_at = data.get("output_recorded_at")
    if output_summary_hash is not None and (
        not isinstance(output_summary_hash, str)
        or _SHA256_RE.fullmatch(output_summary_hash) is None
    ):
        raise ManualCompactionConflict("Stored manual compaction output hash is invalid")
    if output_finish_reason is not None and output_finish_reason != "stop":
        raise ManualCompactionConflict("Stored manual compaction finish reason is invalid")
    if (output_finish_reason is None) != (output_summary_hash is None):
        raise ManualCompactionConflict("Stored manual compaction output provenance is invalid")
    if (output_finish_reason is None) != (output_recorded_at is None) or (
        output_recorded_at is not None
        and (
            not isinstance(output_recorded_at, int)
            or isinstance(output_recorded_at, bool)
            or output_recorded_at < 1
        )
    ):
        raise ManualCompactionConflict("Stored manual compaction output provenance is invalid")
    terminal_reason = data.get("terminal_reason")
    finished_at = data.get("finished_at")
    if terminal_reason is not None and terminal_reason not in _TERMINAL_REASONS:
        raise ManualCompactionConflict("Stored manual compaction terminal reason is invalid")
    if finished_at is not None and (
        not isinstance(finished_at, int) or isinstance(finished_at, bool) or finished_at < 1
    ):
        raise ManualCompactionConflict("Stored manual compaction finished time is invalid")
    if state in ("cancelled", "failed") and (terminal_reason is None or finished_at is None):
        raise ManualCompactionConflict("Stored manual compaction terminal state is incomplete")
    if state not in ("cancelled", "failed") and (
        terminal_reason is not None or finished_at is not None
    ):
        raise ManualCompactionConflict("Stored manual compaction terminal state is invalid")

    def positive_int(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 1

    def optional_positive_int(value: Any) -> bool:
        return value is None or positive_int(value)

    for label, value in (
        ("attempt id", data.get("attempt_id")),
        ("thread id", data.get("thread_id")),
        ("command message id", data.get("command_message_id")),
        ("source head message id", data.get("source_head_message_id")),
        ("expected head message id", data.get("expected_head_message_id")),
    ):
        if not isinstance(value, str) or not value:
            raise ManualCompactionConflict(f"Stored manual compaction {label} is invalid")
    for label, value in (
        ("source hash", data.get("source_hash")),
        ("request hash", data.get("request_hash")),
        ("project instruction digest", data.get("project_instruction_digest")),
        ("context digest", data.get("context_digest")),
    ):
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise ManualCompactionConflict(f"Stored manual compaction {label} is invalid")
    request_message_count = data.get("request_message_count")
    project_instruction_revision = data.get("project_instruction_revision")
    revision = data.get("revision")
    created_at = data.get("created_at")
    if not positive_int(request_message_count):
        raise ManualCompactionConflict("Stored manual compaction request message count is invalid")
    if (
        not isinstance(project_instruction_revision, int)
        or isinstance(project_instruction_revision, bool)
        or project_instruction_revision < 0
    ):
        raise ManualCompactionConflict(
            "Stored manual compaction project instruction revision is invalid"
        )
    if not positive_int(revision) or not positive_int(created_at):
        raise ManualCompactionConflict("Stored manual compaction revision metadata is invalid")

    summary_message_id = data.get("summary_message_id")
    summary_hash = data.get("summary_hash")
    if summary_message_id is not None and (
        not isinstance(summary_message_id, str) or not summary_message_id
    ):
        raise ManualCompactionConflict("Stored manual compaction summary message id is invalid")
    if summary_hash is not None and (
        not isinstance(summary_hash, str) or _SHA256_RE.fullmatch(summary_hash) is None
    ):
        raise ManualCompactionConflict("Stored manual compaction summary hash is invalid")
    if (summary_message_id is None) != (summary_hash is None):
        raise ManualCompactionConflict("Stored manual compaction summary identity is invalid")

    archive_status = data.get("archive_status")
    if archive_status not in ("pending", "failed", "skipped", "archived"):
        raise ManualCompactionConflict("Stored manual compaction archive status is invalid")
    started_at = data.get("started_at")
    lease_expires_at = data.get("lease_expires_at")
    cancelled_at = data.get("cancelled_at")
    committed_at = data.get("committed_at")
    if not all(
        optional_positive_int(value)
        for value in (started_at, lease_expires_at, cancelled_at, committed_at)
    ):
        raise ManualCompactionConflict("Stored manual compaction lifecycle time is invalid")

    has_output = output_summary_hash is not None
    if state == "pending" and not (
        started_at is None
        and lease_expires_at is None
        and cancelled_at is None
        and committed_at is None
        and summary_message_id is None
        and not has_output
        and archive_status == "pending"
    ):
        raise ManualCompactionConflict("Stored pending manual compaction is inconsistent")
    if state == "running" and not (
        positive_int(started_at)
        and positive_int(lease_expires_at)
        and cancelled_at is None
        and committed_at is None
        and summary_message_id is None
        and archive_status == "pending"
    ):
        raise ManualCompactionConflict("Stored running manual compaction is inconsistent")
    if state == "active" and not (
        positive_int(started_at)
        and lease_expires_at is None
        and cancelled_at is None
        and positive_int(committed_at)
        and summary_message_id is not None
        and has_output
        and output_finish_reason == "stop"
        and output_summary_hash == summary_hash
    ):
        raise ManualCompactionConflict("Stored active manual compaction is inconsistent")
    if state in ("cancelled", "failed") and not (
        lease_expires_at is None
        and committed_at is None
        and summary_message_id is None
        and archive_status == "pending"
        and not source_message_ids
        and not effective_source_message_ids
        and not archive_payload
        and archive_payload_hash == _EMPTY_JSON_HASH
        and (positive_int(cancelled_at) if state == "cancelled" else cancelled_at is None)
    ):
        raise ManualCompactionConflict("Stored terminal manual compaction is inconsistent")
    return {
        "attemptId": data["attempt_id"],
        "threadId": data["thread_id"],
        "commandMessageId": data["command_message_id"],
        "sourceHeadMessageId": data["source_head_message_id"],
        "expectedHeadMessageId": data["expected_head_message_id"],
        "sourceMessageIds": source_message_ids,
        "effectiveSourceMessageIds": effective_source_message_ids,
        "sourceHash": data["source_hash"],
        "requestHash": data.get("request_hash"),
        "requestMessageCount": data.get("request_message_count"),
        "projectInstructionDigest": data.get("project_instruction_digest"),
        "projectInstructionRevision": data.get("project_instruction_revision"),
        "contextDigest": data.get("context_digest"),
        "archivePayload": archive_payload,
        "archivePayloadHash": archive_payload_hash,
        "revision": revision,
        "state": state,
        "commandHash": command_hash,
        "commandHashState": command_hash_state,
        "summaryMessageId": summary_message_id,
        "summaryHash": summary_hash,
        "outputSummaryHash": output_summary_hash,
        "outputFinishReason": output_finish_reason,
        "outputRecordedAt": output_recorded_at,
        "archiveStatus": archive_status,
        "createdAt": created_at,
        "startedAt": started_at,
        "leaseExpiresAt": lease_expires_at,
        "cancelledAt": cancelled_at,
        "committedAt": committed_at,
        "terminalReason": terminal_reason,
        "finishedAt": finished_at,
    }


def _lease_expired(attempt: dict[str, Any], now: int) -> bool:
    expires_at = attempt.get("leaseExpiresAt")
    return (
        attempt.get("state") == "running"
        and isinstance(expires_at, int)
        and not isinstance(expires_at, bool)
        and expires_at <= now
    )


def _scrub_terminal_attempt(conn, attempt_id: str) -> None:
    conn.execute(
        "UPDATE manual_compactions SET source_message_ids_json = '[]', "
        "effective_source_message_ids_json = '[]', archive_payload_json = '[]', "
        "archive_payload_hash = ? WHERE attempt_id = ? "
        "AND state IN ('cancelled', 'failed')",
        (_EMPTY_JSON_HASH, attempt_id),
    )


def _terminalize_attempt(
    conn,
    attempt_id: str,
    now: int,
    *,
    state: str,
    reason: str,
    allowed_states: tuple[str, ...] = ("pending", "running"),
) -> bool:
    if state not in ("cancelled", "failed"):
        raise ValueError("Manual compaction terminal state is invalid")
    if reason not in _TERMINAL_REASONS:
        raise ValueError("Manual compaction terminal reason is invalid")
    placeholders = ", ".join("?" for _ in allowed_states)
    changed = conn.execute(
        "UPDATE manual_compactions SET state = ?, lease_expires_at = NULL, "
        "cancelled_at = CASE WHEN ? = 'cancelled' THEN ? ELSE cancelled_at END, "
        "terminal_reason = ?, finished_at = ? "
        f"WHERE attempt_id = ? AND state IN ({placeholders})",
        (state, state, now, reason, now, attempt_id, *allowed_states),
    ).rowcount
    if changed:
        _scrub_terminal_attempt(conn, attempt_id)
    return changed == 1


def _referenced_manual_compaction_attempt_ids(conn, thread_id: str | None) -> set[str]:
    """Decode every scoped message before retention decides that a fence is unreferenced."""
    rows = conn.execute(
        "SELECT id, thread_id, parent_id, role, content_json, attachments_json, "
        "metadata_json, created_at FROM chat_messages "
        + ("" if thread_id is None else "WHERE thread_id = ?"),
        () if thread_id is None else (thread_id,),
    ).fetchall()
    referenced: set[str] = set()
    for row in rows:
        message = _message_dict(row)
        metadata = message.get("metadata")
        if not isinstance(metadata, dict) or MANUAL_COMPACTION_CLIENT_KEY not in metadata:
            continue
        record = _manual_compaction_client_record(message)
        if record is None:
            raise ManualCompactionConflict("Stored manual compaction client metadata is invalid")
        referenced.add(str(record["attemptId"]))
    return referenced


def _cleanup_manual_compaction_attempts(conn, thread_id: str | None, now: int) -> None:
    # Discover references first. Any malformed message or fence aborts the writer
    # transaction before terminalization, scrubbing, or retention can mutate state.
    referenced_attempt_ids = _referenced_manual_compaction_attempt_ids(conn, thread_id)
    scope_sql = "1 = 1" if thread_id is None else "thread_id = ?"
    scope_params: tuple[Any, ...] = () if thread_id is None else (thread_id,)
    pending_cutoff = now - MANUAL_COMPACTION_PENDING_TTL_MS
    expired = conn.execute(
        "SELECT attempt_id, state FROM manual_compactions WHERE "
        f"{scope_sql} AND ((state = 'pending' AND created_at <= ?) "
        "OR (state = 'running' AND lease_expires_at IS NOT NULL AND lease_expires_at <= ?))",
        (*scope_params, pending_cutoff, now),
    ).fetchall()
    for row in expired:
        reason = "pending_expired" if row["state"] == "pending" else "lease_expired"
        _terminalize_attempt(
            conn,
            str(row["attempt_id"]),
            now,
            state = "failed",
            reason = reason,
            allowed_states = (str(row["state"]),),
        )
    terminal_rows = conn.execute(
        "SELECT attempt_id, state, terminal_reason FROM manual_compactions WHERE "
        f"{scope_sql} AND state IN ('cancelled', 'failed')",
        scope_params,
    ).fetchall()
    for row in terminal_rows:
        reason = row["terminal_reason"]
        if reason not in _TERMINAL_REASONS:
            reason = "migrated_cancelled" if row["state"] == "cancelled" else "migrated_failed"
        conn.execute(
            "UPDATE manual_compactions SET terminal_reason = ?, "
            "finished_at = COALESCE(finished_at, cancelled_at, committed_at, created_at), "
            "cancelled_at = CASE WHEN state = 'cancelled' "
            "THEN COALESCE(cancelled_at, finished_at, created_at) ELSE NULL END, "
            "committed_at = NULL, summary_message_id = NULL, summary_hash = NULL, "
            "lease_expires_at = NULL, archive_status = 'pending' "
            "WHERE attempt_id = ? AND state IN ('cancelled', 'failed')",
            (reason, row["attempt_id"]),
        )
    conn.execute(
        "UPDATE manual_compactions SET source_message_ids_json = '[]', "
        "effective_source_message_ids_json = '[]', archive_payload_json = '[]', "
        "archive_payload_hash = ? WHERE "
        f"{scope_sql} "
        "AND state IN ('cancelled', 'failed')",
        (_EMPTY_JSON_HASH, *scope_params),
    )
    terminal_cutoff = now - MANUAL_COMPACTION_TERMINAL_RETENTION_MS
    terminal_rows = conn.execute(
        "SELECT attempt_id, thread_id, finished_at, created_at "
        "FROM manual_compactions WHERE "
        f"{scope_sql} AND state IN ('cancelled', 'failed') "
        "ORDER BY thread_id, finished_at DESC, created_at DESC, attempt_id DESC",
        scope_params,
    ).fetchall()
    retained_unreferenced: dict[str, int] = {}
    delete_ids: list[str] = []
    for row in terminal_rows:
        attempt_id = str(row["attempt_id"])
        if attempt_id in referenced_attempt_ids:
            continue
        row_thread_id = str(row["thread_id"])
        rank = retained_unreferenced.get(row_thread_id, 0)
        retained_unreferenced[row_thread_id] = rank + 1
        finished_at = int(row["finished_at"] or row["created_at"] or 0)
        if (
            finished_at <= terminal_cutoff
            or rank >= MAX_MANUAL_COMPACTION_TERMINAL_ATTEMPTS_PER_THREAD
        ):
            delete_ids.append(attempt_id)
    for start in range(0, len(delete_ids), 500):
        chunk = delete_ids[start : start + 500]
        placeholders = ",".join("?" for _ in chunk)
        conn.execute(
            f"DELETE FROM manual_compactions WHERE attempt_id IN ({placeholders})",
            tuple(chunk),
        )


def reconcile_all_manual_compaction_attempts_in_connection(
    conn,
    *,
    now: int | None = None,
    process_restart: bool = False,
) -> None:
    """Idempotently expire and bound durable attempts for every thread."""
    current = int(time.time() * 1000) if now is None else now
    if process_restart:
        running = conn.execute(
            "SELECT attempt_id FROM manual_compactions WHERE state = 'running'"
        ).fetchall()
        for row in running:
            _terminalize_attempt(
                conn,
                str(row["attempt_id"]),
                current,
                state = "failed",
                reason = "inference_failed",
                allowed_states = ("running",),
            )
    _cleanup_manual_compaction_attempts(
        conn,
        None,
        current,
    )


def cleanup_all_manual_compaction_attempts() -> None:
    conn = _db().get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        reconcile_all_manual_compaction_attempts_in_connection(conn)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def cleanup_manual_compaction_attempts(thread_id: str) -> None:
    thread_id = _bounded_id(thread_id, "thread id")
    conn = _db().get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        _cleanup_manual_compaction_attempts(conn, thread_id, int(time.time() * 1000))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _cancel_attempt(
    conn,
    attempt_id: str,
    now: int,
    *,
    reason: str = "cancelled",
) -> None:
    if not _terminalize_attempt(
        conn,
        attempt_id,
        now,
        state = "cancelled",
        reason = reason,
    ):
        raise ManualCompactionConflict("Manual compaction attempt changed during replacement")


def fail_manual_compaction_attempt(
    attempt_id: str,
    reason: str,
    *,
    cancelled: bool = False,
) -> dict[str, Any] | None:
    """Promptly close a claimed attempt after an inference boundary fails."""
    attempt_id = _bounded_id(attempt_id, "attempt id")
    if reason not in _INFERENCE_TERMINAL_REASONS:
        reason = "inference_cancelled" if cancelled else "inference_failed"
    conn = _db().get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM manual_compactions WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        if row is None:
            conn.commit()
            return None
        attempt = _attempt_from_row(row)
        target = "cancelled" if cancelled else "failed"
        if attempt["state"] in ("cancelled", "failed"):
            conn.commit()
            return attempt
        if attempt["state"] != "running":
            conn.commit()
            return attempt
        now = int(time.time() * 1000)
        if not _terminalize_attempt(
            conn,
            attempt_id,
            now,
            state = target,
            reason = reason,
            allowed_states = ("running",),
        ):
            raise ManualCompactionConflict("Manual compaction failure lost its state race")
        updated = conn.execute(
            "SELECT * FROM manual_compactions WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        conn.commit()
        return _attempt_from_row(updated)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def bind_manual_compaction_command(
    thread_id: str,
    *,
    attempt_id: str,
    command_message_id: str,
    summary_message_id: str,
    attempt_sequence: int,
    expected_attempt_id: str | None = None,
    expected_attempt_sequence: int | None = None,
) -> dict[str, Any]:
    """CAS the client-owned identity before prepare can create server state."""
    thread_id = _bounded_id(thread_id, "thread id")
    attempt_id = _bounded_id(attempt_id, "attempt id")
    command_message_id = _bounded_id(command_message_id, "command message id")
    summary_message_id = _bounded_id(summary_message_id, "summary message id")
    if summary_message_id == command_message_id:
        raise ManualCompactionError("summary message id must differ from the command message id")
    if (
        not isinstance(attempt_sequence, int)
        or isinstance(attempt_sequence, bool)
        or attempt_sequence < 1
        or attempt_sequence > 2**53 - 1
    ):
        raise ManualCompactionError("attempt sequence must be a positive safe integer")
    if (expected_attempt_id is None) != (expected_attempt_sequence is None):
        raise ManualCompactionError(
            "expected attempt id and sequence must both be present or both be absent"
        )
    if expected_attempt_id is not None:
        expected_attempt_id = _bounded_id(expected_attempt_id, "expected attempt id")
        if (
            not isinstance(expected_attempt_sequence, int)
            or isinstance(expected_attempt_sequence, bool)
            or expected_attempt_sequence < 1
            or expected_attempt_sequence > 2**53 - 1
        ):
            raise ManualCompactionError("expected attempt sequence must be a positive safe integer")

    candidate = {
        "schemaVersion": MANUAL_COMPACTION_SCHEMA_VERSION,
        "state": "bound",
        "threadId": thread_id,
        "commandMessageId": command_message_id,
        "attemptId": attempt_id,
        "summaryMessageId": summary_message_id,
        "attemptSequence": attempt_sequence,
        "expectedHeadMessageId": command_message_id,
    }
    conn = _db().get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        now = int(time.time() * 1000)
        terminalized_pending = False
        _cleanup_manual_compaction_attempts(conn, thread_id, now)
        branch = _complete_command_branch(conn, thread_id, command_message_id)
        command = branch[-1]
        if not _literal_compact(command):
            raise ManualCompactionConflict("The fenced command must be the literal /compact")
        current = _manual_compaction_client_record(command)
        if current is not None and _client_record_identity(current) == _client_record_identity(
            candidate
        ):
            _reserve_manual_compaction_attempt(conn, current, created_at = now)
            conn.commit()
            return command

        _require_branch_lifecycle_quiescent(conn, thread_id, branch)

        if current is None:
            if expected_attempt_id is not None or attempt_sequence != 1:
                raise ManualCompactionConflict("The /compact command binding CAS is stale")
        else:
            if (
                current["attemptId"] != expected_attempt_id
                or current["attemptSequence"] != expected_attempt_sequence
                or attempt_sequence != current["attemptSequence"] + 1
                or attempt_id == current["attemptId"]
            ):
                raise ManualCompactionConflict("The /compact command binding CAS is stale")
            attempt_row = conn.execute(
                "SELECT * FROM manual_compactions WHERE attempt_id = ?",
                (current["attemptId"],),
            ).fetchone()
            if attempt_row is None:
                raise ManualCompactionConflict(
                    "The bound /compact attempt must be recovered before it can be replaced"
                )
            current_attempt = _attempt_from_row(attempt_row)
            if (
                current_attempt["threadId"] != thread_id
                or current_attempt["commandMessageId"] != command_message_id
            ):
                raise ManualCompactionConflict("Stored /compact binding authority is inconsistent")
            if current_attempt["state"] == "pending" and not _pending_attempt_is_valid(
                conn, current_attempt
            ):
                now = int(time.time() * 1000)
                if not _terminalize_attempt(
                    conn,
                    current_attempt["attemptId"],
                    now,
                    state = "failed",
                    reason = "prepare_invalidated",
                    allowed_states = ("pending",),
                ):
                    raise ManualCompactionConflict(
                        "The invalidated /compact attempt changed during replacement"
                    )
                terminalized_pending = True
                current_attempt = _attempt_from_row(
                    conn.execute(
                        "SELECT * FROM manual_compactions WHERE attempt_id = ?",
                        (current_attempt["attemptId"],),
                    ).fetchone()
                )
            if current_attempt["state"] not in ("cancelled", "failed"):
                raise ManualCompactionConflict(
                    "A live /compact attempt cannot be replaced through command metadata"
                )
            command_hash_state = current_attempt.get("commandHashState")
            if command_hash_state not in ("prepared", "summary_saved"):
                raise ManualCompactionConflict(
                    "The terminal /compact command has no proven command digest"
                )
            _require_command_hash(current_attempt, command, state = command_hash_state)
            source = branch[:-1]
            if (
                not source
                or source[-1]["id"] != current_attempt["sourceHeadMessageId"]
                or current_attempt["expectedHeadMessageId"] != command_message_id
                or canonical_source_hash(source) != current_attempt["sourceHash"]
            ):
                raise ManualCompactionConflict(
                    "The terminal /compact command ancestry changed before replacement"
                )
            _reserve_manual_compaction_attempt(
                conn, current, created_at = current_attempt["createdAt"]
            )

        incoming_attempt = conn.execute(
            "SELECT thread_id, command_message_id FROM manual_compactions WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        if incoming_attempt is not None:
            raise ManualCompactionConflict("Attempt id is already used for another binding")
        child = conn.execute(
            "SELECT id FROM chat_messages WHERE thread_id = ? AND parent_id = ? LIMIT 1",
            (thread_id, command_message_id),
        ).fetchone()
        if child is not None:
            if terminalized_pending:
                conn.commit()
            raise ManualCompactionConflict("The /compact command is no longer the branch head")
        reserved_summary = conn.execute(
            "SELECT thread_id, parent_id FROM chat_messages WHERE id = ?",
            (summary_message_id,),
        ).fetchone()
        if reserved_summary is not None:
            raise ManualCompactionConflict("Summary message id is already in use")
        _reserve_manual_compaction_attempt(conn, candidate, created_at = now)
        updated = _write_manual_compaction_client_record(conn, command, candidate)
        conn.commit()
        return updated
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _pending_attempt_is_valid(conn, attempt: dict[str, Any]) -> bool:
    try:
        source_ids = list(attempt["sourceMessageIds"])
        branch = _exact_branch(
            conn,
            attempt["threadId"],
            [*source_ids, attempt["commandMessageId"]],
            attempt["expectedHeadMessageId"],
        )
        command = branch[-1]
        if not _literal_compact(command):
            return False
        if canonical_source_hash(branch[:-1]) != attempt["sourceHash"]:
            return False
        if _next_branch_revision(conn, attempt["threadId"], branch[:-1]) != attempt["revision"]:
            return False
        child = conn.execute(
            "SELECT id FROM chat_messages WHERE thread_id = ? AND parent_id = ? LIMIT 1",
            (attempt["threadId"], attempt["commandMessageId"]),
        ).fetchone()
        if child is not None:
            return False
        _recheck_project_context(conn, attempt["threadId"], attempt)
        effective = _effective_branch(conn, attempt["threadId"], branch[:-1], command)
        if [message["id"] for message in effective[:-1]] != attempt["effectiveSourceMessageIds"]:
            return False
        archive_payload, _archive_json, archive_payload_hash = _archive_payload(effective)
        return (
            archive_payload == attempt["archivePayload"]
            and archive_payload_hash == attempt["archivePayloadHash"]
        )
    except ManualCompactionError:
        return False


def _manual_compaction_envelope(attempt: dict[str, Any]) -> dict[str, Any]:
    return {
        "attemptId": attempt["attemptId"],
        "threadId": attempt["threadId"],
        "commandMessageId": attempt["commandMessageId"],
        "expectedHeadMessageId": attempt["expectedHeadMessageId"],
        "sourceHash": attempt["sourceHash"],
        "requestHash": attempt["requestHash"],
        "requestMessageCount": attempt["requestMessageCount"],
        "projectInstructionDigest": attempt["projectInstructionDigest"],
        "projectInstructionRevision": attempt["projectInstructionRevision"],
        "contextDigest": attempt["contextDigest"],
        "revision": attempt["revision"],
    }


def _canonical_command_hash(command: dict[str, Any], *, state: str) -> str:
    try:
        return _db().canonical_chat_message_hash(command, state = state)
    except (KeyError, TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ManualCompactionConflict(
            "Stored /compact command cannot be hashed canonically"
        ) from exc


def _require_command_hash(attempt: dict[str, Any], command: dict[str, Any], *, state: str) -> None:
    if attempt.get("commandHashState") != state:
        raise ManualCompactionConflict("The /compact command hash state is stale")
    if not hmac.compare_digest(
        str(attempt.get("commandHash") or ""), _canonical_command_hash(command, state = state)
    ):
        raise ManualCompactionConflict("The /compact command changed after prepare")


def _advance_prepared_client_record(
    conn, command: dict[str, Any], attempt: dict[str, Any]
) -> dict[str, Any]:
    current = _manual_compaction_client_record(command)
    if current is None or (
        current["threadId"] != attempt["threadId"]
        or current["commandMessageId"] != attempt["commandMessageId"]
        or current["attemptId"] != attempt["attemptId"]
    ):
        raise ManualCompactionConflict("The /compact command binding changed before prepare")
    if current["state"] not in ("bound", "prepared"):
        raise ManualCompactionConflict("The /compact command is not in a preparable state")
    prepared = {key: current[key] for key in _CLIENT_BASE_KEYS if key != "state"}
    prepared.update(
        {
            "state": "prepared",
            "sourceHeadMessageId": attempt["sourceHeadMessageId"],
            "envelope": _manual_compaction_envelope(attempt),
        }
    )
    if current["state"] == "prepared":
        if current != prepared:
            raise ManualCompactionConflict("Stored /compact prepared metadata is inconsistent")
        return command
    return _write_manual_compaction_client_record(conn, command, prepared)


def _advance_summary_saved_client_record(
    conn,
    command: dict[str, Any],
    attempt: dict[str, Any],
    *,
    summary_message_id: str,
    summary_hash: str,
) -> dict[str, Any]:
    current = _manual_compaction_client_record(command)
    if current is None or (
        current["threadId"] != attempt["threadId"]
        or current["commandMessageId"] != attempt["commandMessageId"]
        or current["attemptId"] != attempt["attemptId"]
        or current["summaryMessageId"] != summary_message_id
    ):
        raise ManualCompactionConflict("The /compact command binding changed before commit")
    saved = {
        **{key: current[key] for key in _CLIENT_BASE_KEYS if key != "state"},
        "state": "summary_saved",
        "sourceHeadMessageId": attempt["sourceHeadMessageId"],
        "envelope": _manual_compaction_envelope(attempt),
        "summaryHash": summary_hash,
    }
    if current["state"] == "summary_saved":
        if current != saved:
            raise ManualCompactionConflict("Stored /compact summary metadata is inconsistent")
        return command
    if current["state"] != "prepared":
        raise ManualCompactionConflict("The /compact command is not prepared for commit")
    return _write_manual_compaction_client_record(conn, command, saved)


def prepare_manual_compaction(
    thread_id: str,
    *,
    attempt_id: str,
    command_message_id: str,
    expected_head_message_id: str,
    message_ids: list[str],
    request_messages: list[Any],
) -> dict[str, Any]:
    thread_id = _bounded_id(thread_id, "thread id")
    attempt_id = _bounded_id(attempt_id, "attempt id")
    command_message_id = _bounded_id(command_message_id, "command message id")
    if command_message_id != expected_head_message_id:
        raise ManualCompactionConflict("The /compact command must be the expected branch head")
    conn = _db().get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        now = int(time.time() * 1000)
        _cleanup_manual_compaction_attempts(conn, thread_id, now)
        existing = conn.execute(
            "SELECT * FROM manual_compactions WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        branch = _exact_branch(conn, thread_id, message_ids, expected_head_message_id)
        command = branch[-1]
        if command["id"] != command_message_id or not _literal_compact(command):
            raise ManualCompactionConflict("The branch head must be the literal /compact command")
        binding = _manual_compaction_client_record(command)
        if binding is None or (
            binding["threadId"] != thread_id
            or binding["commandMessageId"] != command_message_id
            or binding["attemptId"] != attempt_id
            or binding["expectedHeadMessageId"] != expected_head_message_id
            or binding["state"] not in ("bound", "prepared")
        ):
            raise ManualCompactionConflict(
                "The /compact command must be fenced to this attempt before prepare"
            )
        _require_manual_compaction_attempt_reservation(conn, binding)
        child = conn.execute(
            "SELECT id FROM chat_messages WHERE thread_id = ? AND parent_id = ? LIMIT 1",
            (thread_id, command_message_id),
        ).fetchone()
        if child is not None:
            raise ManualCompactionConflict("The /compact command is no longer the branch head")
        source = branch[:-1]
        source_hash = canonical_source_hash(source)
        effective_branch = _effective_branch(conn, thread_id, source, command)
        request_encoded, request_branch = _canonical_request(request_messages)
        project_context = _validate_project_instructions(conn, thread_id, request_branch)
        _validate_request_covers_branch(request_branch, effective_branch)
        archive_payload, archive_payload_json, archive_payload_hash = _archive_payload(
            effective_branch
        )
        request_hash = hashlib.sha256(request_encoded).hexdigest()
        request_message_count = len(request_branch)
        revision = _next_branch_revision(conn, thread_id, source)
        source_head_message_id = source[-1]["id"]
        candidate = {
            "attemptId": attempt_id,
            "threadId": thread_id,
            "commandMessageId": command_message_id,
            "sourceHeadMessageId": source_head_message_id,
            "expectedHeadMessageId": expected_head_message_id,
            "sourceMessageIds": [message["id"] for message in source],
            "effectiveSourceMessageIds": [message["id"] for message in effective_branch[:-1]],
            "sourceHash": source_hash,
            "requestHash": request_hash,
            "requestMessageCount": request_message_count,
            "projectInstructionDigest": project_context["projectInstructionDigest"],
            "projectInstructionRevision": project_context["projectInstructionRevision"],
            "contextDigest": project_context["contextDigest"],
            "archivePayload": archive_payload,
            "archivePayloadHash": archive_payload_hash,
            "revision": revision,
        }
        if existing is not None:
            stored = _attempt_from_row(existing)
            identity_keys = (
                "attemptId",
                "threadId",
                "commandMessageId",
                "sourceHeadMessageId",
                "expectedHeadMessageId",
                "sourceHash",
                "requestHash",
                "requestMessageCount",
                "projectInstructionDigest",
                "projectInstructionRevision",
                "contextDigest",
                "revision",
            )
            compare_keys = (
                candidate if stored["state"] not in ("cancelled", "failed") else identity_keys
            )
            for key in compare_keys:
                value = candidate[key]
                if stored.get(key) != value:
                    raise ManualCompactionConflict("Attempt id is already used for another branch")
            if _lease_expired(stored, now):
                _terminalize_attempt(
                    conn,
                    attempt_id,
                    now,
                    state = "failed",
                    reason = "lease_expired",
                    allowed_states = ("running",),
                )
                stored = _attempt_from_row(
                    conn.execute(
                        "SELECT * FROM manual_compactions WHERE attempt_id = ?", (attempt_id,)
                    ).fetchone()
                )
            elif stored["state"] in ("cancelled", "failed"):
                raise ManualCompactionConflict(
                    "Manual compaction attempt is terminal; prepare a new attempt id"
                )
            prepared_command = _advance_prepared_client_record(conn, command, stored)
            _require_command_hash(stored, prepared_command, state = "prepared")
            conn.commit()
            return stored
        occupied = conn.execute(
            "SELECT * FROM manual_compactions "
            "WHERE thread_id = ? AND revision = ? AND command_message_id = ? "
            "AND state IN ('pending', 'running', 'active')",
            (thread_id, revision, command_message_id),
        ).fetchone()
        if occupied is not None:
            occupied_attempt = _attempt_from_row(occupied)
            if occupied_attempt["state"] == "pending" or _lease_expired(occupied_attempt, now):
                _cancel_attempt(
                    conn,
                    occupied_attempt["attemptId"],
                    now,
                    reason = "replaced",
                )
                _cleanup_manual_compaction_attempts(conn, thread_id, now)
            else:
                raise ManualCompactionConflict(
                    "This /compact branch already has a running or active attempt"
                )
        result = {
            **candidate,
            "state": "pending",
            "summaryMessageId": None,
            "summaryHash": None,
            "outputSummaryHash": None,
            "outputFinishReason": None,
            "outputRecordedAt": None,
            "archiveStatus": "pending",
            "createdAt": now,
            "startedAt": None,
            "leaseExpiresAt": None,
            "cancelledAt": None,
            "committedAt": None,
            "terminalReason": None,
            "finishedAt": None,
        }
        prepared_command = _advance_prepared_client_record(conn, command, result)
        command_hash = _canonical_command_hash(prepared_command, state = "prepared")
        result.update(
            {
                "commandHash": command_hash,
                "commandHashState": "prepared",
            }
        )
        conn.execute(
            """
            INSERT INTO manual_compactions (
                attempt_id, thread_id, command_message_id, source_head_message_id,
                expected_head_message_id, source_message_ids_json,
                effective_source_message_ids_json, source_hash, request_hash,
                request_message_count, project_instruction_digest,
                project_instruction_revision, context_digest, archive_payload_json,
                archive_payload_hash, revision, state, command_hash, command_hash_state,
                archive_status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?,
                'prepared', 'pending', ?)
            """,
            (
                attempt_id,
                thread_id,
                command_message_id,
                source_head_message_id,
                expected_head_message_id,
                _strict_json_dumps(candidate["sourceMessageIds"]),
                _strict_json_dumps(candidate["effectiveSourceMessageIds"]),
                source_hash,
                request_hash,
                request_message_count,
                candidate["projectInstructionDigest"],
                candidate["projectInstructionRevision"],
                candidate["contextDigest"],
                archive_payload_json,
                archive_payload_hash,
                revision,
                command_hash,
                now,
            ),
        )
        conn.commit()
        return result
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_manual_compaction_attempt(attempt_id: str) -> dict[str, Any] | None:
    attempt_id = _bounded_id(attempt_id, "attempt id")
    conn = _db().get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        identity = conn.execute(
            "SELECT thread_id FROM manual_compactions WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        if identity is not None:
            _cleanup_manual_compaction_attempts(
                conn,
                str(identity["thread_id"]),
                int(time.time() * 1000),
            )
        row = conn.execute(
            "SELECT * FROM manual_compactions WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        attempt = _attempt_from_row(row) if row is not None else None
        conn.commit()
        return attempt
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def record_manual_compaction_output(
    attempt_id: str, *, text: Any, finish_reason: str
) -> dict[str, Any]:
    """Persist server-observed terminal output before a client may commit it."""
    attempt_id = _bounded_id(attempt_id, "attempt id")
    if finish_reason not in (
        "stop",
        "length",
        "tool_calls",
        "content_filter",
        "function_call",
    ):
        raise ManualCompactionError("Manual compaction finish reason is invalid")
    output_hash: str | None = None
    if isinstance(text, str):
        try:
            output_hash = summary_hash(text)
        except ManualCompactionError:
            output_hash = None
    conn = _db().get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM manual_compactions WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        if row is None:
            raise ManualCompactionNotFound("Manual compaction attempt not found")
        attempt = _attempt_from_row(row)
        if attempt["state"] != "running":
            raise ManualCompactionConflict(
                "Manual compaction output does not belong to a running attempt"
            )
        if attempt["outputFinishReason"] is not None:
            if (
                attempt["outputFinishReason"] != finish_reason
                or attempt["outputSummaryHash"] != output_hash
            ):
                raise ManualCompactionConflict(
                    "Manual compaction output was already finalized differently"
                )
            conn.commit()
            return attempt
        now = int(time.time() * 1000)
        if finish_reason != "stop" or output_hash is None:
            reason = (
                f"finish_{finish_reason}" if finish_reason != "stop" else "invalid_summary_output"
            )
            if not _terminalize_attempt(
                conn,
                attempt_id,
                now,
                state = "failed",
                reason = reason,
                allowed_states = ("running",),
            ):
                raise ManualCompactionConflict(
                    "Manual compaction output lost its terminal state race"
                )
            updated = conn.execute(
                "SELECT * FROM manual_compactions WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
            conn.commit()
            return _attempt_from_row(updated)
        changed = conn.execute(
            "UPDATE manual_compactions SET output_summary_hash = ?, "
            "output_finish_reason = ?, output_recorded_at = ? "
            "WHERE attempt_id = ? AND state = 'running' "
            "AND output_finish_reason IS NULL",
            (output_hash, finish_reason, now, attempt_id),
        ).rowcount
        if changed != 1:
            raise ManualCompactionConflict("Manual compaction output lost its finalization race")
        conn.commit()
        return {
            **attempt,
            "outputSummaryHash": output_hash,
            "outputFinishReason": finish_reason,
            "outputRecordedAt": now,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _envelope_dict(envelope: Any) -> dict[str, Any]:
    if hasattr(envelope, "model_dump"):
        return envelope.model_dump(by_alias = True)
    if isinstance(envelope, dict):
        return dict(envelope)
    raise ManualCompactionError("manual_compaction must be an object")


def _stored_studio_tool_history(branch: Iterable[dict[str, Any]]) -> bool | None:
    """Derive the frontend ownership marker from the pinned stored branch."""
    saw_replayed_call = False
    for message in branch:
        if message.get("role") != "assistant":
            continue
        for part in _parts(message.get("content"), label = "Stored assistant content"):
            if part.get("type") != "tool-call":
                continue
            builtin, native = _tool_builtin(part)
            if (builtin and not native) or (not builtin and part.get("result") is None):
                continue
            saw_replayed_call = True
            provenance = part.get("provenance")
            if not isinstance(provenance, dict) or provenance.get("source") != "local":
                return None
    return True if saw_replayed_call else None


def _rewrite_claimed_payload(payload: Any, *, studio_tool_history: bool | None = None) -> None:
    # The claim secret and durable envelope terminate at this security boundary.
    # Downstream providers receive only the rewritten handoff request.
    payload.manual_compaction = None
    payload.messages[-1].content = MANUAL_COMPACTION_HANDOFF_INSTRUCTION
    payload.tools = None
    payload.tool_choice = "none"
    payload.enable_tools = False
    payload.enabled_tools = []
    payload.mcp_enabled = False
    payload.deep_research_armed = False
    payload.confirm_tool_calls = False
    payload.permission_mode = "off"
    payload.bypass_permissions = False
    payload.auto_heal_tool_calls = False
    payload.nudge_tool_calls = False
    payload.max_tool_calls_per_message = 0
    payload.tool_call_timeout = 1
    payload.run_tools_locally = False
    payload.studio_tool_history = studio_tool_history
    payload.openai_code_exec_container_id = None
    payload.anthropic_code_exec_container_id = None
    payload.rag_scope = None
    payload.image_base64 = None
    payload.audio_base64 = None
    payload.video_base64 = None
    payload.context_overflow = "error"
    payload.context_policy = None
    payload.compaction_headroom_ratio = None
    payload.compaction_threshold = None
    payload.max_tokens = None
    payload.max_completion_tokens = MAX_MANUAL_COMPACTION_SUMMARY_TOKENS
    payload.enable_thinking = False
    payload.reasoning_effort = "none"
    payload.preserve_thinking = False
    payload.thinking = None
    payload.continue_final_message = False
    payload.response_format = None
    payload.n = 1
    payload.stop = None
    payload.stream_options = None
    payload.logprobs = False
    payload.top_logprobs = None
    payload.parallel_tool_calls = False
    extra = getattr(payload, "__pydantic_extra__", None)
    if isinstance(extra, dict):
        for key in list(extra):
            normalized = str(key).lower().replace("-", "_")
            if (
                normalized == "context_management"
                or normalized == "chat_template_kwargs"
                or "reason" in normalized
                or "think" in normalized
            ):
                extra.pop(key, None)


def validate_and_rewrite_manual_compaction_request(payload: Any) -> dict[str, Any] | None:
    envelope_value = getattr(payload, "manual_compaction", None)
    if envelope_value is None:
        return None
    envelope = _envelope_dict(envelope_value)
    attempt_id = _bounded_id(envelope.get("attemptId", ""), "attempt id")
    claim_token_hash = _claim_token_hash(envelope.get("claimId"))
    request_encoded, request_branch = _canonical_request(getattr(payload, "messages", []))
    now = int(time.time() * 1000)
    conn = _db().get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM manual_compactions WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        if row is None:
            raise ManualCompactionNotFound("Manual compaction attempt not found")
        attempt = _attempt_from_row(row)
        expected = {
            "threadId": attempt["threadId"],
            "commandMessageId": attempt["commandMessageId"],
            "expectedHeadMessageId": attempt["expectedHeadMessageId"],
            "sourceHash": attempt["sourceHash"],
            "requestHash": attempt["requestHash"],
            "requestMessageCount": attempt["requestMessageCount"],
            "projectInstructionDigest": attempt["projectInstructionDigest"],
            "projectInstructionRevision": attempt["projectInstructionRevision"],
            "contextDigest": attempt["contextDigest"],
            "revision": attempt["revision"],
        }
        for key, value in expected.items():
            if envelope.get(key) != value:
                raise ManualCompactionConflict(
                    "Manual compaction configuration drifted after prepare"
                )
        if attempt["state"] == "running":
            if _lease_expired(attempt, now):
                _terminalize_attempt(
                    conn,
                    attempt_id,
                    now,
                    state = "failed",
                    reason = "lease_expired",
                    allowed_states = ("running",),
                )
                conn.commit()
                raise ManualCompactionConflict(
                    "Manual compaction inference lease expired; prepare a new attempt id"
                )
            raise ManualCompactionConflict("Manual compaction attempt is already running")
        if attempt["state"] != "pending":
            raise ManualCompactionConflict("Manual compaction attempt is no longer claimable")
        if attempt["createdAt"] <= now - MANUAL_COMPACTION_PENDING_TTL_MS:
            _terminalize_attempt(
                conn,
                attempt_id,
                now,
                state = "failed",
                reason = "pending_expired",
                allowed_states = ("pending",),
            )
            conn.commit()
            raise ManualCompactionConflict(
                "Manual compaction prepare lease expired; prepare a new attempt id"
            )
        if getattr(payload, "thread_id", None) != attempt["threadId"]:
            raise ManualCompactionConflict("Manual compaction thread does not match the request")
        source_ids = list(attempt["sourceMessageIds"])
        branch = _exact_branch(
            conn,
            attempt["threadId"],
            [*source_ids, attempt["commandMessageId"]],
            attempt["expectedHeadMessageId"],
        )
        if canonical_source_hash(branch[:-1]) != attempt["sourceHash"]:
            raise ManualCompactionConflict("Manual compaction source changed after prepare")
        if _next_branch_revision(conn, attempt["threadId"], branch[:-1]) != attempt["revision"]:
            raise ManualCompactionConflict(
                "Manual compaction branch revision changed after prepare"
            )
        command = branch[-1]
        if not _literal_compact(command):
            raise ManualCompactionConflict("The prepared /compact command changed")
        command_binding = _manual_compaction_client_record(command)
        if command_binding is None or (
            command_binding["threadId"] != attempt["threadId"]
            or command_binding["commandMessageId"] != attempt["commandMessageId"]
            or command_binding["attemptId"] != attempt_id
            or command_binding["state"] != "prepared"
        ):
            raise ManualCompactionConflict("The prepared /compact command binding changed")
        _require_manual_compaction_attempt_reservation(conn, command_binding)
        _require_command_hash(attempt, command, state = "prepared")
        child = conn.execute(
            "SELECT id FROM chat_messages WHERE thread_id = ? AND parent_id = ? LIMIT 1",
            (attempt["threadId"], attempt["commandMessageId"]),
        ).fetchone()
        if child is not None:
            if not _terminalize_attempt(
                conn,
                attempt_id,
                now,
                state = "failed",
                reason = "prepare_invalidated",
                allowed_states = ("pending",),
            ):
                raise ManualCompactionConflict(
                    "The invalidated /compact attempt changed during claim"
                )
            conn.commit()
            raise ManualCompactionConflict(
                "The prepared /compact command is no longer the branch head"
            )
        _recheck_project_context(conn, attempt["threadId"], attempt)
        _validate_project_instructions(conn, attempt["threadId"], list(payload.messages))
        effective_branch = _effective_branch(conn, attempt["threadId"], branch[:-1], branch[-1])
        if [message["id"] for message in effective_branch[:-1]] != attempt[
            "effectiveSourceMessageIds"
        ]:
            raise ManualCompactionConflict("Manual compaction effective branch changed")
        if len(request_branch) != attempt["requestMessageCount"]:
            raise ManualCompactionConflict(
                "Manual compaction requires the complete untruncated branch"
            )
        if hashlib.sha256(request_encoded).hexdigest() != attempt["requestHash"]:
            raise ManualCompactionConflict("Manual compaction request changed after prepare")
        _validate_request_covers_branch(request_branch, effective_branch)
        if _content_text(_request_content(request_branch[-1])) != "/compact":
            raise ManualCompactionConflict(
                "Manual compaction request must end with the literal /compact command"
            )
        lease_expires_at = now + MANUAL_COMPACTION_LEASE_MS
        changed = conn.execute(
            "UPDATE manual_compactions SET state = 'running', started_at = ?, "
            "lease_expires_at = ?, cancelled_at = NULL, claim_token_hash = ? "
            "WHERE attempt_id = ? AND state = 'pending'",
            (now, lease_expires_at, claim_token_hash, attempt_id),
        ).rowcount
        if changed != 1:
            raise ManualCompactionConflict("Manual compaction inference claim was lost")
        conn.commit()
        attempt.update(
            {
                "state": "running",
                "startedAt": now,
                "leaseExpiresAt": lease_expires_at,
                "cancelledAt": None,
            }
        )
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    try:
        _rewrite_claimed_payload(
            payload,
            studio_tool_history = _stored_studio_tool_history(effective_branch[:-1]),
        )
    except BaseException as exc:
        try:
            fail_manual_compaction_attempt(
                attempt_id,
                "request_rewrite_failed",
                cancelled = isinstance(exc, asyncio.CancelledError),
            )
        except Exception:
            pass
        raise
    return attempt


def _summary_payload(text: str) -> bytes:
    encoded = _strict_utf8(
        text,
        label = "Manual compaction summary",
        max_bytes = MAX_MANUAL_COMPACTION_SUMMARY_BYTES,
    )
    if not encoded or not text.strip():
        raise ManualCompactionConflict("Manual compaction summary must not be blank")
    return json.dumps(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "attachments": None,
            "providerMetadata": None,
        },
        ensure_ascii = False,
        allow_nan = False,
        sort_keys = True,
        separators = (",", ":"),
    ).encode("utf-8", errors = "strict")


def _strict_summary(message: dict[str, Any], *, active: bool) -> tuple[str, str, str]:
    if message.get("role") != "assistant":
        raise ManualCompactionConflict("Manual compaction summary must be an assistant message")
    content = message.get("content")
    if isinstance(content, str):
        text = content
    elif (
        isinstance(content, list)
        and len(content) == 1
        and isinstance(content[0], dict)
        and set(content[0]) == {"type", "text"}
        and content[0].get("type") == "text"
        and isinstance(content[0].get("text"), str)
    ):
        text = content[0]["text"]
    else:
        raise ManualCompactionConflict(
            "Manual compaction summary must contain one text-only content part"
        )
    if message.get("attachments") is not None:
        raise ManualCompactionConflict("Manual compaction summary cannot contain attachments")
    metadata = message.get("metadata")
    if active:
        if not (
            isinstance(metadata, dict)
            and set(metadata) == {"manualCompaction"}
            and isinstance(metadata.get("manualCompaction"), dict)
        ):
            raise ManualCompactionConflict("Active manual compaction summary metadata is invalid")
    elif metadata not in (None, {}):
        raise ManualCompactionConflict(
            "Manual compaction summary cannot contain provider or generation metadata"
        )
    canonical_content = json.dumps(
        [{"type": "text", "text": text}],
        ensure_ascii = False,
        allow_nan = False,
        separators = (",", ":"),
    )
    return text, canonical_content, hashlib.sha256(_summary_payload(text)).hexdigest()


def summary_hash(text: str) -> str:
    return hashlib.sha256(_summary_payload(text)).hexdigest()


def cancel_manual_compaction(
    thread_id: str, *, attempt_id: str, command_message_id: str, claim_id: str
) -> dict[str, Any]:
    """Cancel only the running inference request that owns the claim secret."""
    thread_id = _bounded_id(thread_id, "thread id")
    attempt_id = _bounded_id(attempt_id, "attempt id")
    command_message_id = _bounded_id(command_message_id, "command message id")
    candidate_claim_hash = _claim_token_hash(claim_id)
    conn = _db().get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM manual_compactions WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        if row is None:
            raise ManualCompactionNotFound("Manual compaction attempt not found")
        attempt = _attempt_from_row(row)
        if attempt["threadId"] != thread_id or attempt["commandMessageId"] != command_message_id:
            raise ManualCompactionConflict("Manual compaction cancellation is stale")
        command_row = conn.execute(
            "SELECT * FROM chat_messages WHERE thread_id = ? AND id = ?",
            (thread_id, command_message_id),
        ).fetchone()
        if command_row is None:
            raise ManualCompactionConflict("The /compact command is missing during cancellation")
        command_binding = _manual_compaction_client_record(_message_dict(command_row))
        if command_binding is None or (
            command_binding["threadId"] != thread_id
            or command_binding["commandMessageId"] != command_message_id
            or command_binding["attemptId"] != attempt_id
        ):
            raise ManualCompactionConflict("Manual compaction cancellation binding is stale")
        _require_manual_compaction_attempt_reservation(conn, command_binding)
        if attempt["state"] == "active":
            raise ManualCompactionConflict("An active manual compaction cannot be cancelled")
        stored_claim_hash = row["claim_token_hash"]
        if not (
            isinstance(stored_claim_hash, str)
            and _SHA256_RE.fullmatch(stored_claim_hash) is not None
            and hmac.compare_digest(stored_claim_hash, candidate_claim_hash)
        ):
            raise ManualCompactionConflict(
                "Manual compaction cancellation is not owned by this inference request"
            )
        if attempt["state"] in ("cancelled", "failed"):
            conn.commit()
            return attempt
        if attempt["state"] != "running":
            raise ManualCompactionConflict(
                "A pending manual compaction cannot be cancelled without an inference claim"
            )
        now = int(time.time() * 1000)
        _cancel_attempt(conn, attempt_id, now)
        updated = conn.execute(
            "SELECT * FROM manual_compactions WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        conn.commit()
        return _attempt_from_row(updated)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def commit_manual_compaction(
    thread_id: str,
    *,
    attempt_id: str,
    command_message_id: str,
    summary_message_id: str,
    expected_head_message_id: str,
    expected_revision: int,
    expected_summary_hash: str,
) -> dict[str, Any]:
    thread_id = _bounded_id(thread_id, "thread id")
    attempt_id = _bounded_id(attempt_id, "attempt id")
    command_message_id = _bounded_id(command_message_id, "command message id")
    summary_message_id = _bounded_id(summary_message_id, "summary message id")
    expected_head_message_id = _bounded_id(expected_head_message_id, "expected head message id")
    if (
        not isinstance(expected_revision, int)
        or isinstance(expected_revision, bool)
        or expected_revision < 1
    ):
        raise ManualCompactionError("expected revision must be a positive integer")
    if (
        not isinstance(expected_summary_hash, str)
        or _SHA256_RE.fullmatch(expected_summary_hash) is None
    ):
        raise ManualCompactionError("expected summary hash must be a lowercase SHA-256 digest")

    conn = _db().get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM manual_compactions WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        if row is None:
            raise ManualCompactionNotFound("Manual compaction attempt not found")
        attempt = _attempt_from_row(row)
        identity = (
            attempt["threadId"] == thread_id
            and attempt["commandMessageId"] == command_message_id
            and attempt["revision"] == expected_revision
        )
        if not identity:
            raise ManualCompactionConflict("Manual compaction commit is stale")
        active = attempt["state"] == "active"
        if attempt["state"] == "cancelled":
            raise ManualCompactionConflict("Manual compaction attempt was cancelled")
        if attempt["state"] == "failed":
            reason = attempt.get("terminalReason") or "inference failed"
            raise ManualCompactionConflict(f"Manual compaction attempt failed: {reason}")
        if not active and attempt["state"] != "running":
            raise ManualCompactionConflict(
                "Manual compaction attempt must claim inference before commit"
            )
        if not active:
            now = int(time.time() * 1000)
            if _lease_expired(attempt, now):
                _terminalize_attempt(
                    conn,
                    attempt_id,
                    now,
                    state = "failed",
                    reason = "lease_expired",
                    allowed_states = ("running",),
                )
                conn.commit()
                raise ManualCompactionConflict(
                    "Manual compaction inference lease expired; prepare a new attempt id"
                )
        if active and not (
            attempt["summaryMessageId"] == summary_message_id
            and attempt["summaryHash"] == expected_summary_hash
            and expected_head_message_id == summary_message_id
        ):
            raise ManualCompactionConflict(
                "Manual compaction attempt was already committed differently"
            )
        source_ids = list(attempt["sourceMessageIds"])
        branch = _exact_branch(
            conn,
            thread_id,
            [*source_ids, command_message_id, summary_message_id],
            expected_head_message_id,
        )
        if not active:
            child = conn.execute(
                "SELECT id FROM chat_messages WHERE thread_id = ? AND parent_id = ? LIMIT 1",
                (thread_id, summary_message_id),
            ).fetchone()
            if child is not None:
                raise ManualCompactionConflict(
                    "Manual compaction summary is no longer the branch head"
                )
        command = branch[-2]
        summary = branch[-1]
        if command["id"] != command_message_id or not _literal_compact(command):
            raise ManualCompactionConflict("The literal /compact command changed before commit")
        command_binding = _manual_compaction_client_record(command)
        expected_binding_state = "summary_saved" if active else "prepared"
        if command_binding is None or (
            command_binding["threadId"] != thread_id
            or command_binding["commandMessageId"] != command_message_id
            or command_binding["attemptId"] != attempt_id
            or command_binding["summaryMessageId"] != summary_message_id
            or command_binding["state"] != expected_binding_state
        ):
            raise ManualCompactionConflict("The /compact command binding changed before commit")
        _require_manual_compaction_attempt_reservation(conn, command_binding)
        _require_command_hash(
            attempt,
            command,
            state = "summary_saved" if active else "prepared",
        )
        if summary.get("parentId") != command_message_id:
            raise ManualCompactionConflict("Manual compaction summary is not the command child")
        current_source_hash = canonical_source_hash(branch[:-2])
        if current_source_hash != attempt["sourceHash"]:
            raise ManualCompactionConflict("Manual compaction source changed before commit")
        if _next_branch_revision(conn, thread_id, branch[:-2]) != expected_revision:
            raise ManualCompactionConflict(
                "Manual compaction branch revision changed before commit"
            )
        if not active:
            _recheck_project_context(conn, thread_id, attempt)
        effective_branch = _effective_branch(conn, thread_id, branch[:-2], command)
        if [message["id"] for message in effective_branch[:-1]] != attempt[
            "effectiveSourceMessageIds"
        ]:
            raise ManualCompactionConflict("Manual compaction effective branch changed")
        archive_payload, _archive_json, archive_payload_hash = _archive_payload(effective_branch)
        if (
            archive_payload != attempt["archivePayload"]
            or archive_payload_hash != attempt["archivePayloadHash"]
        ):
            raise ManualCompactionConflict("Manual compaction archive payload changed")
        text, canonical_content_json, actual_summary_hash = _strict_summary(summary, active = active)
        if actual_summary_hash != expected_summary_hash:
            raise ManualCompactionConflict("Manual compaction summary hash does not match")
        if attempt["outputFinishReason"] is None:
            raise ManualCompactionConflict(
                "Manual compaction output has no server-observed terminal event"
            )
        if attempt["outputFinishReason"] != "stop":
            raise ManualCompactionConflict("Manual compaction output did not finish cleanly")
        if (
            attempt["outputSummaryHash"] is None
            or attempt["outputSummaryHash"] != actual_summary_hash
        ):
            raise ManualCompactionConflict(
                "Manual compaction summary does not match the server-observed output"
            )
        if active:
            active_metadata = _active_compaction_metadata(summary)
            if active_metadata is None or any(
                active_metadata.get(key) != value
                for key, value in {
                    "attemptId": attempt_id,
                    "threadId": thread_id,
                    "revision": expected_revision,
                    "commandMessageId": command_message_id,
                    "sourceHeadMessageId": attempt["sourceHeadMessageId"],
                    "sourceHash": current_source_hash,
                    "projectInstructionDigest": attempt["projectInstructionDigest"],
                    "projectInstructionRevision": attempt["projectInstructionRevision"],
                    "contextDigest": attempt["contextDigest"],
                    "archivePayloadHash": attempt["archivePayloadHash"],
                    "outputSummaryHash": attempt["outputSummaryHash"],
                    "outputFinishReason": attempt["outputFinishReason"],
                    "summaryHash": actual_summary_hash,
                }.items()
            ):
                raise ManualCompactionConflict(
                    "Stored manual compaction metadata changed after commit"
                )
            saved_command = _advance_summary_saved_client_record(
                conn,
                command,
                attempt,
                summary_message_id = summary_message_id,
                summary_hash = actual_summary_hash,
            )
            _require_command_hash(attempt, saved_command, state = "summary_saved")
            conn.commit()
            if attempt["archiveStatus"] in ("pending", "failed"):
                return attempt
            return attempt

        metadata = summary.get("metadata")
        metadata = dict(metadata) if isinstance(metadata, dict) else {}
        metadata["manualCompaction"] = {
            "schemaVersion": MANUAL_COMPACTION_SCHEMA_VERSION,
            "state": "active",
            "attemptId": attempt_id,
            "threadId": thread_id,
            "revision": expected_revision,
            "commandMessageId": command_message_id,
            "sourceHeadMessageId": attempt["sourceHeadMessageId"],
            "summaryMessageId": summary_message_id,
            "sourceHash": current_source_hash,
            "requestHash": attempt["requestHash"],
            "requestMessageCount": attempt["requestMessageCount"],
            "projectInstructionDigest": attempt["projectInstructionDigest"],
            "projectInstructionRevision": attempt["projectInstructionRevision"],
            "contextDigest": attempt["contextDigest"],
            "archivePayloadHash": attempt["archivePayloadHash"],
            "outputSummaryHash": attempt["outputSummaryHash"],
            "outputFinishReason": attempt["outputFinishReason"],
            "summaryHash": actual_summary_hash,
        }
        now = int(time.time() * 1000)
        conn.execute(
            "UPDATE chat_messages SET content_json = ?, attachments_json = NULL, metadata_json = ? "
            "WHERE thread_id = ? AND id = ?",
            (
                canonical_content_json,
                _strict_json_dumps(metadata),
                thread_id,
                summary_message_id,
            ),
        )
        saved_command = _advance_summary_saved_client_record(
            conn,
            command,
            attempt,
            summary_message_id = summary_message_id,
            summary_hash = actual_summary_hash,
        )
        saved_command_hash = _canonical_command_hash(saved_command, state = "summary_saved")
        changed = conn.execute(
            """
            UPDATE manual_compactions
            SET state = 'active', summary_message_id = ?, summary_hash = ?,
                command_hash = ?, command_hash_state = 'summary_saved',
                lease_expires_at = NULL, claim_token_hash = NULL, committed_at = ?
            WHERE attempt_id = ? AND state = 'running' AND revision = ?
            """,
            (
                summary_message_id,
                actual_summary_hash,
                saved_command_hash,
                now,
                attempt_id,
                expected_revision,
            ),
        ).rowcount
        if changed != 1:
            raise ManualCompactionConflict("Manual compaction commit lost its revision race")
        conn.commit()
        return {
            **attempt,
            "state": "active",
            "summaryMessageId": summary_message_id,
            "summaryHash": actual_summary_hash,
            "commandHash": saved_command_hash,
            "commandHashState": "summary_saved",
            "leaseExpiresAt": None,
            "committedAt": now,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def archive_manual_compaction_best_effort(result: dict[str, Any]) -> str:
    """Archive originals after activation. A failed optional archive never rolls back the summary."""
    status = "failed"
    try:
        from core.rag import conversation_archive
        if not conversation_archive.enabled() or not conversation_archive.can_archive(
            result["threadId"]
        ):
            status = "skipped"
        else:
            source = result["archivePayload"]
            written = conversation_archive.archive_turns(
                result["threadId"],
                source,
                live = [],
                branch = source,
            )
            if conversation_archive.degraded():
                status = "failed"
            elif written > 0 or conversation_archive.turns_archived(
                result["threadId"],
                source,
                live = [],
                branch = source,
            ):
                status = "archived"
            else:
                status = "failed"
    except Exception:  # noqa: BLE001 -- the archive is explicitly best effort
        status = "failed"
    conn = None
    try:
        conn = _db().get_connection()
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "UPDATE manual_compactions SET archive_status = CASE "
            "WHEN archive_status = 'archived' OR ? = 'archived' THEN 'archived' "
            "WHEN archive_status = 'skipped' OR ? = 'skipped' THEN 'skipped' "
            "ELSE 'failed' END WHERE attempt_id = ? AND state = 'active'",
            (status, status, result["attemptId"]),
        )
        row = conn.execute(
            "SELECT archive_status FROM manual_compactions WHERE attempt_id = ? AND state = 'active'",
            (result["attemptId"],),
        ).fetchone()
        conn.commit()
        if row is not None:
            status = str(row["archive_status"])
    except Exception:  # noqa: BLE001 -- durable retry remains pending or failed
        if conn is not None:
            try:
                conn.rollback()
            except Exception:  # noqa: BLE001 -- preserve the original best-effort boundary
                pass
        status = "failed"
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001 -- close failure cannot undo activation
                status = "failed"
    return status


def rewrite_forked_manual_compaction_metadata(
    *, conn, source_rows: list[Any], id_map: dict[str, str], new_thread_id: str
) -> dict[str, str]:
    """Strip source authority and restore only complete, proven active checkpoints."""
    source_messages = [_message_dict(row) for row in source_rows]
    cloned: list[dict[str, Any]] = []
    rewritten: dict[str, str] = {}
    for message in source_messages:
        metadata = message.get("metadata")
        next_metadata = dict(metadata) if isinstance(metadata, dict) else None
        had_authority = False
        if isinstance(next_metadata, dict):
            for key in (MANUAL_COMPACTION_CLIENT_KEY, "manualCompaction"):
                if key in next_metadata:
                    had_authority = True
                    next_metadata.pop(key, None)
        cloned_message = {
            **message,
            "id": id_map[message["id"]],
            "threadId": new_thread_id,
            "parentId": (
                message.get("parentId")
                if message.get("parentId") in (None, "")
                else id_map.get(message.get("parentId"))
            ),
            "metadata": next_metadata,
        }
        cloned.append(cloned_message)
        if had_authority:
            rewritten[message["id"]] = _strict_json_dumps(next_metadata)

    active_keys = {
        "schemaVersion",
        "state",
        "attemptId",
        "threadId",
        "revision",
        "commandMessageId",
        "sourceHeadMessageId",
        "summaryMessageId",
        "sourceHash",
        "requestHash",
        "requestMessageCount",
        "projectInstructionDigest",
        "projectInstructionRevision",
        "contextDigest",
        "archivePayloadHash",
        "outputSummaryHash",
        "outputFinishReason",
        "summaryHash",
    }
    for index, message in enumerate(source_messages):
        metadata = message.get("metadata")
        raw = metadata.get("manualCompaction") if isinstance(metadata, dict) else None
        if (
            not isinstance(raw, dict)
            or set(raw) != active_keys
            or raw.get("schemaVersion") != MANUAL_COMPACTION_SCHEMA_VERSION
            or raw.get("state") != "active"
            or index < 2
        ):
            continue
        summary_old = message["id"]
        source_head_old = raw.get("sourceHeadMessageId")
        command_old = raw.get("commandMessageId")
        command_index = index - 1
        command_message = source_messages[command_index]
        if (
            command_old not in id_map
            or source_head_old not in id_map
            or command_message["id"] != command_old
            or message.get("parentId") != command_old
            or source_messages[command_index - 1]["id"] != source_head_old
        ):
            continue
        try:
            client = _manual_compaction_client_record(command_message)
            old_attempt_row = conn.execute(
                "SELECT * FROM manual_compactions WHERE attempt_id = ? AND state = 'active'",
                (raw.get("attemptId"),),
            ).fetchone()
            if client is None or old_attempt_row is None:
                continue
            old_attempt = _attempt_from_row(old_attempt_row)
            source_ids = [candidate["id"] for candidate in source_messages[:command_index]]
            if (
                client["state"] != "summary_saved"
                or client["attemptId"] != raw.get("attemptId")
                or client["summaryMessageId"] != summary_old
                or client["summaryHash"] != raw.get("summaryHash")
                or old_attempt["threadId"] != message["threadId"]
                or old_attempt["commandMessageId"] != command_old
                or old_attempt["sourceHeadMessageId"] != source_head_old
                or old_attempt["expectedHeadMessageId"] != command_old
                or old_attempt["summaryMessageId"] != summary_old
                or old_attempt["sourceMessageIds"] != source_ids
                or not all(
                    source_id in id_map for source_id in old_attempt["effectiveSourceMessageIds"]
                )
            ):
                continue
            source_hash = canonical_source_hash(source_messages[:command_index])
            _text, _content_json, stored_summary_hash = _strict_summary(message, active = True)
            effective = _effective_branch(
                conn,
                message["threadId"],
                source_messages[:command_index],
                command_message,
            )
            archive_payload, _archive_json, archive_payload_hash = _archive_payload(effective)
            if (
                source_hash != old_attempt["sourceHash"]
                or stored_summary_hash != raw.get("summaryHash")
                or [candidate["id"] for candidate in effective[:-1]]
                != old_attempt["effectiveSourceMessageIds"]
                or archive_payload != old_attempt["archivePayload"]
                or archive_payload_hash != old_attempt["archivePayloadHash"]
            ):
                continue
            expected_active = {
                "schemaVersion": MANUAL_COMPACTION_SCHEMA_VERSION,
                "state": "active",
                "attemptId": old_attempt["attemptId"],
                "threadId": old_attempt["threadId"],
                "revision": old_attempt["revision"],
                "commandMessageId": old_attempt["commandMessageId"],
                "sourceHeadMessageId": old_attempt["sourceHeadMessageId"],
                "summaryMessageId": old_attempt["summaryMessageId"],
                "sourceHash": old_attempt["sourceHash"],
                "requestHash": old_attempt["requestHash"],
                "requestMessageCount": old_attempt["requestMessageCount"],
                "projectInstructionDigest": old_attempt["projectInstructionDigest"],
                "projectInstructionRevision": old_attempt["projectInstructionRevision"],
                "contextDigest": old_attempt["contextDigest"],
                "archivePayloadHash": old_attempt["archivePayloadHash"],
                "outputSummaryHash": old_attempt["outputSummaryHash"],
                "outputFinishReason": old_attempt["outputFinishReason"],
                "summaryHash": old_attempt["summaryHash"],
            }
            if raw != expected_active:
                continue
            _require_command_hash(old_attempt, command_message, state = "summary_saved")
        except ManualCompactionError:
            continue

        next_attempt_id = (
            "fork-"
            + hashlib.sha256(
                f"{new_thread_id}\0{raw.get('attemptId', '')}\0{id_map[summary_old]}".encode()
            ).hexdigest()[:32]
        )
        next_source_hash = canonical_source_hash(cloned[:command_index])
        next_raw = dict(raw)
        next_raw.update(
            {
                "attemptId": next_attempt_id,
                "threadId": new_thread_id,
                "commandMessageId": id_map[command_old],
                "sourceHeadMessageId": id_map[source_head_old],
                "summaryMessageId": id_map[summary_old],
                "sourceHash": next_source_hash,
            }
        )
        next_metadata = dict(metadata)
        next_metadata["manualCompaction"] = next_raw
        cloned[index]["metadata"] = next_metadata
        rewritten[summary_old] = _strict_json_dumps(next_metadata)

        next_envelope = dict(client["envelope"])
        next_envelope.update(
            {
                "attemptId": next_attempt_id,
                "threadId": new_thread_id,
                "commandMessageId": id_map[command_old],
                "expectedHeadMessageId": id_map[command_old],
                "sourceHash": next_source_hash,
            }
        )
        next_client = dict(client)
        next_client.update(
            {
                "threadId": new_thread_id,
                "commandMessageId": id_map[command_old],
                "attemptId": next_attempt_id,
                "summaryMessageId": id_map[summary_old],
                "attemptSequence": 1,
                "expectedHeadMessageId": id_map[command_old],
                "sourceHeadMessageId": id_map[source_head_old],
                "envelope": next_envelope,
            }
        )
        command_metadata = dict(command_message.get("metadata") or {})
        command_metadata.pop("manualCompaction", None)
        command_metadata[MANUAL_COMPACTION_CLIENT_KEY] = next_client
        cloned[command_index]["metadata"] = command_metadata
        rewritten[command_old] = _strict_json_dumps(command_metadata)
    return rewritten
