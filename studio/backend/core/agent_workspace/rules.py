# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Trusted project command rules with bounded, non-executing discovery."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shlex
import stat
from pathlib import Path
from typing import Optional, Sequence

from .common import AgentWorkspaceError


MAX_RULE_FILES = 32
MAX_RULE_DIRECTORY_ENTRIES = 256
MAX_RULE_FILE_BYTES = 64 * 1024
MAX_RULE_TOTAL_BYTES = 256 * 1024
MAX_RULES = 256
MAX_PATTERN_TOKENS = 64
MAX_PATTERN_ALTERNATIVES = 32
MAX_TOKEN_CHARACTERS = 4096
MAX_EXAMPLES_PER_KIND = 64
MAX_EXAMPLE_CHARACTERS = 8192
MAX_JUSTIFICATION_CHARACTERS = 4096

_DECISION_PRIORITY = {"allow": 0, "prompt": 1, "forbidden": 2}
_ALLOWED_FIELDS = frozenset({"pattern", "decision", "justification", "match", "not_match"})
_SHELL_NAMES = frozenset({"bash", "sh", "zsh"})
_SIMPLE_SEPARATORS = frozenset({";", "&&", "||", "|", "&"})
_ADVANCED_SHELL = re.compile(
    r"(?:\$|`|[()<>*?{}\[\]]|(?:^|[;&|\s])[A-Za-z_][A-Za-z0-9_]*=)",
)
_SHELL_CONTROL_WORDS = frozenset(
    {
        "!",
        "[[",
        "]]",
        "case",
        "coproc",
        "do",
        "done",
        "elif",
        "else",
        "esac",
        "fi",
        "for",
        "function",
        "if",
        "in",
        "select",
        "then",
        "time",
        "until",
        "while",
    }
)


def _policy_hash(files: Sequence[dict]) -> str:
    payload = [
        {
            "path": str(item.get("path") or ""),
            "sha256": str(item.get("sha256") or ""),
        }
        for item in files
    ]
    encoded = json.dumps(payload, separators = (",", ":"), sort_keys = True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def secure_command_rule_traversal_supported() -> bool:
    """Return whether descriptor-relative rule discovery is available."""
    return (
        os.name != "nt"
        and os.open in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
        and os.scandir in os.supports_fd
        and hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NONBLOCK")
        and hasattr(os, "O_NOFOLLOW")
    )


def _open_flags(*flags: int) -> int:
    value = 0
    for flag in flags:
        value |= flag
    value |= getattr(os, "O_CLOEXEC", 0)
    return value


def _normalize_identity(expected_identity: tuple[int, int]) -> tuple[int, int]:
    try:
        identity = int(expected_identity[0]), int(expected_identity[1])
    except (IndexError, TypeError, ValueError) as exc:
        raise AgentWorkspaceError("Project root identity is invalid.") from exc
    if min(identity) < 0:
        raise AgentWorkspaceError("Project root identity is invalid.")
    return identity


def _open_verified_root(root: Path, expected_identity: tuple[int, int]) -> tuple[Path, int]:
    descriptor = None
    try:
        if root.is_symlink():
            raise AgentWorkspaceError("Symbolic-link project roots are not supported.")
        resolved = root.resolve(strict = True)
        descriptor = os.open(
            resolved,
            _open_flags(os.O_RDONLY, os.O_DIRECTORY, os.O_NOFOLLOW),
        )
        path_metadata = resolved.stat(follow_symlinks = False)
        opened_metadata = os.fstat(descriptor)
        path_identity = (int(path_metadata.st_dev), int(path_metadata.st_ino))
        opened_identity = (int(opened_metadata.st_dev), int(opened_metadata.st_ino))
        if (
            not stat.S_ISDIR(path_metadata.st_mode)
            or not stat.S_ISDIR(opened_metadata.st_mode)
            or path_identity != opened_identity
            or opened_identity != _normalize_identity(expected_identity)
        ):
            raise AgentWorkspaceError("Project root identity changed.")
        return resolved, descriptor
    except AgentWorkspaceError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise AgentWorkspaceError("Project command rules are unavailable or unsafe.") from exc


def _open_directory_at(parent_fd: int, name: str) -> int:
    descriptor = os.open(
        name,
        _open_flags(os.O_RDONLY, os.O_DIRECTORY, os.O_NOFOLLOW),
        dir_fd = parent_fd,
    )
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise NotADirectoryError(name)
    return descriptor


def _identity(metadata: os.stat_result) -> tuple[int, int]:
    return int(metadata.st_dev), int(metadata.st_ino)


def _verify_directory_at(parent_fd: int, name: str, expected_fd: int) -> None:
    current = None
    try:
        current = _open_directory_at(parent_fd, name)
        if _identity(os.fstat(current)) != _identity(os.fstat(expected_fd)):
            raise AgentWorkspaceError("Project command rule directory changed during discovery.")
    except AgentWorkspaceError:
        raise
    except OSError as exc:
        raise AgentWorkspaceError(
            "Project command rule directory changed during discovery."
        ) from exc
    finally:
        if current is not None:
            os.close(current)


def _safe_rule_name(name: str) -> bool:
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        return False
    try:
        name.encode("utf-8", errors = "strict")
    except UnicodeEncodeError:
        return False
    return not any(ord(character) < 32 or ord(character) == 127 for character in name)


def _rule_file_names(rules_fd: int, max_directory_entries: int) -> list[str]:
    names: list[str] = []
    count = 0
    try:
        with os.scandir(rules_fd) as entries:
            for entry in entries:
                count += 1
                if count > max_directory_entries:
                    raise AgentWorkspaceError(
                        "Project command rule directory exceeds the entry limit."
                    )
                if entry.name.endswith(".rules"):
                    if not _safe_rule_name(entry.name):
                        raise AgentWorkspaceError("Project command rule filename is invalid.")
                    names.append(entry.name)
    except AgentWorkspaceError:
        raise
    except OSError as exc:
        raise AgentWorkspaceError("Project command rules could not be enumerated safely.") from exc
    return sorted(names)


def _read_rule_file(
    rules_fd: int, name: str, *, file_limit: int, total_remaining: int
) -> tuple[bytes, os.stat_result]:
    descriptor = None
    try:
        descriptor = os.open(
            name,
            _open_flags(os.O_RDONLY, os.O_NOFOLLOW, os.O_NONBLOCK),
            dir_fd = rules_fd,
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AgentWorkspaceError("Project command rule paths must be regular files.")
        limit = min(file_limit, total_remaining)
        if int(before.st_size) > limit:
            raise AgentWorkspaceError("Project command rules exceed the byte limit.")
        raw = bytearray()
        while len(raw) <= limit:
            chunk = os.read(descriptor, min(8192, limit + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
        if len(raw) > limit:
            raise AgentWorkspaceError("Project command rules exceed the byte limit.")
        before_identity = (
            int(before.st_dev),
            int(before.st_ino),
            int(before.st_size),
            int(before.st_mtime_ns),
            int(before.st_ctime_ns),
        )
        after_identity = (
            int(after.st_dev),
            int(after.st_ino),
            int(after.st_size),
            int(after.st_mtime_ns),
            int(after.st_ctime_ns),
        )
        if before_identity != after_identity:
            raise AgentWorkspaceError("Project command rule file changed during discovery.")
        path_metadata = os.stat(name, dir_fd = rules_fd, follow_symlinks = False)
        if not stat.S_ISREG(path_metadata.st_mode) or _identity(path_metadata) != _identity(after):
            raise AgentWorkspaceError("Project command rule file changed during discovery.")
        return bytes(raw), after
    except AgentWorkspaceError:
        raise
    except OSError as exc:
        raise AgentWorkspaceError("Project command rule file is unavailable or unsafe.") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _string(
    node: ast.AST,
    field: str,
    *,
    maximum: int = MAX_TOKEN_CHARACTERS,
) -> str:
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        raise ValueError(f"{field} must contain only string literals")
    value = node.value
    if not value or len(value) > maximum or "\x00" in value:
        raise ValueError(f"{field} contains an invalid string")
    return value


def _string_list(
    node: ast.AST, field: str, *, maximum_items: int, maximum_characters: int
) -> tuple[str, ...]:
    if not isinstance(node, (ast.List, ast.Tuple)):
        raise ValueError(f"{field} must be a list of string literals")
    if len(node.elts) > maximum_items:
        raise ValueError(f"{field} exceeds its item limit")
    return tuple(_string(item, field, maximum = maximum_characters) for item in node.elts)


def _pattern(node: ast.AST) -> tuple[tuple[str, ...], ...]:
    if not isinstance(node, (ast.List, ast.Tuple)) or not node.elts:
        raise ValueError("pattern must be a non-empty list")
    if len(node.elts) > MAX_PATTERN_TOKENS:
        raise ValueError("pattern exceeds its token limit")
    positions: list[tuple[str, ...]] = []
    for item in node.elts:
        if isinstance(item, ast.Constant):
            positions.append((_string(item, "pattern"),))
            continue
        alternatives = _string_list(
            item,
            "pattern",
            maximum_items = MAX_PATTERN_ALTERNATIVES,
            maximum_characters = MAX_TOKEN_CHARACTERS,
        )
        if not alternatives:
            raise ValueError("pattern alternatives must not be empty")
        positions.append(tuple(dict.fromkeys(alternatives)))
    return tuple(positions)


def _display_pattern(pattern: tuple[tuple[str, ...], ...]) -> list[object]:
    return [position[0] if len(position) == 1 else list(position) for position in pattern]


def command_matches_pattern(argv: Sequence[str], pattern: Sequence[object]) -> bool:
    """Match one already-tokenized command against an exact argument prefix."""
    command = tuple(argv)
    if len(command) < len(pattern):
        return False
    for index, raw_position in enumerate(pattern):
        if isinstance(raw_position, str):
            alternatives = (raw_position,)
        elif isinstance(raw_position, (list, tuple)) and all(
            isinstance(value, str) for value in raw_position
        ):
            alternatives = tuple(raw_position)
        else:
            raise AgentWorkspaceError("Project command rule pattern is invalid.")
        if not alternatives or command[index] not in alternatives:
            return False
    return True


def _parse_example(example: str, field: str) -> tuple[str, ...]:
    if any(ord(character) < 32 and character not in {"\t"} for character in example):
        raise ValueError(f"{field} contains control characters")
    try:
        argv = tuple(shlex.split(example, posix = True))
    except ValueError as exc:
        raise ValueError(f"{field} contains an invalid command example") from exc
    if not argv:
        raise ValueError(f"{field} contains an empty command example")
    return argv


def _parse_rule(call: ast.Call, path: str, index: int) -> dict:
    if not isinstance(call.func, ast.Name) or call.func.id != "prefix_rule" or call.args:
        raise ValueError("only prefix_rule(...) calls with named fields are allowed")
    values: dict[str, ast.AST] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            raise ValueError("expanded prefix_rule fields are not allowed")
        if keyword.arg not in _ALLOWED_FIELDS:
            raise ValueError(f"prefix_rule has unsupported field {keyword.arg!r}")
        if keyword.arg in values:
            raise ValueError(f"prefix_rule repeats field {keyword.arg!r}")
        values[keyword.arg] = keyword.value
    if "pattern" not in values:
        raise ValueError("prefix_rule is missing pattern")

    pattern = _pattern(values["pattern"])
    decision = _string(values["decision"], "decision") if "decision" in values else "allow"
    if decision not in _DECISION_PRIORITY:
        raise ValueError("decision must be allow, prompt, or forbidden")
    justification: Optional[str] = None
    if "justification" in values:
        justification = _string(
            values["justification"],
            "justification",
            maximum = MAX_JUSTIFICATION_CHARACTERS,
        )
        if any(
            ord(character) < 32 and character not in {"\t", "\n"} for character in justification
        ):
            raise ValueError("justification contains control characters")
    matches = (
        _string_list(
            values["match"],
            "match",
            maximum_items = MAX_EXAMPLES_PER_KIND,
            maximum_characters = MAX_EXAMPLE_CHARACTERS,
        )
        if "match" in values
        else ()
    )
    not_matches = (
        _string_list(
            values["not_match"],
            "not_match",
            maximum_items = MAX_EXAMPLES_PER_KIND,
            maximum_characters = MAX_EXAMPLE_CHARACTERS,
        )
        if "not_match" in values
        else ()
    )
    displayed_pattern = _display_pattern(pattern)
    for example in matches:
        if not command_matches_pattern(_parse_example(example, "match"), displayed_pattern):
            raise ValueError(f"match example does not match pattern: {example!r}")
    for example in not_matches:
        if command_matches_pattern(_parse_example(example, "not_match"), displayed_pattern):
            raise ValueError(f"not_match example matches pattern: {example!r}")

    return {
        "id": f"{path}:{index}",
        "path": path,
        "line": int(getattr(call, "lineno", 1)),
        "pattern": displayed_pattern,
        "decision": decision,
        "justification": justification,
        "match": list(matches),
        "notMatch": list(not_matches),
    }


def _parse_rule_file(raw: bytes, path: str) -> list[dict]:
    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AgentWorkspaceError(f"{path} must be valid UTF-8.") from exc
    try:
        module = ast.parse(source, filename = path, mode = "exec")
    except SyntaxError as exc:
        line = int(exc.lineno or 1)
        raise AgentWorkspaceError(f"{path}:{line} has invalid rule syntax.") from exc
    rules: list[dict] = []
    for statement in module.body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            line = int(getattr(statement, "lineno", 1))
            raise AgentWorkspaceError(f"{path}:{line} may contain only prefix_rule(...) calls.")
        try:
            rules.append(_parse_rule(statement.value, path, len(rules) + 1))
        except ValueError as exc:
            line = int(getattr(statement, "lineno", 1))
            raise AgentWorkspaceError(f"{path}:{line} {exc}.") from exc
    return rules


def discover_project_command_rules(
    root: Path | str,
    *,
    expected_identity: tuple[int, int],
    project_trusted: bool,
    max_files: int = MAX_RULE_FILES,
    max_directory_entries: int = MAX_RULE_DIRECTORY_ENTRIES,
    max_file_bytes: int = MAX_RULE_FILE_BYTES,
    max_total_bytes: int = MAX_RULE_TOTAL_BYTES,
    max_rules: int = MAX_RULES,
) -> dict:
    """Load trusted `.codex/rules/*.rules` files as a stable bounded snapshot."""
    if not project_trusted:
        raise AgentWorkspaceError("Project command rules require a trusted project workspace.")
    if (
        min(
            max_files,
            max_directory_entries,
            max_file_bytes,
            max_total_bytes,
            max_rules,
        )
        <= 0
    ):
        raise AgentWorkspaceError("Project command rule limits must be positive.")
    if not secure_command_rule_traversal_supported():
        raise AgentWorkspaceError(
            "Secure project command rules are not available on this platform yet."
        )

    resolved, root_fd = _open_verified_root(Path(root), expected_identity)
    codex_fd = rules_fd = None
    try:
        try:
            codex_fd = _open_directory_at(root_fd, ".codex")
        except FileNotFoundError:
            return {
                "trusted": True,
                "rules": [],
                "files": [],
                "bytesRead": 0,
                "policyHash": _policy_hash([]),
            }
        try:
            rules_fd = _open_directory_at(codex_fd, "rules")
        except FileNotFoundError:
            return {
                "trusted": True,
                "rules": [],
                "files": [],
                "bytesRead": 0,
                "policyHash": _policy_hash([]),
            }
        names = _rule_file_names(rules_fd, max_directory_entries)
        if len(names) > max_files:
            raise AgentWorkspaceError("Project command rules exceed the file limit.")

        loaded_rules: list[dict] = []
        files: list[dict] = []
        file_identities: dict[str, tuple[int, int]] = {}
        used = 0
        for name in names:
            raw, metadata = _read_rule_file(
                rules_fd,
                name,
                file_limit = max_file_bytes,
                total_remaining = max_total_bytes - used,
            )
            path = f".codex/rules/{name}"
            parsed = _parse_rule_file(raw, path)
            if len(loaded_rules) + len(parsed) > max_rules:
                raise AgentWorkspaceError("Project command rules exceed the rule limit.")
            loaded_rules.extend(parsed)
            used += len(raw)
            file_identities[name] = _identity(metadata)
            files.append(
                {
                    "path": path,
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "ruleCount": len(parsed),
                }
            )

        _verify_directory_at(root_fd, ".codex", codex_fd)
        _verify_directory_at(codex_fd, "rules", rules_fd)
        for name, expected in file_identities.items():
            try:
                metadata = os.stat(name, dir_fd = rules_fd, follow_symlinks = False)
            except OSError as exc:
                raise AgentWorkspaceError(
                    "Project command rule file changed during discovery."
                ) from exc
            if not stat.S_ISREG(metadata.st_mode) or _identity(metadata) != expected:
                raise AgentWorkspaceError("Project command rule file changed during discovery.")
        final_root = resolved.stat(follow_symlinks = False)
        if _identity(final_root) != _identity(os.fstat(root_fd)):
            raise AgentWorkspaceError("Project root identity changed during discovery.")
        return {
            "trusted": True,
            "rules": loaded_rules,
            "files": files,
            "bytesRead": used,
            "policyHash": _policy_hash(files),
        }
    except AgentWorkspaceError:
        raise
    except OSError as exc:
        raise AgentWorkspaceError("Project command rules are unavailable or unsafe.") from exc
    finally:
        if rules_fd is not None:
            os.close(rules_fd)
        if codex_fd is not None:
            os.close(codex_fd)
        os.close(root_fd)


def evaluate_project_command_rules(discovered: dict, argv: Sequence[str]) -> dict:
    """Evaluate one argv vector and return all matches plus the strictest decision."""
    if not discovered.get("trusted"):
        raise AgentWorkspaceError("Project command rules are not from a trusted workspace.")
    if (
        isinstance(argv, (str, bytes))
        or not argv
        or not all(isinstance(value, str) for value in argv)
        or not argv[0]
        or any("\x00" in value for value in argv)
    ):
        raise AgentWorkspaceError("Command arguments are invalid.")
    matches = [
        rule
        for rule in discovered.get("rules") or []
        if command_matches_pattern(argv, rule.get("pattern") or [])
    ]
    decision = None
    if matches:
        decision = max(matches, key = lambda rule: _DECISION_PRIORITY[rule["decision"]])["decision"]
    return {
        "decision": decision,
        "matchedRules": matches,
    }


def _shell_payload_index(argv: Sequence[str]) -> Optional[int]:
    if not argv or os.path.basename(argv[0]).casefold() not in _SHELL_NAMES:
        return None
    for index, token in enumerate(argv[1:], start = 1):
        if token == "--":
            return None
        if not token.startswith("-") or token == "-":
            return None
        if token.startswith("--"):
            if token == "--command" and index + 1 < len(argv):
                return index + 1
            continue
        if "c" in token[1:]:
            return index + 1 if index + 1 < len(argv) else None
    return None


def split_terminal_command_for_rules(command: str, *, _depth: int = 0) -> list[list[str]]:
    """Split only plain shell chains into the argv vectors Codex rules evaluate.

    Advanced shell syntax remains one ``bash -lc`` invocation. Rules never
    pretend to understand expansion, redirection, globbing, assignments, or
    control flow that cannot be parsed safely.
    """
    if not isinstance(command, str) or not command.strip() or "\x00" in command or _depth > 4:
        raise AgentWorkspaceError("Terminal command is invalid for project rule evaluation.")
    if "\n" in command or "\r" in command or _ADVANCED_SHELL.search(command):
        return [["bash", "-lc", command]]
    try:
        lexer = shlex.shlex(command, posix = True, punctuation_chars = ";&|")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return [["bash", "-lc", command]]
    if not tokens:
        raise AgentWorkspaceError("Terminal command is invalid for project rule evaluation.")
    if any(token.casefold() in _SHELL_CONTROL_WORDS for token in tokens):
        return [["bash", "-lc", command]]

    commands: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token in _SIMPLE_SEPARATORS or (token and not set(token) - set(";&|")):
            if not current:
                return [["bash", "-lc", command]]
            commands.append(current)
            current = []
            continue
        if token in {"(", ")"}:
            return [["bash", "-lc", command]]
        current.append(token)
    if not current:
        return [["bash", "-lc", command]]
    commands.append(current)

    expanded: list[list[str]] = []
    for argv in commands:
        payload_index = _shell_payload_index(argv)
        if payload_index is None:
            expanded.append(argv)
            continue
        expanded.extend(split_terminal_command_for_rules(argv[payload_index], _depth = _depth + 1))
    return expanded


def evaluate_terminal_command_rules(discovered: dict, command: str) -> dict:
    """Evaluate every safely separable command and keep the strictest result."""
    commands = split_terminal_command_for_rules(command)
    evaluations = [evaluate_project_command_rules(discovered, argv) for argv in commands]
    matches = [rule for result in evaluations for rule in result["matchedRules"]]
    decisions = [result["decision"] for result in evaluations]
    restrictive = [decision for decision in decisions if decision in {"prompt", "forbidden"}]
    if restrictive:
        decision = max(restrictive, key = _DECISION_PRIORITY.__getitem__)
    elif decisions and all(decision == "allow" for decision in decisions):
        decision = "allow"
    else:
        decision = None
    return {
        "decision": decision,
        "commands": commands,
        "matchedRules": matches,
        "policyHash": discovered.get("policyHash"),
    }


__all__ = [
    "MAX_RULE_DIRECTORY_ENTRIES",
    "MAX_RULE_FILE_BYTES",
    "MAX_RULE_FILES",
    "MAX_RULE_TOTAL_BYTES",
    "MAX_RULES",
    "command_matches_pattern",
    "discover_project_command_rules",
    "evaluate_project_command_rules",
    "evaluate_terminal_command_rules",
    "secure_command_rule_traversal_supported",
    "split_terminal_command_for_rules",
]
