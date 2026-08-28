# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Focused coverage for trusted project command rules."""

import os
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

from core.agent_workspace import rules
from core.agent_workspace.common import AgentWorkspaceError
from core.agent_workspace.rules import (
    command_matches_pattern,
    discover_project_command_rules,
    evaluate_project_command_rules,
    evaluate_terminal_command_rules,
    split_terminal_command_for_rules,
)
from core.inference import tools
from routes import project_guidance


def _identity(root: Path) -> tuple[int, int]:
    metadata = root.stat()
    return int(metadata.st_dev), int(metadata.st_ino)


def _write_rules(root: Path, name: str, source: str) -> Path:
    path = root / ".codex" / "rules" / name
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(source, encoding = "utf-8")
    return path


@contextmanager
def _workspace_access(root: Path):
    metadata = root.stat()
    yield type(
        "Workspace",
        (),
        {
            "root": root,
            "device_id": metadata.st_dev,
            "file_id": metadata.st_ino,
        },
    )()


def _discover(root: Path, **limits) -> dict:
    return discover_project_command_rules(
        root,
        expected_identity = _identity(root),
        project_trusted = True,
        **limits,
    )


def test_discovers_sorted_rules_and_applies_most_restrictive_decision(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    _write_rules(
        root,
        "20-restrict.rules",
        """
prefix_rule(
    pattern = ["gh", ["pr", "issue"], "view"],
    decision = "prompt",
    justification = "Review remote access.",
    match = ["gh pr view 17", "gh issue view 2"],
    not_match = ["gh pr list", "git status"],
)
prefix_rule(pattern = ["gh", "pr"], decision = "forbidden")
""",
    )
    _write_rules(
        root,
        "10-allow.rules",
        'prefix_rule(pattern = ["gh"], justification = "Known client")\n',
    )

    discovered = _discover(root)

    assert [item["path"] for item in discovered["files"]] == [
        ".codex/rules/10-allow.rules",
        ".codex/rules/20-restrict.rules",
    ]
    assert [item["decision"] for item in discovered["rules"]] == ["allow", "prompt", "forbidden"]
    assert discovered["rules"][1]["pattern"] == ["gh", ["pr", "issue"], "view"]
    assert discovered["rules"][1]["line"] == 2
    assert discovered["bytesRead"] == sum(item["bytes"] for item in discovered["files"])
    assert all(len(item["sha256"]) == 64 for item in discovered["files"])

    evaluated = evaluate_project_command_rules(discovered, ["gh", "pr", "view", "17"])
    assert evaluated["decision"] == "forbidden"
    assert [item["decision"] for item in evaluated["matchedRules"]] == [
        "allow",
        "prompt",
        "forbidden",
    ]
    assert (
        evaluate_project_command_rules(discovered, ["gh", "issue", "view", "2"])["decision"]
        == "prompt"
    )
    assert evaluate_project_command_rules(discovered, ["git", "status"]) == {
        "decision": None,
        "matchedRules": [],
    }


def test_pattern_matching_uses_exact_argv_prefixes_and_unions():
    pattern = ["git", ["status", "diff"]]

    assert command_matches_pattern(["git", "status"], pattern)
    assert command_matches_pattern(["git", "diff", "--stat"], pattern)
    assert not command_matches_pattern(["git"], pattern)
    assert not command_matches_pattern(["sudo", "git", "status"], pattern)
    assert not command_matches_pattern(["git", "commit"], pattern)
    with pytest.raises(AgentWorkspaceError, match = "pattern is invalid"):
        command_matches_pattern(["git", "status"], [object()])


def test_terminal_rule_evaluation_splits_plain_chains_and_nested_shells(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    _write_rules(
        root,
        "default.rules",
        """
prefix_rule(pattern = ["git", "status"], decision = "allow")
prefix_rule(pattern = ["rg"], decision = "allow")
prefix_rule(pattern = ["rm"], decision = "forbidden")
""",
    )
    discovered = _discover(root)

    assert split_terminal_command_for_rules("git status && rg TODO .") == [
        ["git", "status"],
        ["rg", "TODO", "."],
    ]
    assert (
        evaluate_terminal_command_rules(discovered, "git status && rg TODO .")["decision"]
        == "allow"
    )
    blocked = evaluate_terminal_command_rules(discovered, "git status; bash -lc 'rm -rf build'")
    assert blocked["decision"] == "forbidden"
    assert blocked["commands"] == [["git", "status"], ["rm", "-rf", "build"]]


def test_terminal_rule_evaluation_keeps_advanced_shell_as_one_wrapper(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    _write_rules(
        root,
        "default.rules",
        """
prefix_rule(pattern = ["bash"], decision = "prompt")
prefix_rule(pattern = ["rm"], decision = "forbidden")
""",
    )

    evaluated = evaluate_terminal_command_rules(_discover(root), 'rm "$TARGET"')

    assert evaluated["commands"] == [["bash", "-lc", 'rm "$TARGET"']]
    assert evaluated["decision"] == "prompt"


@pytest.mark.parametrize(
    "command",
    [
        "(rm -rf build)",
        "if true; then rm -rf build; fi",
        'for path in build; do rm -rf "$path"; done',
        "! rm -rf build",
    ],
)
def test_terminal_rule_evaluation_falls_back_for_shell_control_syntax(tmp_path, command):
    root = tmp_path / "repository"
    root.mkdir()
    _write_rules(
        root,
        "default.rules",
        """
prefix_rule(pattern = ["bash"], decision = "prompt")
prefix_rule(pattern = ["rm"], decision = "forbidden")
""",
    )

    evaluated = evaluate_terminal_command_rules(_discover(root), command)

    assert evaluated["commands"] == [["bash", "-lc", command]]
    assert evaluated["decision"] == "prompt"


def test_terminal_allow_requires_every_command_in_a_plain_chain_to_match(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    _write_rules(
        root,
        "default.rules",
        'prefix_rule(pattern = ["git", "status"], decision = "allow")\n',
    )

    evaluated = evaluate_terminal_command_rules(_discover(root), "git status && python -m pytest")

    assert evaluated["decision"] is None


def test_full_access_terminal_enforces_forbidden_rule_before_process_setup(monkeypatch):
    monkeypatch.setattr(
        tools,
        "project_terminal_rule_policy",
        lambda *_args, **_kwargs: {
            "decision": "forbidden",
            "matchedRules": [{"justification": "Use the reviewed cleanup command instead."}],
        },
    )
    monkeypatch.setattr(
        tools,
        "_harden_parent_against_proc_env_leak",
        lambda: (_ for _ in ()).throw(AssertionError("process setup must not run")),
    )

    result = tools._bash_exec(
        "rm -rf build",
        session_id = "project-example",
        disable_sandbox = True,
    )

    assert result == ("Blocked by project command rules. Use the reviewed cleanup command instead.")


def test_full_access_project_policy_capability_failure_is_forbidden(monkeypatch):
    monkeypatch.setattr(rules, "secure_command_rule_traversal_supported", lambda: False)

    policy = tools.project_terminal_rule_policy(
        "project-example",
        "git status",
        outside_sandbox = True,
    )

    assert policy["decision"] == "forbidden"
    assert "unavailable on this platform" in policy["error"]


def test_full_access_terminal_rejects_policy_change_between_approval_and_spawn(
    tmp_path, monkeypatch
):
    policies = iter(
        [
            {
                "decision": "allow",
                "matchedRules": [],
                "policyHash": "a" * 64,
            },
            {
                "decision": "prompt",
                "matchedRules": [],
                "policyHash": "b" * 64,
            },
        ]
    )
    monkeypatch.setattr(
        tools,
        "project_terminal_rule_policy",
        lambda *_args, **_kwargs: next(policies),
    )
    monkeypatch.setattr(tools, "_harden_parent_against_proc_env_leak", lambda: True)
    monkeypatch.setattr(tools, "_get_workdir", lambda _session_id: str(tmp_path))
    monkeypatch.setattr(tools, "_tracks_workspace_artifacts", lambda _session_id: False)
    monkeypatch.setattr(tools, "_build_bypass_env", lambda _workdir: {})
    monkeypatch.setattr(tools, "_call_started", lambda _workdir: "call")
    monkeypatch.setattr(tools, "_call_finished", lambda _token: None)
    monkeypatch.setattr(
        tools.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("policy drift reached process creation")
        ),
    )

    result = tools._bash_exec(
        "git status",
        session_id = "project-example",
        disable_sandbox = True,
        project_rule_proof = {"policyHash": "a" * 64, "approved": False},
    )

    assert "Blocked by project command rules" in result
    assert "changed or was not reviewed" in result


def test_full_access_terminal_rechecks_policy_immediately_before_legacy_popen(
    tmp_path, monkeypatch
):
    events = []
    policy = {
        "decision": "allow",
        "matchedRules": [],
        "policyHash": "a" * 64,
    }

    def resolve_policy(*_args, **_kwargs):
        events.append("preflight_policy" if not events else "late_policy")
        return dict(policy)

    def resolve_workdir(_session_id):
        events.append("workdir")
        return str(tmp_path)

    def start_call(_workdir):
        events.append("call_started")
        return "call"

    def build_environment(_workdir):
        events.append("environment")
        return {}

    def popen(*_args, **_kwargs):
        events.append("popen")
        raise OSError("popen reached")

    def finish_call(_token):
        events.append("call_finished")

    monkeypatch.setattr(tools, "project_terminal_rule_policy", resolve_policy)
    monkeypatch.setattr(tools, "_harden_parent_against_proc_env_leak", lambda: True)
    monkeypatch.setattr(tools, "_get_workdir", resolve_workdir)
    monkeypatch.setattr(tools, "_tracks_workspace_artifacts", lambda _session_id: False)
    monkeypatch.setattr(tools, "_build_bypass_env", build_environment)
    monkeypatch.setattr(tools, "_call_started", start_call)
    monkeypatch.setattr(tools, "_call_finished", finish_call)
    monkeypatch.setattr(tools.subprocess, "Popen", popen)

    result = tools._bash_exec(
        "git status",
        session_id = "project-example",
        disable_sandbox = True,
        project_rule_proof = {
            "policyHash": policy["policyHash"],
            "command": "git status",
            "argv": list(tools._get_shell_cmd("git status")),
            "approved": False,
        },
    )

    assert "Execution error: popen reached" in result
    assert events == [
        "preflight_policy",
        "workdir",
        "call_started",
        "environment",
        "late_policy",
        "popen",
        "call_finished",
    ]


def test_execute_tool_forwards_policy_proof_only_to_terminal(monkeypatch):
    observed = {}

    def bash(
        command,
        *_args,
        project_rule_proof = None,
        **_kwargs,
    ):
        observed["terminal"] = (command, project_rule_proof)
        return "terminal-ok"

    def python(code, *_args, **kwargs):
        assert "project_rule_proof" not in kwargs
        observed["python"] = code
        return "python-ok"

    monkeypatch.setattr(tools, "_bash_exec", bash)
    monkeypatch.setattr(tools, "_python_exec", python)
    proof = {"policyHash": "a" * 64, "approved": True}

    assert (
        tools.execute_tool(
            "terminal",
            {"command": "git status"},
            project_rule_proof = proof,
        )
        == "terminal-ok"
    )
    assert tools.execute_tool("python", {"code": "print(1)"}) == "python-ok"
    assert observed == {"terminal": ("git status", proof), "python": "print(1)"}


def test_project_rules_route_returns_only_the_bounded_snapshot(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    _write_rules(root, "default.rules", 'prefix_rule(pattern = ["git", "status"])\n')
    monkeypatch.setattr(
        project_guidance,
        "get_chat_project",
        lambda _project_id: {"id": "rules-project", "archived": False},
    )
    monkeypatch.setattr(
        project_guidance,
        "project_workspace_access",
        lambda _project_id: _workspace_access(root),
    )

    response = project_guidance.project_rules("rules-project", _current_subject = "tester")

    assert response["trusted"] is True
    assert response["rules"][0]["pattern"] == ["git", "status"]
    assert response["rules"][0]["decision"] == "allow"
    assert response["files"][0]["path"] == ".codex/rules/default.rules"


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("value = 1\n", "may contain only"),
        ("other(pattern = ['git'])\n", "only prefix_rule"),
        ("prefix_rule(['git'])\n", "named fields"),
        ("prefix_rule()\n", "missing pattern"),
        ("prefix_rule(pattern = [])\n", "non-empty"),
        ("prefix_rule(pattern = ['git', []])\n", "must not be empty"),
        ("prefix_rule(pattern = ['git'], decision = 'maybe')\n", "decision must"),
        ("prefix_rule(pattern = ['git'], extra = 'value')\n", "unsupported field"),
        (
            "prefix_rule(pattern = ['git'], match = ['gh status'])\n",
            "match example does not match",
        ),
        (
            "prefix_rule(pattern = ['git'], not_match = ['git status'])\n",
            "not_match example matches",
        ),
        ("prefix_rule(pattern = build_pattern())\n", "pattern must"),
        ("prefix_rule(pattern = [f'{1}'])\n", "string literals"),
    ],
)
def test_rejects_non_rule_starlark_and_invalid_fields(tmp_path, source, message):
    root = tmp_path / "repository"
    root.mkdir()
    _write_rules(root, "invalid.rules", source)

    with pytest.raises(AgentWorkspaceError, match = message):
        _discover(root)


def test_parser_never_executes_rule_source(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    marker = tmp_path / "executed"
    _write_rules(
        root,
        "malicious.rules",
        f"__import__('pathlib').Path({str(marker)!r}).write_text('bad')\n",
    )

    with pytest.raises(AgentWorkspaceError, match = "only prefix_rule"):
        _discover(root)

    assert not marker.exists()


def test_requires_explicit_project_trust_and_exact_root_identity(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    _write_rules(root, "default.rules", "prefix_rule(pattern = ['git'])\n")
    identity = _identity(root)

    with pytest.raises(AgentWorkspaceError, match = "trusted project"):
        discover_project_command_rules(
            root,
            expected_identity = identity,
            project_trusted = False,
        )
    with pytest.raises(AgentWorkspaceError, match = "identity changed"):
        discover_project_command_rules(
            root,
            expected_identity = (identity[0], identity[1] + 1),
            project_trusted = True,
        )


def test_missing_rule_directories_return_an_empty_trusted_snapshot(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()

    discovered = _discover(root)
    assert discovered == {
        "trusted": True,
        "rules": [],
        "files": [],
        "bytesRead": 0,
        "policyHash": discovered["policyHash"],
    }
    assert len(discovered["policyHash"]) == 64
    (root / ".codex").mkdir()
    assert _discover(root)["rules"] == []


def test_fails_closed_when_secure_traversal_is_unavailable(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    monkeypatch.setattr(rules, "secure_command_rule_traversal_supported", lambda: False)

    with pytest.raises(AgentWorkspaceError, match = "not available on this platform"):
        _discover(root)


@pytest.mark.skipif(os.name == "nt", reason = "POSIX symbolic links required")
def test_rejects_symbolic_link_roots_directories_and_rule_files(tmp_path):
    real_root = tmp_path / "real"
    real_root.mkdir()
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory = True)
    with pytest.raises(AgentWorkspaceError, match = "Symbolic-link project roots"):
        discover_project_command_rules(
            linked_root,
            expected_identity = _identity(real_root),
            project_trusted = True,
        )

    external = tmp_path / "external"
    external.mkdir()
    (real_root / ".codex").symlink_to(external, target_is_directory = True)
    with pytest.raises(AgentWorkspaceError, match = "unavailable or unsafe"):
        _discover(real_root)

    (real_root / ".codex").unlink()
    rules_directory = real_root / ".codex" / "rules"
    rules_directory.mkdir(parents = True)
    external_rule = tmp_path / "external.rules"
    external_rule.write_text("prefix_rule(pattern = ['rm'], decision = 'forbidden')\n")
    (rules_directory / "linked.rules").symlink_to(external_rule)
    with pytest.raises(AgentWorkspaceError, match = "unavailable or unsafe"):
        _discover(real_root)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason = "POSIX FIFO required")
def test_rule_fifo_fails_without_blocking(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    path = root / ".codex" / "rules" / "pipe.rules"
    path.parent.mkdir(parents = True)
    os.mkfifo(path)
    outcome = {}

    def discover():
        try:
            outcome["value"] = _discover(root)
        except BaseException as exc:
            outcome["error"] = exc

    worker = threading.Thread(target = discover, daemon = True)
    worker.start()
    worker.join(timeout = 1)

    assert not worker.is_alive(), "rule FIFO traversal blocked"
    assert isinstance(outcome.get("error"), AgentWorkspaceError)
    assert "regular files" in str(outcome["error"])


def test_fails_closed_when_files_bytes_rules_or_entries_exceed_bounds(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    _write_rules(root, "a.rules", "prefix_rule(pattern = ['git'])\n")
    _write_rules(root, "b.rules", "prefix_rule(pattern = ['gh'])\n")

    with pytest.raises(AgentWorkspaceError, match = "file limit"):
        _discover(root, max_files = 1)
    with pytest.raises(AgentWorkspaceError, match = "byte limit"):
        _discover(root, max_file_bytes = 4)
    with pytest.raises(AgentWorkspaceError, match = "byte limit"):
        _discover(root, max_total_bytes = 40)
    with pytest.raises(AgentWorkspaceError, match = "rule limit"):
        _discover(root, max_rules = 1)
    with pytest.raises(AgentWorkspaceError, match = "entry limit"):
        _discover(root, max_directory_entries = 1)


def test_detects_project_root_drift_during_discovery(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    _write_rules(root, "default.rules", "prefix_rule(pattern = ['git'])\n")
    original_read = rules._read_rule_file
    moved = tmp_path / "moved"

    def read_then_replace(*args, **kwargs):
        result = original_read(*args, **kwargs)
        root.rename(moved)
        root.mkdir()
        return result

    monkeypatch.setattr(rules, "_read_rule_file", read_then_replace)

    with pytest.raises(AgentWorkspaceError, match = "root identity changed"):
        _discover(root)


def test_evaluator_rejects_untrusted_snapshots_and_invalid_argv():
    with pytest.raises(AgentWorkspaceError, match = "not from a trusted"):
        evaluate_project_command_rules({"trusted": False, "rules": []}, ["git"])
    with pytest.raises(AgentWorkspaceError, match = "arguments are invalid"):
        evaluate_project_command_rules({"trusted": True, "rules": []}, "git status")
    with pytest.raises(AgentWorkspaceError, match = "arguments are invalid"):
        evaluate_project_command_rules({"trusted": True, "rules": []}, [])
    with pytest.raises(AgentWorkspaceError, match = "arguments are invalid"):
        evaluate_project_command_rules({"trusted": True, "rules": []}, ["git", "bad\x00arg"])
