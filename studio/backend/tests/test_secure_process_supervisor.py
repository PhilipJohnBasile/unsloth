# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Security contract for project-scoped supervised processes."""

from __future__ import annotations

import contextlib
import inspect
import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from core.agent_workspace import common, execution, mutation, supervisor
from core.agent_workspace.common import AgentWorkspaceError, ProjectWorkspace
from core.agent_workspace.execution import ExecutionBoundaryStatus, ProjectExecutionUnavailable


def test_cancel_during_host_command_revalidation_prevents_popen(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    metadata = root.stat()
    workspace = ProjectWorkspace(
        project_id = "project-cancel",
        root = root,
        kind = "folder",
        device_id = metadata.st_dev,
        file_id = metadata.st_ino,
        revision = 7,
    )
    capability = supervisor._ProjectCommandCapability(
        project_id = workspace.project_id,
        command = "echo safe",
        argv = ("echo", "safe"),
        policy_hash = "a" * 64,
        workspace_identity = (metadata.st_dev, metadata.st_ino),
        workspace_revision = workspace.revision,
        approved = True,
    )
    cancel_event = threading.Event()
    revalidation_started = threading.Event()
    release_revalidation = threading.Event()
    failures = []

    @contextlib.contextmanager
    def workspace_access(project_id):
        assert project_id == workspace.project_id
        yield workspace

    def revalidate(bound_capability, bound_workspace):
        assert bound_capability is capability
        assert bound_workspace is workspace
        revalidation_started.set()
        assert release_revalidation.wait(2)

    def invoke():
        try:
            supervisor._spawn_authorized_project_host_command(
                capability,
                {"cwd": str(root)},
                cancel_event,
            )
        except BaseException as exc:  # noqa: BLE001 - asserted below
            failures.append(exc)

    monkeypatch.setattr(common, "project_workspace_access", workspace_access)
    monkeypatch.setattr(supervisor, "_revalidate_project_command_capability", revalidate)
    monkeypatch.setattr(supervisor, "spawn_on_lifetime_thread", lambda callback: callback())
    monkeypatch.setattr(
        supervisor.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("cancelled host command reached Popen"),
    )

    worker = threading.Thread(target = invoke)
    worker.start()
    assert revalidation_started.wait(2)
    cancel_event.set()
    release_revalidation.set()
    worker.join(2)

    assert not worker.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], InterruptedError)


@pytest.mark.skipif(os.name != "posix", reason = "POSIX execution fence required")
def test_interprocess_finalizer_fence_blocks_reclaim_until_slow_child_is_dead(tmp_path):
    backend = Path(__file__).resolve().parents[1]
    ready = tmp_path / "ready"
    child_dead = tmp_path / "child-dead"
    observed = tmp_path / "observed.json"
    fence_id = "generation:SessionEnd"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(backend), environment.get("PYTHONPATH", "")) if part
    )
    environment["UNSLOTH_STUDIO_HOME"] = str(tmp_path / "studio-home")
    owner_code = textwrap.dedent(
        """
        import subprocess, sys, threading, time
        from pathlib import Path
        from core.agent_workspace import supervisor

        fence_id, ready, child_dead = sys.argv[1:]
        descriptor = supervisor._acquire_project_execution_fence(
            fence_id, threading.Event(), time.monotonic() + 10
        )
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.6)"])
        Path(ready).write_text("ready", encoding="utf-8")
        child.wait(timeout=5)
        Path(child_dead).write_text("dead", encoding="utf-8")
        supervisor._release_project_execution_fence(descriptor)
        """
    )
    reclaimer_code = textwrap.dedent(
        """
        import json, sys, threading, time
        from pathlib import Path
        from core.agent_workspace import supervisor

        fence_id, ready, child_dead, observed = sys.argv[1:]
        deadline = time.monotonic() + 10
        while not Path(ready).exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        started = time.monotonic()
        descriptor = supervisor._acquire_project_execution_fence(
            fence_id, threading.Event(), deadline
        )
        Path(observed).write_text(
            json.dumps({
                "elapsed": time.monotonic() - started,
                "child_dead": Path(child_dead).exists(),
            }),
            encoding="utf-8",
        )
        supervisor._release_project_execution_fence(descriptor)
        """
    )
    owner = subprocess.Popen(
        [sys.executable, "-c", owner_code, fence_id, str(ready), str(child_dead)],
        cwd = str(backend),
        env = environment,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
    )
    reclaimer = subprocess.Popen(
        [
            sys.executable,
            "-c",
            reclaimer_code,
            fence_id,
            str(ready),
            str(child_dead),
            str(observed),
        ],
        cwd = str(backend),
        env = environment,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
    )
    try:
        owner_stdout, owner_stderr = owner.communicate(timeout = 10)
        reclaimer_stdout, reclaimer_stderr = reclaimer.communicate(timeout = 10)
    finally:
        for process in (owner, reclaimer):
            if process.poll() is None:
                process.kill()
                process.wait(timeout = 5)
    assert owner.returncode == 0, (owner_stdout, owner_stderr)
    assert reclaimer.returncode == 0, (reclaimer_stdout, reclaimer_stderr)
    result = json.loads(observed.read_text(encoding = "utf-8"))
    assert result["child_dead"] is True
    assert result["elapsed"] >= 0.4


@pytest.mark.skipif(os.name != "posix", reason = "POSIX execution fence required")
def test_interprocess_finalizer_fence_survives_owner_crash_until_child_is_dead(tmp_path):
    backend = Path(__file__).resolve().parents[1]
    ready = tmp_path / "ready"
    child_dead = tmp_path / "child-dead"
    child_pid = tmp_path / "child-pid"
    observed = tmp_path / "observed.json"
    delivery_id = "crashed-generation:SessionEnd"
    fence_id = json.dumps(
        [delivery_id, "SessionEnd:0:0"],
        separators = (",", ":"),
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(backend), environment.get("PYTHONPATH", "")) if part
    )
    environment["UNSLOTH_STUDIO_HOME"] = str(tmp_path / "studio-home")
    owner_code = textwrap.dedent(
        """
        import os, subprocess, sys, threading, time
        from pathlib import Path
        from core.agent_workspace import supervisor

        fence_id, ready, child_dead, child_pid = sys.argv[1:]
        descriptor = supervisor._acquire_project_execution_fence(
            fence_id, threading.Event(), time.monotonic() + 10
        )
        child_code = (
            "import sys,time; from pathlib import Path; "
            "time.sleep(2); Path(sys.argv[1]).write_text('dead', encoding='utf-8')"
        )
        child = subprocess.Popen(
            [sys.executable, "-c", child_code, child_dead],
            pass_fds=(descriptor,),
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        Path(child_pid).write_text(str(child.pid), encoding="utf-8")
        Path(ready).write_text("ready", encoding="utf-8")
        os._exit(23)
        """
    )
    reclaimer_code = textwrap.dedent(
        """
        import json, sys, threading, time
        from pathlib import Path
        from core.agent_workspace import supervisor

        fence_id, ready, child_dead, observed = sys.argv[1:]
        deadline = time.monotonic() + 10
        while not Path(ready).exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        started = time.monotonic()
        descriptor = supervisor._acquire_project_execution_fence(
            fence_id, threading.Event(), deadline
        )
        Path(observed).write_text(
            json.dumps({
                "elapsed": time.monotonic() - started,
                "child_dead": Path(child_dead).exists(),
            }),
            encoding="utf-8",
        )
        supervisor._release_project_execution_fence(descriptor)
        """
    )
    owner = subprocess.Popen(
        [
            sys.executable,
            "-c",
            owner_code,
            fence_id,
            str(ready),
            str(child_dead),
            str(child_pid),
        ],
        cwd = str(backend),
        env = environment,
        stdin = subprocess.DEVNULL,
        stdout = subprocess.DEVNULL,
        stderr = subprocess.DEVNULL,
    )
    reclaimer = subprocess.Popen(
        [
            sys.executable,
            "-c",
            reclaimer_code,
            fence_id,
            str(ready),
            str(child_dead),
            str(observed),
        ],
        cwd = str(backend),
        env = environment,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
    )
    try:
        assert owner.wait(timeout = 5) == 23
        reclaimer_stdout, reclaimer_stderr = reclaimer.communicate(timeout = 10)
    finally:
        for process in (owner, reclaimer):
            if process.poll() is None:
                process.kill()
                process.wait(timeout = 5)
        if child_pid.exists():
            with contextlib.suppress(OSError, ProcessLookupError, ValueError):
                os.kill(int(child_pid.read_text(encoding = "utf-8")), signal.SIGKILL)
    assert reclaimer.returncode == 0, (reclaimer_stdout, reclaimer_stderr)
    result = json.loads(observed.read_text(encoding = "utf-8"))
    assert result["child_dead"] is True
    assert result["elapsed"] >= 1.5


@pytest.mark.skipif(os.name != "posix", reason = "POSIX execution fence required")
def test_composite_finalizer_fence_blocks_same_handler_retry_not_sibling(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio-home"))
    delivery_id = "generation:SessionEnd"
    retry_fence_id = json.dumps(
        [delivery_id, "SessionEnd:1:4"],
        separators = (",", ":"),
    )
    sibling_fence_id = json.dumps(
        [delivery_id, "SessionEnd:3:5"],
        separators = (",", ":"),
    )
    assert supervisor._project_execution_fence_path(
        retry_fence_id
    ) != supervisor._project_execution_fence_path(sibling_fence_id)

    owner = supervisor._acquire_project_execution_fence(
        retry_fence_id,
        threading.Event(),
        time.monotonic() + 3,
    )
    retry_started = threading.Event()
    retry_acquired = threading.Event()
    sibling_acquired = threading.Event()
    sibling_release = threading.Event()
    retry_release = threading.Event()
    failures = []

    def acquire(fence_id, *, started, acquired, release):
        descriptor = None
        try:
            started.set()
            descriptor = supervisor._acquire_project_execution_fence(
                fence_id,
                threading.Event(),
                time.monotonic() + 3,
            )
            acquired.set()
            assert release.wait(2)
        except BaseException as exc:  # noqa: BLE001 - surface worker failures in the test
            failures.append(exc)
        finally:
            if descriptor is not None:
                supervisor._release_project_execution_fence(descriptor)

    retry = threading.Thread(
        target = acquire,
        kwargs = {
            "fence_id": retry_fence_id,
            "started": retry_started,
            "acquired": retry_acquired,
            "release": retry_release,
        },
    )
    sibling = threading.Thread(
        target = acquire,
        kwargs = {
            "fence_id": sibling_fence_id,
            "started": threading.Event(),
            "acquired": sibling_acquired,
            "release": sibling_release,
        },
    )
    try:
        retry.start()
        sibling.start()
        assert retry_started.wait(1)
        assert sibling_acquired.wait(1)
        assert not retry_acquired.wait(0.1)
        sibling_release.set()
        supervisor._release_project_execution_fence(owner)
        owner = None
        assert retry_acquired.wait(1)
    finally:
        sibling_release.set()
        retry_release.set()
        if owner is not None:
            supervisor._release_project_execution_fence(owner)
        retry.join(2)
        sibling.join(2)
    assert not retry.is_alive()
    assert not sibling.is_alive()
    assert failures == []


def _workspace(root: Path, project_id: str = "supervised-project") -> ProjectWorkspace:
    metadata = root.stat(follow_symlinks = False)
    return ProjectWorkspace(
        project_id = project_id,
        root = root.resolve(strict = True),
        kind = "folder",
        device_id = int(metadata.st_dev),
        file_id = int(metadata.st_ino),
    )


class _LocalBoundary:
    backend = "bubblewrap"

    def __init__(self, workspace: ProjectWorkspace, scratch: Path) -> None:
        self.root = workspace.root
        self.root_identity = (int(workspace.device_id), int(workspace.file_id))
        self.scratch = scratch
        self.closed = False
        self.slot = False

    def acquire_execution_slot(
        self,
        cancel_event,
        deadline = None,
    ) -> bool:
        if not mutation.acquire_workspace_mutation_slot(
            self.root_identity, cancel_event, deadline = deadline
        ):
            return False
        self.slot = True
        return True

    def apply_environment(self, environment):
        isolated = dict(environment)
        isolated.update(
            {
                "HOME": str(self.scratch),
                "TMP": str(self.scratch),
                "TEMP": str(self.scratch),
                "TMPDIR": str(self.scratch),
            }
        )
        return isolated

    def wrap_argv(self, argv):
        return list(argv)

    def popen_kwargs(self, preexec_fn):
        options = {"cwd": str(self.root)}
        if preexec_fn is not None:
            options["preexec_fn"] = preexec_fn
        return options

    def close(self):
        if self.slot:
            mutation.release_workspace_mutation_slot(self.root_identity)
            self.slot = False
        self.closed = True


class _LocalLifecycle:
    instances = []

    def __init__(self) -> None:
        self.closed = False
        self.bound = False
        self.instances.append(self)

    def wrap_argv(self, argv):
        return list(argv)

    def add_popen_options(self, options):
        return dict(options)

    def after_spawn(self, _process):
        return None

    def bind(
        self,
        _process,
        cancel_event = None,
        deadline = None,
    ):
        assert deadline is None or deadline > 0
        self.bound = True
        return cancel_event is None or not cancel_event.is_set()

    def release(self, cancel_event):
        assert self.bound is True
        return cancel_event is None or not cancel_event.is_set()

    def terminate_and_prove(self, process):
        return supervisor._terminate_and_reap(process)

    def close(self):
        self.closed = True


@pytest.fixture
def local_supervisor(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    scratch = tmp_path / "scratch"
    root.mkdir()
    scratch.mkdir()
    workspace = _workspace(root)
    lease_active = {"value": False}
    boundaries = []

    @contextlib.contextmanager
    def access(project_id: str, **_kwargs):
        assert project_id == workspace.project_id
        assert lease_active["value"] is False
        lease_active["value"] = True
        try:
            yield workspace
        finally:
            lease_active["value"] = False

    def open_boundary(received: ProjectWorkspace, **_kwargs):
        assert received is workspace
        boundary = _LocalBoundary(received, scratch)
        boundaries.append(boundary)
        return boundary

    monkeypatch.setattr(common, "project_workspace_access", access)
    monkeypatch.setattr(execution.ProjectExecutionBoundary, "open", open_boundary)
    monkeypatch.setattr(
        supervisor,
        "supervised_process_status",
        lambda: ExecutionBoundaryStatus(True, "bubblewrap", None),
    )
    monkeypatch.setattr(supervisor, "initialize_parent_lifetime", lambda: None)
    monkeypatch.setattr(supervisor, "adopt_pid", lambda _pid: None)
    monkeypatch.setattr(supervisor, "forget_pid", lambda _pid: None)
    monkeypatch.setattr(supervisor, "spawn_on_lifetime_thread", lambda spawn: spawn())
    monkeypatch.setattr(supervisor, "_start_quarantine_retry_owner", lambda: None)
    _LocalLifecycle.instances.clear()
    monkeypatch.setattr(supervisor, "_BubblewrapLifecycle", _LocalLifecycle)
    return workspace, lease_active, boundaries


def test_supervisor_owns_workspace_lease_slot_and_minimal_environment(
    local_supervisor, monkeypatch
):
    workspace, lease_active, boundaries = local_supervisor
    parent_only = {
        "OPENAI_API_KEY": "operator-secret",
        "AWS_PROFILE": "operator-profile",
        "CLOUDSDK_CONFIG": "/operator/gcloud",
        "GIT_CONFIG_GLOBAL": "/operator/gitconfig",
        "LD_PRELOAD": "/operator/inject.so",
        "DYLD_INSERT_LIBRARIES": "/operator/inject.dylib",
        "PYTHONPATH": "/operator/python",
        "PYTHONHOME": "/operator/home",
        "NODE_OPTIONS": "--require=/operator/node.js",
        "BASH_ENV": "/operator/bashrc",
        "SSH_AUTH_SOCK": "/operator/agent.sock",
    }
    for name, value in parent_only.items():
        monkeypatch.setenv(name, value)

    source = (
        "import json, os; "
        "print(json.dumps({key: os.environ.get(key) for key in "
        f"{list(parent_only)!r}}})); "
        "print(os.environ['UNSLOTH_STUDIO_PROJECT_ID'])"
    )
    real_popen = supervisor.subprocess.Popen

    def checked_popen(*args, **kwargs):
        assert lease_active["value"] is True
        assert boundaries[-1].slot is True
        assert kwargs["stdin"] is subprocess.DEVNULL
        assert kwargs["stderr"] is subprocess.STDOUT
        assert kwargs["env"]["HOME"] == str(boundaries[-1].scratch)
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(supervisor.subprocess, "Popen", checked_popen)
    result = supervisor.run_project_process(
        workspace.project_id,
        [sys.executable, "-c", source],
    )

    lines = result.output.splitlines()
    assert result.status == "passed"
    assert result.exit_code == 0
    child_environment = json.loads(lines[0])
    assert {key: child_environment[key] for key in parent_only if key != "PYTHONPATH"} == {
        key: None for key in parent_only if key != "PYTHONPATH"
    }
    assert "/operator/python" not in child_environment["PYTHONPATH"]
    assert str(workspace.root) in child_environment["PYTHONPATH"]
    assert str(boundaries[-1].scratch) in child_environment["PYTHONPATH"]
    assert lines[1] == workspace.project_id
    assert lease_active["value"] is False
    assert boundaries[-1].closed is True
    assert boundaries[-1].slot is False

    assert mutation.acquire_workspace_mutation_slot(boundaries[-1].root_identity) is True
    mutation.release_workspace_mutation_slot(boundaries[-1].root_identity)


def test_project_python_preserves_import_cwd_file_and_streaming_contract(local_supervisor):
    workspace, _lease_active, boundaries = local_supervisor
    (workspace.root / "helper.py").write_text("VALUE = 42\n", encoding = "utf-8")
    streamed = []
    source = (
        "from pathlib import Path; import helper; "
        "print(Path.cwd()); print(helper.VALUE); print(Path(__file__).name); "
        "Path('/mnt/data/remapped.txt').write_text('mapped', encoding='utf-8')"
    )

    result = supervisor.run_project_python(
        workspace.project_id,
        source,
        output_callback = streamed.append,
    )

    assert result.status == "passed"
    assert result.exit_code == 0
    assert "42" in result.output
    assert str(workspace.root) in result.output
    assert "studio_exec_" in result.output
    assert "".join(streamed) == result.output
    assert (workspace.root / "remapped.txt").read_text(encoding = "utf-8") == "mapped"
    assert not list(boundaries[-1].scratch.glob("studio_exec_*.py"))


def test_supervisor_bounds_combined_stdout_and_stderr(local_supervisor):
    workspace, _lease_active, _boundaries = local_supervisor
    streamed = []
    source = (
        "import sys; print('o' * 2048); print('e' * 2048, file=sys.stderr); raise SystemExit(7)"
    )

    result = supervisor.run_project_process(
        workspace.project_id,
        [sys.executable, "-c", source],
        output_limit_bytes = 1024,
        output_callback = streamed.append,
    )

    assert result.status == "failed"
    assert result.exit_code == 7
    assert result.output_bytes >= 4098
    assert len(result.output.removesuffix(result.truncation_notice).encode("utf-8")) == 1024
    assert result.output_truncated is True
    assert result.truncation_notice
    assert "".join(streamed) == result.output
    assert "".join(streamed).count(result.truncation_notice) == 1


def test_supervisor_bounds_rendered_invalid_utf8(local_supervisor):
    workspace, _lease_active, _boundaries = local_supervisor
    streamed = []
    result = supervisor.run_project_process(
        workspace.project_id,
        [sys.executable, "-c", "import os; os.write(1, b'\\xff' * 1024)"],
        output_limit_bytes = 1024,
        output_callback = streamed.append,
    )

    assert result.status == "passed"
    assert result.output_bytes == 1024
    assert len(result.output.removesuffix(result.truncation_notice).encode("utf-8")) <= 1024
    assert result.output_truncated is True
    assert result.output.count(result.truncation_notice) == 1
    assert "".join(streamed) == result.output


def test_slot_wait_is_preparation_and_honors_cancellation_before_popen(
    local_supervisor, monkeypatch
):
    workspace, _lease_active, boundaries = local_supervisor
    identity = (int(workspace.device_id), int(workspace.file_id))
    assert mutation.acquire_workspace_mutation_slot(identity) is True
    cancellation = threading.Event()
    timer = threading.Timer(0.08, cancellation.set)
    monkeypatch.setattr(
        supervisor.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("spawned before acquiring the mutation slot"),
    )
    timer.start()
    try:
        result = supervisor.run_project_process(
            workspace.project_id,
            [sys.executable, "-c", "print('late')"],
            timeout_seconds = 0.5,
            cancel_event = cancellation,
        )
    finally:
        timer.join(timeout = 1)
        mutation.release_workspace_mutation_slot(identity)

    assert result.status == "cancelled"
    assert result.exit_code is None
    assert boundaries[-1].closed is True
    assert "absolute deadline" in supervisor.run_project_process.__doc__


def test_pre_cancel_returns_before_capability_probe(monkeypatch):
    cancellation = threading.Event()
    cancellation.set()
    monkeypatch.setattr(
        supervisor,
        "supervised_process_status",
        lambda: pytest.fail("probed the host for a pre-cancelled command"),
    )

    result = supervisor.run_project_process(
        "project",
        ["echo", "cancelled"],
        cancel_event = cancellation,
    )

    assert result == supervisor.ProjectProcessResult("cancelled", None, "", 0, False)


@pytest.mark.skipif(not hasattr(os, "pipe2"), reason = "POSIX lifecycle pipes")
def test_bubblewrap_block_fd_is_blocking_until_pidfd_binding():
    lifecycle = supervisor._BubblewrapLifecycle()
    try:
        assert os.get_blocking(lifecycle._status_read) is False
        assert os.get_blocking(lifecycle._block_read) is True
        options = lifecycle.add_popen_options({"pass_fds": ()})
        assert lifecycle._status_write in options["pass_fds"]
        assert lifecycle._block_read in options["pass_fds"]
        assert lifecycle._block_write in options["pass_fds"]
    finally:
        lifecycle.close()


@pytest.mark.skipif(not hasattr(os, "pipe2"), reason = "POSIX lifecycle pipes")
def test_bubblewrap_inherits_execution_fence_until_sandbox_exit(tmp_path):
    fence_path = tmp_path / "execution.lock"
    descriptor = os.open(fence_path, os.O_RDWR | os.O_CREAT, 0o600)
    lifecycle = supervisor._BubblewrapLifecycle()
    try:
        lifecycle.attach_execution_fence(descriptor)
        command = lifecycle.wrap_argv(["bwrap", "--", "/bin/true"])
        options = lifecycle.add_popen_options({"pass_fds": ()})
        position = command.index("--sync-fd")
        assert command[position + 1] == str(descriptor)
        assert descriptor in options["pass_fds"]
    finally:
        lifecycle.close()
        os.close(descriptor)


@pytest.mark.skipif(not hasattr(os, "pipe2"), reason = "POSIX lifecycle pipes")
def test_invalid_pidfd_never_counts_as_process_exit():
    descriptor, peer = os.pipe()
    os.close(descriptor)
    try:
        with pytest.raises(
            supervisor.ProjectProcessContainmentError,
            match = "pidfd became invalid",
        ):
            supervisor._BubblewrapLifecycle._pidfd_ready(descriptor, 0)
    finally:
        os.close(peer)


def _install_test_pipe2(monkeypatch) -> None:
    if hasattr(os, "pipe2"):
        return

    def pipe2(flags):
        read_descriptor, write_descriptor = os.pipe()
        if flags & getattr(os, "O_NONBLOCK", 0):
            os.set_blocking(read_descriptor, False)
            os.set_blocking(write_descriptor, False)
        return read_descriptor, write_descriptor

    monkeypatch.setattr(supervisor.os, "pipe2", pipe2, raising = False)


@pytest.mark.skipif(
    os.name != "posix",
    reason = "POSIX process-group setup cleanup",
)
@pytest.mark.parametrize(
    "status_payload",
    [None, b"{malformed status}\n"],
    ids = ["eof", "malformed"],
)
def test_unbound_status_failure_proves_group_before_idempotent_release(status_payload, monkeypatch):
    _install_test_pipe2(monkeypatch)
    lifecycle = supervisor._BubblewrapLifecycle()
    if status_payload is not None:
        os.write(lifecycle._status_write, status_payload)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout = subprocess.PIPE,
        start_new_session = True,
    )
    lifecycle.after_spawn(process)
    calls = {"unbound": 0, "boundary": 0, "lease": 0}
    real_unbound_cleanup = lifecycle.terminate_unbound_and_prove

    def count_unbound_cleanup(received_process):
        calls["unbound"] += 1
        return real_unbound_cleanup(received_process)

    class Boundary:
        def close(self):
            calls["boundary"] += 1
            if calls["boundary"] == 1:
                raise RuntimeError("transient boundary close failure")

    class Lease:
        def __exit__(self, _kind, _value, _traceback):
            calls["lease"] += 1

    monkeypatch.setattr(lifecycle, "terminate_unbound_and_prove", count_unbound_cleanup)
    run = supervisor._QuarantinedRun(
        lease = Lease(),
        boundary = Boundary(),
        lifecycle = lifecycle,
        process = process,
        after_spawn_done = True,
    )
    try:
        with pytest.raises(RuntimeError, match = "transient boundary close failure"):
            supervisor._advance_quarantined_run(run)

        assert process.poll() is not None
        assert lifecycle._released is False
        assert run.tree_proven is True
        assert run.lifecycle_closed is True
        assert run.boundary_closed is False
        assert run.lease_released is False

        supervisor._advance_quarantined_run(run)
        supervisor._advance_quarantined_run(run)

        assert calls == {"unbound": 1, "boundary": 2, "lease": 1}
        assert run.boundary_closed is True
        assert run.lease_released is True
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout = 5)
        lifecycle.close()


@pytest.mark.skipif(
    os.name != "posix",
    reason = "POSIX process-group setup cleanup",
)
def test_unbound_status_failure_keeps_locks_when_group_death_is_unproven(monkeypatch):
    _install_test_pipe2(monkeypatch)
    lifecycle = supervisor._BubblewrapLifecycle()
    process = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        stdout = subprocess.PIPE,
        start_new_session = True,
    )
    lifecycle.after_spawn(process)
    process.wait(timeout = 5)
    released = {"boundary": False, "lease": False}

    class Boundary:
        def close(self):
            released["boundary"] = True

    class Lease:
        def __exit__(self, _kind, _value, _traceback):
            released["lease"] = True

    monkeypatch.setattr(
        lifecycle,
        "_wait_for_process_group_exit",
        lambda *_args, **_kwargs: False,
    )
    run = supervisor._QuarantinedRun(
        lease = Lease(),
        boundary = Boundary(),
        lifecycle = lifecycle,
        process = process,
        after_spawn_done = True,
    )
    try:
        for _attempt in range(2):
            with pytest.raises(
                supervisor.ProjectProcessContainmentError,
                match = "remains quarantined",
            ):
                supervisor._advance_quarantined_run(run)
            assert run.tree_proven is False
            assert released == {"boundary": False, "lease": False}
    finally:
        lifecycle.close()


@pytest.mark.skipif(not hasattr(os, "pipe2"), reason = "POSIX lifecycle pipes")
def test_bubblewrap_bind_latches_cancellation_while_status_is_pending(monkeypatch):
    lifecycle = supervisor._BubblewrapLifecycle()
    cancellation = threading.Event()
    bound = []

    def report_status():
        cancellation.set()
        time.sleep(0.1)
        cancellation.clear()
        time.sleep(0.1)
        os.write(
            lifecycle._status_write,
            b'{"child-pid": 4242, "pid-namespace": 1234}\n',
        )

    monkeypatch.setattr(
        lifecycle,
        "_bind_child",
        lambda process, document: bound.append((process, document)),
    )
    reporter = threading.Thread(target = report_status)
    reporter.start()
    process = object()
    try:
        assert lifecycle.bind(process, cancellation) is False
    finally:
        reporter.join(timeout = 1)
        lifecycle.close()

    assert bound == [(process, {"child-pid": 4242, "pid-namespace": 1234})]


def test_supervisor_times_out_and_releases_only_after_reap(local_supervisor, monkeypatch):
    workspace, _lease_active, boundaries = local_supervisor
    observed = {}
    real_popen = supervisor.subprocess.Popen

    def capture_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        observed["process"] = process
        return process

    monkeypatch.setattr(supervisor.subprocess, "Popen", capture_popen)
    started = time.monotonic()
    result = supervisor.run_project_process(
        workspace.project_id,
        [sys.executable, "-c", "import time; time.sleep(30)"],
        timeout_seconds = 0.1,
    )

    assert time.monotonic() - started < 5
    assert result.status == "timed_out"
    assert result.exit_code is None
    assert observed["process"].poll() is not None
    assert boundaries[-1].closed is True


def test_timeout_during_pre_spawn_revalidation_never_starts_process(local_supervisor, monkeypatch):
    workspace, _lease_active, _boundaries = local_supervisor
    revalidation_started = threading.Event()
    release_revalidation = threading.Event()
    spawned = []

    def before_spawn(_workspace):
        revalidation_started.set()
        release_revalidation.wait(1)

    monkeypatch.setattr(
        supervisor.subprocess,
        "Popen",
        lambda *args, **kwargs: spawned.append((args, kwargs)),
    )
    timer = threading.Thread(
        target = lambda: (
            revalidation_started.wait(1),
            time.sleep(0.05),
            release_revalidation.set(),
        ),
        daemon = True,
    )
    timer.start()
    result = supervisor._run_project_process(
        workspace.project_id,
        (sys.executable, "-c", "raise SystemExit('must not run')"),
        timeout_seconds = 0.01,
        output_limit_bytes = supervisor.DEFAULT_OUTPUT_LIMIT_BYTES,
        cancel_event = None,
        output_callback = None,
        input_bytes = None,
        separate_stderr = False,
        before_spawn = before_spawn,
        absolute_deadline = time.monotonic() + 0.01,
    )
    timer.join(1)
    deadline = time.monotonic() + 1
    while supervisor.retry_quarantined_project_processes() and time.monotonic() < deadline:
        time.sleep(0.01)

    assert result.status == "timed_out"
    assert revalidation_started.is_set()
    assert spawned == []


def test_stuck_security_probe_releases_caller_capacity_before_deadline(
    local_supervisor, monkeypatch
):
    workspace, lease_active, _boundaries = local_supervisor
    entered = threading.Event()
    release = threading.Event()
    spawned = []
    probe_calls = 0
    real_popen = supervisor.subprocess.Popen

    def probe():
        nonlocal probe_calls
        probe_calls += 1
        if probe_calls > 1:
            return ExecutionBoundaryStatus(True, "bubblewrap", None)
        entered.set()
        release.wait(1)
        return ExecutionBoundaryStatus(True, "bubblewrap", None)

    def capture_popen(*args, **kwargs):
        spawned.append(tuple(args[0]))
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(supervisor, "supervised_process_status", probe)
    monkeypatch.setattr(supervisor.subprocess, "Popen", capture_popen)

    result = supervisor.run_project_process(
        "supervised-project",
        [sys.executable, "-c", "raise SystemExit('must not run')"],
        timeout_seconds = 0.02,
    )

    assert entered.is_set()
    assert result.status == "timed_out"
    assert lease_active["value"] is False
    assert spawned == []

    successor = supervisor.run_project_process(
        workspace.project_id,
        [sys.executable, "-c", "print('successor')"],
        timeout_seconds = 1,
    )
    assert successor.status == "passed"
    assert successor.output.strip() == "successor"
    assert len(spawned) == 1

    release.set()
    time.sleep(0.05)
    assert len(spawned) == 1


def test_stuck_boundary_setup_releases_lease_and_closes_late_boundary(
    local_supervisor, monkeypatch
):
    workspace, lease_active, boundaries = local_supervisor
    original_open = execution.ProjectExecutionBoundary.open
    entered = threading.Event()
    release = threading.Event()
    spawned = []
    open_calls = 0
    real_popen = supervisor.subprocess.Popen

    def blocked_open(*args, **kwargs):
        nonlocal open_calls
        open_calls += 1
        if open_calls > 1:
            return original_open(*args, **kwargs)
        entered.set()
        release.wait(1)
        return original_open(*args, **kwargs)

    def capture_popen(*args, **kwargs):
        spawned.append(tuple(args[0]))
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(execution.ProjectExecutionBoundary, "open", blocked_open)
    monkeypatch.setattr(supervisor.subprocess, "Popen", capture_popen)

    result = supervisor.run_project_process(
        workspace.project_id,
        [sys.executable, "-c", "raise SystemExit('must not run')"],
        timeout_seconds = 0.02,
    )

    assert entered.is_set()
    assert result.status == "timed_out"
    assert lease_active["value"] is False
    assert spawned == []

    successor = supervisor.run_project_process(
        workspace.project_id,
        [sys.executable, "-c", "print('successor')"],
        timeout_seconds = 1,
    )
    assert successor.status == "passed"
    assert successor.output.strip() == "successor"
    assert len(spawned) == 1

    release.set()
    deadline = time.monotonic() + 1
    while len(boundaries) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(boundaries) == 2
    assert all(boundary.closed for boundary in boundaries)
    assert len(spawned) == 1


def test_stuck_workspace_lease_enter_returns_by_deadline_and_releases_late_acquisition(
    local_supervisor, monkeypatch
):
    workspace, _lease_active, boundaries = local_supervisor
    entered = threading.Event()
    release = threading.Event()
    exited = threading.Event()
    state = {"active": False, "exits": 0}
    spawned = []

    class BlockedLease:
        def __enter__(self):
            entered.set()
            assert release.wait(2)
            state["active"] = True
            return workspace

        def __exit__(self, *_args):
            assert state["active"] is True
            state["active"] = False
            state["exits"] += 1
            exited.set()

    monkeypatch.setattr(
        common,
        "project_workspace_access",
        lambda _project_id, **_kwargs: BlockedLease(),
    )
    monkeypatch.setattr(
        supervisor.subprocess,
        "Popen",
        lambda *args, **kwargs: spawned.append((args, kwargs)),
    )

    started = time.monotonic()
    try:
        result = supervisor.run_project_process(
            workspace.project_id,
            [sys.executable, "-c", "raise SystemExit('must not run')"],
            timeout_seconds = 0.03,
        )
        assert time.monotonic() - started < 0.5
        assert result.status == "timed_out"
        assert entered.is_set()
        assert boundaries == []
        assert spawned == []
    finally:
        release.set()

    assert exited.wait(1)
    assert state == {"active": False, "exits": 1}
    assert spawned == []


def test_stuck_popen_options_returns_by_deadline_and_closes_late_authority(
    local_supervisor, monkeypatch
):
    workspace, lease_active, boundaries = local_supervisor
    entered = threading.Event()
    release = threading.Event()
    spawned = []
    real_popen_options = supervisor._popen_options

    def blocked_popen_options(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return real_popen_options(*args, **kwargs)

    monkeypatch.setattr(supervisor, "_popen_options", blocked_popen_options)
    monkeypatch.setattr(
        supervisor.subprocess,
        "Popen",
        lambda *args, **kwargs: spawned.append((args, kwargs)),
    )

    started = time.monotonic()
    try:
        result = supervisor.run_project_process(
            workspace.project_id,
            [sys.executable, "-c", "raise SystemExit('must not run')"],
            timeout_seconds = 0.03,
        )
        assert time.monotonic() - started < 0.5
        assert result.status == "timed_out"
        assert entered.is_set()
        assert lease_active["value"] is True
        assert boundaries[-1].closed is False
        assert _LocalLifecycle.instances[-1].closed is False
        assert spawned == []
    finally:
        release.set()

    deadline = time.monotonic() + 1
    while lease_active["value"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert lease_active["value"] is False
    assert boundaries[-1].closed is True
    assert _LocalLifecycle.instances[-1].closed is True
    assert spawned == []


def test_preparation_detach_atomically_claims_result_before_delayed_notification(monkeypatch):
    publish_reached = threading.Event()
    release_notification = threading.Event()
    real_done = threading.Event()
    closed = []

    class DelayedNotification:
        def set(self):
            publish_reached.set()
            release_notification.wait(1)
            real_done.set()

        def wait(self, timeout):
            return real_done.wait(timeout)

    class LateBoundary:
        def close(self):
            closed.append(self)

    boundary = LateBoundary()
    monkeypatch.setattr(
        supervisor,
        "_new_preparation_event",
        lambda: DelayedNotification(),
    )
    try:
        with pytest.raises(TimeoutError, match = "exceeded its deadline"):
            supervisor._bounded_preparation(
                lambda: boundary,
                cancel_event = None,
                deadline = time.monotonic() + 0.02,
                label = "delayed publication",
                close_late_result = lambda value: value.close(),
            )
        assert publish_reached.is_set()
        assert closed == [boundary]
    finally:
        release_notification.set()


def test_supervisor_cancellation_is_bounded_and_deterministic(local_supervisor, monkeypatch):
    workspace, _lease_active, boundaries = local_supervisor
    cancellation = threading.Event()
    spawned = threading.Event()
    real_popen = supervisor.subprocess.Popen

    def capture_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        spawned.set()
        return process

    monkeypatch.setattr(supervisor.subprocess, "Popen", capture_popen)
    timer = threading.Thread(
        target = lambda: (spawned.wait(timeout = 3), cancellation.set()),
        daemon = True,
    )
    timer.start()
    result = supervisor.run_project_process(
        workspace.project_id,
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cancel_event = cancellation,
    )
    timer.join(timeout = 1)

    assert result.status == "cancelled"
    assert result.exit_code is None
    assert boundaries[-1].closed is True


def test_cancellation_during_lifecycle_bind_never_releases_command(local_supervisor, monkeypatch):
    workspace, _lease_active, boundaries = local_supervisor
    cancellation = threading.Event()
    original_bind = _LocalLifecycle.bind
    original_release = _LocalLifecycle.release
    release_called = {"value": False}

    def cancel_during_bind(
        self,
        process,
        cancel_event = None,
        deadline = None,
    ):
        assert cancel_event is cancellation
        cancellation.set()
        bound = original_bind(self, process, cancel_event, deadline)
        cancellation.clear()
        return bound

    def observe_release(self, cancel_event):
        release_called["value"] = True
        return original_release(self, cancel_event)

    monkeypatch.setattr(_LocalLifecycle, "bind", cancel_during_bind)
    monkeypatch.setattr(_LocalLifecycle, "release", observe_release)

    result = supervisor.run_project_process(
        workspace.project_id,
        [sys.executable, "-c", "raise SystemExit('must not run')"],
        cancel_event = cancellation,
    )

    assert result.status == "cancelled"
    assert release_called["value"] is False
    assert boundaries[-1].closed is True


def test_post_spawn_exception_reaps_before_boundary_release(local_supervisor, monkeypatch):
    workspace, _lease_active, boundaries = local_supervisor
    observed = {}
    real_popen = supervisor.subprocess.Popen

    def capture_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        observed["process"] = process
        return process

    monkeypatch.setattr(supervisor.subprocess, "Popen", capture_popen)
    monkeypatch.setattr(
        supervisor,
        "adopt_pid",
        lambda _pid: (_ for _ in ()).throw(RuntimeError("adoption failed")),
    )

    with pytest.raises(RuntimeError, match = "adoption failed"):
        supervisor.run_project_process(
            workspace.project_id,
            [sys.executable, "-c", "import time; time.sleep(30)"],
        )

    assert observed["process"].poll() is not None
    assert boundaries[-1].closed is True
    assert boundaries[-1].slot is False


def test_reaping_failure_quarantines_process_lease_slot_and_descriptors(
    local_supervisor, monkeypatch
):
    workspace, lease_active, boundaries = local_supervisor
    observed = {}
    real_popen = supervisor.subprocess.Popen
    real_terminate = _LocalLifecycle.terminate_and_prove
    failed_once = {"value": False}

    def capture_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        observed["process"] = process
        return process

    def fail_once(self, process):
        if not failed_once["value"]:
            failed_once["value"] = True
            raise RuntimeError("reap proof failed")
        return real_terminate(self, process)

    monkeypatch.setattr(supervisor.subprocess, "Popen", capture_popen)
    monkeypatch.setattr(_LocalLifecycle, "terminate_and_prove", fail_once)

    with pytest.raises(
        supervisor.ProjectProcessContainmentError,
        match = "workspace lease and mutation slot remain locked",
    ):
        supervisor.run_project_process(
            workspace.project_id,
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout_seconds = 0.05,
        )

    assert observed["process"].poll() is None
    assert lease_active["value"] is True
    assert boundaries[-1].closed is False
    assert boundaries[-1].slot is True
    assert _LocalLifecycle.instances[-1].closed is False
    cancelled = threading.Event()
    cancelled.set()
    assert (
        mutation.acquire_workspace_mutation_slot(boundaries[-1].root_identity, cancelled) is False
    )

    assert supervisor.retry_quarantined_project_processes() == 0
    assert observed["process"].poll() is not None
    assert lease_active["value"] is False
    assert boundaries[-1].closed is True
    assert boundaries[-1].slot is False
    assert _LocalLifecycle.instances[-1].closed is True


def test_failed_quarantine_retry_keeps_lease_and_slot_locked(local_supervisor, monkeypatch):
    workspace, lease_active, boundaries = local_supervisor
    real_terminate = _LocalLifecycle.terminate_and_prove
    cleanup_allowed = {"value": False}

    def fail_until_allowed(self, process):
        if not cleanup_allowed["value"]:
            raise RuntimeError("pidfd cleanup is temporarily unavailable")
        return real_terminate(self, process)

    monkeypatch.setattr(_LocalLifecycle, "terminate_and_prove", fail_until_allowed)

    with pytest.raises(supervisor.ProjectProcessContainmentError):
        supervisor.run_project_process(
            workspace.project_id,
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout_seconds = 0.05,
        )

    assert supervisor.retry_quarantined_project_processes() == 1
    assert lease_active["value"] is True
    assert boundaries[-1].slot is True
    assert boundaries[-1].closed is False

    cleanup_allowed["value"] = True
    assert supervisor.retry_quarantined_project_processes() == 0
    assert lease_active["value"] is False
    assert boundaries[-1].slot is False
    assert boundaries[-1].closed is True


@pytest.mark.parametrize("stop_kind", ["cancel", "preparation_timeout"])
def test_stuck_spawn_returns_bounded_but_keeps_lease_and_slot(
    local_supervisor, monkeypatch, stop_kind
):
    workspace, lease_active, boundaries = local_supervisor
    real_popen = supervisor.subprocess.Popen
    entered = threading.Event()
    allow_spawn = threading.Event()
    cancellation = threading.Event()

    def blocked_popen(*args, **kwargs):
        entered.set()
        assert allow_spawn.wait(5)
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(supervisor.subprocess, "Popen", blocked_popen)
    canceller = None
    if stop_kind == "cancel":

        def cancel_after_start():
            assert entered.wait(2)
            cancellation.set()

        canceller = threading.Thread(target = cancel_after_start)
        canceller.start()
    else:
        monkeypatch.setattr(supervisor, "_SPAWN_WAIT_SECONDS", 0.05)
    started = time.monotonic()
    try:
        if stop_kind == "cancel":
            result = supervisor.run_project_process(
                workspace.project_id,
                [sys.executable, "-c", "import time; time.sleep(30)"],
                cancel_event = cancellation,
            )
        else:
            with pytest.raises(
                supervisor.ProjectProcessContainmentError,
                match = "spawn is still resolving",
            ):
                supervisor.run_project_process(
                    workspace.project_id,
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                )
    finally:
        allow_spawn.set()
        if canceller is not None:
            canceller.join(timeout = 2)

    assert time.monotonic() - started < 1
    if stop_kind == "cancel":
        assert result.status == "cancelled"
    assert lease_active["value"] is True
    assert boundaries[-1].slot is True
    with supervisor._QUARANTINE_LOCK:
        run = supervisor._QUARANTINED_RUNS[-1]
    assert run.spawn_attempt is not None
    assert run.spawn_attempt._done.wait(2)
    assert supervisor.retry_quarantined_project_processes() == 0
    assert lease_active["value"] is False
    assert boundaries[-1].slot is False


def test_quarantine_retries_only_the_failed_release_phase(local_supervisor, monkeypatch):
    workspace, lease_active, _boundaries = local_supervisor
    lifecycle_calls = {"terminate": 0, "close": 0}
    boundary_calls = {"close": 0}
    lease_calls = {"exit": 0}
    real_terminate = _LocalLifecycle.terminate_and_prove
    real_lifecycle_close = _LocalLifecycle.close
    real_boundary_close = _LocalBoundary.close

    class FailingLease:
        def __enter__(self):
            assert lease_active["value"] is False
            lease_active["value"] = True
            return workspace

        def __exit__(self, _kind, _value, _traceback):
            lease_calls["exit"] += 1
            if lease_calls["exit"] == 1:
                raise RuntimeError("lease close failed")
            lease_active["value"] = False

    def fail_first_termination(self, process):
        lifecycle_calls["terminate"] += 1
        if lifecycle_calls["terminate"] == 1:
            raise RuntimeError("pidfd proof failed")
        return real_terminate(self, process)

    def count_lifecycle_close(self):
        lifecycle_calls["close"] += 1
        return real_lifecycle_close(self)

    def fail_first_boundary_close(self):
        boundary_calls["close"] += 1
        if boundary_calls["close"] == 1:
            raise RuntimeError("boundary close failed")
        return real_boundary_close(self)

    monkeypatch.setattr(
        common,
        "project_workspace_access",
        lambda _project_id, **_kwargs: FailingLease(),
    )
    monkeypatch.setattr(_LocalLifecycle, "terminate_and_prove", fail_first_termination)
    monkeypatch.setattr(_LocalLifecycle, "close", count_lifecycle_close)
    monkeypatch.setattr(_LocalBoundary, "close", fail_first_boundary_close)

    with pytest.raises(supervisor.ProjectProcessContainmentError):
        supervisor.run_project_process(
            workspace.project_id,
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout_seconds = 0.05,
        )

    assert supervisor.retry_quarantined_project_processes() == 1
    with supervisor._QUARANTINE_LOCK:
        run = supervisor._QUARANTINED_RUNS[-1]
    assert run.tree_proven is True
    assert run.lifecycle_closed is True
    assert run.boundary_closed is False
    assert run.lease_released is False

    assert supervisor.retry_quarantined_project_processes() == 1
    assert run.boundary_closed is True
    assert run.lease_released is False

    assert supervisor.retry_quarantined_project_processes() == 0
    assert lifecycle_calls == {"terminate": 2, "close": 1}
    assert boundary_calls == {"close": 2}
    assert lease_calls == {"exit": 2}
    assert lease_active["value"] is False


def test_normal_service_owner_retries_quarantine_until_release():
    released = threading.Event()
    calls = {"boundary": 0}

    class Lifecycle:
        def close(self):
            return None

    class Boundary:
        def close(self):
            calls["boundary"] += 1
            if calls["boundary"] == 1:
                raise RuntimeError("transient boundary failure")

    class Lease:
        def __exit__(self, _kind, _value, _traceback):
            released.set()

    with supervisor._QUARANTINE_LOCK:
        assert not supervisor._QUARANTINED_RUNS
    supervisor._register_quarantined_run(
        supervisor._QuarantinedRun(
            lease = Lease(),
            boundary = Boundary(),
            lifecycle = Lifecycle(),
            tree_proven = True,
        )
    )

    assert released.wait(2)
    assert calls["boundary"] == 2
    assert supervisor.retry_quarantined_project_processes() == 0


@pytest.mark.parametrize(
    "platform_status",
    [
        ExecutionBoundaryStatus(
            False,
            None,
            "Project command execution is disabled until a Windows filesystem sandbox is available.",
        ),
        ExecutionBoundaryStatus(True, "sandbox-exec", None),
    ],
)
def test_unsupported_process_boundaries_fail_before_workspace_or_popen(
    platform_status, monkeypatch
):
    monkeypatch.setattr(
        execution,
        "execution_boundary_status",
        lambda probe = True: platform_status,
    )
    monkeypatch.setattr(
        common,
        "project_workspace_access",
        lambda _project_id: pytest.fail("resolved a workspace without process containment"),
    )
    monkeypatch.setattr(
        supervisor.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("spawned without process containment"),
    )

    with pytest.raises(ProjectExecutionUnavailable):
        supervisor.run_project_process("project", ["echo", "unsafe"])


def test_failed_exact_lifecycle_probe_precedes_workspace_and_user_popen(monkeypatch):
    monkeypatch.setattr(supervisor.os, "pidfd_open", lambda *_args: 1, raising = False)
    monkeypatch.setattr(
        supervisor.signal,
        "pidfd_send_signal",
        lambda *_args: None,
        raising = False,
    )
    monkeypatch.setattr(
        execution,
        "execution_boundary_status",
        lambda probe = False: ExecutionBoundaryStatus(True, "bubblewrap", None),
    )
    monkeypatch.setattr(execution, "_bubblewrap_path", lambda: "/usr/bin/bwrap")
    monkeypatch.setattr(supervisor, "_bubblewrap_identity", lambda _path: (1, 2, 3, 4))
    monkeypatch.setattr(
        supervisor,
        "_probe_supervised_backend",
        lambda _path, _identity: (False, "pidfd lifecycle probe failed"),
    )
    monkeypatch.setattr(
        common,
        "project_workspace_access",
        lambda _project_id: pytest.fail("resolved a workspace after a failed lifecycle probe"),
    )
    monkeypatch.setattr(
        supervisor.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("spawned the caller's command after probe failure"),
    )

    with pytest.raises(ProjectExecutionUnavailable, match = "pidfd lifecycle probe failed"):
        supervisor.run_project_process("project", ["echo", "unsafe"])


def test_supervisor_rejects_invalid_inputs_before_capability_probe(monkeypatch):
    monkeypatch.setattr(
        supervisor,
        "supervised_process_status",
        lambda: pytest.fail("probed the host before validating caller input"),
    )

    with pytest.raises(AgentWorkspaceError, match = "sequence of strings"):
        supervisor.run_project_process("project", "echo unsafe")
    with pytest.raises(AgentWorkspaceError, match = "invalid item"):
        supervisor.run_project_process("project", ["echo", "bad\x00item"])
    with pytest.raises(AgentWorkspaceError, match = "timeout"):
        supervisor.run_project_process("project", ["echo"], timeout_seconds = float("inf"))
    with pytest.raises(AgentWorkspaceError, match = "output limit"):
        supervisor.run_project_process("project", ["echo"], output_limit_bytes = True)
    with pytest.raises(AgentWorkspaceError, match = "Project id"):
        supervisor.run_project_process(" project\n", ["echo"])


def test_public_api_has_no_root_identity_or_boundary_injection():
    parameters = inspect.signature(supervisor.run_project_process).parameters

    assert set(parameters) == {
        "project_id",
        "argv",
        "timeout_seconds",
        "output_limit_bytes",
        "cancel_event",
        "output_callback",
    }
    assert "root" not in inspect.signature(supervisor.run_project_python).parameters
    assert "identity" not in inspect.signature(supervisor.run_project_python).parameters
    assert "boundary" not in inspect.signature(supervisor.run_project_python).parameters


@pytest.mark.skipif(sys.platform != "win32", reason = "native Windows process boundary")
def test_native_windows_supervisor_fails_before_popen(monkeypatch):
    monkeypatch.setattr(
        supervisor.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("spawned on unsupported Windows containment"),
    )

    with pytest.raises(ProjectExecutionUnavailable, match = "Windows"):
        supervisor.run_project_process("unresolved", ["cmd.exe", "/c", "echo unsafe"])


@pytest.mark.skipif(sys.platform != "darwin", reason = "native macOS process boundary")
def test_native_macos_rejects_sandbox_exec_before_detached_setsid_payload(monkeypatch):
    status = execution.execution_boundary_status()
    if not status.available:
        if os.environ.get("UNSLOTH_SECURE_BOUNDARY_REQUIRED") == "1":
            pytest.fail(status.reason or "the required macOS filesystem boundary is unavailable")
        pytest.skip(status.reason or "macOS sandbox-exec is unavailable")
    assert status.backend == "sandbox-exec"
    monkeypatch.setattr(
        supervisor.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("started the detached setsid payload"),
    )
    payload = (
        "import subprocess, sys; "
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], "
        "start_new_session=True)"
    )

    with pytest.raises(ProjectExecutionUnavailable, match = "detached descendants"):
        supervisor.run_project_process("unresolved", [sys.executable, "-c", payload])


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "native Linux PID namespace")
def test_native_linux_bubblewrap_kills_detached_setsid_descendant(tmp_path, monkeypatch):
    status = supervisor.supervised_process_status()
    if not status.available:
        if os.environ.get("UNSLOTH_SECURE_BOUNDARY_REQUIRED") == "1":
            pytest.fail(status.reason or "the required Linux process boundary is unavailable")
        pytest.skip(status.reason or "Linux bubblewrap is unavailable")

    root = tmp_path / "repository"
    root.mkdir()
    workspace = _workspace(root)

    @contextlib.contextmanager
    def access(project_id: str):
        assert project_id == workspace.project_id
        yield workspace

    monkeypatch.setattr(common, "project_workspace_access", access)
    slot_released = root / "slot-released.txt"
    marker = root / "late-write.txt"
    child_code = (
        "import time; from pathlib import Path; "
        f"released=Path({str(slot_released)!r}); marker=Path({str(marker)!r}); "
        "deadline=time.monotonic()+10; "
        'exec("while time.monotonic() < deadline and not released.exists():\\n'
        '    time.sleep(0.005)"); '
        "marker.write_text('escaped', encoding='utf-8') if released.exists() else None"
    )
    payload = (
        "import subprocess, sys; "
        f"code={child_code!r}; "
        "child=subprocess.Popen([sys.executable, '-c', code], start_new_session=True, "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
        "print(child.pid, flush=True); "
        "import time; time.sleep(30)"
    )

    result = supervisor.run_project_process(
        workspace.project_id,
        [sys.executable, "-c", payload],
        timeout_seconds = 0.2,
    )
    identity = (int(workspace.device_id), int(workspace.file_id))
    assert mutation.acquire_workspace_mutation_slot(identity) is True
    try:
        slot_released.write_text("released", encoding = "utf-8")
    finally:
        mutation.release_workspace_mutation_slot(identity)
    time.sleep(0.5)

    assert result.status == "timed_out", result.output
    assert result.exit_code is None
    assert result.output.strip().isdigit()
    assert not marker.exists()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "native Linux PID namespace")
@pytest.mark.parametrize("release_state", ["before", "after"])
def test_native_linux_owner_sigkill_contains_project_command(tmp_path, release_state):
    status = supervisor.supervised_process_status()
    if not status.available:
        if os.environ.get("UNSLOTH_SECURE_BOUNDARY_REQUIRED") == "1":
            pytest.fail(status.reason or "the required Linux process boundary is unavailable")
        pytest.skip(status.reason or "Linux bubblewrap is unavailable")

    backend = Path(__file__).resolve().parents[1]
    root = tmp_path / "repository"
    record_dir = tmp_path / "child-records"
    ready = tmp_path / "bound.txt"
    marker = root / "must-not-run.txt"
    trigger = root / "owner-dead.txt"
    root.mkdir()
    record_dir.mkdir()
    owner_code = textwrap.dedent(
        """
        import contextlib
        import sys
        import time
        from pathlib import Path

        from core.agent_workspace import common, supervisor
        from core.agent_workspace.common import ProjectWorkspace

        root = Path(sys.argv[1]).resolve(strict=True)
        ready = Path(sys.argv[2])
        marker = Path(sys.argv[3])
        trigger = Path(sys.argv[4])
        release_state = sys.argv[5]
        metadata = root.stat(follow_symlinks=False)
        workspace = ProjectWorkspace(
            project_id="owner-crash-project",
            root=root,
            kind="folder",
            device_id=int(metadata.st_dev),
            file_id=int(metadata.st_ino),
        )

        status = supervisor.supervised_process_status()
        if not status.available:
            raise RuntimeError(status.reason or "supervisor unavailable")

        @contextlib.contextmanager
        def access(project_id):
            assert project_id == workspace.project_id
            yield workspace

        original_release = supervisor._BubblewrapLifecycle.release

        def stop_at_release(self, cancel_event):
            if release_state == "before":
                ready.write_text("bound", encoding="utf-8")
            else:
                original_release(self, cancel_event)
                ready.write_text("released", encoding="utf-8")
            while True:
                time.sleep(0.05)

        common.project_workspace_access = access
        supervisor._BubblewrapLifecycle.release = stop_at_release
        if release_state == "before":
            payload = (
                "from pathlib import Path; "
                f"Path({str(marker)!r}).write_text('released', encoding='utf-8')"
            )
        else:
            payload = (
                "import time; from pathlib import Path; "
                f"trigger=Path({str(trigger)!r}); marker=Path({str(marker)!r}); "
                "deadline=time.monotonic()+10; "
                'exec("while time.monotonic() < deadline and not trigger.exists():\\n'
                '    time.sleep(0.005)"); '
                "marker.write_text('survived', encoding='utf-8') if trigger.exists() else None"
            )
        supervisor.run_project_process(
            workspace.project_id,
            [sys.executable, "-c", payload],
            timeout_seconds=30,
        )
        """
    )
    owner_environment = dict(os.environ)
    owner_environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(backend), owner_environment.get("PYTHONPATH", "")) if part
    )
    owner_environment["UNSLOTH_STUDIO_CHILD_RECORD"] = str(record_dir)
    owner_environment["UNSLOTH_STUDIO_DISABLE_DEVICE_PROBE"] = "1"
    owner = subprocess.Popen(
        [
            sys.executable,
            "-c",
            owner_code,
            str(root),
            str(ready),
            str(marker),
            str(trigger),
            release_state,
        ],
        cwd = str(backend),
        env = owner_environment,
        stderr = subprocess.PIPE,
        stdout = subprocess.PIPE,
        text = True,
    )
    try:
        deadline = time.monotonic() + 15
        while not ready.exists() and owner.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if not ready.exists():
            stdout, stderr = owner.communicate(timeout = 2)
            pytest.fail(
                f"owner did not reach the blocked lifecycle: stdout={stdout!r} stderr={stderr!r}"
            )

        os.kill(owner.pid, signal.SIGKILL)
        owner.wait(timeout = 5)
        if release_state == "after":
            trigger.write_text("owner dead", encoding = "utf-8")
        time.sleep(0.3)
        assert not marker.exists()

        records = list(record_dir.glob("*.json"))
        assert len(records) == 1
        record = json.loads(records[0].read_text(encoding = "utf-8"))
        children = record.get("children")
        assert isinstance(children, list) and len(children) == 1
        recorded_pid = children[0]["pid"]
        recorded_pgid = children[0]["pgid"]
        assert recorded_pid >= 2
        assert recorded_pgid == recorded_pid

        reaper_code = (
            "import json; "
            "from utils.process_lifetime import reap_recorded_children; "
            "print(json.dumps(reap_recorded_children(timeout=2)))"
        )
        reaper = subprocess.run(
            [sys.executable, "-c", reaper_code],
            cwd = str(backend),
            env = owner_environment,
            capture_output = True,
            check = True,
            text = True,
            timeout = 10,
        )
        if release_state == "before":
            assert recorded_pid in json.loads(reaper.stdout)
        group_deadline = time.monotonic() + 5
        while time.monotonic() < group_deadline:
            try:
                os.killpg(recorded_pgid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            pytest.fail("the recorded blocked process group survived startup recovery")
        assert not list(record_dir.glob("*.json"))
        assert not marker.exists()
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout = 5)
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from utils.process_lifetime import reap_recorded_children; "
                "reap_recorded_children(timeout=1)",
            ],
            cwd = str(backend),
            env = owner_environment,
            check = False,
            timeout = 10,
        )
