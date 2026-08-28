# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Supervise one bounded command inside a persisted project workspace.

The public entry point accepts a project id and argv only. Root selection,
identity validation, workspace leasing, serialization, environment construction,
and the OS boundary stay inside this module so a caller cannot replace any part
of the security decision.

Linux bubblewrap is currently the only supported backend. Its PID namespace
provides the kernel lifecycle boundary needed to prove that detached descendants
are gone before the workspace mutation slot is released. macOS sandbox-exec
confines filesystem access but has no equivalent process-tree lifecycle primitive,
so supervised commands fail closed there. Windows remains unavailable until it
has both filesystem and process-tree containment.
"""

from __future__ import annotations

import contextlib
import codecs
import ctypes
import functools
import hashlib
import json
import math
import os
import select
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from utils.process_lifetime import (
    adopt_pid,
    child_popen_kwargs,
    forget_pid,
    initialize_parent_lifetime,
    spawn_on_lifetime_thread,
)

from . import common, execution
from .common import AgentWorkspaceError
from .execution import ExecutionBoundaryStatus, ProjectExecutionUnavailable

try:
    import resource as _resource
except ImportError:  # pragma: no cover - unavailable on Windows, which fails earlier
    _resource = None

try:
    _libc = ctypes.CDLL("libc.so.6", use_errno = True) if sys.platform.startswith("linux") else None
except OSError:
    _libc = None


DEFAULT_OUTPUT_LIMIT_BYTES = 256 * 1024
MAX_OUTPUT_LIMIT_BYTES = 4 * 1024 * 1024
MAX_TIMEOUT_SECONDS = 3600.0
MAX_ARGV_ITEMS = 256
MAX_ARGV_BYTES = 64 * 1024
MAX_PYTHON_SOURCE_BYTES = 4 * 1024 * 1024
MAX_STDIN_BYTES = 256 * 1024
_READ_CHUNK_BYTES = 64 * 1024
_READ_CHUNKS_PER_POLL = 16
_PREPARATION_WORKERS = 4
# A caller may abandon a preparation owner after its absolute deadline. Keep
# normal concurrent work small, but retain a separate hard cap for owners that
# are still unwinding an uninterruptible filesystem or host probe. This lets a
# single stuck owner release admission capacity without permitting unbounded
# daemon-thread growth.
_PREPARATION_CAPACITY = threading.BoundedSemaphore(_PREPARATION_WORKERS)
_PREPARATION_TOTAL_OWNERS = 16
_PREPARATION_OWNER_CAPACITY = threading.BoundedSemaphore(_PREPARATION_TOTAL_OWNERS)
_POLL_SECONDS = 0.02
_GRACEFUL_STOP_SECONDS = 0.5
_FORCED_STOP_SECONDS = 5.0
_LIFECYCLE_START_SECONDS = 5.0
_LIFECYCLE_PROBE_SECONDS = 5.0
_SPAWN_WAIT_SECONDS = 5.0
_MAX_STATUS_BYTES = 64 * 1024
_QUARANTINE_RETRY_INITIAL_SECONDS = 0.05
_QUARANTINE_RETRY_MAX_SECONDS = 2.0


class ProjectProcessContainmentError(AgentWorkspaceError):
    """A spawned process remains quarantined behind its lease and mutation slot."""


@dataclass(frozen = True)
class ProjectProcessResult:
    """Bounded and stable evidence from one supervised project command."""

    status: str
    exit_code: Optional[int]
    output: str
    output_bytes: int
    output_truncated: bool
    truncation_notice: str = ""
    stderr: str = ""
    stderr_bytes: int = 0
    stderr_truncated: bool = False


@dataclass(frozen = True)
class _ProjectCommandCapability:
    project_id: str
    command: str
    argv: tuple[str, ...]
    policy_hash: str
    workspace_identity: tuple[int, int]
    workspace_revision: int
    approved: bool


class _SpawnCancelledBeforeStart(RuntimeError):
    """The queued lifetime-thread job was cancelled before it called Popen."""


class _QuarantinePending(RuntimeError):
    """Cleanup cannot advance until an owned asynchronous phase resolves."""


class _SpawnAttempt:
    """Own a possibly blocked lifetime-thread Popen without blocking its caller."""

    def __init__(
        self, spawn: Callable[[], subprocess.Popen], on_spawned: Callable[[subprocess.Popen], None]
    ) -> None:
        self._spawn = spawn
        self._on_spawned = on_spawned
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._cancelled_before_start = False
        self._started = False
        self._process: Optional[subprocess.Popen] = None
        self._error: Optional[BaseException] = None
        self._owner = threading.Thread(
            target = self._run,
            name = "unsloth-project-spawn-owner",
            daemon = True,
        )
        self._owner.start()

    def _spawn_owned(self) -> subprocess.Popen:
        with self._lock:
            if self._cancelled_before_start:
                raise _SpawnCancelledBeforeStart()
            self._started = True
        process = self._spawn()
        with self._lock:
            self._process = process
        self._on_spawned(process)
        return process

    def _run(self) -> None:
        try:
            spawn_on_lifetime_thread(self._spawn_owned)
        except BaseException as exc:
            with self._lock:
                self._error = exc
        finally:
            self._done.set()

    def cancel_before_start(self) -> bool:
        """Prevent Popen if its lifetime-thread job has not started."""
        with self._lock:
            if self._started or self._done.is_set():
                return False
            self._cancelled_before_start = True
            return True

    def wait(self, cancel_event: Optional[threading.Event], timeout_seconds: float) -> str:
        deadline = time.monotonic() + timeout_seconds
        while not self._done.is_set():
            if cancel_event is not None and cancel_event.is_set():
                return "cancelled" if self.cancel_before_start() else "pending"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "not_started" if self.cancel_before_start() else "pending"
            self._done.wait(min(_POLL_SECONDS, remaining))
        return "resolved"

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def process(self) -> Optional[subprocess.Popen]:
        with self._lock:
            return self._process

    def result(self) -> subprocess.Popen:
        if not self._done.is_set():
            raise RuntimeError("The project process spawn is still pending.")
        with self._lock:
            error = self._error
            process = self._process
        if error is not None:
            raise error
        if process is None:
            raise RuntimeError("The project process spawn returned no process.")
        return process


@dataclass
class _SpawnOwnership:
    adopted: bool = False
    after_spawn_done: bool = False


@dataclass(eq = False)
class _QuarantinedRun:
    lease: object
    boundary: object
    lifecycle: object
    process: Optional[subprocess.Popen] = None
    spawn_attempt: Optional[_SpawnAttempt] = None
    spawn_ownership: Optional[_SpawnOwnership] = None
    scratch_script: Optional[str] = None
    adopted: bool = False
    after_spawn_done: bool = False
    lifecycle_bound: bool = False
    tree_proven: bool = False
    stdout_closed: bool = False
    scratch_removed: bool = False
    pid_forgotten: bool = False
    lifecycle_closed: bool = False
    boundary_closed: bool = False
    lease_released: bool = False
    execution_fence_fd: Optional[int] = None
    execution_fence_released: bool = False


_QUARANTINE_LOCK = threading.Lock()
_QUARANTINE_CLEANUP_LOCK = threading.Lock()
_QUARANTINED_RUNS: list[_QuarantinedRun] = []
_QUARANTINE_WAKE = threading.Event()
_QUARANTINE_OWNER_LOCK = threading.Lock()
_QUARANTINE_OWNER: Optional[threading.Thread] = None


class _StopSignal:
    def __init__(self, cancel_event: Optional[threading.Event], deadline: float) -> None:
        self._cancel_event = cancel_event
        self._deadline = deadline

    def reason(self) -> Optional[str]:
        if self._cancel_event is not None and self._cancel_event.is_set():
            return "cancelled"
        if time.monotonic() >= self._deadline:
            return "timed_out"
        return None

    def is_set(self) -> bool:
        return self.reason() is not None

    def wait(self, timeout: float) -> bool:
        remaining = max(0.0, self._deadline - time.monotonic())
        bounded = min(max(0.0, timeout), remaining)
        if self._cancel_event is not None:
            self._cancel_event.wait(bounded)
        elif bounded:
            time.sleep(bounded)
        return self.is_set()


def _bounded_preparation(
    callback,
    *,
    cancel_event: Optional[threading.Event],
    deadline: float,
    label: str,
    close_late_result = None,
    on_detach = None,
    on_late_finish = None,
):
    """Run potentially blocking pre-spawn setup under bounded global ownership."""
    if not _PREPARATION_CAPACITY.acquire(blocking = False):
        raise ProjectExecutionUnavailable(f"Project process {label} capacity is full.")
    if not _PREPARATION_OWNER_CAPACITY.acquire(blocking = False):
        _PREPARATION_CAPACITY.release()
        raise ProjectExecutionUnavailable(f"Project process {label} owner capacity is full.")

    lock = threading.Lock()
    done = _new_preparation_event()
    state = {
        "active_released": False,
        "disposition": "unclaimed",
        "finished": False,
        "result": None,
        "error": None,
    }

    def release_active() -> None:
        with lock:
            if state["active_released"]:
                return
            state["active_released"] = True
        _PREPARATION_CAPACITY.release()

    def close_late(value) -> None:
        if close_late_result is None:
            return
        try:
            close_late_result(value)
        except BaseException:
            pass

    def finish_late() -> None:
        if on_late_finish is None:
            return
        try:
            on_late_finish()
        except BaseException:
            pass

    def owned() -> None:
        value = None
        error = None
        try:
            value = callback()
        except BaseException as exc:
            error = exc
        with lock:
            state["result"] = value
            state["error"] = error
            state["finished"] = True
            detached = state["disposition"] == "detached"
        release_active()
        if detached:
            if error is None:
                close_late(value)
            finish_late()
        _PREPARATION_OWNER_CAPACITY.release()
        done.set()

    try:
        threading.Thread(
            target = owned,
            name = f"unsloth-project-{label.replace(' ', '-')}",
            daemon = True,
        ).start()
    except BaseException:
        release_active()
        _PREPARATION_OWNER_CAPACITY.release()
        raise

    def detach():
        value = None
        error = None
        finished = False
        with lock:
            if state["disposition"] == "claimed":
                return None, None, False
            if state["disposition"] != "detached":
                state["disposition"] = "detached"
                if on_detach is not None:
                    try:
                        on_detach()
                    except BaseException:
                        pass
            if state["finished"]:
                finished = True
                value = state["result"]
                error = state["error"]
        # The owner retains the global total-owner slot until it exits, but it
        # no longer consumes one of the normal concurrent preparation slots.
        release_active()
        return value, error, finished

    def abandon(kind: str) -> None:
        value, error, finished = detach()
        # A result published just before the caller detached is now owned by
        # this path. Closing it here is the atomic counterpart to the worker
        # closing a result published just after detachment.
        if error is None and value is not None:
            close_late(value)
        if finished:
            finish_late()
        if kind == "cancelled":
            raise InterruptedError(f"Project process {label} was cancelled.")
        raise TimeoutError(f"Project process {label} exceeded its deadline.")

    while True:
        if cancel_event is not None and cancel_event.is_set():
            abandon("cancelled")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            abandon("timed_out")
        if not done.wait(min(_POLL_SECONDS, remaining)):
            continue
        stop_kind = None
        with lock:
            if state["disposition"] == "detached":
                continue
            # A successful wake only proves publication. The caller may have
            # been descheduled after that wake until cancellation or the
            # absolute deadline had already won. Claiming here without a fresh
            # check would let late preparation mutate admission state.
            if cancel_event is not None and cancel_event.is_set():
                stop_kind = "cancelled"
            elif time.monotonic() >= deadline:
                stop_kind = "timed_out"
            else:
                state["disposition"] = "claimed"
                error = state["error"]
                value = state["result"]
        if stop_kind is not None:
            abandon(stop_kind)
        if error is not None:
            raise error
        return value


def _new_preparation_event():
    return threading.Event()


class _OutputBuffer:
    def __init__(
        self,
        limit: int,
        output_callback = None,
    ) -> None:
        self._limit = limit
        self._captured = bytearray()
        self._total = 0
        self._possibly_incomplete = False
        self._output_callback = output_callback
        self._stream_decoder = (
            codecs.getincrementaldecoder("utf-8")(errors = "replace")
            if output_callback is not None
            else None
        )
        self._stream_finished = False
        self._stream_rendered_bytes = 0
        self._truncation_notified = False
        self._truncation_notice = (
            f"\n[Process output was truncated. The capture limit was {limit} bytes.]\n"
        )

    def _stream(
        self,
        chunk: bytes,
        *,
        final: bool = False,
    ) -> None:
        if self._stream_decoder is None or self._stream_finished:
            return
        rendered = self._stream_decoder.decode(chunk, final = final)
        if final:
            self._stream_finished = True
        encoded = rendered.encode("utf-8")
        remaining = self._limit - self._stream_rendered_bytes
        if len(encoded) > remaining:
            rendered = encoded[: max(0, remaining)].decode("utf-8", errors = "ignore")
            encoded = rendered.encode("utf-8")
        self._stream_rendered_bytes += len(encoded)
        if not rendered:
            return
        try:
            self._output_callback(rendered)
        except Exception:
            pass

    def _notify_truncation(self) -> None:
        if self._truncation_notified:
            return
        self._stream(b"", final = True)
        self._truncation_notified = True
        if self._output_callback is not None:
            try:
                self._output_callback(self._truncation_notice)
            except Exception:
                pass

    def read_available(
        self,
        descriptor: int,
        *,
        max_chunks: int = _READ_CHUNKS_PER_POLL,
    ) -> bool:
        """Drain available bytes without letting an output flood hide cancellation."""
        chunks = 0
        while chunks < max_chunks:
            try:
                chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            except BlockingIOError:
                self._possibly_incomplete = False
                return False
            except InterruptedError:
                continue
            if not chunk:
                self._possibly_incomplete = False
                self._stream(b"", final = True)
                return True
            chunks += 1
            self._total += len(chunk)
            remaining = self._limit - len(self._captured)
            if remaining > 0:
                retained = chunk[:remaining]
                self._captured.extend(retained)
                self._stream(retained)
            if len(chunk) > max(0, remaining):
                self._notify_truncation()
        self._possibly_incomplete = True
        return False

    def result(self) -> tuple[str, int, bool, str]:
        self._stream(b"", final = True)
        rendered = bytes(self._captured).decode("utf-8", errors = "replace")
        encoded = rendered.encode("utf-8")
        normalized_truncated = len(encoded) > self._limit
        if normalized_truncated:
            rendered = encoded[: self._limit].decode("utf-8", errors = "ignore")
        truncated = (
            self._total > len(self._captured) or self._possibly_incomplete or normalized_truncated
        )
        if truncated:
            self._notify_truncation()
            rendered += self._truncation_notice
        return (
            rendered,
            self._total,
            truncated,
            self._truncation_notice if truncated else "",
        )


class _BubblewrapLifecycle:
    """Bind the run to bubblewrap's namespace init through a kernel pidfd."""

    def __init__(self) -> None:
        close_on_exec = getattr(os, "O_CLOEXEC", 0)
        self._status_read, self._status_write = os.pipe2(
            close_on_exec | getattr(os, "O_NONBLOCK", 0)
        )
        self._block_read, self._block_write = os.pipe2(close_on_exec)
        self._pidfd: Optional[int] = None
        self._child_pid: Optional[int] = None
        self._monitor_group: Optional[int] = None
        self._monitor_group_verified = False
        self._released = False
        self._unbound_group_proven = False
        self._closed = False
        self._execution_fence_fd: Optional[int] = None

    def attach_execution_fence(self, descriptor: int) -> None:
        """Keep the interprocess finalizer lock alive in bubblewrap itself."""
        if self._execution_fence_fd is not None or self._released:
            raise ProjectProcessContainmentError(
                "The project execution fence cannot be replaced after lifecycle setup."
            )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ProjectExecutionUnavailable("Project execution fence file is unsafe.")
        self._execution_fence_fd = descriptor

    def wrap_argv(self, argv: Sequence[str]) -> list[str]:
        command = list(argv)
        if not command:
            raise ProjectExecutionUnavailable("The bubblewrap command is unavailable.")
        lifecycle_options = [
            command[0],
            "--json-status-fd",
            str(self._status_write),
            "--block-fd",
            str(self._block_read),
        ]
        if self._execution_fence_fd is not None:
            # bubblewrap retains --sync-fd in its namespace monitor until every
            # sandbox process is gone. The inherited flock therefore survives
            # a Studio owner crash and cannot be reclaimed while the old hook
            # tree remains alive.
            lifecycle_options.extend(("--sync-fd", str(self._execution_fence_fd)))
        lifecycle_options.extend(command[1:])
        return lifecycle_options

    def add_popen_options(self, options: dict) -> dict:
        prepared = dict(options)
        passed = tuple(prepared.get("pass_fds", ()))
        # The setup child must retain a writer until it consumes an explicit
        # release byte. bubblewrap treats EOF on --block-fd as success, so if
        # Studio owned the only writer an owner crash would start the command.
        # The retained writer instead leaves the setup child safely blocked;
        # its recorded process group is reaped by startup crash recovery.
        prepared["pass_fds"] = tuple(
            dict.fromkeys(
                (
                    *passed,
                    self._status_write,
                    self._block_read,
                    self._block_write,
                    *((self._execution_fence_fd,) if self._execution_fence_fd is not None else ()),
                )
            )
        )
        return prepared

    def after_spawn(self, process: subprocess.Popen) -> None:
        """Capture the fresh monitor group before lifecycle status can fail."""
        self._monitor_group = int(process.pid)
        try:
            group = os.getpgid(process.pid)
        except ProcessLookupError:
            # The monitor may fail before the parent gets scheduled. Keep its
            # expected fresh-session group for a read-only disappearance proof,
            # but do not signal that numeric group without a live identity check.
            self._monitor_group_verified = False
        else:
            if group != process.pid:
                raise ProjectProcessContainmentError(
                    "The bubblewrap monitor did not start in its own process group."
                )
            self._monitor_group_verified = True
        finally:
            self._close_fd("_status_write")
            self._close_fd("_block_read")

    @staticmethod
    def _status_fields(pid: int) -> dict[str, str]:
        fields: dict[str, str] = {}
        with open(f"/proc/{pid}/status", encoding = "utf-8") as stream:
            for line in stream:
                name, separator, value = line.partition(":")
                if separator:
                    fields[name] = value.strip()
        return fields

    def _bind_child(self, process: subprocess.Popen, document: dict) -> None:
        raw_pid = document.get("child-pid")
        raw_namespace = document.get("pid-namespace")
        if (
            isinstance(raw_pid, bool)
            or not isinstance(raw_pid, int)
            or raw_pid <= 1
            or isinstance(raw_namespace, bool)
            or not isinstance(raw_namespace, int)
            or raw_namespace <= 0
        ):
            raise ProjectExecutionUnavailable(
                "bubblewrap did not provide a verifiable PID namespace identity."
            )
        pidfd = os.pidfd_open(raw_pid, 0)
        try:
            fields = self._status_fields(raw_pid)
            parent_pid = int(fields.get("PPid", "0"))
            namespace_pids = [int(value) for value in fields.get("NSpid", "").split()]
            namespace_inode = int(os.stat(f"/proc/{raw_pid}/ns/pid").st_ino)
            if (
                parent_pid != process.pid
                or not namespace_pids
                or namespace_pids[-1] != 1
                or namespace_inode != raw_namespace
            ):
                raise ProjectExecutionUnavailable(
                    "bubblewrap returned an unverifiable PID namespace init."
                )
        except Exception:
            os.close(pidfd)
            raise
        self._child_pid = raw_pid
        self._pidfd = pidfd

    def bind(
        self,
        process: subprocess.Popen,
        cancel_event: Optional[threading.Event] = None,
        deadline: Optional[float] = None,
    ) -> bool:
        """Read child-pid while --block-fd keeps the sandboxed command stopped."""
        deadline = min(
            time.monotonic() + _LIFECYCLE_START_SECONDS,
            deadline if deadline is not None else math.inf,
        )
        pending = bytearray()
        cancelled = cancel_event is not None and cancel_event.is_set()
        poller = select.poll()
        poller.register(self._status_read, select.POLLIN | select.POLLHUP | select.POLLERR)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Project command lifecycle binding exceeded its deadline.")
            try:
                events = poller.poll(max(1, min(50, int(remaining * 1000))))
            except InterruptedError:
                continue
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
            if not events:
                continue
            try:
                chunk = os.read(self._status_read, 4096)
            except BlockingIOError:
                continue
            if not chunk:
                if not pending:
                    raise ProjectExecutionUnavailable(
                        "bubblewrap exited before reporting a supervised PID namespace."
                    )
                raise ProjectExecutionUnavailable(
                    "bubblewrap returned incomplete lifecycle status."
                )
            pending.extend(chunk)
            if len(pending) > _MAX_STATUS_BYTES:
                raise ProjectExecutionUnavailable("bubblewrap lifecycle status is too large.")
            while b"\n" in pending:
                raw_line, _, remainder = pending.partition(b"\n")
                pending = bytearray(remainder)
                try:
                    document = json.loads(raw_line)
                except (UnicodeError, ValueError) as exc:
                    raise ProjectExecutionUnavailable(
                        "bubblewrap returned invalid lifecycle status."
                    ) from exc
                if not isinstance(document, dict):
                    raise ProjectExecutionUnavailable(
                        "bubblewrap returned invalid lifecycle status."
                    )
                if "child-pid" not in document:
                    continue
                self._bind_child(process, document)
                return not cancelled and not (cancel_event is not None and cancel_event.is_set())

    def release(self, cancel_event: Optional[threading.Event]) -> bool:
        """Release the command only after lifecycle and cancellation checks."""
        if self._pidfd is None:
            raise ProjectProcessContainmentError(
                "The sandbox process tree was not bound before command release."
            )
        if cancel_event is not None and cancel_event.is_set():
            return False
        os.write(self._block_write, b"1")
        self._released = True
        self._close_fd("_block_write")
        return True

    @staticmethod
    def _process_group_exists(group: int) -> bool:
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError as exc:
            raise ProjectProcessContainmentError(
                "The bubblewrap setup process group could not be inspected."
            ) from exc
        return True

    @classmethod
    def _wait_for_process_group_exit(
        cls, process: subprocess.Popen, group: int, timeout_seconds: float
    ) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while True:
            process.poll()
            if not cls._process_group_exists(group):
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(_POLL_SECONDS, remaining))

    def terminate_unbound_and_prove(self, process: subprocess.Popen) -> int:
        """Prove failed setup is gone without ever releasing the user command."""
        if self._unbound_group_proven:
            return_code = process.poll()
            if return_code is None:
                raise ProjectProcessContainmentError(
                    "The proven bubblewrap setup process was unexpectedly still running."
                )
            return int(return_code)
        if self._released:
            raise ProjectProcessContainmentError(
                "A released project command cannot use unbound setup cleanup."
            )
        if self._pidfd is not None:
            return self.terminate_and_prove(process)
        group = self._monitor_group
        if group is None or group <= 1 or group != process.pid:
            raise ProjectProcessContainmentError(
                "The failed bubblewrap setup process group was not captured."
            )

        monitor_alive = process.poll() is None
        if monitor_alive:
            try:
                current_group = os.getpgid(process.pid)
            except ProcessLookupError:
                monitor_alive = False
            else:
                if current_group != group:
                    raise ProjectProcessContainmentError(
                        "The bubblewrap monitor process group changed before cleanup."
                    )
                self._monitor_group_verified = True

        if monitor_alive and self._monitor_group_verified:
            try:
                os.killpg(group, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError as exc:
                raise ProjectProcessContainmentError(
                    "The failed bubblewrap setup process group could not be terminated."
                ) from exc
            if not self._wait_for_process_group_exit(
                process,
                group,
                _GRACEFUL_STOP_SECONDS,
            ):
                try:
                    os.killpg(group, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError as exc:
                    raise ProjectProcessContainmentError(
                        "The failed bubblewrap setup process group could not be killed."
                    ) from exc
                if not self._wait_for_process_group_exit(
                    process,
                    group,
                    _FORCED_STOP_SECONDS,
                ):
                    raise ProjectProcessContainmentError(
                        "The failed bubblewrap setup process group did not exit."
                    )
        elif not self._wait_for_process_group_exit(process, group, _GRACEFUL_STOP_SECONDS):
            # Once the monitor has exited, a numeric process group is no longer
            # anchored strongly enough to signal. Keep every workspace lock if
            # passive disappearance cannot prove that the old group is gone.
            raise ProjectProcessContainmentError(
                "The failed bubblewrap setup process group could not be proven dead."
            )

        return_code = _terminate_and_reap(process)
        if self._process_group_exists(group):
            raise ProjectProcessContainmentError(
                "The failed bubblewrap setup process group survived monitor reap."
            )
        self._unbound_group_proven = True
        return return_code

    @staticmethod
    def _pidfd_ready(pidfd: int, timeout_seconds: Optional[float]) -> bool:
        poller = select.poll()
        exited = select.POLLIN | select.POLLHUP
        poller.register(pidfd, exited | select.POLLERR)
        timeout_ms = -1 if timeout_seconds is None else max(0, int(timeout_seconds * 1000))
        while True:
            try:
                events = poller.poll(timeout_ms)
            except InterruptedError:
                continue
            if not events:
                return False
            if any(event & exited for _descriptor, event in events):
                return True
            raise ProjectProcessContainmentError(
                "The namespace init pidfd became invalid before exit was proven."
            )

    def _signal_init(self, requested_signal: int) -> None:
        if self._pidfd is None or self._pidfd_ready(self._pidfd, 0):
            return
        try:
            signal.pidfd_send_signal(self._pidfd, requested_signal, None, 0)
        except ProcessLookupError:
            pass

    def prove_signal_capability(self) -> None:
        """Exercise the exact pidfd signal syscall before a project command runs."""
        if self._pidfd is None or self._pidfd_ready(self._pidfd, 0):
            raise ProjectExecutionUnavailable(
                "The bubblewrap namespace init exited before lifecycle probing completed."
            )
        try:
            signal.pidfd_send_signal(self._pidfd, 0, None, 0)
        except OSError as exc:
            raise ProjectExecutionUnavailable(
                "The host refused pidfd signalling for supervised project commands."
            ) from exc

    def terminate_and_prove(self, process: subprocess.Popen) -> int:
        """Wait for the exact namespace init to die, then reap its monitor."""
        if self._pidfd is None:
            _terminate_and_reap(process)
            raise ProjectProcessContainmentError(
                "The sandbox process tree could not be identified and remains quarantined."
            )
        if not self._pidfd_ready(self._pidfd, 0):
            self._signal_init(signal.SIGTERM)
            if not self._pidfd_ready(self._pidfd, _GRACEFUL_STOP_SECONDS):
                self._signal_init(signal.SIGKILL)
                if not self._pidfd_ready(self._pidfd, _FORCED_STOP_SECONDS):
                    raise ProjectProcessContainmentError(
                        "The sandbox process tree could not be terminated."
                    )
        return _terminate_and_reap(process)

    def _close_fd(self, attribute: str) -> None:
        descriptor = getattr(self, attribute)
        if descriptor is None:
            return
        setattr(self, attribute, None)
        with contextlib.suppress(OSError):
            os.close(descriptor)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for attribute in (
            "_status_read",
            "_status_write",
            "_block_read",
            "_block_write",
            "_pidfd",
        ):
            self._close_fd(attribute)


def _bubblewrap_identity(executable: str) -> Optional[tuple[int, int, int, int]]:
    try:
        metadata = os.stat(executable, follow_symlinks = False)
    except OSError:
        return None
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mtime_ns),
        int(metadata.st_size),
    )


@functools.lru_cache(maxsize = 4)
def _probe_supervised_backend(
    executable: str, executable_identity: tuple[int, int, int, int]
) -> tuple[bool, Optional[str]]:
    """Exercise the exact bubblewrap, pidfd, and teardown path without a workspace."""
    del executable_identity
    lifecycle: Optional[_BubblewrapLifecycle] = None
    process: Optional[subprocess.Popen] = None
    adopted = False
    cleanup_proven = False
    try:
        lifecycle = _BubblewrapLifecycle()
        command = lifecycle.wrap_argv(
            [
                executable,
                "--die-with-parent",
                "--unshare-all",
                "--ro-bind",
                "/",
                "/",
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--",
                "/bin/true",
            ]
        )
        lifetime = child_popen_kwargs()
        options = {
            "env": {
                "HOME": "/",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/bin:/bin",
            },
            "stderr": subprocess.DEVNULL,
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "start_new_session": True,
        }
        options.update(lifetime)
        options = lifecycle.add_popen_options(options)
        initialize_parent_lifetime()
        process = spawn_on_lifetime_thread(lambda: subprocess.Popen(command, **options))
        adopt_pid(process.pid)
        adopted = True
        lifecycle.after_spawn(process)
        if not lifecycle.bind(process):
            raise ProjectExecutionUnavailable("The lifecycle probe was unexpectedly cancelled.")
        lifecycle.prove_signal_capability()
        if not lifecycle.release(None):
            raise ProjectExecutionUnavailable("The lifecycle probe could not be released.")
        try:
            exit_code = process.wait(timeout = _LIFECYCLE_PROBE_SECONDS)
        except subprocess.TimeoutExpired as exc:
            raise ProjectExecutionUnavailable(
                "The supervised bubblewrap lifecycle probe timed out."
            ) from exc
        lifecycle.terminate_and_prove(process)
        cleanup_proven = True
        if exit_code != 0:
            raise ProjectExecutionUnavailable(
                "bubblewrap failed the supervised process lifecycle probe."
            )
        return True, None
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        if isinstance(exc, ProjectExecutionUnavailable):
            reason = str(exc)
        else:
            reason = "The host cannot establish the supervised bubblewrap lifecycle."
        return False, reason
    finally:
        if process is not None and lifecycle is not None and not cleanup_proven:
            try:
                lifecycle.terminate_and_prove(process)
            except BaseException:
                pass
            else:
                cleanup_proven = True
        if adopted and cleanup_proven and process is not None:
            with contextlib.suppress(Exception):
                forget_pid(process.pid)
        if lifecycle is not None:
            lifecycle.close()


def supervised_process_status(*, probe: bool = True) -> ExecutionBoundaryStatus:
    """Report whether the host has a boundary with full descendant containment."""
    status = execution.execution_boundary_status(probe = False)
    if not status.available:
        return status
    if status.backend != "bubblewrap":
        return ExecutionBoundaryStatus(
            False,
            None,
            "Secure supervised project commands require Linux bubblewrap process isolation. "
            "macOS sandbox-exec cannot prove detached descendants are gone.",
        )
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        return ExecutionBoundaryStatus(
            False,
            None,
            "Secure supervised project commands require Linux pidfd lifecycle support.",
        )
    if probe:
        executable = execution._bubblewrap_path()
        identity = _bubblewrap_identity(executable) if executable is not None else None
        if executable is None or identity is None:
            return ExecutionBoundaryStatus(
                False,
                None,
                "bubblewrap disappeared before lifecycle probing.",
            )
        available, reason = _probe_supervised_backend(executable, identity)
        if not available:
            return ExecutionBoundaryStatus(False, None, reason)
    return status


def _normalized_argv(argv: Sequence[str]) -> tuple[str, ...]:
    if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence):
        raise AgentWorkspaceError("Project command argv must be a sequence of strings.")
    if not argv or len(argv) > MAX_ARGV_ITEMS:
        raise AgentWorkspaceError(
            f"Project command argv must contain between 1 and {MAX_ARGV_ITEMS} items."
        )
    normalized: list[str] = []
    size = 0
    for item in argv:
        if not isinstance(item, str) or not item or "\x00" in item:
            raise AgentWorkspaceError("Project command argv contains an invalid item.")
        try:
            size += len(item.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise AgentWorkspaceError("Project command argv must be valid UTF-8 text.") from exc
        if size > MAX_ARGV_BYTES:
            raise AgentWorkspaceError("Project command argv is too large.")
        normalized.append(item)
    return tuple(normalized)


def _timeout(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AgentWorkspaceError("Project command timeout must be a number.")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0 or parsed > MAX_TIMEOUT_SECONDS:
        raise AgentWorkspaceError(
            f"Project command timeout must be greater than zero and at most "
            f"{int(MAX_TIMEOUT_SECONDS)} seconds."
        )
    return parsed


def _project_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 256
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise AgentWorkspaceError("Project id is invalid.")
    return value


def _python_source(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise AgentWorkspaceError("Project Python source is invalid.")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise AgentWorkspaceError("Project Python source must be valid UTF-8 text.") from exc
    if len(encoded) > MAX_PYTHON_SOURCE_BYTES:
        raise AgentWorkspaceError("Project Python source is too large.")
    return value


def _output_limit(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AgentWorkspaceError("Project command output limit must be an integer.")
    if value < 1 or value > MAX_OUTPUT_LIMIT_BYTES:
        raise AgentWorkspaceError(
            f"Project command output limit must be between 1 and {MAX_OUTPUT_LIMIT_BYTES} bytes."
        )
    return value


def _sandbox_preexec() -> None:
    """Apply the existing sandbox resource policy before bubblewrap starts."""
    with contextlib.suppress(OSError):
        os.umask(0o077)
    if _libc is not None:
        with contextlib.suppress(OSError, AttributeError):
            _libc.prctl(38, 1, 0, 0, 0)
    if _resource is None:
        return

    def configured(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, str(default)))
        except ValueError:
            return default

    limits = (
        ("RLIMIT_NPROC", configured("UNSLOTH_STUDIO_SANDBOX_NPROC", 10000)),
        ("RLIMIT_FSIZE", 100 * 1024 * 1024),
        (
            "RLIMIT_AS",
            configured("UNSLOTH_STUDIO_SANDBOX_AS_GB", 8) * 1024 * 1024 * 1024,
        ),
        ("RLIMIT_CPU", configured("UNSLOTH_STUDIO_SANDBOX_CPU_S", 600)),
    )
    for name, requested in limits:
        limit = getattr(_resource, name, None)
        if limit is None:
            continue
        with contextlib.suppress(ValueError, OSError):
            _resource.setrlimit(limit, (requested, requested))
    nofile_limit = getattr(_resource, "RLIMIT_NOFILE", None)
    if nofile_limit is not None:
        with contextlib.suppress(ValueError, OSError):
            requested = configured("UNSLOTH_STUDIO_SANDBOX_NOFILE", 16384)
            _soft, hard = _resource.getrlimit(nofile_limit)
            target = requested if hard == _resource.RLIM_INFINITY else min(requested, hard)
            _resource.setrlimit(nofile_limit, (target, target))


def _install_sitecustomize(boundary) -> str:
    """Copy the trusted path-remapping shim into the mounted scratch directory."""
    source = os.path.realpath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "inference",
            "sandbox_site",
            "sitecustomize.py",
        )
    )
    target_dir = os.path.join(str(boundary.scratch), "pythonpath")
    os.mkdir(target_dir, 0o700)
    target = os.path.join(target_dir, "sitecustomize.py")
    source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        metadata = os.fstat(source_fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 1024 * 1024:
            raise ProjectExecutionUnavailable("The Python sandbox shim is unavailable.")
        payload = bytearray()
        while len(payload) <= 1024 * 1024:
            chunk = os.read(source_fd, min(64 * 1024, 1024 * 1024 + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        if len(payload) != metadata.st_size:
            raise ProjectExecutionUnavailable("The Python sandbox shim changed during setup.")
    finally:
        os.close(source_fd)
    target_fd = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(target_fd, view)
            if written <= 0:
                raise ProjectExecutionUnavailable("The Python sandbox shim could not be copied.")
            view = view[written:]
    finally:
        os.close(target_fd)
    return target_dir


def _minimal_environment(boundary, workspace, project_id: str) -> dict[str, str]:
    """Build from scratch so parent credentials and code-loading hooks cannot leak."""
    executable_dir = os.path.dirname(os.path.realpath(sys.executable))
    path = os.pathsep.join(
        dict.fromkeys(part for part in (executable_dir, "/usr/bin", "/bin") if part)
    )
    pythonpath = os.pathsep.join((str(workspace.root), _install_sitecustomize(boundary)))
    return boundary.apply_environment(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "HOME": str(boundary.scratch),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "MPLBACKEND": "Agg",
            "NO_COLOR": "1",
            "PATH": path,
            "PYTHONPATH": pythonpath,
            "PYTHONIOENCODING": "utf-8",
            "TERM": "dumb",
            "TZ": "UTC",
            "UNSLOTH_STUDIO_PROJECT_ID": project_id,
        }
    )


def _signal_process_group(process: subprocess.Popen, requested_signal: int) -> None:
    """Signal only the fresh session created for this direct child."""
    getpgid = getattr(os, "getpgid", None)
    killpg = getattr(os, "killpg", None)
    if getpgid is None or killpg is None:
        with contextlib.suppress(OSError, ProcessLookupError, ValueError):
            process.send_signal(requested_signal)
        return
    try:
        group = getpgid(process.pid)
    except (OSError, ProcessLookupError):
        group = None
    if group == process.pid:
        try:
            killpg(group, requested_signal)
            return
        except (OSError, ProcessLookupError):
            pass
    with contextlib.suppress(OSError, ProcessLookupError):
        process.send_signal(requested_signal)


def _terminate_and_reap(process: subprocess.Popen) -> int:
    """Synchronously end the bubblewrap namespace before its slot can be released."""
    return_code = process.poll()
    if return_code is None:
        _signal_process_group(process, signal.SIGTERM)
        try:
            return_code = process.wait(timeout = _GRACEFUL_STOP_SECONDS)
        except subprocess.TimeoutExpired:
            _signal_process_group(process, signal.SIGKILL)
            with contextlib.suppress(OSError, ProcessLookupError):
                process.kill()
            try:
                return_code = process.wait(timeout = _FORCED_STOP_SECONDS)
            except subprocess.TimeoutExpired as exc:
                raise ProjectProcessContainmentError(
                    "The bubblewrap monitor did not exit after SIGKILL."
                ) from exc
    if return_code is None:
        raise RuntimeError("The supervised process could not be reaped.")
    return int(return_code)


def _popen_options(
    boundary,
    environment: dict[str, str],
    lifecycle: _BubblewrapLifecycle,
    *,
    input_pipe: bool = False,
    separate_stderr: bool = False,
) -> dict:
    lifetime = child_popen_kwargs(_sandbox_preexec)
    options = {
        "bufsize": 0,
        "env": environment,
        "stderr": subprocess.PIPE if separate_stderr else subprocess.STDOUT,
        "stdin": subprocess.PIPE if input_pipe else subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "start_new_session": True,
    }
    options.update(boundary.popen_kwargs(lifetime.pop("preexec_fn", None)))
    options.update(lifetime)
    return lifecycle.add_popen_options(options)


def _quarantine_run(
    lease,
    boundary,
    lifecycle,
    process: Optional[subprocess.Popen] = None,
    *,
    spawn_attempt: Optional[_SpawnAttempt] = None,
    spawn_ownership: Optional[_SpawnOwnership] = None,
    scratch_script: Optional[str] = None,
    adopted: bool = False,
    after_spawn_done: bool = False,
    lifecycle_bound: bool = False,
    execution_fence_fd: Optional[int] = None,
) -> None:
    run = _QuarantinedRun(
        lease = lease,
        boundary = boundary,
        lifecycle = lifecycle,
        process = process,
        spawn_attempt = spawn_attempt,
        spawn_ownership = spawn_ownership,
        scratch_script = scratch_script,
        adopted = adopted,
        after_spawn_done = after_spawn_done,
        lifecycle_bound = lifecycle_bound,
        execution_fence_fd = execution_fence_fd,
    )
    _register_quarantined_run(run)


def _register_quarantined_run(run: _QuarantinedRun) -> None:
    with _QUARANTINE_LOCK:
        _QUARANTINED_RUNS.append(run)
    _QUARANTINE_WAKE.set()
    _start_quarantine_retry_owner()


def _resolve_quarantined_spawn(run: _QuarantinedRun) -> None:
    attempt = run.spawn_attempt
    if attempt is None:
        return
    if not attempt.done:
        raise _QuarantinePending("The project process spawn is still resolving.")
    run.process = attempt.process
    run.spawn_attempt = None
    if run.spawn_ownership is not None:
        run.adopted = run.adopted or run.spawn_ownership.adopted
        run.after_spawn_done = run.after_spawn_done or run.spawn_ownership.after_spawn_done
        run.spawn_ownership = None
    try:
        attempt.result()
    except BaseException:
        if run.process is None:
            run.tree_proven = True
            return


def _prepare_quarantined_process(run: _QuarantinedRun) -> None:
    process = run.process
    if process is None or run.tree_proven:
        return
    if not run.after_spawn_done:
        run.lifecycle.after_spawn(process)
        run.after_spawn_done = True
    if not run.adopted:
        try:
            adopt_pid(process.pid)
        except BaseException:
            pass
        else:
            run.adopted = True
    if not run.lifecycle_bound:
        try:
            run.lifecycle.bind(process, None)
        except BaseException:
            try:
                run.lifecycle.terminate_unbound_and_prove(process)
            except BaseException as cleanup_error:
                raise ProjectProcessContainmentError(
                    "The failed bubblewrap setup process group remains quarantined."
                ) from cleanup_error
            run.tree_proven = True
            return
        else:
            run.lifecycle_bound = True


def _advance_quarantined_run(run: _QuarantinedRun) -> None:
    """Advance cleanup phases without replaying an already completed phase."""
    _resolve_quarantined_spawn(run)
    _prepare_quarantined_process(run)
    process = run.process
    if not run.tree_proven:
        if process is not None:
            run.lifecycle.terminate_and_prove(process)
        run.tree_proven = True
    if not run.stdout_closed:
        if process is not None:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    with contextlib.suppress(OSError, ValueError):
                        stream.close()
        run.stdout_closed = True
    if not run.scratch_removed:
        if run.scratch_script is not None:
            with contextlib.suppress(OSError):
                os.unlink(run.scratch_script)
        run.scratch_removed = True
    if not run.pid_forgotten:
        if process is not None and run.adopted:
            with contextlib.suppress(Exception):
                forget_pid(process.pid)
        run.pid_forgotten = True
    if not run.execution_fence_released:
        if run.execution_fence_fd is not None:
            _release_project_execution_fence(run.execution_fence_fd)
            run.execution_fence_fd = None
        run.execution_fence_released = True
    if not run.lifecycle_closed:
        run.lifecycle.close()
        run.lifecycle_closed = True
    if not run.boundary_closed:
        run.boundary.close()
        run.boundary_closed = True
    if not run.lease_released:
        run.lease.__exit__(None, None, None)
        run.lease_released = True


def _quarantine_retry_loop() -> None:
    delay = _QUARANTINE_RETRY_INITIAL_SECONDS
    while True:
        _QUARANTINE_WAKE.wait()
        _QUARANTINE_WAKE.clear()
        while retry_quarantined_project_processes():
            _QUARANTINE_WAKE.wait(delay)
            _QUARANTINE_WAKE.clear()
            delay = min(delay * 2, _QUARANTINE_RETRY_MAX_SECONDS)
        delay = _QUARANTINE_RETRY_INITIAL_SECONDS


def _start_quarantine_retry_owner() -> None:
    global _QUARANTINE_OWNER
    with _QUARANTINE_OWNER_LOCK:
        if _QUARANTINE_OWNER is not None and _QUARANTINE_OWNER.is_alive():
            return
        owner = threading.Thread(
            target = _quarantine_retry_loop,
            name = "unsloth-project-quarantine",
            daemon = True,
        )
        _QUARANTINE_OWNER = owner
        owner.start()


def retry_quarantined_project_processes() -> int:
    """Retry cleanup without releasing an unproven workspace slot.

    Returns the number of runs that remain quarantined. This is safe to call
    repeatedly and is intended for the orderly backend shutdown path.
    """
    with _QUARANTINE_CLEANUP_LOCK:
        with _QUARANTINE_LOCK:
            pending = list(_QUARANTINED_RUNS)
        completed: set[int] = set()
        for run in pending:
            try:
                _advance_quarantined_run(run)
            except BaseException:
                continue
            completed.add(id(run))
        if completed:
            with _QUARANTINE_LOCK:
                _QUARANTINED_RUNS[:] = [
                    run for run in _QUARANTINED_RUNS if id(run) not in completed
                ]
        with _QUARANTINE_LOCK:
            return len(_QUARANTINED_RUNS)


def run_project_process(
    project_id: str,
    argv: Sequence[str],
    *,
    timeout_seconds: Optional[float] = 300,
    output_limit_bytes: int = DEFAULT_OUTPUT_LIMIT_BYTES,
    cancel_event: Optional[threading.Event] = None,
    output_callback = None,
) -> ProjectProcessResult:
    """Run argv under the persisted project's lease and secure process boundary.

    ``timeout_seconds`` is one absolute deadline from the security probe through
    workspace admission, boundary construction, hard-link scanning, spawn,
    namespace binding, output collection, and proven teardown. Preparation runs
    under bounded owners so timeout or cancellation can release caller capacity
    without permitting a late project process spawn.
    """
    return _run_project_process(
        _project_id(project_id),
        _normalized_argv(argv),
        timeout_seconds = timeout_seconds,
        output_limit_bytes = output_limit_bytes,
        cancel_event = cancel_event,
        output_callback = output_callback,
        input_bytes = None,
        separate_stderr = False,
        before_spawn = None,
    )


def run_project_python(
    project_id: str,
    source: str,
    *,
    timeout_seconds: Optional[float] = 300,
    output_limit_bytes: int = DEFAULT_OUTPUT_LIMIT_BYTES,
    cancel_event: Optional[threading.Event] = None,
    output_callback = None,
) -> ProjectProcessResult:
    """Run Python source with project imports and the trusted path-remapping shim."""
    return _run_project_process(
        _project_id(project_id),
        (),
        python_source = _python_source(source),
        timeout_seconds = timeout_seconds,
        output_limit_bytes = output_limit_bytes,
        cancel_event = cancel_event,
        output_callback = output_callback,
        input_bytes = None,
        separate_stderr = False,
        before_spawn = None,
    )


def _run_trusted_project_hook_process(
    invocation,
    *,
    timeout_seconds: float,
    output_limit_bytes: int,
    cancel_event: Optional[threading.Event],
    end_snapshot = None,
) -> ProjectProcessResult:
    """Run one hash-bound hook without exposing a configurable security boundary."""
    from .hook_runtime import (
        _HookInvocation,
        HookSessionEndSnapshot,
        _END_SNAPSHOT_SEAL,
        _hook_invocation_process_spec,
        _revalidate_hook_invocation,
    )

    if not isinstance(invocation, _HookInvocation):
        raise AgentWorkspaceError("Project hook invocation authority is invalid.")
    argv, event_input = _hook_invocation_process_spec(invocation)
    if not isinstance(event_input, bytes) or len(event_input) > MAX_STDIN_BYTES:
        raise AgentWorkspaceError("Project hook input exceeds the supported size limit.")
    workspace_override = None
    captured_end = end_snapshot is not None
    execution_fence_id = None
    if captured_end:
        if (
            not isinstance(end_snapshot, HookSessionEndSnapshot)
            or end_snapshot._seal is not _END_SNAPSHOT_SEAL
            or invocation not in end_snapshot.invocations
            or end_snapshot.token.project_id != invocation.project_id
            or invocation.event != "SessionEnd"
        ):
            raise AgentWorkspaceError("Project hook end authority is invalid.")
        workspace_override = end_snapshot.workspace
        try:
            payload = json.loads(event_input)
        except (TypeError, ValueError) as exc:
            raise AgentWorkspaceError("Project hook input authority is invalid.") from exc
        delivery_id = payload.get("delivery_id") if isinstance(payload, dict) else None
        expected_delivery_id = f"{end_snapshot.token.generation}:SessionEnd"
        if delivery_id != expected_delivery_id:
            raise AgentWorkspaceError("Project hook end delivery identity is invalid.")
        # A delivery fans out to sibling handlers. The client-visible id stays
        # identical across all of them and every retry, while the internal
        # owner-death fence must serialize only retries of the same handler.
        execution_fence_id = json.dumps(
            (delivery_id, invocation.handler_id),
            ensure_ascii = False,
            separators = (",", ":"),
        )

    def revalidate(workspace) -> None:
        _revalidate_hook_invocation(invocation, workspace, captured_end = captured_end)

    return _run_project_process(
        _project_id(invocation.project_id),
        _normalized_argv(argv),
        timeout_seconds = timeout_seconds,
        output_limit_bytes = output_limit_bytes,
        cancel_event = cancel_event,
        output_callback = None,
        input_bytes = event_input,
        separate_stderr = True,
        before_spawn = revalidate,
        workspace_override = workspace_override,
        absolute_deadline = invocation.deadline_monotonic,
        execution_fence_id = execution_fence_id,
    )


def _bind_project_command_capability(
    project_id: str, command: str, argv: Sequence[str], proof: dict
) -> _ProjectCommandCapability:
    """Bind one reviewed full-access command to its exact policy snapshot."""
    normalized_project_id = _project_id(project_id)
    normalized_argv = _normalized_argv(argv)
    if not isinstance(command, str) or not command.strip() or "\x00" in command:
        raise AgentWorkspaceError("Project command authority is invalid.")
    if not isinstance(proof, dict):
        raise AgentWorkspaceError("Project command approval proof is missing.")
    if proof.get("projectId") != normalized_project_id or proof.get("command") != command:
        raise AgentWorkspaceError("Project command approval does not match this command.")
    if tuple(proof.get("argv") or ()) != normalized_argv:
        raise AgentWorkspaceError("Project command approval does not match this argv.")
    identity = proof.get("workspaceIdentity")
    if not isinstance(identity, (list, tuple)) or len(identity) != 2:
        raise AgentWorkspaceError("Project command workspace proof is invalid.")
    policy_hash = proof.get("policyHash")
    if not isinstance(policy_hash, str) or not policy_hash:
        raise AgentWorkspaceError("Project command policy proof is invalid.")
    return _ProjectCommandCapability(
        project_id = normalized_project_id,
        command = command,
        argv = normalized_argv,
        policy_hash = policy_hash,
        workspace_identity = (int(identity[0]), int(identity[1])),
        workspace_revision = int(proof.get("workspaceRevision")),
        approved = proof.get("approved") is True,
    )


def _revalidate_project_command_capability(
    capability: _ProjectCommandCapability, workspace
) -> None:
    from .rules import (
        discover_project_command_rules,
        evaluate_terminal_command_rules,
        secure_command_rule_traversal_supported,
    )

    identity = (int(workspace.device_id), int(workspace.file_id))
    if (
        workspace.project_id != capability.project_id
        or identity != capability.workspace_identity
        or int(workspace.revision) != capability.workspace_revision
    ):
        raise AgentWorkspaceError("Project command workspace changed before execution.")
    if not secure_command_rule_traversal_supported():
        raise ProjectExecutionUnavailable(
            "Secure project command policy is unavailable on this platform."
        )
    discovered = discover_project_command_rules(
        workspace.root,
        expected_identity = identity,
        project_trusted = True,
    )
    policy = evaluate_terminal_command_rules(discovered, capability.command)
    if policy.get("policyHash") != capability.policy_hash:
        raise AgentWorkspaceError("Project command policy changed before execution.")
    decision = policy.get("decision")
    if decision == "forbidden":
        raise AgentWorkspaceError("Project command is forbidden by the current policy.")
    if decision == "prompt" and not capability.approved:
        raise AgentWorkspaceError("Project command approval is missing for the current policy.")


def _spawn_authorized_project_host_command(
    capability: _ProjectCommandCapability,
    popen_options: dict,
    cancel_event: Optional[threading.Event] = None,
) -> subprocess.Popen:
    """Spawn one full-access project command after lifetime-thread revalidation."""
    if not isinstance(capability, _ProjectCommandCapability):
        raise AgentWorkspaceError("Project command capability is invalid.")
    if not isinstance(popen_options, dict):
        raise AgentWorkspaceError("Project command process options are invalid.")
    if cancel_event is not None and not isinstance(cancel_event, threading.Event):
        raise AgentWorkspaceError("Project command cancellation must use a threading event.")
    options = dict(popen_options)

    def spawn() -> subprocess.Popen:
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Project command was cancelled before launch.")
        with common.project_workspace_access(capability.project_id) as workspace:
            _revalidate_project_command_capability(capability, workspace)
            if os.path.realpath(str(options.get("cwd") or "")) != os.path.realpath(
                str(workspace.root)
            ):
                raise AgentWorkspaceError("Project command working directory changed.")
            # Revalidation can block on filesystem and policy reads. Cancellation
            # that wins while those checks run must prevent the host process from
            # crossing the final launch boundary.
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Project command was cancelled before launch.")
            return subprocess.Popen(capability.argv, **options)

    return spawn_on_lifetime_thread(spawn)


def _project_execution_fence_path(fence_id: str) -> str:
    if (
        not isinstance(fence_id, str)
        or not fence_id
        or len(fence_id.encode("utf-8", errors = "strict")) > 1024
    ):
        raise AgentWorkspaceError("Project execution fence identity is invalid.")
    from utils.paths.storage_roots import studio_root  # noqa: PLC0415

    directory = os.path.join(str(studio_root()), "project-hook-execution-fences")
    os.makedirs(directory, mode = 0o700, exist_ok = True)
    directory_metadata = os.lstat(directory)
    if not stat.S_ISDIR(directory_metadata.st_mode) or stat.S_ISLNK(directory_metadata.st_mode):
        raise ProjectExecutionUnavailable("Project execution fence directory is unsafe.")
    # Keep the complete digest. Deliberately sharding identities onto a small
    # fixed lock set lets unrelated sibling handlers serialize and can make one
    # miss its deadline behind another. The full digest gives every stable
    # composite delivery/handler identity its own process-shared inode.
    digest = hashlib.sha256(fence_id.encode("utf-8")).hexdigest()
    return os.path.join(directory, f"{digest}.lock")


def _acquire_project_execution_fence(
    fence_id: str, cancel_event: Optional[threading.Event], deadline: float
) -> int:
    """Acquire a process-wide finalizer fence until the prior tree is dead."""
    if os.name != "posix":
        raise ProjectExecutionUnavailable(
            "Project execution fencing is unavailable on this platform."
        )
    try:
        import fcntl  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - non-POSIX fails above
        raise ProjectExecutionUnavailable(
            "Project execution fencing is unavailable on this platform."
        ) from exc
    path = _project_execution_fence_path(fence_id)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ProjectExecutionUnavailable("Project execution fence file is unsafe.")
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Project execution fence wait was cancelled.")
            if time.monotonic() >= deadline:
                raise TimeoutError("Project execution fence wait exceeded its deadline.")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return descriptor
            except BlockingIOError:
                if cancel_event is not None:
                    cancel_event.wait(min(_POLL_SECONDS, max(0.0, deadline - time.monotonic())))
                else:
                    time.sleep(min(_POLL_SECONDS, max(0.0, deadline - time.monotonic())))
    except BaseException:
        os.close(descriptor)
        raise


def _release_project_execution_fence(descriptor: int) -> None:
    try:
        import fcntl  # noqa: PLC0415
        with contextlib.suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        with contextlib.suppress(OSError):
            os.close(descriptor)


def _run_project_process(
    project_id: str,
    command: tuple[str, ...],
    *,
    python_source: Optional[str] = None,
    timeout_seconds: Optional[float],
    output_limit_bytes: int,
    cancel_event: Optional[threading.Event],
    output_callback,
    input_bytes: Optional[bytes],
    separate_stderr: bool,
    before_spawn,
    workspace_override = None,
    absolute_deadline: Optional[float] = None,
    execution_fence_id: Optional[str] = None,
) -> ProjectProcessResult:
    timeout = _timeout(timeout_seconds)
    deadline = (
        absolute_deadline
        if absolute_deadline is not None
        else math.inf
        if timeout is None
        else time.monotonic() + timeout
    )
    limit = _output_limit(output_limit_bytes)
    if input_bytes is not None:
        if not isinstance(input_bytes, bytes) or len(input_bytes) > MAX_STDIN_BYTES:
            raise AgentWorkspaceError("Project command input exceeds the supported size limit.")
    if not isinstance(separate_stderr, bool):
        raise AgentWorkspaceError("Project command output mode is invalid.")
    if cancel_event is not None and not isinstance(cancel_event, threading.Event):
        raise AgentWorkspaceError("Project command cancellation must use a threading event.")
    if cancel_event is not None and cancel_event.is_set():
        return ProjectProcessResult("cancelled", None, "", 0, False)

    try:
        status = _bounded_preparation(
            supervised_process_status,
            cancel_event = cancel_event,
            deadline = deadline,
            label = "security probe",
        )
    except InterruptedError:
        return ProjectProcessResult("cancelled", None, "", 0, False)
    except TimeoutError:
        return ProjectProcessResult("timed_out", None, "", 0, False)
    if not status.available:
        raise ProjectExecutionUnavailable(
            status.reason or "Secure supervised project commands are unavailable."
        )

    if workspace_override is not None:
        if (
            not isinstance(workspace_override, common.ProjectWorkspace)
            or workspace_override.project_id != project_id
        ):
            raise AgentWorkspaceError("Project workspace snapshot is invalid.")
        lease = contextlib.nullcontext(workspace_override)
    else:
        lease = common.project_workspace_access(
            project_id,
            cancel_event = cancel_event,
            deadline = deadline,
        )
    lease_entered = False
    boundary = None
    lifecycle: Optional[_BubblewrapLifecycle] = None
    process: Optional[subprocess.Popen] = None
    spawn_attempt: Optional[_SpawnAttempt] = None
    spawn_ownership = _SpawnOwnership()
    quarantined = False
    adopted = False
    after_spawn_done = False
    lifecycle_bound = False
    output = _OutputBuffer(limit, output_callback)
    stderr_output = _OutputBuffer(limit)
    scratch_script: Optional[str] = None
    execution_fence_fd: Optional[int] = None
    popen_preparation_detached = False
    result_status = "failed"
    exit_code: Optional[int] = None

    def close_pre_spawn_resources() -> None:
        nonlocal lease_entered, boundary, lifecycle, scratch_script, execution_fence_fd
        try:
            if lifecycle is not None:
                lifecycle.close()
        finally:
            lifecycle = None
            try:
                if scratch_script is not None:
                    with contextlib.suppress(OSError):
                        os.unlink(scratch_script)
                    scratch_script = None
                if boundary is not None:
                    try:
                        boundary.close()
                    finally:
                        boundary = None
            finally:
                try:
                    if lease_entered:
                        lease.__exit__(None, None, None)
                        lease_entered = False
                finally:
                    if execution_fence_fd is not None:
                        _release_project_execution_fence(execution_fence_fd)
                        execution_fence_fd = None

    def mark_popen_preparation_detached() -> None:
        nonlocal popen_preparation_detached
        popen_preparation_detached = True

    try:
        workspace = _bounded_preparation(
            lease.__enter__,
            cancel_event = cancel_event,
            deadline = deadline,
            label = "workspace lease acquisition",
            close_late_result = lambda _workspace: lease.__exit__(None, None, None),
        )
        lease_entered = True
        if time.monotonic() >= deadline:
            return ProjectProcessResult("timed_out", None, "", 0, False)
        if cancel_event is not None and cancel_event.is_set():
            return ProjectProcessResult("cancelled", None, "", 0, False)

        boundary = _bounded_preparation(
            lambda: execution.ProjectExecutionBoundary.open(
                workspace,
                cancel_event = cancel_event,
                deadline = deadline,
            ),
            cancel_event = cancel_event,
            deadline = deadline,
            label = "execution boundary setup",
            close_late_result = lambda opened: opened.close(),
        )
        if boundary.backend != "bubblewrap":
            raise ProjectExecutionUnavailable(
                "The opened project boundary cannot contain detached processes."
            )
        if not boundary.acquire_execution_slot(cancel_event, deadline):
            return ProjectProcessResult(
                "cancelled" if cancel_event is not None and cancel_event.is_set() else "timed_out",
                None,
                "",
                0,
                False,
            )
        if cancel_event is not None and cancel_event.is_set():
            return ProjectProcessResult("cancelled", None, "", 0, False)

        if execution_fence_id is not None:
            execution_fence_fd = _acquire_project_execution_fence(
                execution_fence_id,
                cancel_event,
                deadline,
            )
            if cancel_event is not None and cancel_event.is_set():
                return ProjectProcessResult("cancelled", None, "", 0, False)

        lifecycle = _BubblewrapLifecycle()
        if execution_fence_fd is not None:
            lifecycle.attach_execution_fence(execution_fence_fd)
        if python_source is not None:
            descriptor, scratch_script = tempfile.mkstemp(
                suffix = ".py",
                prefix = "studio_exec_",
                dir = str(boundary.scratch),
            )
            try:
                payload = python_source.encode("utf-8")
                view = memoryview(payload)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise AgentWorkspaceError("Project Python source could not be written.")
                    view = view[written:]
            finally:
                os.close(descriptor)
            command = (os.path.realpath(sys.executable), "-u", scratch_script)
        wrapped = lifecycle.wrap_argv(boundary.wrap_argv(command))
        environment = _minimal_environment(boundary, workspace, workspace.project_id)
        options = _bounded_preparation(
            lambda: _popen_options(
                boundary,
                environment,
                lifecycle,
                input_pipe = input_bytes is not None,
                separate_stderr = separate_stderr,
            ),
            cancel_event = cancel_event,
            deadline = deadline,
            label = "process option preparation",
            on_detach = mark_popen_preparation_detached,
            on_late_finish = close_pre_spawn_resources,
        )
        if cancel_event is not None and cancel_event.is_set():
            return ProjectProcessResult("cancelled", None, "", 0, False)

        initialize_parent_lifetime()

        def own_spawned_process(spawned: subprocess.Popen) -> None:
            # Persist the fresh session leader on the lifetime thread before the
            # bounded caller can stop waiting. If Studio dies while setup is
            # blocked, startup recovery can kill the recorded process group.
            adopt_pid(spawned.pid)
            spawn_ownership.adopted = True
            lifecycle.after_spawn(spawned)
            spawn_ownership.after_spawn_done = True

        def spawn_process() -> subprocess.Popen:
            # Hook authority is checked on the lifetime thread in the same call
            # that creates the process. No queue wait can sit between this check
            # and Popen.
            if before_spawn is not None:
                before_spawn(workspace)
            # Revalidation may block. Authority expiring while it runs must not
            # leave a late lifetime-thread Popen after the caller has already
            # returned timeout or cancellation and quarantined the boundary.
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Project process was cancelled before spawn.")
            if time.monotonic() >= deadline:
                raise TimeoutError("Project process deadline expired before spawn.")
            return subprocess.Popen(wrapped, **options)

        spawn_attempt = _SpawnAttempt(spawn_process, own_spawned_process)
        spawn_state = spawn_attempt.wait(
            cancel_event,
            max(0.0, min(_SPAWN_WAIT_SECONDS, deadline - time.monotonic())),
        )
        if spawn_state == "cancelled":
            return ProjectProcessResult("cancelled", None, "", 0, False)
        if spawn_state == "not_started":
            if time.monotonic() >= deadline:
                return ProjectProcessResult("timed_out", None, "", 0, False)
            raise ProjectExecutionUnavailable(
                "The project process spawn did not start within its preparation limit."
            )
        if spawn_state == "pending":
            _quarantine_run(
                lease,
                boundary,
                lifecycle,
                spawn_attempt = spawn_attempt,
                spawn_ownership = spawn_ownership,
                scratch_script = scratch_script,
                execution_fence_fd = execution_fence_fd,
            )
            execution_fence_fd = None
            quarantined = True
            if cancel_event is not None and cancel_event.is_set():
                return ProjectProcessResult("cancelled", None, "", 0, False)
            if time.monotonic() >= deadline:
                return ProjectProcessResult("timed_out", None, "", 0, False)
            raise ProjectProcessContainmentError(
                "The project process spawn is still resolving. Its workspace lease and "
                "mutation slot remain locked."
            )
        process = spawn_attempt.process
        adopted = spawn_ownership.adopted
        after_spawn_done = spawn_ownership.after_spawn_done
        process = spawn_attempt.result()
        bound_without_cancellation = lifecycle.bind(process, cancel_event, deadline)
        lifecycle_bound = True
        if process.stdout is None:
            raise RuntimeError("The supervised process has no output pipe.")
        descriptor = process.stdout.fileno()
        os.set_blocking(descriptor, False)
        stderr_descriptor = None
        if separate_stderr:
            if process.stderr is None:
                raise RuntimeError("The supervised process has no stderr pipe.")
            stderr_descriptor = process.stderr.fileno()
            os.set_blocking(stderr_descriptor, False)
        stdin_descriptor = None
        input_offset = 0
        if input_bytes is not None:
            if process.stdin is None:
                raise RuntimeError("The supervised process has no input pipe.")
            stdin_descriptor = process.stdin.fileno()
            os.set_blocking(stdin_descriptor, False)
            if not input_bytes:
                process.stdin.close()
                stdin_descriptor = None

        if not bound_without_cancellation or not lifecycle.release(cancel_event):
            result_status = "cancelled"
        else:
            stop = _StopSignal(cancel_event, deadline)
            while True:
                output.read_available(descriptor)
                if stderr_descriptor is not None:
                    stderr_output.read_available(stderr_descriptor)
                if stdin_descriptor is not None:
                    try:
                        written = os.write(stdin_descriptor, input_bytes[input_offset:])
                    except (BlockingIOError, InterruptedError):
                        written = 0
                    except BrokenPipeError:
                        written = len(input_bytes) - input_offset
                    input_offset += written
                    if input_offset >= len(input_bytes):
                        with contextlib.suppress(OSError, ValueError):
                            process.stdin.close()
                        stdin_descriptor = None
                observed_exit = process.poll()
                if observed_exit is not None:
                    exit_code = int(observed_exit)
                    result_status = "passed" if exit_code == 0 else "failed"
                    break
                reason = stop.reason()
                if reason is not None:
                    result_status = reason
                    break
                stop.wait(_POLL_SECONDS)
    except TimeoutError:
        result_status = "timed_out"
    except InterruptedError:
        result_status = "cancelled"
    finally:
        if process is not None:
            cleanup_run = _QuarantinedRun(
                lease = lease,
                boundary = boundary,
                lifecycle = lifecycle,
                process = process,
                scratch_script = scratch_script,
                adopted = adopted or spawn_ownership.adopted,
                after_spawn_done = after_spawn_done or spawn_ownership.after_spawn_done,
                lifecycle_bound = lifecycle_bound,
                execution_fence_fd = execution_fence_fd,
            )
            execution_fence_fd = None
            try:
                if lifecycle is None:
                    raise ProjectProcessContainmentError(
                        "The spawned process has no lifecycle boundary."
                    )
                _prepare_quarantined_process(cleanup_run)
                if cleanup_run.tree_proven:
                    reaped_code = process.poll()
                    if reaped_code is None:
                        raise ProjectProcessContainmentError(
                            "The failed bubblewrap setup monitor was not reaped."
                        )
                else:
                    reaped_code = lifecycle.terminate_and_prove(process)
                    cleanup_run.tree_proven = True
                if process.stdout is not None:
                    with contextlib.suppress(OSError, ValueError):
                        output.read_available(process.stdout.fileno(), max_chunks = 64)
                if separate_stderr and process.stderr is not None:
                    with contextlib.suppress(OSError, ValueError):
                        stderr_output.read_available(process.stderr.fileno(), max_chunks = 64)
                _advance_quarantined_run(cleanup_run)
            except BaseException as exc:
                if (
                    lease_entered
                    and boundary is not None
                    and lifecycle is not None
                    and not quarantined
                ):
                    _register_quarantined_run(cleanup_run)
                    quarantined = True
                raise ProjectProcessContainmentError(
                    "The process tree could not be proven dead. Its workspace lease and "
                    "mutation slot remain locked."
                ) from exc
            if result_status in {"passed", "failed"} and exit_code is None:
                exit_code = reaped_code
                result_status = "passed" if exit_code == 0 else "failed"
            quarantined = True
        if not quarantined and not popen_preparation_detached:
            close_pre_spawn_resources()

    rendered, output_bytes, truncated, truncation_notice = output.result()
    rendered_stderr, stderr_bytes, stderr_truncated, _stderr_notice = stderr_output.result()
    if result_status in {"cancelled", "timed_out"}:
        exit_code = None
    return ProjectProcessResult(
        status = result_status,
        exit_code = exit_code,
        output = rendered,
        output_bytes = output_bytes,
        output_truncated = truncated,
        truncation_notice = truncation_notice,
        stderr = rendered_stderr,
        stderr_bytes = stderr_bytes,
        stderr_truncated = stderr_truncated,
    )


__all__ = [
    "DEFAULT_OUTPUT_LIMIT_BYTES",
    "MAX_ARGV_BYTES",
    "MAX_ARGV_ITEMS",
    "MAX_OUTPUT_LIMIT_BYTES",
    "MAX_PYTHON_SOURCE_BYTES",
    "MAX_TIMEOUT_SECONDS",
    "ProjectProcessContainmentError",
    "ProjectProcessResult",
    "retry_quarantined_project_processes",
    "run_project_process",
    "run_project_python",
    "supervised_process_status",
]
