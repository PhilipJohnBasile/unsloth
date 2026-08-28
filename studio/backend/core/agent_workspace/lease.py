# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Full-request project workspace mutation fence."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Optional

from .guidance import project_id_from_session


_ACQUISITION_ACK_SECONDS = 5.0
_OWNER_POLL_SECONDS = 0.05


def _publish_context_acquisition(publication: asyncio.Future) -> None:
    if not publication.done():
        publication.set_result(None)


async def _acquire_context_owned(context) -> None:
    """Transfer a blocking context entry to the caller or a late releaser."""
    loop = asyncio.get_running_loop()
    publication = loop.create_future()
    state_changed = threading.Condition()
    state = {
        "disposition": "unclaimed",
        "error": None,
    }

    def owner() -> None:
        error = None
        try:
            context.__enter__()
        except BaseException as exc:
            error = exc
        with state_changed:
            state["error"] = error
        try:
            loop.call_soon_threadsafe(_publish_context_acquisition, publication)
        except RuntimeError:
            with state_changed:
                if state["disposition"] == "unclaimed":
                    state["disposition"] = "detached"
                    state_changed.notify_all()
        if error is not None:
            return
        acknowledgement_deadline = time.monotonic() + _ACQUISITION_ACK_SECONDS
        with state_changed:
            while state["disposition"] == "unclaimed":
                remaining = acknowledgement_deadline - time.monotonic()
                if remaining <= 0 or loop.is_closed() or not loop.is_running():
                    state["disposition"] = "detached"
                    state_changed.notify_all()
                    break
                state_changed.wait(min(_OWNER_POLL_SECONDS, remaining))
            detached = state["disposition"] == "detached"
        if detached:
            try:
                context.__exit__(None, None, None)
            except BaseException:
                pass

    loop.run_in_executor(None, owner)
    try:
        await asyncio.shield(publication)
    except BaseException:
        with state_changed:
            if state["disposition"] == "unclaimed":
                state["disposition"] = "detached"
                state_changed.notify_all()
        raise
    with state_changed:
        if state["disposition"] != "unclaimed":
            raise RuntimeError("Project workspace lease acquisition was abandoned.")
        state["disposition"] = "claimed"
        state_changed.notify_all()
        error = state["error"]
    if error is not None:
        raise error


class ProjectWorkspaceRequestLease:
    def __init__(self, context) -> None:
        self._context = context
        self._released = False
        self._lock = threading.Lock()
        self._release_worker = None

    def _release_owned(self):
        error = None
        try:
            self._context.__exit__(None, None, None)
        except BaseException as exc:
            error = exc
        with self._lock:
            if error is None:
                self._released = True
            self._release_worker = None
        return error

    @classmethod
    async def acquire(cls, session_id: Optional[str]):
        project_id = await asyncio.to_thread(project_id_from_session, session_id)
        if project_id is None:
            return None
        from core.inference.tools import project_workspace_in_flight

        context = project_workspace_in_flight(project_id)
        await _acquire_context_owned(context)
        return cls(context)

    async def release(self) -> None:
        loop = asyncio.get_running_loop()
        with self._lock:
            if self._released:
                return
            worker = self._release_worker
            if worker is None:
                worker = _submit_release_owner(loop, self._release_owned)
                self._release_worker = worker
        error = await asyncio.shield(worker)
        if error is not None:
            raise error


def _submit_release_owner(loop, callback):
    return loop.run_in_executor(None, callback)


__all__ = ["ProjectWorkspaceRequestLease"]
