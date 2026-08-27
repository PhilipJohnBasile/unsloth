# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Full-request project workspace mutation fence."""

from __future__ import annotations

import asyncio
import threading
from typing import Optional

from .guidance import project_id_from_session


class ProjectWorkspaceRequestLease:
    def __init__(self, context) -> None:
        self._context = context
        self._released = False
        self._lock = threading.Lock()

    @classmethod
    async def acquire(cls, session_id: Optional[str]):
        project_id = await asyncio.to_thread(project_id_from_session, session_id)
        if project_id is None:
            return None
        from core.inference.tools import project_workspace_in_flight

        context = project_workspace_in_flight(project_id)
        await asyncio.to_thread(context.__enter__)
        return cls(context)

    async def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        await asyncio.to_thread(self._context.__exit__, None, None, None)


__all__ = ["ProjectWorkspaceRequestLease"]
