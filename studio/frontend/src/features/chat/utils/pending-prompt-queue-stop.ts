// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type PendingPromptQueueStopDetail = {
  threadIds?: string[];
  temporaryOnly?: boolean;
  localOnly?: boolean;
};

/** Cancel queue factories that have not finished settings hydration yet. */
export function cancelPendingPromptQueueFactoriesForStop<
  T extends { temporary: boolean; cancelled: boolean },
>(
  pendingFactories: Map<string, T>,
  aliases: string[],
  detail: PendingPromptQueueStopDetail,
): void {
  const { threadIds, temporaryOnly, localOnly } = detail;
  if (localOnly) {
    // Advancing the model boundary invalidates local factories once hydrated.
    // External factories must remain intact.
    return;
  }
  if (
    threadIds &&
    threadIds.length > 0 &&
    !threadIds.some((threadId) => aliases.includes(threadId))
  ) {
    return;
  }
  for (const [key, reservation] of pendingFactories) {
    if (temporaryOnly && !reservation.temporary) continue;
    reservation.cancelled = true;
    pendingFactories.delete(key);
  }
}
