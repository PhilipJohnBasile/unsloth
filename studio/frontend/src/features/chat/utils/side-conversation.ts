// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ThreadMessage } from "@assistant-ui/core";
import {
  discardQueuedChatRunSettings,
  registerQueuedChatRunSettings,
  updateQueuedChatRunSettings,
  type QueuedChatRunSettings,
} from "./queued-chat-run-settings.ts";

export const SIDE_CONVERSATION_INSTRUCTION =
  "This is an ephemeral side conversation. The earlier messages are a read-only snapshot of the parent conversation. Keep this side conversation separate: do not imply that its messages or conclusions were added to the parent conversation.";

export type ParsedSideCommand = {
  matched: boolean;
  prompt: string;
};

type SideConversationBase = {
  owner: number;
  parentThreadId: string;
  parentRemoteId: string;
  inheritedMessages: readonly ThreadMessage[];
  parentSettings: QueuedChatRunSettings;
  runSettings: QueuedChatRunSettings;
  pendingPrompt: string | null;
  createdAt: number;
};

export type SideConversationSession = SideConversationBase & {
  sideThreadId: string | null;
  phase: "launching" | "active" | "cleanup";
  settingsRegistrationId: number | null;
  parentSettingsRestored: boolean;
};

export type SideConversation = SideConversationSession & {
  sideThreadId: string;
  phase: "active";
};

export type ClaimedSideConversation = SideConversationSession & {
  sideThreadId: string;
};

export type SideConversationDisposalActions = {
  currentThreadId: () => string | null | undefined;
  cancelSideRun: (sideThreadId: string) => void;
  stopSideQueues: (sideThreadId: string) => void;
  abortPendingSideTransition: (
    sideThreadId: string,
  ) => void | Promise<void>;
  switchToParent: (parentThreadId: string) => Promise<void>;
  restoreParentPresentation: (conversation: ClaimedSideConversation) => void;
  disposeSide: (sideThreadId: string) => Promise<void>;
  clearSideUsage: (sideThreadId: string) => void;
};

export type SideConversationDisposalOptions = {
  /**
   * A detached assistant-ui runtime can leave its original switch promise
   * pending. Bound that third-party promise so the owner fence can still be
   * released after the runtime has been quarantined.
   */
  launchTransitionWaitMs?: number;
};

let activeSideConversation: SideConversationSession | null = null;
let nextSideConversationOwner = 1;
const listeners = new Set<() => void>();
const launchTransitions = new Map<number, Promise<void>>();
const cleanupWaiters = new Map<number, Set<() => void>>();
const SIDE_COMMAND_PATTERN = /^\s*\/side(?:\s+([\s\S]*?))?\s*$/;
const DEFAULT_LAUNCH_TRANSITION_DISPOSAL_WAIT_MS = 500;

function publish(): void {
  for (const listener of listeners) {
    listener();
  }
}

function resolveCleanupWaiters(owner: number): void {
  const waiters = cleanupWaiters.get(owner);
  cleanupWaiters.delete(owner);
  for (const resolve of waiters ?? []) resolve();
}

export function subscribeSideConversation(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function parseSideCommand(value: string): ParsedSideCommand {
  const match = SIDE_COMMAND_PATTERN.exec(value);
  if (!match) {
    return { matched: false, prompt: "" };
  }
  return { matched: true, prompt: (match[1] ?? "").trim() };
}

export function snapshotStableSideHistory(
  messages: readonly ThreadMessage[],
): readonly ThreadMessage[] {
  return Object.freeze(
    messages
      .filter(
        (message) =>
          message.role !== "assistant" ||
          message.status?.type === "complete",
      )
      .map((message) => structuredClone(message)),
  );
}

export function beginSideConversationLaunch(args: {
  parentThreadId: string;
  parentRemoteId: string;
  inheritedMessages: readonly ThreadMessage[];
  parentSettings: QueuedChatRunSettings;
  pendingPrompt?: string;
  createdAt?: number;
}): number {
  if (activeSideConversation) {
    throw new Error("A side conversation is already active.");
  }
  const owner = nextSideConversationOwner++;
  activeSideConversation = {
    ...args,
    owner,
    runSettings: args.parentSettings,
    pendingPrompt: args.pendingPrompt?.trim() || null,
    sideThreadId: null,
    phase: "launching",
    settingsRegistrationId: null,
    parentSettingsRestored: false,
    createdAt: args.createdAt ?? Date.now(),
  };
  publish();
  return owner;
}

export function claimSideConversationLaunchThread(
  owner: number,
  sideThreadId: string,
): void {
  const launch = activeSideConversation;
  if (
    !launch ||
    launch.owner !== owner ||
    launch.sideThreadId !== null ||
    launch.phase !== "launching"
  ) {
    throw new Error("No side conversation launch is pending.");
  }
  const settingsRegistrationId = registerQueuedChatRunSettings(
    [sideThreadId],
    launch.runSettings,
  );
  activeSideConversation = {
    ...launch,
    sideThreadId,
    settingsRegistrationId,
  };
  publish();
}

export function finishSideConversationLaunch(
  owner: number,
  sideThreadId: string,
): SideConversation {
  const launch = activeSideConversation;
  if (
    !launch ||
    launch.owner !== owner ||
    launch.sideThreadId !== sideThreadId ||
    launch.phase !== "launching"
  ) {
    throw new Error("No side conversation launch is pending.");
  }
  activeSideConversation = { ...launch, phase: "active" };
  publish();
  return activeSideConversation as SideConversation;
}

/**
 * A reserved assistant-ui thread is not deletable until initialize() changes
 * it from `new` to `regular`. Track the whole initialize plus switch transition
 * so every cleanup door waits for the same operation before inspecting the
 * current thread or deleting it.
 */
export async function activateSideConversationAfterInitialization(
  owner: number,
  sideThreadId: string,
  initialize: () => Promise<unknown>,
  switchToSide: () => Promise<unknown>,
): Promise<SideConversation> {
  const transition = (async () => {
    const [initialized, switched] = await Promise.allSettled([
      initialize(),
      switchToSide(),
    ]);
    if (initialized.status === "rejected") throw initialized.reason;
    if (switched.status === "rejected") throw switched.reason;
  })();
  launchTransitions.set(owner, transition);
  void transition.then(
    () => {
      if (launchTransitions.get(owner) === transition) {
        launchTransitions.delete(owner);
      }
    },
    () => {
      if (launchTransitions.get(owner) === transition) {
        launchTransitions.delete(owner);
      }
    },
  );
  await transition;
  return finishSideConversationLaunch(owner, sideThreadId);
}

export function isSideConversationOwner(owner: number): boolean {
  return activeSideConversation?.owner === owner;
}

export function getSideConversationSession(): SideConversationSession | null {
  return activeSideConversation;
}

export function updateSideConversationRunSettings(
  sideThreadId: string,
  settings: QueuedChatRunSettings,
): boolean {
  const session = activeSideConversation;
  if (
    !session ||
    session.sideThreadId !== sideThreadId ||
    session.settingsRegistrationId === null ||
    session.phase === "cleanup"
  ) {
    return false;
  }
  activeSideConversation = { ...session, runSettings: settings };
  updateQueuedChatRunSettings(session.settingsRegistrationId, settings);
  publish();
  return true;
}

/**
 * Fence every pending or active callback behind one owner. A launch with no
 * claimed thread can be forgotten immediately. Once a runtime thread exists,
 * ownership stays in the cleanup phase until its deletion is confirmed.
 */
export function requestSideConversationCleanup(
  owner: number,
): SideConversationSession | null {
  const session = activeSideConversation;
  if (!session || session.owner !== owner) return null;
  if (session.sideThreadId === null) {
    activeSideConversation = null;
    launchTransitions.delete(owner);
    publish();
    resolveCleanupWaiters(owner);
    return null;
  }
  if (session.phase !== "cleanup") {
    activeSideConversation = { ...session, phase: "cleanup" };
    publish();
  }
  return activeSideConversation;
}

/** Atomically hands the inline prompt to the mounted, active side Composer. */
export function takeSideConversationPendingPrompt(
  owner: number,
  sideThreadId: string,
): string | null {
  const session = activeSideConversation;
  if (
    !session ||
    session.owner !== owner ||
    session.sideThreadId !== sideThreadId ||
    session.phase !== "active" ||
    session.pendingPrompt === null
  ) {
    return null;
  }
  const prompt = session.pendingPrompt;
  activeSideConversation = { ...session, pendingPrompt: null };
  publish();
  return prompt;
}

/** Claim the one allowed restore so cleanup retries cannot overwrite later parent edits. */
export function takeSideConversationParentSettingsForRestore(
  owner: number,
): QueuedChatRunSettings | null {
  const session = activeSideConversation;
  if (!session || session.owner !== owner || session.parentSettingsRestored) {
    return null;
  }
  activeSideConversation = { ...session, parentSettingsRestored: true };
  publish();
  return session.parentSettings;
}

export function completeSideConversationCleanup(
  owner: number,
  sideThreadId: string,
): boolean {
  const session = activeSideConversation;
  if (
    !session ||
    session.owner !== owner ||
    session.sideThreadId !== sideThreadId ||
    session.phase !== "cleanup"
  ) {
    return false;
  }
  if (session.settingsRegistrationId !== null) {
    discardQueuedChatRunSettings(session.settingsRegistrationId);
  }
  activeSideConversation = null;
  launchTransitions.delete(owner);
  publish();
  resolveCleanupWaiters(owner);
  return true;
}

export function waitForSideConversationCleanup(owner: number): Promise<void> {
  if (activeSideConversation?.owner !== owner) return Promise.resolve();
  return new Promise((resolve) => {
    const waiters = cleanupWaiters.get(owner) ?? new Set<() => void>();
    waiters.add(resolve);
    cleanupWaiters.set(owner, waiters);
  });
}

/**
 * The only side disposal transition. In particular, delete() must never be
 * allowed to auto-switch to an unrelated blank thread, and a late launch must
 * never land after deletion.
 */
export async function disposeSideConversation(
  owner: number,
  actions: SideConversationDisposalActions,
  options: SideConversationDisposalOptions = {},
): Promise<boolean> {
  let session = requestSideConversationCleanup(owner);
  if (!session?.sideThreadId) return false;

  const sideThreadId = session.sideThreadId;
  actions.cancelSideRun(sideThreadId);
  actions.stopSideQueues(sideThreadId);
  await actions.abortPendingSideTransition(sideThreadId);

  const launchTransition = launchTransitions.get(owner);
  if (launchTransition) {
    let waitTimer: ReturnType<typeof setTimeout> | undefined;
    const transitionSettled = await Promise.race([
      launchTransition.then(
        () => true,
        () => true,
      ),
      new Promise<false>((resolve) => {
        waitTimer = setTimeout(
          () => resolve(false),
          options.launchTransitionWaitMs ??
            DEFAULT_LAUNCH_TRANSITION_DISPOSAL_WAIT_MS,
        );
      }),
    ]);
    if (waitTimer) clearTimeout(waitTimer);
    if (!transitionSettled) {
      // The first abort may have raced initialize() changing `new` to
      // `regular`. Try the quarantine again at the deadline, then stop
      // waiting on assistant-ui's unresolved switch. The owner check in
      // finishSideConversationLaunch rejects any callback that arrives later.
      await actions.abortPendingSideTransition(sideThreadId);
      if (launchTransitions.get(owner) === launchTransition) {
        launchTransitions.delete(owner);
      }
    }
  }

  session = activeSideConversation;
  if (
    !session ||
    session.owner !== owner ||
    session.sideThreadId !== sideThreadId ||
    session.phase !== "cleanup"
  ) {
    return false;
  }
  const claimed = session as ClaimedSideConversation;

  if (actions.currentThreadId() === sideThreadId) {
    await actions.switchToParent(claimed.parentThreadId);
  }
  actions.restoreParentPresentation(claimed);
  await actions.disposeSide(sideThreadId);
  actions.clearSideUsage(sideThreadId);
  if (!completeSideConversationCleanup(owner, sideThreadId)) {
    throw new Error("Side chat cleanup ownership changed.");
  }
  return true;
}

export function cancelSideConversationLaunch(owner: number): void {
  requestSideConversationCleanup(owner);
}

export function endSideConversation(sideThreadId: string): void {
  const session = activeSideConversation;
  if (!session || session.sideThreadId !== sideThreadId) return;
  requestSideConversationCleanup(session.owner);
  completeSideConversationCleanup(session.owner, sideThreadId);
}

export function getActiveSideConversation(): SideConversation | null {
  return activeSideConversation?.phase === "active" &&
    activeSideConversation.sideThreadId
    ? (activeSideConversation as SideConversation)
    : null;
}

/** The resident model is installation-wide, so every side lifecycle phase owns this fence. */
export function sideConversationBlocksModelLifecycle(): boolean {
  return activeSideConversation !== null;
}

export function createSideConversationLaunchDeadline(
  owner: number,
  timeoutMs: number,
  onExpired: (session: SideConversationSession | null) => void,
): () => void {
  const timer = setTimeout(() => {
    if (!isSideConversationOwner(owner)) return;
    onExpired(requestSideConversationCleanup(owner));
  }, timeoutMs);
  return () => clearTimeout(timer);
}

export function sideConversationForThread(
  threadId: string | null | undefined,
): ClaimedSideConversation | null {
  const session = activeSideConversation;
  return session?.sideThreadId === threadId
    ? (session as ClaimedSideConversation)
    : null;
}

/**
 * The persisted thread whose project and workspace authority a run belongs to.
 * Side threads are intentionally incognito and have no database row, so callers
 * must resolve authority through the saved parent rather than through the side
 * thread or whichever project is currently open in the composer.
 */
export function sideConversationAuthorityThreadId(
  threadId: string | undefined,
): string | undefined {
  return sideConversationForThread(threadId)?.parentRemoteId ?? threadId;
}

export function sideConversationVisibleThreadId(
  requestedThreadId: string | null | undefined,
  mountedThreadId: string | null | undefined,
): string | null | undefined {
  const side = sideConversationForThread(mountedThreadId);
  if (
    !side ||
    (requestedThreadId !== side.parentRemoteId &&
      requestedThreadId !== side.parentThreadId &&
      requestedThreadId !== side.sideThreadId)
  ) {
    return requestedThreadId;
  }
  return side.sideThreadId;
}

export function isSideConversationThread(
  threadId: string | null | undefined,
): boolean {
  const session = activeSideConversation;
  return Boolean(
    threadId && session?.sideThreadId && session.sideThreadId === threadId,
  );
}

/** Keeps the route-owned parent switch from reclaiming the runtime while `/side` is open. */
export function sideConversationOwnsRuntimeSwitch(
  routeThreadId: string,
  runtimeThreadId: string | null | undefined,
): boolean {
  const active = activeSideConversation;
  if (!active || active.parentThreadId !== routeThreadId) {
    return false;
  }
  return active.sideThreadId !== null && active.sideThreadId === runtimeThreadId;
}

export function sideConversationMessagesForRun<T extends ThreadMessage>(
  threadId: string | null | undefined,
  messages: readonly T[],
): readonly T[] {
  const side = sideConversationForThread(threadId);
  if (!side) {
    return messages;
  }
  return [...(side.inheritedMessages as readonly T[]), ...messages];
}

export function sideConversationEffectiveMessagesForRun<
  T extends ThreadMessage,
>(
  threadId: string | null | undefined,
  survivingMessages: readonly T[],
  reprune: (messages: readonly T[]) => readonly T[],
): readonly T[] {
  const withInherited = sideConversationMessagesForRun(
    threadId,
    survivingMessages,
  );
  return withInherited === survivingMessages
    ? survivingMessages
    : reprune(withInherited);
}

export function sideConversationInstructionForThread(
  threadId: string | null | undefined,
): string {
  return isSideConversationThread(threadId)
    ? SIDE_CONVERSATION_INSTRUCTION
    : "";
}
