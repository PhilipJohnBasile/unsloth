// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type { ThreadMessage } from "@assistant-ui/core";
import type { QueuedChatRunSettings } from "../src/features/chat/utils/queued-chat-run-settings.ts";
import {
  SIDE_CONVERSATION_INSTRUCTION,
  activateSideConversationAfterInitialization,
  beginSideConversationLaunch,
  cancelSideConversationLaunch,
  claimSideConversationLaunchThread,
  completeSideConversationCleanup,
  createSideConversationLaunchDeadline,
  disposeSideConversation,
  endSideConversation,
  finishSideConversationLaunch,
  getActiveSideConversation,
  getSideConversationSession,
  isSideConversationOwner,
  isSideConversationThread,
  parseSideCommand,
  requestSideConversationCleanup,
  sideConversationAuthorityThreadId,
  sideConversationBlocksModelLifecycle,
  sideConversationEffectiveMessagesForRun,
  sideConversationInstructionForThread,
  sideConversationMessagesForRun,
  sideConversationOwnsRuntimeSwitch,
  snapshotStableSideHistory,
  takeSideConversationParentSettingsForRestore,
  updateSideConversationRunSettings,
  waitForSideConversationCleanup,
} from "../src/features/chat/utils/side-conversation.ts";
import { sandboxSessionIdFor } from "../src/components/assistant-ui/sandbox-files.ts";
import { consumeQueuedChatRunSettings } from "../src/features/chat/utils/queued-chat-run-settings.ts";
import { studioToolHistoryRequestFields } from "../src/features/chat/utils/studio-tool-history.ts";
import {
  cancelPendingPromptQueueFactoriesForStop,
} from "../src/features/chat/utils/pending-prompt-queue-stop.ts";

const ALREADY_ACTIVE_PATTERN = /already active/;
const PARENT_SETTINGS = {
  params: { checkpoint: "parent-model", temperature: 0.7 },
  toolsEnabled: true,
  permissionMode: "ask",
} as unknown as QueuedChatRunSettings;

test("an exact side queue stop cancels a hydration-pending factory", () => {
  const sideReservation = { temporary: false, cancelled: false };
  const otherReservation = { temporary: false, cancelled: false };
  const unrelatedReservation = { temporary: false, cancelled: false };
  const pending = new Map([
    ["side", sideReservation],
    ["other", otherReservation],
  ]);
  const detail = { threadIds: ["side-local"] };

  cancelPendingPromptQueueFactoriesForStop(
    pending,
    ["side-local", "side-remote"],
    detail,
  );

  assert.equal(sideReservation.cancelled, true);
  assert.equal(otherReservation.cancelled, true);
  assert.equal(pending.size, 0);

  const unrelated = new Map([["unrelated", unrelatedReservation]]);
  cancelPendingPromptQueueFactoriesForStop(
    unrelated,
    ["other-thread"],
    detail,
  );
  assert.equal(unrelatedReservation.cancelled, false);
  assert.equal(unrelated.size, 1);
});

function message(
  id: string,
  role: "user" | "assistant",
  text: string,
  status?: "running" | "requires-action" | "complete" | "incomplete",
): ThreadMessage {
  return {
    id,
    role,
    content: [{ type: "text", text }],
    createdAt: new Date(0),
    metadata: { custom: {} },
    ...(role === "user"
      ? { attachments: [] }
      : {
          status:
            status === "running"
              ? { type: "running" }
              : status === "requires-action"
                ? { type: "requires-action", reason: "tool-calls" }
                : status === "incomplete"
                  ? { type: "incomplete", reason: "cancelled" }
                  : { type: "complete", reason: "stop" },
        }),
  } as ThreadMessage;
}

function resetSide(): void {
  const session = getSideConversationSession();
  if (!session) return;
  if (session.sideThreadId) endSideConversation(session.sideThreadId);
  else cancelSideConversationLaunch(session.owner);
}

test.afterEach(resetSide);

test("parses only a complete /side command and preserves an optional prompt", () => {
  assert.deepEqual(parseSideCommand("/side"), { matched: true, prompt: "" });
  assert.deepEqual(parseSideCommand("  /side   inspect this\ncarefully  "), {
    matched: true,
    prompt: "inspect this\ncarefully",
  });
  assert.equal(parseSideCommand("/sidebar").matched, false);
  assert.equal(parseSideCommand("please /side this").matched, false);
  assert.equal(parseSideCommand("/SIDE").matched, false);
});

test("takes a detached stable snapshot without partial assistant output", () => {
  const parent = [
    message("u1", "user", "first"),
    message("a1", "assistant", "done", "complete"),
    message("u2", "user", "current"),
    message("a2", "assistant", "partial", "running"),
    message("a3", "assistant", "tool pending", "requires-action"),
    message("a4", "assistant", "cancelled partial", "incomplete"),
  ];

  const snapshot = snapshotStableSideHistory(parent);
  assert.deepEqual(
    snapshot.map((item) => item.id),
    ["u1", "a1", "u2"],
  );
  assert.notEqual(snapshot[0], parent[0]);
  (parent[0].content[0] as { text: string }).text = "mutated";
  assert.equal((snapshot[0]?.content[0] as { text: string }).text, "first");
});

test("the route fence applies only while its own parent is launching or open", () => {
  const owner = beginSideConversationLaunch({
    parentThreadId: "parent-local",
    parentRemoteId: "parent-remote",
    inheritedMessages: [],
    parentSettings: PARENT_SETTINGS,
    createdAt: 1,
  });
  assert.equal(
    sideConversationOwnsRuntimeSwitch("parent-local", "new-local"),
    false,
  );
  assert.equal(
    sideConversationOwnsRuntimeSwitch("another-parent", "new-local"),
    false,
  );

  claimSideConversationLaunchThread(owner, "side-local");
  assert.equal(
    sideConversationOwnsRuntimeSwitch("parent-local", "side-local"),
    true,
  );
  finishSideConversationLaunch(owner, "side-local");
  assert.equal(
    sideConversationOwnsRuntimeSwitch("parent-local", "unrelated"),
    false,
  );
});

test("side runs receive inherited context without adding it to the visible transcript", () => {
  const inherited = [message("parent", "user", "parent context")];
  const owner = beginSideConversationLaunch({
    parentThreadId: "parent-local",
    parentRemoteId: "parent-remote",
    inheritedMessages: inherited,
    parentSettings: PARENT_SETTINGS,
    createdAt: 1,
  });
  claimSideConversationLaunchThread(owner, "side-local");
  finishSideConversationLaunch(owner, "side-local");
  const visible = [message("side", "user", "side prompt")];

  const outbound = sideConversationMessagesForRun("side-local", visible);
  assert.deepEqual(
    outbound.map((item) => item.id),
    ["parent", "side"],
  );
  assert.deepEqual(
    visible.map((item) => item.id),
    ["side"],
    "the side transcript remains separate",
  );
  assert.equal(
    sideConversationInstructionForThread("side-local"),
    SIDE_CONVERSATION_INSTRUCTION,
  );
  assert.deepEqual(
    sideConversationMessagesForRun("parent-local", visible),
    visible,
  );
  assert.equal(sideConversationInstructionForThread("parent-local"), "");
});

test("effective side history carries inherited modality and tool ownership", () => {
  const inheritedImage = {
    ...message("parent-image", "user", "inspect this"),
    content: [
      {
        type: "image",
        image: "data:image/png;base64,cGFyZW50",
      },
    ],
  } as unknown as ThreadMessage;
  const inheritedTool = {
    ...message("parent-tool", "assistant", "", "complete"),
    content: [
      {
        type: "tool-call",
        toolName: "python",
        toolCallId: "call-parent",
        args: { code: "print(1)" },
        result: "1",
        provenance: { source: "local" },
      },
    ],
  } as unknown as ThreadMessage;
  const owner = beginSideConversationLaunch({
    parentThreadId: "parent-local",
    parentRemoteId: "parent-remote",
    inheritedMessages: [inheritedImage, inheritedTool],
    parentSettings: PARENT_SETTINGS,
  });
  claimSideConversationLaunchThread(owner, "side-local");
  finishSideConversationLaunch(owner, "side-local");

  const visible = [message("side-user", "user", "continue")];
  const effective = sideConversationEffectiveMessagesForRun(
    "side-local",
    visible,
    (messages) => messages,
  );

  assert.equal(
    effective.some((item) =>
      item.content.some((part) => part.type === "image"),
    ),
    true,
    "the inherited image must reach the adapter's modality gate",
  );
  assert.deepEqual(
    studioToolHistoryRequestFields(effective),
    { studio_tool_history: true },
  );
  assert.deepEqual(
    visible.map((item) => item.id),
    ["side-user"],
    "effective payload derivation must not mutate the visible side transcript",
  );
});

test("side project authority comes only from the persisted parent", () => {
  const owner = beginSideConversationLaunch({
    parentThreadId: "parent-local",
    parentRemoteId: "parent-remote",
    inheritedMessages: [message("parent", "user", "parent context")],
    parentSettings: PARENT_SETTINGS,
    createdAt: 1,
  });
  claimSideConversationLaunchThread(owner, "side-local");
  finishSideConversationLaunch(owner, "side-local");

  const persistedThreads = new Map([
    ["parent-remote", { projectId: "parent-project" }],
    ["side-local", { projectId: "wrong-side-project" }],
  ]);
  const projects = new Map([
    ["parent-project", { instructions: "Use the parent project rules." }],
    ["composer-project", { instructions: "Wrong active composer rules." }],
  ]);
  const activeComposerProjectId = "composer-project";

  const authorityThreadId = sideConversationAuthorityThreadId("side-local");
  const projectId = authorityThreadId
    ? (persistedThreads.get(authorityThreadId)?.projectId ?? null)
    : activeComposerProjectId;

  assert.equal(authorityThreadId, "parent-remote");
  assert.equal(projectId, "parent-project");
  assert.equal(
    projects.get(projectId ?? "")?.instructions,
    "Use the parent project rules.",
  );
  assert.equal(
    sandboxSessionIdFor(authorityThreadId, projectId),
    "project-parent-project",
  );
  assert.equal(
    sandboxSessionIdFor(authorityThreadId, null),
    "parent-remote",
    "a side outside a project still shares the saved parent's sandbox",
  );
  assert.deepEqual(
    sideConversationMessagesForRun("side-local", [
      message("side", "user", "side prompt"),
    ]).map((item) => item.id),
    ["parent", "side"],
  );

  requestSideConversationCleanup(owner);
  assert.equal(
    sideConversationAuthorityThreadId("side-local"),
    "parent-remote",
    "project and sandbox authority survive until deletion completes",
  );
  assert.equal(
    sandboxSessionIdFor(
      sideConversationAuthorityThreadId("side-local"),
      projectId,
    ),
    "project-parent-project",
  );
  assert.deepEqual(
    sideConversationMessagesForRun("side-local", [
      message("cleanup-side", "user", "cleanup"),
    ]).map((item) => item.id),
    ["parent", "cleanup-side"],
  );
  assert.equal(
    sideConversationInstructionForThread("side-local"),
    SIDE_CONVERSATION_INSTRUCTION,
  );
  completeSideConversationCleanup(owner, "side-local");
});

test("nested side launches fail and ending the side releases the session", () => {
  const owner = beginSideConversationLaunch({
    parentThreadId: "parent-local",
    parentRemoteId: "parent-remote",
    inheritedMessages: [],
    parentSettings: PARENT_SETTINGS,
  });
  claimSideConversationLaunchThread(owner, "side-local");
  finishSideConversationLaunch(owner, "side-local");

  assert.throws(
    () =>
      beginSideConversationLaunch({
        parentThreadId: "side-local",
        parentRemoteId: "side-local",
        inheritedMessages: [],
        parentSettings: PARENT_SETTINGS,
      }),
    ALREADY_ACTIVE_PATTERN,
  );
  endSideConversation("side-local");
  assert.equal(getActiveSideConversation(), null);
});

test("a successful return releases the launch gate for a second side", () => {
  const firstOwner = beginSideConversationLaunch({
    parentThreadId: "parent-local",
    parentRemoteId: "parent-remote",
    inheritedMessages: [],
    parentSettings: PARENT_SETTINGS,
  });
  claimSideConversationLaunchThread(firstOwner, "first-side");
  finishSideConversationLaunch(firstOwner, "first-side");
  requestSideConversationCleanup(firstOwner);
  assert.equal(
    completeSideConversationCleanup(firstOwner, "first-side"),
    true,
  );

  const secondOwner = beginSideConversationLaunch({
    parentThreadId: "parent-local",
    parentRemoteId: "parent-remote",
    inheritedMessages: [],
    parentSettings: PARENT_SETTINGS,
  });
  claimSideConversationLaunchThread(secondOwner, "second-side");
  finishSideConversationLaunch(secondOwner, "second-side");
  assert.equal(getActiveSideConversation()?.sideThreadId, "second-side");
});

test("the side owns one mutable run snapshot until cleanup succeeds", () => {
  const owner = beginSideConversationLaunch({
    parentThreadId: "parent-local",
    parentRemoteId: "parent-remote",
    inheritedMessages: [],
    parentSettings: PARENT_SETTINGS,
  });
  claimSideConversationLaunchThread(owner, "side-local");
  finishSideConversationLaunch(owner, "side-local");

  assert.equal(
    consumeQueuedChatRunSettings("side-local")?.params.checkpoint,
    "parent-model",
  );
  assert.equal(
    consumeQueuedChatRunSettings("side-local")?.params.checkpoint,
    "parent-model",
    "tool-loop adapter calls reuse the same snapshot",
  );

  const edited = {
    ...PARENT_SETTINGS,
    params: { ...PARENT_SETTINGS.params, temperature: 0.2 },
    toolsEnabled: false,
  };
  assert.equal(
    updateSideConversationRunSettings("side-local", edited),
    true,
  );
  assert.equal(
    consumeQueuedChatRunSettings("side-local")?.params.temperature,
    0.2,
  );
  assert.equal(consumeQueuedChatRunSettings("side-local")?.toolsEnabled, false);

  const cleanup = requestSideConversationCleanup(owner);
  assert.equal(cleanup?.phase, "cleanup");
  assert.equal(
    isSideConversationThread("side-local"),
    true,
    "cleanup retains the incognito side identity until deletion succeeds",
  );
  assert.equal(
    takeSideConversationParentSettingsForRestore(owner),
    PARENT_SETTINGS,
  );
  assert.equal(
    takeSideConversationParentSettingsForRestore(owner),
    null,
    "a cleanup retry must not restore over later parent edits",
  );
  assert.equal(
    consumeQueuedChatRunSettings("side-local")?.params.temperature,
    0.2,
    "failed cleanup keeps the side snapshot and ownership retryable",
  );
  assert.equal(isSideConversationOwner(owner), true);
  assert.equal(
    sideConversationAuthorityThreadId("side-local"),
    "parent-remote",
    "cleanup retains project and sandbox authority until disposal succeeds",
  );
  assert.deepEqual(
    sideConversationMessagesForRun("side-local", [
      message("side", "user", "still cleaning"),
    ]).map((item) => item.id),
    ["side"],
  );
  assert.equal(
    sideConversationInstructionForThread("side-local"),
    SIDE_CONVERSATION_INSTRUCTION,
  );

  assert.equal(completeSideConversationCleanup(owner, "side-local"), true);
  assert.equal(consumeQueuedChatRunSettings("side-local"), null);
  assert.equal(isSideConversationOwner(owner), false);
  assert.equal(isSideConversationThread("side-local"), false);
});

test("a blank side initializes before activation and cleanup is one ordered transition", async () => {
  const owner = beginSideConversationLaunch({
    parentThreadId: "parent-local",
    parentRemoteId: "parent-remote",
    inheritedMessages: [message("parent", "user", "context")],
    parentSettings: PARENT_SETTINGS,
  });
  claimSideConversationLaunchThread(owner, "side-local");
  const lifecycle: string[] = [];
  let initialized = false;

  const side = await activateSideConversationAfterInitialization(
    owner,
    "side-local",
    async () => {
      lifecycle.push("initialize");
      initialized = true;
    },
    async () => {
      assert.equal(initialized, true);
      lifecycle.push("open-side");
    },
  );
  assert.equal(side.phase, "active");
  assert.equal(sideConversationBlocksModelLifecycle(), true);

  const cleanup = disposeSideConversation(owner, {
    currentThreadId: () => "side-local",
    cancelSideRun: () => lifecycle.push("cancel"),
    stopSideQueues: (threadId) => lifecycle.push(`stop-queues:${threadId}`),
    abortPendingSideTransition: () => {
      lifecycle.push("abort-transition");
    },
    switchToParent: async () => {
      lifecycle.push("switch-parent");
    },
    restoreParentPresentation: () => lifecycle.push("restore-parent"),
    disposeSide: async () => {
      assert.equal(initialized, true);
      lifecycle.push("delete-side");
    },
    clearSideUsage: () => lifecycle.push("clear-usage"),
  });
  await cleanup;

  assert.deepEqual(lifecycle, [
    "initialize",
    "open-side",
    "cancel",
    "stop-queues:side-local",
    "abort-transition",
    "switch-parent",
    "restore-parent",
    "delete-side",
    "clear-usage",
  ]);
  assert.equal(getSideConversationSession(), null);
  assert.equal(sideConversationBlocksModelLifecycle(), false);
});

test("cleanup waits for a failed initialize and safely releases the reusable new slot", async () => {
  const owner = beginSideConversationLaunch({
    parentThreadId: "parent-local",
    parentRemoteId: "parent-remote",
    inheritedMessages: [],
    parentSettings: PARENT_SETTINGS,
  });
  claimSideConversationLaunchThread(owner, "side-local");
  const lifecycle: string[] = [];
  let rejectInitialize!: (error: Error) => void;
  const initialize = new Promise<void>((_resolve, reject) => {
    rejectInitialize = reject;
  });

  const activation = activateSideConversationAfterInitialization(
    owner,
    "side-local",
    () => initialize,
    async () => lifecycle.push("reserved-side-settled"),
  );
  const cleanup = disposeSideConversation(owner, {
    currentThreadId: () => "side-local",
    cancelSideRun: () => lifecycle.push("cancel"),
    stopSideQueues: (threadId) => lifecycle.push(`stop-queues:${threadId}`),
    abortPendingSideTransition: () => {
      lifecycle.push("abort-transition");
    },
    switchToParent: async () => {
      lifecycle.push("switch-parent");
    },
    restoreParentPresentation: () => lifecycle.push("restore-parent"),
    disposeSide: async () => {
      lifecycle.push("release-new-slot");
    },
    clearSideUsage: () => lifecycle.push("clear-usage"),
  });

  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.deepEqual(lifecycle, [
    "reserved-side-settled",
    "cancel",
    "stop-queues:side-local",
    "abort-transition",
  ]);
  rejectInitialize(new Error("initialize failed"));
  await assert.rejects(activation, /initialize failed/);
  await cleanup;
  await waitForSideConversationCleanup(owner);

  assert.deepEqual(lifecycle, [
    "reserved-side-settled",
    "cancel",
    "stop-queues:side-local",
    "abort-transition",
    "switch-parent",
    "restore-parent",
    "release-new-slot",
    "clear-usage",
  ]);
  assert.equal(getSideConversationSession(), null);
});

test("cleanup aborts a reserved side whose runtime switch never mounted", async () => {
  const owner = beginSideConversationLaunch({
    parentThreadId: "parent-local",
    parentRemoteId: "parent-remote",
    inheritedMessages: [],
    parentSettings: PARENT_SETTINGS,
  });
  claimSideConversationLaunchThread(owner, "side-local");
  let rejectSwitch!: (error: Error) => void;
  const neverMounted = new Promise<void>((_resolve, reject) => {
    rejectSwitch = reject;
  });
  const lifecycle: string[] = [];
  const activation = activateSideConversationAfterInitialization(
    owner,
    "side-local",
    async () => undefined,
    () => neverMounted,
  );
  const activationRejected = assert.rejects(activation, /mount cancelled/);

  const cleanup = disposeSideConversation(owner, {
    currentThreadId: () => "parent-local",
    cancelSideRun: () => lifecycle.push("cancel"),
    stopSideQueues: (threadId) => lifecycle.push(`stop-queues:${threadId}`),
    abortPendingSideTransition: () => {
      lifecycle.push("abort-transition");
      rejectSwitch(new Error("mount cancelled"));
    },
    switchToParent: async () => {
      lifecycle.push("unexpected-switch");
    },
    restoreParentPresentation: () => lifecycle.push("restore-parent"),
    disposeSide: async () => {
      lifecycle.push("delete-side");
    },
    clearSideUsage: () => lifecycle.push("clear-usage"),
  });

  await activationRejected;
  await cleanup;
  assert.deepEqual(lifecycle, [
    "cancel",
    "stop-queues:side-local",
    "abort-transition",
    "restore-parent",
    "delete-side",
    "clear-usage",
  ]);
  assert.equal(getSideConversationSession(), null);
});

test("a claimed launch with a stuck mount is quarantined and releases its owner", async () => {
  const owner = beginSideConversationLaunch({
    parentThreadId: "parent-local",
    parentRemoteId: "parent-remote",
    inheritedMessages: [],
    parentSettings: PARENT_SETTINGS,
  });
  claimSideConversationLaunchThread(owner, "stuck-side");
  let resolveMount!: () => void;
  const mount = new Promise<void>((resolve) => {
    resolveMount = resolve;
  });
  const activation = activateSideConversationAfterInitialization(
    owner,
    "stuck-side",
    async () => undefined,
    () => mount,
  );
  const activationRejected = assert.rejects(
    activation,
    /No side conversation launch is pending/,
  );
  const lifecycle: string[] = [];

  await disposeSideConversation(
    owner,
    {
      currentThreadId: () => "parent-local",
      cancelSideRun: () => lifecycle.push("cancel"),
      stopSideQueues: (threadId) => lifecycle.push(`stop-queues:${threadId}`),
      abortPendingSideTransition: () => {
        lifecycle.push("quarantine");
      },
      switchToParent: async () => {
        lifecycle.push("unexpected-switch");
      },
      restoreParentPresentation: () => lifecycle.push("restore-parent"),
      disposeSide: async () => {
        lifecycle.push("delete-side");
      },
      clearSideUsage: () => lifecycle.push("clear-usage"),
    },
    { launchTransitionWaitMs: 5 },
  );

  assert.deepEqual(lifecycle, [
    "cancel",
    "stop-queues:stuck-side",
    "quarantine",
    "quarantine",
    "restore-parent",
    "delete-side",
    "clear-usage",
  ]);
  assert.equal(getSideConversationSession(), null);

  const nextOwner = beginSideConversationLaunch({
    parentThreadId: "next-parent",
    parentRemoteId: "next-parent-remote",
    inheritedMessages: [],
    parentSettings: PARENT_SETTINGS,
  });
  resolveMount();
  await activationRejected;
  assert.equal(
    getSideConversationSession()?.owner,
    nextOwner,
    "a late mount callback cannot reclaim the released owner",
  );
});

test("a cancelled generation rejects every late launch callback", () => {
  const staleOwner = beginSideConversationLaunch({
    parentThreadId: "parent-local",
    parentRemoteId: "parent-remote",
    inheritedMessages: [],
    parentSettings: PARENT_SETTINGS,
  });
  cancelSideConversationLaunch(staleOwner);

  const currentOwner = beginSideConversationLaunch({
    parentThreadId: "other-parent",
    parentRemoteId: "other-parent-remote",
    inheritedMessages: [],
    parentSettings: PARENT_SETTINGS,
  });
  assert.throws(
    () => claimSideConversationLaunchThread(staleOwner, "late-side"),
    /No side conversation launch is pending/,
  );
  assert.equal(getSideConversationSession()?.owner, currentOwner);

  claimSideConversationLaunchThread(currentOwner, "current-side");
  requestSideConversationCleanup(currentOwner);
  assert.throws(
    () => finishSideConversationLaunch(currentOwner, "current-side"),
    /No side conversation launch is pending/,
  );
  assert.equal(getSideConversationSession()?.phase, "cleanup");
  completeSideConversationCleanup(currentOwner, "current-side");
});

test("a never-settling launch expires behind its owner fence", async () => {
  const owner = beginSideConversationLaunch({
    parentThreadId: "parent-local",
    parentRemoteId: "parent-remote",
    inheritedMessages: [],
    parentSettings: PARENT_SETTINGS,
  });
  let expirations = 0;
  createSideConversationLaunchDeadline(owner, 5, () => {
    expirations += 1;
  });

  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.equal(expirations, 1);
  assert.equal(getSideConversationSession(), null);
  assert.equal(isSideConversationOwner(owner), false);
});
