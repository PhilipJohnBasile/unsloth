// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

const { store: localStorageFake } = installLocalStorageFake();
localStorageFake.set("unsloth_chat_settings_imported_to_studio_db", "true");
register("./thread-sampling-resolver.mjs", import.meta.url);

const { settingsHttp } = await import("./helpers/store-stubs/settings-http.ts");
const { threadRows } = await import(
  "./helpers/store-stubs/chat-history-storage.ts"
);
const { restoreSideConversationParentSettings, useChatRuntimeStore } =
  await import("../src/features/chat/stores/chat-runtime-store.ts");
const { snapshotQueuedChatRunSettings, consumeQueuedChatRunSettings } =
  await import("../src/features/chat/utils/queued-chat-run-settings.ts");
const {
  beginSideConversationLaunch,
  claimSideConversationLaunchThread,
  completeSideConversationCleanup,
  disposeSideConversation,
  finishSideConversationLaunch,
  getSideConversationSession,
  requestSideConversationCleanup,
  sideConversationVisibleThreadId,
} = await import("../src/features/chat/utils/side-conversation.ts");

const PARENT = "saved-parent";
const SIDE = "incognito-side";

async function flushSideEdits(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 0));
}

test.before(async () => {
  settingsHttp.settings = {
    inferenceParams: { temperature: 0.7 },
    toolsEnabled: true,
    permissionMode: "ask",
  };
  await useChatRuntimeStore.getState().hydratePersistedSettings();
});

test.afterEach(() => {
  const session = getSideConversationSession();
  if (!session) return;
  requestSideConversationCleanup(session.owner);
  if (session.sideThreadId) {
    completeSideConversationCleanup(session.owner, session.sideThreadId);
  }
});

test("side controls update only the side run snapshot", async () => {
  threadRows.reset();
  const store = useChatRuntimeStore.getState();
  store.setActiveThreadId(PARENT);
  store.applyThreadScopedSettings(PARENT, {
    temperature: 0.7,
    toolsEnabled: true,
    permissionMode: "ask",
  });
  const parentSettings = snapshotQueuedChatRunSettings(
    useChatRuntimeStore.getState(),
  );

  const owner = beginSideConversationLaunch({
    parentThreadId: "parent-local",
    parentRemoteId: PARENT,
    inheritedMessages: [],
    parentSettings,
  });
  claimSideConversationLaunchThread(owner, SIDE);
  finishSideConversationLaunch(owner, SIDE);

  useChatRuntimeStore.getState().setToolsEnabled(false);
  useChatRuntimeStore.getState().setPermissionMode("off");
  useChatRuntimeStore.getState().setParams({
    ...useChatRuntimeStore.getState().params,
    temperature: 0.2,
  });
  await flushSideEdits();

  const sideSettings = consumeQueuedChatRunSettings(SIDE);
  assert.equal(sideSettings?.toolsEnabled, false);
  assert.equal(sideSettings?.permissionMode, "off");
  assert.equal(sideSettings?.params.temperature, 0.2);
  assert.equal(
    consumeQueuedChatRunSettings(SIDE),
    sideSettings,
    "tool-loop calls keep using the side-owned snapshot",
  );

  await new Promise((resolve) => setTimeout(resolve, 600));
  assert.deepEqual(
    threadRows.writesFor(PARENT),
    [],
    "side edits persisted onto the saved parent row",
  );

  restoreSideConversationParentSettings(parentSettings);
  assert.equal(useChatRuntimeStore.getState().toolsEnabled, true);
  assert.equal(useChatRuntimeStore.getState().permissionMode, "ask");
  assert.equal(useChatRuntimeStore.getState().params.temperature, 0.7);
});

test("standalone side params edits update the side run snapshot", async () => {
  threadRows.reset();
  const store = useChatRuntimeStore.getState();
  store.setActiveThreadId(PARENT);
  const parentSettings = snapshotQueuedChatRunSettings(
    useChatRuntimeStore.getState(),
  );
  const owner = beginSideConversationLaunch({
    parentThreadId: "parent-local",
    parentRemoteId: PARENT,
    inheritedMessages: [],
    parentSettings,
  });
  claimSideConversationLaunchThread(owner, SIDE);
  finishSideConversationLaunch(owner, SIDE);

  const initialParams = useChatRuntimeStore.getState().params;
  const nextMaxTokens = initialParams.maxTokens === 4096 ? 4097 : 4096;
  useChatRuntimeStore.getState().setParams({
    ...initialParams,
    maxTokens: nextMaxTokens,
  });
  await flushSideEdits();
  assert.equal(
    consumeQueuedChatRunSettings(SIDE)?.params.maxTokens,
    nextMaxTokens,
    "Max Tokens changed by itself must reach the side run",
  );

  const nextFastMode = !useChatRuntimeStore.getState().params.fastMode;
  useChatRuntimeStore.getState().setParams({
    ...useChatRuntimeStore.getState().params,
    fastMode: nextFastMode,
  });
  await flushSideEdits();
  const sideSettings = consumeQueuedChatRunSettings(SIDE);
  assert.equal(
    sideSettings?.params.maxTokens,
    nextMaxTokens,
    "the later params snapshot must retain the earlier side edit",
  );
  assert.equal(
    sideSettings?.params.fastMode,
    nextFastMode,
    "Fast mode changed by itself must reach the side run",
  );

  await new Promise((resolve) => setTimeout(resolve, 600));
  assert.deepEqual(threadRows.writesFor(PARENT), []);
  restoreSideConversationParentSettings(parentSettings);
});

test("every visible queued setter stays in the side snapshot", async () => {
  await new Promise((resolve) => setTimeout(resolve, 500));
  threadRows.reset();
  settingsHttp.puts.length = 0;
  const store = useChatRuntimeStore.getState();
  store.setActiveThreadId(PARENT);
  const parentSettings = snapshotQueuedChatRunSettings(store);
  const storageBefore = new Map(localStorageFake);

  const owner = beginSideConversationLaunch({
    parentThreadId: "parent-local",
    parentRemoteId: PARENT,
    inheritedMessages: [],
    parentSettings,
  });
  claimSideConversationLaunchThread(owner, SIDE);
  finishSideConversationLaunch(owner, SIDE);

  const live = useChatRuntimeStore.getState();
  const nextPreserveThinking = !live.preserveThinking;
  const nextAutoHeal = !live.autoHealToolCalls;
  const nextNudge = !live.nudgeToolCalls;
  const nextAutoCompact = !live.autoCompactEnabled;
  const nextReasoningStyle =
    live.reasoningStyle === "enable_thinking"
      ? "reasoning_effort"
      : "enable_thinking";
  live.setPreserveThinking(nextPreserveThinking);
  live.setReasoningStyle(nextReasoningStyle);
  live.setResearchWebsitePolicy({
    allowedDomains: ["docs.example.test"],
    blockedDomains: ["blocked.example.test"],
  });
  live.setResearchModelTimeoutSeconds(120);
  live.setAutoHealToolCalls(nextAutoHeal);
  live.setNudgeToolCalls(nextNudge);
  live.setAutoCompactEnabled(nextAutoCompact);
  live.setContextPolicy("rolling");
  live.setCompactionHeadroomRatio(0.1);
  live.setMaxToolCallsPerMessage(17);
  live.setToolCallTimeout(33);
  live.setBypassPermissions(true);
  live.setBypassPermissions(false);
  await flushSideEdits();

  assert.equal(
    useChatRuntimeStore.getState().preserveThinking,
    nextPreserveThinking,
  );
  const side = consumeQueuedChatRunSettings(SIDE);
  assert.equal(side?.toolCallTimeout, 33);
  assert.equal(side?.preserveThinking, nextPreserveThinking);
  assert.equal(side?.reasoningStyle, nextReasoningStyle);
  assert.equal(side?.bypassPermissions, false);
  assert.deepEqual(side?.researchWebsitePolicy, {
    allowedDomains: ["docs.example.test"],
    blockedDomains: ["blocked.example.test"],
  });
  assert.equal(side?.researchModelTimeoutSeconds, 120);
  assert.equal(side?.autoHealToolCalls, nextAutoHeal);
  assert.equal(side?.nudgeToolCalls, nextNudge);
  assert.equal(side?.autoCompactEnabled, nextAutoCompact);
  assert.equal(side?.contextPolicy, "rolling");
  assert.equal(side?.compactionHeadroomRatio, 0.1);
  assert.equal(side?.maxToolCallsPerMessage, 17);

  await new Promise((resolve) => setTimeout(resolve, 600));
  assert.deepEqual(threadRows.writesFor(PARENT), []);
  assert.deepEqual(settingsHttp.puts, []);
  assert.deepEqual(
    [...localStorageFake.entries()],
    [...storageBefore.entries()],
    "side edits must not change installation storage",
  );

  restoreSideConversationParentSettings(parentSettings);
});

test("visible side usage and model state never overwrite the parent's usage entry", async () => {
  const parentUsage = {
    promptTokens: 100,
    completionTokens: 20,
    totalTokens: 120,
    cachedTokens: 5,
    cacheWriteTokens: 0,
  };
  const sideUsage = {
    promptTokens: 40,
    completionTokens: 10,
    totalTokens: 50,
    cachedTokens: 0,
    cacheWriteTokens: 0,
  };
  useChatRuntimeStore.setState({
    activeThreadId: PARENT,
    contextUsage: parentUsage,
    contextUsageByThreadId: { [PARENT]: parentUsage },
  });
  const owner = beginSideConversationLaunch({
    parentThreadId: "parent-local",
    parentRemoteId: PARENT,
    inheritedMessages: [],
    parentSettings: snapshotQueuedChatRunSettings(
      useChatRuntimeStore.getState(),
    ),
  });
  claimSideConversationLaunchThread(owner, SIDE);
  finishSideConversationLaunch(owner, SIDE);
  const parentSettings = snapshotQueuedChatRunSettings(
    useChatRuntimeStore.getState(),
  );

  assert.equal(sideConversationVisibleThreadId(PARENT, SIDE), SIDE);

  useChatRuntimeStore
    .getState()
    .setThreadContextUsage(SIDE, sideUsage, { visible: true });

  const state = useChatRuntimeStore.getState();
  assert.deepEqual(state.contextUsage, sideUsage);
  assert.deepEqual(state.contextUsageByThreadId[SIDE], sideUsage);
  assert.deepEqual(state.contextUsageByThreadId[PARENT], parentUsage);

  const completedParentUsage = { ...parentUsage, totalTokens: 150 };
  useChatRuntimeStore
    .getState()
    .setThreadContextUsage(PARENT, completedParentUsage, { visible: false });
  assert.deepEqual(
    useChatRuntimeStore.getState().contextUsage,
    sideUsage,
    "a parent completion cannot repaint the visible side usage",
  );
  assert.deepEqual(
    useChatRuntimeStore.getState().contextUsageByThreadId[PARENT],
    completedParentUsage,
  );

  const checkpointBeforeSide = useChatRuntimeStore.getState().params.checkpoint;
  useChatRuntimeStore.getState().setCheckpoint("side-only-model", null);
  await flushSideEdits();
  const afterSideModelEdit = useChatRuntimeStore.getState();
  assert.deepEqual(
    afterSideModelEdit.contextUsageByThreadId[PARENT],
    completedParentUsage,
  );
  assert.equal(afterSideModelEdit.params.checkpoint, checkpointBeforeSide);
  assert.equal(
    consumeQueuedChatRunSettings(SIDE)?.params.checkpoint,
    checkpointBeforeSide,
  );
  assert.equal(
    useChatRuntimeStore.getState().beginModelLoading(),
    null,
    "the installation-wide model lifecycle is fenced while side owns it",
  );
  useChatRuntimeStore.getState().setParams(
    {
      ...useChatRuntimeStore.getState().params,
      checkpoint: "status-race-model",
    },
    { fromModelDefaults: true },
  );
  assert.equal(
    useChatRuntimeStore.getState().params.checkpoint,
    checkpointBeforeSide,
  );
  useChatRuntimeStore.getState().setParams({
    ...useChatRuntimeStore.getState().params,
    temperature: 0.15,
  });
  await flushSideEdits();
  assert.equal(consumeQueuedChatRunSettings(SIDE)?.params.temperature, 0.15);

  await disposeSideConversation(owner, {
    currentThreadId: () => SIDE,
    cancelSideRun: () => undefined,
    stopSideQueues: () => undefined,
    abortPendingSideTransition: () => undefined,
    switchToParent: async () => undefined,
    restoreParentPresentation: () =>
      restoreSideConversationParentSettings(parentSettings),
    disposeSide: async () => undefined,
    clearSideUsage: (threadId) =>
      useChatRuntimeStore.getState().clearThreadContextUsage(threadId),
  });
  assert.equal(
    useChatRuntimeStore.getState().contextUsageByThreadId[SIDE],
    undefined,
  );
  assert.deepEqual(
    useChatRuntimeStore.getState().contextUsageByThreadId[PARENT],
    completedParentUsage,
  );
  restoreSideConversationParentSettings(parentSettings);
});
