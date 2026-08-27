// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const thread = readFileSync(
  new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
  "utf8",
);
const adapter = readFileSync(
  new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
  "utf8",
);
const sharedComposer = readFileSync(
  new URL("../src/features/chat/shared-composer.tsx", import.meta.url),
  "utf8",
);
const chatPage = readFileSync(
  new URL("../src/features/chat/chat-page.tsx", import.meta.url),
  "utf8",
);
const sideConversation = readFileSync(
  new URL(
    "../src/features/chat/utils/side-conversation.ts",
    import.meta.url,
  ),
  "utf8",
);
const chatStore = readFileSync(
  new URL(
    "../src/features/chat/stores/chat-runtime-store.ts",
    import.meta.url,
  ),
  "utf8",
);
const modelRuntime = readFileSync(
  new URL(
    "../src/features/chat/hooks/use-chat-model-runtime.ts",
    import.meta.url,
  ),
  "utf8",
);
const inferenceStatus = readFileSync(
  new URL(
    "../src/features/chat/lib/apply-inference-status-to-store.ts",
    import.meta.url,
  ),
  "utf8",
);
const promptQueueBoundary = readFileSync(
  new URL(
    "../src/features/chat/utils/pending-prompt-queue-stop.ts",
    import.meta.url,
  ),
  "utf8",
);

function blockBetween(source: string, startNeedle: string, endNeedle: string) {
  const start = source.indexOf(startNeedle);
  assert.ok(start >= 0, `${startNeedle} not found`);
  const end = source.indexOf(endNeedle, start);
  assert.ok(end > start, `${endNeedle} not found after ${startNeedle}`);
  return source.slice(start, end);
}

test("resolveProjectId reads side authority from the persisted parent", () => {
  const beforeComposerFallback = blockBetween(
    adapter,
    "export async function resolveProjectId(",
    "const composerProjectId =",
  );
  assert.match(
    beforeComposerFallback,
    /const authorityThreadId = sideConversationAuthorityThreadId\(threadId\);/,
  );
  assert.match(
    beforeComposerFallback,
    /const parentThread = await getStoredChatThread\(authorityThreadId\);/,
  );
  assert.doesNotMatch(
    beforeComposerFallback,
    /readThreadRecord\?\.\(\)/,
    "the side-bound reader must not stand in for the persisted parent row",
  );
  assert.match(
    beforeComposerFallback,
    /return parentThread\?\.projectId \?\? null;/,
  );

  const sandboxResolver = blockBetween(
    adapter,
    "async function resolveSandboxSessionId(",
    "/** Wait for an in-progress model load",
  );
  assert.match(
    sandboxResolver,
    /sandboxSessionIdFor\(\s*sideConversationAuthorityThreadId\(threadId\),\s*projectId,\s*\)/,
  );
});

test("the single-chat composer intercepts /side before normal send", () => {
  const submit = blockBetween(
    thread,
    "const handleSubmit = useCallback(",
    "const stopQueue = useCallback(",
  );
  assert.match(
    submit,
    /const sideCommand = parseSideCommand\(composerText\);\s*if \(sideCommand\.matched\) \{\s*event\.preventDefault\(\);\s*startSideConversation\(composerText, sideCommand\.prompt\);\s*return;/,
  );
});

test("launch tags and initializes the reserved thread before activation", () => {
  const launch = blockBetween(
    thread,
    "const startSideConversation = useCallback(",
    "const handleSubmit = useCallback(",
  );
  const mark = launch.indexOf("markThreadIncognito(claimedSideThreadId)");
  const finish = launch.indexOf(
    "activateSideConversationAfterInitialization(",
  );
  assert.ok(mark >= 0, "side thread incognito tag not found");
  assert.ok(finish > mark, "the side thread must be tagged before launch finishes");
  assert.match(
    launch,
    /getItemById\(claimedSideThreadId!\)\.initialize\(\)/,
  );
  assert.match(launch, /pendingPrompt: prompt/);
  assert.doesNotMatch(
    launch,
    /requestAnimationFrame/,
    "the unmounted parent Composer must not own inline prompt delivery",
  );
  assert.match(
    thread,
    /useSideConversationPendingPromptSubmission\(\{[\s\S]*?runtimeThreadId: threadListItemId,[\s\S]*?canDeliverPrompt: \(\) => formRef\.current !== null/,
  );
});

test("the side surface exposes return and deterministic cleanup", () => {
  assert.match(thread, /Return to parent/);
  assert.doesNotMatch(
    blockBetween(
      thread,
      "const returnToParent = useCallback(",
      "useEffect(() => {",
    ),
    /switchToThread/,
    "Return must enter the shared ordered disposal transition",
  );
  const cleanup = blockBetween(
    thread,
    "const discardSideRuntime = useCallback(",
    "const returnToParent = useCallback(",
  );
  assert.match(cleanup, /disposeSideConversation\(cleanupSession\.owner/);
  assert.match(
    cleanup,
    /stopSideQueues: \(threadId\) => requestPromptQueueStop\(\[threadId\]\)/,
    "Return and unmount cleanup must stop materialized queues by the exact side id",
  );
  assert.match(cleanup, /status === "new" \|\| status === "deleted"/);
  assert.match(
    cleanup,
    /item\.getState\(\)\.status === "regular"[\s\S]*?item\.detach\(\)/,
  );
  assert.match(cleanup, /clearThreadContextUsage\(threadId\)/);
  const disposal = blockBetween(
    sideConversation,
    "export async function disposeSideConversation(",
    "export function cancelSideConversationLaunch(",
  );
  const switchPosition = disposal.indexOf("actions.switchToParent(");
  const restorePosition = disposal.indexOf("actions.restoreParentPresentation(");
  const deletePosition = disposal.indexOf("actions.disposeSide(");
  const clearUsagePosition = disposal.indexOf("actions.clearSideUsage(");
  const completePosition = disposal.indexOf("completeSideConversationCleanup(");
  const forgetPosition = cleanup.indexOf("forgetThreadIncognito(sideThreadId)");
  const returnGatePosition = cleanup.indexOf(
    "returningSideRef.current = null",
  );
  assert.ok(switchPosition >= 0 && restorePosition > switchPosition);
  assert.ok(deletePosition > restorePosition);
  assert.ok(clearUsagePosition > deletePosition);
  assert.ok(completePosition > clearUsagePosition);
  assert.ok(forgetPosition >= 0, "incognito tag must clear after disposal resolves");
  assert.ok(
    returnGatePosition > forgetPosition,
    "return gate cleared before the same side completed cleanup",
  );
  assert.match(cleanup, /setReturningToParent\(false\)/);

  const pendingQueueStop = blockBetween(
    promptQueueBoundary,
    "function cancelPendingPromptQueueFactoriesForStop<",
    "\n}",
  );
  assert.match(
    pendingQueueStop,
    /threadIds\.some\(\(threadId\) => aliases\.includes\(threadId\)\)/,
    "an exact side-id stop must cancel hydration-pending queue factories",
  );

  assert.match(
    thread,
    /useSideConversationUnmountCleanup\(runtimeThreadId, discardSideRuntime\);/,
    "unmount must discard an open side thread",
  );
});

test("a mounted parent-to-side switch does not run the unmount cleanup", () => {
  const ownerBoundary = blockBetween(
    thread,
    "const discardSideRuntime = useCallback(",
    "const returnToParent = useCallback(",
  );
  assert.match(
    ownerBoundary,
    /useSideConversationUnmountCleanup\(runtimeThreadId, discardSideRuntime\);/,
  );
  const keyedSubtree = blockBetween(
    thread,
    "<GeneratedImageOverlayProvider key={runtimeThreadId}",
    "</GeneratedImageOverlayProvider>",
  );
  assert.match(keyedSubtree, /<ThreadComposerDock/);
  const composerLifetime = blockBetween(
    thread,
    "const sideLaunchInFlightRef = useRef(false);",
    "const startSideConversation = useCallback(",
  );
  assert.match(composerLifetime, /useSideComposerMountedRef\(\)/);
  assert.match(
    composerLifetime,
    /const sideLaunchDeadlineRef = useRef/,
  );
  assert.doesNotMatch(composerLifetime, /requestSideConversationCleanup/);
  assert.doesNotMatch(composerLifetime, /sideLaunchDeadlineRef\.current\?\.\(\)/);
});

test("model changes are rejected while a side chat owns the shared runtime", () => {
  const checkpointChange = blockBetween(
    chatPage,
    "const handleCheckpointChange = useCallback(",
    "const handleReloadActiveModel = useCallback(",
  );
  assert.match(checkpointChange, /if \(sideConversationBlocksModelLifecycle\(\)\) \{/);
  assert.match(checkpointChange, /Model switching is unavailable in side chat/);
  const guardPosition = checkpointChange.indexOf(
    "sideConversationBlocksModelLifecycle()",
  );
  const mutationPosition = checkpointChange.indexOf("store.setCheckpoint(value, null)");
  assert.ok(guardPosition >= 0 && mutationPosition > guardPosition);

  assert.match(
    blockBetween(chatStore, "beginModelLoading: () => {", "endModelLoading:"),
    /if \(sideConversationBlocksModelLifecycle\(\)\) return null;/,
  );
  assert.match(
    blockBetween(chatStore, "setParams: (params, options)", "setCustomPresets:"),
    /if \(side && \(checkpointChanged \|\| fromModelDefaults\)\) return state;/,
  );
  assert.match(modelRuntime, /const rejectSideModelChange = \(\): boolean =>/);
  assert.match(
    inferenceStatus,
    /export function applyActiveModelStatusToStore\([\s\S]*?if \(sideConversationBlocksModelLifecycle\(\)\) return;/,
  );
});

test("a parent completion cannot repaint the visible side context usage", () => {
  const usage = blockBetween(
    adapter,
    "const usageKey = liveThreadKey(serverCancel);",
    "if (\n          incompleteReason === null",
  );
  assert.match(
    usage,
    /const visibleThreadId =\s*getSideConversationSession\(\)\?\.sideThreadId \?\?\s*useChatRuntimeStore\.getState\(\)\.activeThreadId;/,
  );
  assert.match(
    usage,
    /const usageThreadIsVisible =\s*visibleThreadId === \(usageThreadKey \?\? activeThreadIdAtRunStart\);/,
  );
});

test("Compare rejects /side instead of leaking it as a model prompt", () => {
  const compareSend = blockBetween(
    sharedComposer,
    "async function send()",
    "const hasCompareHandles =",
  );
  assert.match(
    compareSend,
    /if \(parseSideCommand\(submittedText\)\.matched\) \{/,
  );
  assert.match(compareSend, /Side chat is unavailable in Compare/);
  assert.match(compareSend, /resetPromptQueue\(\);\s*return;/);
});
