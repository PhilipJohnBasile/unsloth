// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { act, createElement, StrictMode, useEffect } from "react";
import { createRoot } from "react-dom/client";
import {
  useSideComposerMountedRef,
  useSideConversationPendingPromptSubmission,
  useSideConversationUnmountCleanup,
} from "../src/features/chat/hooks/use-side-conversation-lifecycle.ts";
import {
  activateSideConversationAfterInitialization,
  beginSideConversationLaunch,
  claimSideConversationLaunchThread,
  completeSideConversationCleanup,
  finishSideConversationLaunch,
  getSideConversationSession,
  requestSideConversationCleanup,
  type SideConversationSession,
} from "../src/features/chat/utils/side-conversation.ts";
import type { QueuedChatRunSettings } from "../src/features/chat/utils/queued-chat-run-settings.ts";
import { DEFAULT_INFERENCE_PARAMS } from "../src/features/chat/types/runtime.ts";

const TEST_SETTINGS = {
  params: { ...DEFAULT_INFERENCE_PARAMS, checkpoint: "mounted-model" },
  activeGgufVariant: null,
  supportsTools: false,
  supportsReasoning: false,
  reasoningAlwaysOn: false,
  reasoningStyle: "enable_thinking",
  supportsReasoningOff: false,
  reasoningEffortLevels: ["low", "medium", "high"],
  supportsPreserveThinking: false,
  reasoningEnabled: false,
  reasoningEffort: "medium",
  preserveThinking: false,
  toolsEnabled: false,
  codeToolsEnabled: false,
  imageToolsEnabled: false,
  artifactsEnabled: false,
  mcpEnabledForChat: false,
  confirmToolCalls: true,
  bypassPermissions: false,
  permissionMode: "ask",
  webFetchToolsEnabled: false,
  deepResearchEnabled: false,
  researchWebsitePolicy: { allowedDomains: [], blockedDomains: [] },
  researchModelTimeoutSeconds: 120,
  ragEnabled: false,
  ragSource: { type: "thread" },
  ragMode: "hybrid",
  ragTopK: 5,
  ragAutoInject: "auto",
  ragAutoInjectMinScore: 0,
  ggufContextLength: null,
  autoHealToolCalls: true,
  nudgeToolCalls: true,
  maxToolCallsPerMessage: 25,
  toolCallTimeout: 5,
  autoCompactEnabled: false,
  contextPolicy: "inherit",
  compactionHeadroomRatio: 0.1,
} satisfies QueuedChatRunSettings;

class TestNode {
  readonly nodeType: number;
  readonly nodeName: string;
  readonly ownerDocument: TestDocument;
  nodeValue: string | null = null;
  parentNode: TestNode | null = null;
  readonly childNodes: TestNode[] = [];
  readonly namespaceURI = "http://www.w3.org/1999/xhtml";

  constructor(nodeType: number, nodeName: string, ownerDocument: TestDocument) {
    this.nodeType = nodeType;
    this.nodeName = nodeName;
    this.ownerDocument = ownerDocument;
  }

  get tagName(): string {
    return this.nodeName;
  }

  addEventListener(): void {}
  removeEventListener(): void {}
  setAttribute(): void {}
  removeAttribute(): void {}

  appendChild(node: TestNode): TestNode {
    node.parentNode = this;
    this.childNodes.push(node);
    return node;
  }

  insertBefore(node: TestNode, before: TestNode): TestNode {
    node.parentNode = this;
    const index = this.childNodes.indexOf(before);
    this.childNodes.splice(index < 0 ? this.childNodes.length : index, 0, node);
    return node;
  }

  removeChild(node: TestNode): TestNode {
    const index = this.childNodes.indexOf(node);
    if (index >= 0) this.childNodes.splice(index, 1);
    node.parentNode = null;
    return node;
  }
}

class TestElement extends TestNode {}
class TestIFrameElement extends TestElement {}

class TestDocument extends TestNode {
  readonly documentElement: TestElement;
  defaultView: typeof globalThis = globalThis;

  constructor() {
    super(9, "#document", null as unknown as TestDocument);
    this.documentElement = new TestElement(1, "HTML", this);
  }

  createElement(name: string): TestElement {
    return new TestElement(1, name.toUpperCase(), this);
  }

  createTextNode(value: string): TestNode {
    const node = new TestNode(3, "#text", this);
    node.nodeValue = value;
    return node;
  }
}

function installTestDom(): { container: Element; restore: () => void } {
  const names = [
    "document",
    "window",
    "Node",
    "Element",
    "HTMLElement",
    "HTMLIFrameElement",
    "IS_REACT_ACT_ENVIRONMENT",
  ] as const;
  const previous = new Map(
    names.map((name) => [name, Object.getOwnPropertyDescriptor(globalThis, name)]),
  );
  const document = new TestDocument();
  const values: Record<(typeof names)[number], unknown> = {
    document,
    window: globalThis,
    Node: TestNode,
    Element: TestElement,
    HTMLElement: TestElement,
    HTMLIFrameElement: TestIFrameElement,
    IS_REACT_ACT_ENVIRONMENT: true,
  };
  for (const name of names) {
    Object.defineProperty(globalThis, name, {
      configurable: true,
      value: values[name],
      writable: true,
    });
  }
  document.defaultView = globalThis;

  return {
    container: new TestElement(1, "DIV", document) as unknown as Element,
    restore: () => {
      for (const name of names) {
        const descriptor = previous.get(name);
        if (descriptor) Object.defineProperty(globalThis, name, descriptor);
        else delete (globalThis as Record<string, unknown>)[name];
      }
    },
  };
}

function createFrameQueue() {
  let nextFrame = 1;
  const callbacks = new Map<number, () => void>();
  const cancelled = new Set<number>();
  const invoked = new Set<number>();
  return {
    schedule: (callback: () => void) => {
      const frame = nextFrame++;
      callbacks.set(frame, callback);
      return frame;
    },
    cancel: (frame: number) => {
      cancelled.add(frame);
    },
    drainIncludingCancelled: () => {
      while (true) {
        const next = [...callbacks.entries()].find(
          ([frame]) => !invoked.has(frame),
        );
        if (!next) return;
        invoked.add(next[0]);
        next[1]();
      }
    },
    cancelled,
  };
}

test("a keyed side remount survives StrictMode and submits its stored prompt once", async () => {
  const parentThreadId = "mounted-parent";
  const sideThreadId = "mounted-side";
  const pendingPrompt = "inspect this branch";
  const owner = beginSideConversationLaunch({
    parentThreadId,
    parentRemoteId: "mounted-parent-remote",
    inheritedMessages: [],
    parentSettings: TEST_SETTINGS,
    pendingPrompt,
  });
  claimSideConversationLaunchThread(owner, sideThreadId);
  let finishSwitch!: () => void;
  const switchMounted = new Promise<void>((resolve) => {
    finishSwitch = resolve;
  });
  const activation = activateSideConversationAfterInitialization(
    owner,
    sideThreadId,
    async () => undefined,
    () => switchMounted,
  );

  const lifecycle: string[] = [];
  const mountedStates: boolean[] = [];
  const submissions: string[] = [];
  const frames = createFrameQueue();
  const discard = async (conversation: SideConversationSession) => {
    lifecycle.push(`dispose:${conversation.owner}`);
    lifecycle.push(`stop-queues:${conversation.sideThreadId}`);
    requestSideConversationCleanup(conversation.owner);
  };
  const KeyedComposer = ({ id }: { id: string }) => {
    const mountedRef = useSideComposerMountedRef();
    useSideConversationPendingPromptSubmission({
      runtimeThreadId: id,
      canDeliverPrompt: () => true,
      deliverPrompt: (prompt) => submissions.push(`${id}:${prompt}`),
      scheduleFrame: frames.schedule,
      cancelFrame: frames.cancel,
    });
    useEffect(() => {
      mountedStates.push(mountedRef.current);
      lifecycle.push(`mount:${id}`);
      return () => {
        lifecycle.push(`unmount:${id}`);
      };
    }, [id, mountedRef]);
    return createElement("span");
  };
  const MountedThread = ({ runtimeThreadId }: { runtimeThreadId: string }) => {
    useSideConversationUnmountCleanup(runtimeThreadId, discard);
    return createElement(
      "div",
      null,
      createElement(KeyedComposer, {
        id: runtimeThreadId,
        key: runtimeThreadId,
      }),
    );
  };

  const testDom = installTestDom();
  const root = createRoot(testDom.container);
  try {
    await act(async () => {
      root.render(
        createElement(
          StrictMode,
          null,
          createElement(MountedThread, { runtimeThreadId: parentThreadId }),
        ),
      );
    });
    await Promise.resolve();
    assert.equal(getSideConversationSession()?.phase, "launching");
    assert.equal(lifecycle.some((event) => event === `dispose:${owner}`), false);
    assert.equal(mountedStates.every(Boolean), true);

    await act(async () => {
      root.render(
        createElement(
          StrictMode,
          null,
          createElement(MountedThread, { runtimeThreadId: sideThreadId }),
        ),
      );
    });

    assert.ok(lifecycle.includes(`unmount:${parentThreadId}`));
    assert.ok(lifecycle.includes(`mount:${sideThreadId}`));
    assert.equal(lifecycle.some((event) => event === `dispose:${owner}`), false);
    assert.equal(getSideConversationSession()?.owner, owner);
    assert.equal(getSideConversationSession()?.phase, "launching");

    await act(async () => {
      finishSwitch();
      await activation;
    });
    await act(async () => {
      frames.drainIncludingCancelled();
    });
    assert.deepEqual(submissions, [`${sideThreadId}:${pendingPrompt}`]);
    assert.equal(getSideConversationSession()?.pendingPrompt, null);

    await act(async () => {
      root.unmount();
    });
    await Promise.resolve();
    assert.equal(
      lifecycle.filter((event) => event === `dispose:${owner}`).length,
      1,
    );
    assert.ok(lifecycle.includes(`unmount:${sideThreadId}`));
    assert.ok(lifecycle.includes(`stop-queues:${sideThreadId}`));
    assert.equal(getSideConversationSession()?.phase, "cleanup");
  } finally {
    completeSideConversationCleanup(owner, sideThreadId);
    testDom.restore();
  }
});

test("Return before the side Composer consumes its prompt cancels delivery", async () => {
  const parentThreadId = "return-parent";
  const sideThreadId = "return-side";
  const owner = beginSideConversationLaunch({
    parentThreadId,
    parentRemoteId: "return-parent-remote",
    inheritedMessages: [],
    parentSettings: TEST_SETTINGS,
    pendingPrompt: "do not send",
  });
  claimSideConversationLaunchThread(owner, sideThreadId);
  finishSideConversationLaunch(owner, sideThreadId);

  const frames = createFrameQueue();
  const submissions: string[] = [];
  const Composer = ({ id }: { id: string }) => {
    useSideConversationPendingPromptSubmission({
      runtimeThreadId: id,
      canDeliverPrompt: () => true,
      deliverPrompt: (prompt) => submissions.push(prompt),
      scheduleFrame: frames.schedule,
      cancelFrame: frames.cancel,
    });
    return createElement("span");
  };
  const MountedThread = ({ runtimeThreadId }: { runtimeThreadId: string }) =>
    createElement(
      "div",
      null,
      createElement(Composer, { id: runtimeThreadId, key: runtimeThreadId }),
    );

  const testDom = installTestDom();
  const root = createRoot(testDom.container);
  try {
    await act(async () => {
      root.render(createElement(MountedThread, { runtimeThreadId: sideThreadId }));
    });
    await act(async () => {
      requestSideConversationCleanup(owner);
    });
    await act(async () => {
      frames.drainIncludingCancelled();
    });
    assert.equal(frames.cancelled.size > 0, true);
    assert.deepEqual(submissions, []);
    assert.equal(getSideConversationSession()?.pendingPrompt, "do not send");
    await act(async () => {
      root.unmount();
    });
  } finally {
    completeSideConversationCleanup(owner, sideThreadId);
    testDom.restore();
  }
});
