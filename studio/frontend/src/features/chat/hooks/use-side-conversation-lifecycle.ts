// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  useEffect,
  useRef,
  useSyncExternalStore,
  type RefObject,
} from "react";
import {
  getSideConversationSession,
  subscribeSideConversation,
  takeSideConversationPendingPrompt,
  type SideConversationSession,
} from "../utils/side-conversation.ts";

type DiscardSideConversation = (
  conversation: SideConversationSession,
  showError: boolean,
) => Promise<void>;

/**
 * The Thread component outlives its runtime-keyed children. Keep disposal at
 * this boundary so a parent-to-side switch cannot look like a page unmount.
 */
export function useSideConversationUnmountCleanup(
  runtimeThreadId: string | null | undefined,
  discardSideConversation: DiscardSideConversation,
): void {
  const runtimeThreadIdRef = useRef(runtimeThreadId);
  const discardSideConversationRef = useRef(discardSideConversation);
  const mountGenerationRef = useRef(0);

  useEffect(() => {
    runtimeThreadIdRef.current = runtimeThreadId;
  }, [runtimeThreadId]);
  useEffect(() => {
    discardSideConversationRef.current = discardSideConversation;
  }, [discardSideConversation]);
  useEffect(() => {
    const mountGeneration = ++mountGenerationRef.current;
    return () => {
      // StrictMode immediately sets this effect up again after its development
      // cleanup. Defer one microtask so that replay can supersede this mount,
      // while a real Thread unmount remains the final generation.
      queueMicrotask(() => {
        // The latest generation is the fence. Capturing its current value here
        // would make StrictMode's replacement mount indistinguishable.
        // eslint-disable-next-line react-hooks/exhaustive-deps
        if (mountGenerationRef.current !== mountGeneration) return;
        const conversation = getSideConversationSession();
        const mountedThreadId = runtimeThreadIdRef.current;
        if (
          conversation &&
          (conversation.parentThreadId === mountedThreadId ||
            conversation.sideThreadId === mountedThreadId)
        ) {
          void discardSideConversationRef.current(conversation, false);
        }
      });
    };
  }, []);
}

/** Tracks only this composer's UI lifetime. It never owns the side session. */
export function useSideComposerMountedRef(): RefObject<boolean> {
  const mountedRef = useRef(false);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);
  return mountedRef;
}

/** Delivers a stored inline prompt from the newly mounted side Composer once. */
export function useSideConversationPendingPromptSubmission(args: {
  runtimeThreadId: string | null | undefined;
  canDeliverPrompt: () => boolean;
  deliverPrompt: (prompt: string) => void;
  scheduleFrame: (callback: () => void) => number;
  cancelFrame: (frame: number) => void;
}): void {
  const {
    runtimeThreadId,
    canDeliverPrompt,
    deliverPrompt,
    scheduleFrame,
    cancelFrame,
  } = args;
  const sideSession = useSyncExternalStore(
    subscribeSideConversation,
    getSideConversationSession,
    getSideConversationSession,
  );
  const canDeliverPromptRef = useRef(canDeliverPrompt);
  const deliverPromptRef = useRef(deliverPrompt);
  useEffect(() => {
    canDeliverPromptRef.current = canDeliverPrompt;
  }, [canDeliverPrompt]);
  useEffect(() => {
    deliverPromptRef.current = deliverPrompt;
  }, [deliverPrompt]);
  useEffect(() => {
    if (
      sideSession?.phase !== "active" ||
      !sideSession.sideThreadId ||
      sideSession.sideThreadId !== runtimeThreadId ||
      sideSession.pendingPrompt === null
    ) {
      return;
    }
    const owner = sideSession.owner;
    const sideThreadId = sideSession.sideThreadId;
    let innerFrame: number | null = null;
    const outerFrame = scheduleFrame(() => {
      innerFrame = scheduleFrame(() => {
        if (!canDeliverPromptRef.current()) return;
        const prompt = takeSideConversationPendingPrompt(owner, sideThreadId);
        if (prompt !== null) deliverPromptRef.current(prompt);
      });
    });
    return () => {
      cancelFrame(outerFrame);
      if (innerFrame !== null) cancelFrame(innerFrame);
    };
  }, [
    cancelFrame,
    runtimeThreadId,
    scheduleFrame,
    sideSession,
  ]);
}
