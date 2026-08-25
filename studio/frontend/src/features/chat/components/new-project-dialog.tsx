// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useNavigate } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  pickNativeProjectFolder,
  useNativePathLeasesSupported,
} from "@/features/native-intents";
import {
  ProjectSourceDropzone,
  type StagedSource,
  uploadStagedSources,
} from "@/features/rag/components/project-source-dropzone";
import { isTauri } from "@/lib/api-base";
import { toast } from "@/lib/toast";
import { Folder02Icon, FolderOpenIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

import {
  createChatProject,
  openChatProjectFromFolder,
} from "../hooks/use-chat-projects";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import type { ProjectRecord } from "../types";

interface SelectedProjectFolder {
  token: string;
  displayName: string;
}

function currentRoute(): string {
  if (typeof window === "undefined") return "";
  return window.location.pathname + window.location.search;
}

// Create-project dialog for the composer, sidebar, and projects page. Creating
// opens the new project; `onCreated` overrides that for callers with their own
// follow-up (the sidebar's "move this chat to a new project").
export function NewProjectDialog({
  open,
  onOpenChange,
  title = "Create project",
  submitLabel = "Create project",
  onCreated,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title?: string;
  submitLabel?: string;
  onCreated?: (
    project: ProjectRecord,
    context: { stayedOnRoute: boolean },
  ) => void | Promise<void>;
}) {
  const navigate = useNavigate();
  const nativePathLeasesSupported = useNativePathLeasesSupported();
  const [name, setName] = useState("");
  const [staged, setStaged] = useState<StagedSource[]>([]);
  const [selectedFolder, setSelectedFolder] =
    useState<SelectedProjectFolder | null>(null);
  const [selectingFolder, setSelectingFolder] = useState(false);
  const [busy, setBusy] = useState(false);
  // A desktop drop reaches `staged` only once its native registration settles.
  // Creating before then would upload without the files the user just dropped.
  const [stagingDrop, setStagingDrop] = useState(false);
  // Uploads outlive this component, so a slow one must not yank the user to the
  // new project after they have navigated away.
  const mounted = useRef(true);
  useEffect(() => {
    // Set on setup, not just cleared on cleanup: StrictMode replays
    // setup/cleanup/setup, which would otherwise leave this false forever.
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const canChooseFolder = isTauri && nativePathLeasesSupported;

  function reset() {
    setName("");
    setStaged([]);
    setSelectedFolder(null);
    setSelectingFolder(false);
    setStagingDrop(false);
  }

  // Every close path routes through here: callers keep this mounted, so a draft
  // left behind would resurface (and upload) on the next project.
  function close() {
    if (busy) return;
    reset();
    onOpenChange(false);
  }

  async function chooseExistingFolder() {
    if (!canChooseFolder || busy || selectingFolder) return;
    setSelectingFolder(true);
    try {
      const selected = await pickNativeProjectFolder();
      if (!selected) return;
      setSelectedFolder(selected);
      setName((current) => current.trim() || selected.displayName);
    } catch (error) {
      toast.error("Could not select project folder", {
        description: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setSelectingFolder(false);
    }
  }

  async function commitCreate() {
    const trimmed = name.trim();
    if (!trimmed || busy || stagingDrop || selectingFolder) return;
    setBusy(true);
    // Sidebar callers keep this mounted across routes, so unmounting alone
    // cannot tell whether the user has moved on during a slow upload.
    const origin = currentRoute();
    try {
      const project = selectedFolder
        ? await openChatProjectFromFolder(selectedFolder.token, trimmed)
        : await createChatProject(trimmed);
      // Project sources are still RAG inputs. The workspace folder is not
      // ingested; chats and code tools access it directly as their working dir.
      await uploadStagedSources(project.id, staged);
      if (!mounted.current) return;
      const stayedOnRoute = currentRoute() === origin;
      onOpenChange(false);
      reset();
      if (onCreated) {
        await onCreated(project, { stayedOnRoute });
        return;
      }
      if (!stayedOnRoute) return;
      const runtime = useChatRuntimeStore.getState();
      runtime.setActiveThreadId(null);
      runtime.setActiveProjectId(project.id);
      navigate({ to: "/chat", search: { project: project.id } });
    } catch (err) {
      toast.error("Failed to create project", {
        description: err instanceof Error ? err.message : undefined,
      });
    } finally {
      setBusy(false);
    }
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (next) {
          onOpenChange(true);
          return;
        }
        close();
      }}
    >
      <DialogContent className="corner-squircle dialog-soft-surface gap-5 sm:max-w-lg">
        <DialogHeader>
          <DialogTitle className="text-ui-21">{title}</DialogTitle>
        </DialogHeader>
        {/* Name field: folder glyph in its own cell, divided from the input. */}
        <div className="flex items-stretch overflow-hidden rounded-[16px] border border-border bg-background transition-colors focus-within:border-ring has-[input:disabled]:opacity-50 dark:border-transparent dark:bg-white/[0.06]">
          <span className="flex w-9 shrink-0 items-center justify-center text-muted-foreground">
            <HugeiconsIcon
              icon={Folder02Icon}
              strokeWidth={1.75}
              className="size-5"
            />
          </span>
          <span aria-hidden="true" className="my-3 w-px bg-border" />
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                void commitCreate();
              }
            }}
            autoFocus={true}
            disabled={busy}
            maxLength={120}
            placeholder="Project name"
            aria-label="Project name"
            className="min-w-0 flex-1 bg-transparent py-4 pr-4 pl-2.5 text-base outline-none placeholder:text-muted-foreground disabled:cursor-not-allowed"
          />
        </div>

        <div className="space-y-2.5">
          <p className="text-ui-15 font-medium text-foreground">Workspace</p>
          <div className="grid grid-cols-2 gap-2">
            <Button
              type="button"
              variant={selectedFolder ? "outline" : "secondary"}
              disabled={busy}
              onClick={() => setSelectedFolder(null)}
            >
              <HugeiconsIcon icon={Folder02Icon} strokeWidth={1.75} />
              Managed workspace
            </Button>
            <Button
              type="button"
              variant={selectedFolder ? "secondary" : "outline"}
              disabled={!canChooseFolder || busy || selectingFolder}
              onClick={() => void chooseExistingFolder()}
            >
              <HugeiconsIcon icon={FolderOpenIcon} strokeWidth={1.75} />
              {selectingFolder
                ? "Choosing…"
                : selectedFolder
                  ? "Change folder"
                  : "Existing folder"}
            </Button>
          </div>
          {selectedFolder ? (
            <div className="rounded-[16px] border border-border bg-background px-4 py-3 dark:border-white/10">
              <p className="truncate text-sm font-medium" title={selectedFolder.displayName}>
                {selectedFolder.displayName}
              </p>
              <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                Every project chat and code tool works directly in this folder.
                The folder is never uploaded as a project source or deleted by
                Unsloth.
              </p>
            </div>
          ) : (
            <p className="text-xs leading-relaxed text-muted-foreground">
              {canChooseFolder
                ? "Unsloth creates and manages a private workspace. Choose Existing folder to work directly in a repository, like a Codex project."
                : "Existing-folder workspaces require the managed desktop app. Unsloth will create a private workspace here."}
            </p>
          )}
        </div>

        <ProjectSourceDropzone
          staged={staged}
          onChange={setStaged}
          disabled={busy}
          onPendingChange={setStagingDrop}
        />
        <DialogFooter className="flex-wrap gap-2 sm:justify-end">
          <Button type="button" variant="ghost" disabled={busy} onClick={close}>
            Cancel
          </Button>
          <Button
            type="button"
            onClick={() => void commitCreate()}
            disabled={
              !name.trim() || busy || stagingDrop || selectingFolder
            }
          >
            {busy
              ? "Creating…"
              : stagingDrop
                ? "Adding sources…"
                : submitLabel}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
