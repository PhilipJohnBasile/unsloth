// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

function source(relative: string): string {
  return readFileSync(
    fileURLToPath(new URL(relative, import.meta.url)),
    "utf8",
  );
}

test("create project can choose a purpose-bound local workspace", () => {
  const dialog = source("../src/features/chat/components/new-project-dialog.tsx");
  const nativeApi = source("../src/features/native-intents/api.ts");
  const chatApi = source("../src/features/chat/api/chat-api.ts");

  assert.match(dialog, /await pickNativeProjectFolder\(\)/);
  assert.match(dialog, /await openChatProjectFromFolder\(selectedFolder\.token, trimmed\)/);
  assert.match(dialog, /never uploaded as a project source or deleted by/);
  assert.match(nativeApi, /"pick_native_project_folder"/);
  assert.match(chatApi, /"\/api\/chat\/projects\/open-folder"/);
});

test("an unavailable local folder is blocked with a visible error", () => {
  const projectsPage = source("../src/features/chat/projects-page.tsx");
  assert.match(projectsPage, /workspaceAvailable === false/);
  assert.match(projectsPage, /Project folder is unavailable/);
});
