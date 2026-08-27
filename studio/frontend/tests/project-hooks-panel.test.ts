// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const panel = await readFile(
  new URL(
    "../src/features/chat/components/project-hooks-panel.tsx",
    import.meta.url,
  ),
  "utf8",
);
const controls = await readFile(
  new URL(
    "../src/features/chat/components/project-workspace-controls.tsx",
    import.meta.url,
  ),
  "utf8",
);

test("workspace controls mount the hooks browser beside project guidance", () => {
  assert.match(
    controls,
    /import \{ ProjectHooksPanel \} from "\.\/project-hooks-panel";/,
  );
  assert.match(controls, /<ProjectGuidancePanel project=\{project\} \/>/);
  assert.match(controls, /<ProjectHooksPanel project=\{project\} \/>/);
});

test("trust confirmation captures and revalidates the complete identity", () => {
  const refreshStart = panel.indexOf("const refresh = useCallback");
  const effectStart = panel.indexOf("useEffect(() =>", refreshStart);
  const refreshSource = panel.slice(refreshStart, effectStart);
  assert.ok(refreshStart >= 0 && effectStart > refreshStart);
  assert.match(
    refreshSource,
    /Promise\.allSettled\(\[[\s\S]*getProjectHooks\(project\.id\),[\s\S]*getProjectHooksTrustState\(project\.id\)/,
  );
  assert.doesNotMatch(refreshSource, /trustProjectHooks\(/);

  assert.match(panel, /useState<ProjectHooksTrustIdentity \| null>\(null\)/);
  assert.match(
    panel,
    /function trustIdentityFor[\s\S]*return \{[\s\S]*projectId,[\s\S]*workspaceRevision,[\s\S]*contentHash: hooks\.contentHash,[\s\S]*revision: hooks\.revision/,
  );
  assert.match(panel, /const workspaceRevision = hooks\?\.workspaceRevision/);
  assert.match(
    panel,
    /const identity = trustIdentityFor\(project\.id, hooks\)/,
  );
  assert.match(panel, /setTrustConfirmation\(identity\)/);
  assert.match(
    panel,
    /projectHooksTrustIdentityMatches\(identity, \{[\s\S]*projectId: project\.id,[\s\S]*hooks/,
  );
  assert.match(
    panel,
    /trustProjectHooks\(identity\.projectId, \{[\s\S]*workspaceRevision: identity\.workspaceRevision,[\s\S]*contentHash: identity\.contentHash,[\s\S]*revision: identity\.revision/,
  );
  assert.match(
    panel,
    /Project hooks changed\. Refresh and review the current file\./,
  );
  assert.match(
    panel,
    /<AlertDialogTitle>Trust project hooks\?<\/AlertDialogTitle>/,
  );
  assert.match(panel, /value=\{trustConfirmation\.contentHash\}/);
  assert.match(
    panel,
    /Workspace revision \{trustConfirmation\.workspaceRevision\}/,
  );
});

test("panel shows the full hash and isolates visibly escaped project text", () => {
  assert.doesNotMatch(panel, /shortProjectHooksHash/);
  assert.match(panel, /value=\{hooks\.contentHash\}/);
  assert.match(panel, /value=\{storedTrust\.contentHash\}/);
  assert.match(panel, /visibleProjectHookText\(value \?\? null\)/);
  assert.match(panel, /<pre[\s\S]*dir="ltr"/);
  assert.match(panel, /style=\{\{ unicodeBidi: "isolate" \}\}/);
  for (const expression of [
    "hooks.sourcePath",
    "hooks.description",
    "group.matcher",
    "handler.command",
    "handler.commandWindows",
    "handler.statusMessage",
    "handler.id",
  ]) {
    assert.match(
      panel,
      new RegExp(`value=\\{${expression.replace(".", "\\.")}\\}`),
    );
  }
});

test("refresh and mutation clear stale snapshots and pending confirmation", () => {
  const refreshStart = panel.indexOf("const refresh = useCallback");
  const effectStart = panel.indexOf("useEffect(() =>", refreshStart);
  const refreshSource = panel.slice(refreshStart, effectStart);
  assert.match(
    refreshSource,
    /setTrustConfirmation\(null\);[\s\S]*setMutation\(null\);[\s\S]*setHooks\(null\);[\s\S]*setLoading\(true\)/,
  );

  const mutationStart = panel.indexOf("async function applyMutation");
  const trustStart = panel.indexOf("function requestTrust", mutationStart);
  const mutationSource = panel.slice(mutationStart, trustStart);
  assert.match(
    mutationSource,
    /setTrustConfirmation\(null\);[\s\S]*setHooks\(null\);[\s\S]*setMutation\(key\);[\s\S]*setLoading\(true\)/,
  );
  assert.match(mutationSource, /setLoading\(false\)/);
});

test("refresh rejects a hooks snapshot raced by a newer trust revision", () => {
  const refreshStart = panel.indexOf("const refresh = useCallback");
  const effectStart = panel.indexOf("useEffect(() =>", refreshStart);
  const refreshSource = panel.slice(refreshStart, effectStart);
  assert.match(
    refreshSource,
    /trustStateResult\.status === "fulfilled"[\s\S]*trustStateResult\.value\.revision !== hooksResult\.value\.revision/,
  );
  assert.match(
    refreshSource,
    /setHooks\(null\);[\s\S]*setTrustState\(trustStateResult\.value\);[\s\S]*Project hook trust changed during refresh\. Refresh again\./,
  );
  assert.match(
    refreshSource,
    /else \{[\s\S]*setHooks\(hooksResult\.value\);[\s\S]*setTrustState\(trustStateFromHooks\(hooksResult\.value\)\)/,
  );
});

test("loading disables refresh, trust, revoke, and handler actions", () => {
  assert.match(panel, /disabled=\{loading \|\| busy\}/g);
  assert.match(
    panel,
    /disabled=\{loading \|\| busy \|\| !hooks\.contentHash\}/,
  );
  assert.match(
    panel,
    /disabled=\{loading \|\| busy \|\| trustConfirmation === null\}/,
  );
  assert.match(panel, /disabled=\{loading \|\| busy\}/);
  assert.match(panel, /disabled=\{!trusted \|\| disabled\}/);
  assert.match(panel, /disabled=\{loading \|\| busy\}/);
});

test("unavailable workspace fallback surfaces stored trust and can revoke it", () => {
  assert.match(
    panel,
    /return hooks === null[\s\S]*projectAvailable !== false[\s\S]*hooks\.workspaceAvailable !== false/,
  );
  assert.match(
    panel,
    /currentWorkspaceAvailable\([\s\S]*project\.workspaceAvailable,[\s\S]*hooks/,
  );
  assert.match(panel, /storedTrust=\{storedTrust\}/);
  assert.match(
    panel,
    /Stored trust is inactive while the workspace is unavailable\./,
  );
  assert.match(panel, /Stored SHA-256/);
  assert.match(
    panel,
    /const revision = hooks\?\.trusted \? hooks\.revision : storedTrust\?\.revision/,
  );
  assert.match(panel, /revokeProjectHooks\(project\.id, \{ revision \}\)/);
  assert.match(
    panel,
    /Reconnect the project folder to inspect the current hook file/,
  );
});

test("stored trust remains revocable across absent, changed, error, and unavailable states", () => {
  assert.match(panel, /getProjectHooksTrustState/);
  assert.match(
    panel,
    /useState<ProjectHooksTrustState \| null>\([\s\S]*null,[\s\S]*\)/,
  );
  assert.match(
    panel,
    /The current hook file could not be inspected\. Stored trust is inactive until the exact file and workspace can be verified\./,
  );
  assert.match(
    panel,
    /Stored trust is inactive while the project hook file is absent\./,
  );
  assert.match(
    panel,
    /Stored trust does not match the current hook file and workspace, so it is inactive\./,
  );
  assert.match(
    panel,
    /Stored trust is inactive while the workspace is unavailable\./,
  );
  assert.match(
    panel,
    /function StoredTrustNotice[\s\S]*onRevokeTrust\(\)\.catch/,
  );
});

test("handler UI separates saved preference from effective active state", () => {
  assert.match(panel, /checked=\{handler\.enabled\}/);
  assert.match(
    panel,
    /handler\.enabled \? "Enabled" : "Disabled"[\s\S]*preference,[\s\S]*handler\.active \? "trusted and eligible" : "not eligible"/,
  );
  assert.match(
    panel,
    /setProjectHookHandlerEnabled\(projectId, handler\.id, \{[\s\S]*workspaceRevision,[\s\S]*contentHash,[\s\S]*revision,[\s\S]*enabled/,
  );
  assert.match(panel, /background requested \(runtime follow-up\)/);
  assert.match(panel, /foreground requested \(runtime follow-up\)/);
});

test("copy describes review and trust without claiming every event executes", () => {
  assert.match(
    panel,
    /Review configured hook commands and manage exact-file trust\. Hook[\s\S]*execution is a separate lifecycle integration\./,
  );
  assert.match(
    panel,
    /It[\s\S]*does not run hooks\. Command execution and lifecycle wiring are a[\s\S]*separate integration\./,
  );
  assert.match(
    panel,
    /Trusting resets handler preferences and[\s\S]*enables all configured handlers\./,
  );
  assert.match(
    panel,
    /Any lifecycle integration must require trust for this exact file and workspace\./,
  );
  assert.doesNotMatch(panel, /before any hook can run/);
});

test("every async state update is guarded and cleanup retires the panel", () => {
  assert.match(
    panel,
    /const requestGuard = useRef\(new ProjectHooksRequestGuard\(\)\)/,
  );
  assert.match(panel, /const guard = requestGuard\.current/);
  assert.match(panel, /guard\.activate\(\)/);
  assert.match(panel, /guard\.retire\(\)/);
  assert.match(
    panel,
    /const requestRevision = requestGuard\.current\.begin\(\)/g,
  );
  assert.match(
    panel,
    /if \(requestGuard\.current\.accepts\(requestRevision\)\) \{\s*setHooks\(next\)/g,
  );
  assert.match(
    panel,
    /if \(requestGuard\.current\.accepts\(requestRevision\)\) \{[\s\S]*setTrustState\(fallbackTrustState\);[\s\S]*setError\(errorMessage\(nextError\)\)/,
  );
});
