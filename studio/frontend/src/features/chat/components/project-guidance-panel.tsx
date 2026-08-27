// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { toast } from "@/lib/toast";

import {
  type ProjectCommandRules,
  type ProjectInstructions,
  type ProjectSkills,
  getProjectCommandRules,
  getProjectInstructions,
  getProjectSkills,
  initializeProjectAgents,
  subscribeProjectGuidanceUpdated,
} from "../api/project-guidance-api";
import type { ProjectRecord } from "../types";

function rejectedMessage(
  result: PromiseSettledResult<unknown>,
  fallback: string,
): string | null {
  if (result.status === "fulfilled") {
    return null;
  }
  return result.reason instanceof Error ? result.reason.message : fallback;
}

export function ProjectGuidancePanel({
  project,
}: {
  project: ProjectRecord;
}) {
  const [instructions, setInstructions] = useState<ProjectInstructions | null>(
    null,
  );
  const [skills, setSkills] = useState<ProjectSkills | null>(null);
  const [rules, setRules] = useState<ProjectCommandRules | null>(null);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requestRevision = useRef(0);
  const available = project.workspaceAvailable !== false;

  const refresh = useCallback(async () => {
    if (!available) {
      setLoading(false);
      return;
    }
    const revision = ++requestRevision.current;
    setLoading(true);
    setError(null);
    try {
      const results = await Promise.allSettled([
        getProjectInstructions(project.id),
        getProjectSkills(project.id),
        getProjectCommandRules(project.id),
      ] as const);
      if (revision !== requestRevision.current) {
        return;
      }
      const [nextInstructions, nextSkills, nextRules] = results;
      if (nextInstructions.status === "fulfilled") {
        setInstructions(nextInstructions.value);
      }
      if (nextSkills.status === "fulfilled") {
        setSkills(nextSkills.value);
      }
      if (nextRules.status === "fulfilled") {
        setRules(nextRules.value);
      }
      const failures = [
        rejectedMessage(nextInstructions, "Could not load AGENTS.md."),
        rejectedMessage(nextSkills, "Could not load project skills."),
        rejectedMessage(nextRules, "Could not load project command rules."),
      ].filter((message): message is string => message !== null);
      setError(failures.length > 0 ? failures.join(" ") : null);
    } finally {
      if (revision === requestRevision.current) {
        setLoading(false);
      }
    }
  }, [available, project.id]);

  useEffect(() => {
    const initialRefresh = window.setTimeout(() => {
      refresh().catch(() => undefined);
    }, 0);
    const unsubscribe = subscribeProjectGuidanceUpdated(project.id, () => {
      refresh().catch(() => undefined);
    });
    return () => {
      window.clearTimeout(initialRefresh);
      requestRevision.current += 1;
      unsubscribe();
    };
  }, [project.id, refresh]);

  async function createAgentsFile(): Promise<void> {
    if (creating || !available) {
      return;
    }
    setCreating(true);
    try {
      const result = await initializeProjectAgents(project.id);
      setInstructions(result.instructions);
      toast.success(
        result.created ? "Created AGENTS.md" : `${result.path} already exists`,
        {
          description: result.created
            ? "Future agent turns will use these project instructions."
            : "The existing file was left unchanged.",
        },
      );
    } catch (nextError) {
      toast.error("Could not create AGENTS.md", {
        description: nextError instanceof Error ? nextError.message : undefined,
      });
    } finally {
      setCreating(false);
    }
  }

  const rootInstruction = instructions?.layers.find(
    (layer) =>
      layer.scope === "." &&
      (layer.path === "AGENTS.override.md" || layer.path === "AGENTS.md"),
  );

  return (
    <div className="mt-3 border-t border-border/70 pt-3">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-xs font-semibold text-foreground">Agent context</p>
          <p className="mt-0.5 text-xs text-muted-foreground">
            Repository instructions, skills, and command policy for agent turns.
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={
              creating || loading || !available || Boolean(rootInstruction)
            }
            onClick={createAgentsFile}
          >
            {creating
              ? "Creating..."
              : rootInstruction
                ? `${rootInstruction.path} found`
                : "Create AGENTS.md"}
          </Button>
          <Button
            type="button"
            size="sm"
            variant="ghost"
            disabled={loading || !available}
            onClick={refresh}
          >
            {loading ? "Refreshing..." : "Refresh"}
          </Button>
        </div>
      </div>

      {available ? (
        <>
          {error ? (
            <p className="mt-2 text-xs text-destructive" role="alert">
              {error}
            </p>
          ) : null}
          <div className="mt-3 grid gap-3 lg:grid-cols-3">
            <InstructionsCard instructions={instructions} loading={loading} />

            <SkillsCard skills={skills} loading={loading} />

            <RulesCard rules={rules} loading={loading} />
          </div>
        </>
      ) : (
        <p className="mt-2 text-xs text-muted-foreground">
          Reconnect the project folder to resolve its agent context.
        </p>
      )}
    </div>
  );
}

function InstructionsCard({
  instructions,
  loading,
}: {
  instructions: ProjectInstructions | null;
  loading: boolean;
}) {
  return (
    <div className="rounded-xl bg-background/70 px-3 py-2">
      <p className="text-xs font-medium text-foreground">
        AGENTS.md
        {instructions
          ? ` (${instructions.layers.length} layer${instructions.layers.length === 1 ? "" : "s"})`
          : ""}
      </p>
      {instructions && instructions.layers.length > 0 ? (
        <div className="mt-2 space-y-2">
          {instructions.layers.map((layer) => (
            <details key={layer.path}>
              <summary className="cursor-pointer text-xs text-muted-foreground">
                {layer.path}
                {layer.truncated ? " (truncated)" : ""}
              </summary>
              <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap break-words text-[11px] leading-5 text-muted-foreground">
                {layer.content}
              </pre>
            </details>
          ))}
        </div>
      ) : (
        <p className="mt-1 text-xs text-muted-foreground">
          {loading ? "Resolving instructions..." : "No AGENTS.md found."}
        </p>
      )}
      {instructions && instructions.issues.length > 0 ? (
        <p className="mt-2 text-xs text-amber-600 dark:text-amber-400">
          {instructions.issues.length} instruction path
          {instructions.issues.length === 1 ? " was" : "s were"} excluded.
        </p>
      ) : null}
    </div>
  );
}

function SkillsCard({
  skills,
  loading,
}: {
  skills: ProjectSkills | null;
  loading: boolean;
}) {
  return (
    <div className="rounded-xl bg-background/70 px-3 py-2">
      <p className="text-xs font-medium text-foreground">
        Project skills{skills ? ` (${skills.skills.length})` : ""}
      </p>
      {skills && skills.skills.length > 0 ? (
        <div className="mt-2 space-y-2">
          {skills.skills.map((skill) => (
            <div key={`${skill.name}:${skill.path}`}>
              <p className="text-xs font-medium text-foreground">
                ${skill.name}
                {skill.ambiguous ? " (duplicate name)" : ""}
              </p>
              <p className="text-xs text-muted-foreground">
                {skill.description}
              </p>
              <code className="block break-all text-[10px] text-muted-foreground/80">
                {skill.path}
              </code>
              {skill.ambiguous ? (
                <p className="text-[10px] text-amber-600 dark:text-amber-400">
                  Explicit selection is disabled until this duplicate skill name
                  is resolved.
                </p>
              ) : null}
            </div>
          ))}
        </div>
      ) : (
        <p className="mt-1 text-xs text-muted-foreground">
          {loading
            ? "Discovering skills..."
            : "No .agents/skills packages found."}
        </p>
      )}
      {skills && skills.issues.length > 0 ? (
        <p className="mt-2 text-xs text-amber-600 dark:text-amber-400">
          {skills.issues.length} skill path
          {skills.issues.length === 1 ? " was" : "s were"} excluded.
        </p>
      ) : null}
    </div>
  );
}

function RulesCard({
  rules,
  loading,
}: {
  rules: ProjectCommandRules | null;
  loading: boolean;
}) {
  return (
    <div className="rounded-xl bg-background/70 px-3 py-2">
      <p className="text-xs font-medium text-foreground">
        Command rules{rules ? ` (${rules.rules.length})` : ""}
      </p>
      {rules && rules.rules.length > 0 ? (
        <div className="mt-2 space-y-2">
          {rules.rules.map((rule) => (
            <div key={rule.id}>
              <div className="flex items-center gap-2">
                <code className="break-all text-[11px] text-foreground">
                  {rule.pattern
                    .map((part) =>
                      Array.isArray(part) ? `{${part.join("|")}}` : part,
                    )
                    .join(" ")}
                </code>
                <span
                  className={
                    rule.decision === "forbidden"
                      ? "text-[10px] font-medium text-destructive"
                      : rule.decision === "prompt"
                        ? "text-[10px] font-medium text-amber-600 dark:text-amber-400"
                        : "text-[10px] font-medium text-emerald-600 dark:text-emerald-400"
                  }
                >
                  {rule.decision}
                </span>
              </div>
              {rule.justification ? (
                <p className="text-[10px] text-muted-foreground">
                  {rule.justification}
                </p>
              ) : null}
              <code className="block break-all text-[10px] text-muted-foreground/80">
                {rule.path}:{rule.line}
              </code>
            </div>
          ))}
        </div>
      ) : (
        <p className="mt-1 text-xs text-muted-foreground">
          {loading
            ? "Resolving command policy..."
            : "No .codex/rules files found."}
        </p>
      )}
      <p className="mt-2 text-[10px] text-muted-foreground">
        Rules apply when terminal commands leave the project sandbox.
      </p>
    </div>
  );
}
