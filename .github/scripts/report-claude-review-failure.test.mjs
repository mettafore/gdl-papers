import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { failureMessage } from "./report-claude-review-failure.mjs";

test("classifies provider errors without exposing their contents", () => {
  for (const [result, expected] of [
    ["Invalid API key DO_NOT_PRINT_THIS", /authentication failed/],
    ["Credit balance exhausted DO_NOT_PRINT_THIS", /billing or usage limit/],
    ["Model is unavailable DO_NOT_PRINT_THIS", /unavailable model/],
    ["Other failure DO_NOT_PRINT_THIS", /without a recognized provider error/],
  ]) {
    const message = failureMessage([{ type: "result", is_error: true, result }]);
    assert.match(message, expected);
    assert.ok(!message.includes("DO_NOT_PRINT_THIS"));
  }
  assert.match(failureMessage(null), /did not complete a review/);
});

test("CLI writes sanitized diagnostics and tolerates missing output", () => {
  const directory = mkdtempSync(join(tmpdir(), "claude-review-diagnostics-"));
  try {
    const execution = join(directory, "execution.json");
    const summary = join(directory, "summary.md");
    writeFileSync(execution, JSON.stringify([
      { type: "result", is_error: true, result: "OAuth invalid DO_NOT_PRINT_THIS" },
    ]));
    const script = new URL("./report-claude-review-failure.mjs", import.meta.url);
    const run = spawnSync(process.execPath, [fileURLToPath(script)], {
      env: { ...process.env, CLAUDE_EXECUTION_FILE: execution, GITHUB_STEP_SUMMARY: summary },
      encoding: "utf8",
    });
    assert.equal(run.status, 0);
    assert.match(run.stdout, /authentication failed/);
    assert.ok(!run.stdout.includes("DO_NOT_PRINT_THIS"));
    assert.ok(!readFileSync(summary, "utf8").includes("DO_NOT_PRINT_THIS"));
    const missing = spawnSync(process.execPath, [fileURLToPath(script)], {
      env: { ...process.env, CLAUDE_EXECUTION_FILE: join(directory, "missing.json"), GITHUB_STEP_SUMMARY: summary },
      encoding: "utf8",
    });
    assert.equal(missing.status, 0);
    assert.match(missing.stdout, /did not complete a review/);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
