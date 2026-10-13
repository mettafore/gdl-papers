import { appendFileSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

// Return fixed messages only: provider responses can contain credentials or
// untrusted PR text, so neither the payload nor its fields are printed.
export function failureMessage(messages) {
  const results = Array.isArray(messages)
    ? messages.filter((message) => message?.type === "result" && message.is_error)
    : [];
  const text = JSON.stringify(results).toLowerCase();
  if (/authentication|unauthorized|oauth|invalid.{0,30}(api.key|token)|login/.test(text)) {
    return "Claude authentication failed. Run claude setup-token and update CLAUDE_CODE_OAUTH_TOKEN in repository secrets.";
  }
  if (/credit balance|billing|usage limit|rate.limit|quota/.test(text)) {
    return "Claude reported a billing or usage limit. Check the configured account before retrying.";
  }
  if (/model.{0,40}(not found|not supported|unavailable|does not exist)/.test(text)) {
    return "Claude reported an unavailable model. Check account model access before retrying.";
  }
  return "Claude review failed without a recognized provider error. Check authentication and account access; this job did not complete a review.";
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  let messages = [];
  try {
    const path = process.env.CLAUDE_EXECUTION_FILE ||
      resolve(process.env.RUNNER_TEMP || ".", "claude-execution-output.json");
    messages = JSON.parse(readFileSync(path, "utf8"));
  } catch {
    // Missing or malformed execution output is still a failed review.
  }
  const message = failureMessage(messages);
  process.stdout.write(`::error::${message}\n`);
  if (process.env.GITHUB_STEP_SUMMARY) {
    appendFileSync(process.env.GITHUB_STEP_SUMMARY, `### Claude review failed\n\n${message}\n`);
  }
}
