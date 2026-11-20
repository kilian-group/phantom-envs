// The pi extension of phantom_envs.terminal.pi: the episode budgets that pi has no setting for.
// PiHarness sets the limits in the environment; an unset limit is off.
//
//   PHANTOM_MAX_TURNS         stop the agent after this many turns
//   PHANTOM_COMMAND_TIMEOUT   seconds a bash command may run
//   PHANTOM_MAX_OUTPUT_CHARS  characters of a command's output that the model reads
//   PHANTOM_REQUIRED_TOOLS    comma-separated tools the agent must have from its first turn on

const limit = (name) => Number(process.env[name] || 0);

function truncate(text, maxChars) {
  if (text.length <= maxChars) return text;
  return `${text.slice(0, maxChars)}\n[output truncated: showed ${maxChars} of ${text.length} characters]`;
}

export default function (pi) {
  const maxTurns = limit("PHANTOM_MAX_TURNS");
  const commandTimeout = limit("PHANTOM_COMMAND_TIMEOUT");
  const maxOutputChars = limit("PHANTOM_MAX_OUTPUT_CHARS");

  // pi starts without an MCP server's tools if the server is slow to connect. The episode would then
  // run without them, so it ends here instead, with an event that PiHarness reports as an error.
  const requiredTools = (process.env.PHANTOM_REQUIRED_TOOLS || "").split(",").filter(Boolean);
  let checkedTools = false;
  pi.on("turn_start", (_event, ctx) => {
    if (checkedTools) return;
    checkedTools = true;
    const active = new Set(pi.getActiveTools());
    const missing = requiredTools.filter((name) => !active.has(name));
    if (missing.length === 0) return;
    process.stdout.write(`${JSON.stringify({ type: "phantom_missing_tools", tools: missing })}\n`);
    ctx.abort();
  });

  let turns = 0;
  pi.on("turn_end", (_event, ctx) => {
    turns += 1;
    if (maxTurns && turns >= maxTurns) ctx.abort();
  });

  pi.on("tool_call", (event) => {
    if (commandTimeout && event.toolName === "bash") {
      event.input.timeout = Math.min(event.input.timeout || commandTimeout, commandTimeout);
    }
  });

  pi.on("tool_result", (event) => {
    if (!maxOutputChars) return;
    const content = event.content.map((part) =>
      part.type === "text" ? { ...part, text: truncate(part.text, maxOutputChars) } : part,
    );
    return { content };
  });
}
