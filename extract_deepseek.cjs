/* Project the control flow of DeepSeek Harness's ReactLoopAgent from its TypeScript AST. */
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");

function main() {
  const [typescript, file, sourceUrl, output] = process.argv.slice(2);
  if (!output) throw new Error("usage: node extract_deepseek.cjs PATH_TO_TYPESCRIPT AGENT_TS BLOB_URL OUTPUT_JSON");
  const ts = require(typescript.startsWith(".") || path.isAbsolute(typescript) ? path.resolve(typescript) : typescript);
  const raw = fs.readFileSync(file, "utf8");
  const source = ts.createSourceFile(file, raw, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
  if (source.parseDiagnostics.length) throw new Error("TypeScript source has syntax errors");
  const cls = source.statements.find(node => ts.isClassDeclaration(node) && node.name?.text === "ReactLoopAgent");
  if (!cls) throw new Error("ReactLoopAgent class missing");

  function one(items, name) {
    if (items.length !== 1) throw new Error(`Cannot identify ${name}; matched ${items.length} source statements`);
    return items[0];
  }
  function method(name) {
    return one(cls.members.filter(node => ts.isMethodDeclaration(node) && node.name?.getText(source) === name), name);
  }
  function scan(root, predicate) {
    const matches = [];
    function visit(node) {
      if (predicate(node)) matches.push(node);
      ts.forEachChild(node, visit);
    }
    visit(root);
    return matches;
  }
  const text = node => node.getText(source).replace(/\s+/g, " ").trim();
  function search(root, predicate, name) {
    return one(scan(root, predicate), name);
  }
  function call(root, callee, firstArg) {
    return search(root, n => ts.isCallExpression(n) && text(n.expression) === callee
      && (firstArg === undefined || (n.arguments[0] && ts.isStringLiteral(n.arguments[0])
        && n.arguments[0].text === firstArg)), firstArg || callee);
  }
  function condition(root, expression) {
    return search(root, n => ts.isIfStatement(n) && text(n.expression) === expression, expression);
  }
  function evidence(node) {
    const begin = source.getLineAndCharacterOfPosition(node.getStart(source));
    const end = source.getLineAndCharacterOfPosition(node.getEnd() - 1);
    return {file: path.basename(file), line: begin.line + 1, end_line: end.line + 1,
      column: begin.character + 1, url: `${sourceUrl}#L${begin.line + 1}-L${end.line + 1}`};
  }

  const kick = method("kick");
  const turn = method("turn");
  const preStep = method("preStep");
  const step = method("step");
  const drive = search(kick.body, n => ts.isWhileStatement(n) && text(n.expression) === "await this.turn()", "turn loop");
  const stepLoop = search(turn.body, n => ts.isWhileStatement(n) && text(n.expression) === "true", "step loop");
  const retryLoop = search(step.body, n => ts.isWhileStatement(n) && text(n.expression) === "true", "request loop");
  const onPreStep = call(stepLoop, "this.preStep");
  const blocked = condition(stepLoop, "decision.kind === 'reject'");
  const empty = condition(stepLoop, "phase.step === 0 && decision.messages.length === 0");
  const drained = condition(stepLoop, "turnEnds && decision.messages.length === 0");
  const next = search(stepLoop, n => ts.isIfStatement(n) && ts.isBreakStatement(n.thenStatement)
    && text(n.expression) === "turnEnds && this.inbox.nextStep.length === 0", "decision after turn-stopping hook");
  const invokeStep = call(stepLoop, "this.step");
  const stepEnd = call(stepLoop, "this.session.append", "step/end");
  const toolDispose = call(stepLoop, "decision.toolSet.dispose");
  const turnEnd = call(turn, "this.session.append", "turn/end");
  const turnAgain = condition(turn, "!this.inbox.hasPending");
  const turnError = condition(turn, "cause !== undefined");
  const prepare = call(retryLoop, "this.prepareRequest");
  const model = call(retryLoop, "this.loopCtx.llm.stream");
  const modelRoute = condition(retryLoop, "finish.kind === 'error' || finish.kind === 'aborted'");
  const onRequestError = call(modelRoute, "this.dispatch.waterfall", "agent/request-error");
  const retry = condition(modelRoute, "action?.kind !== 'retry'");
  const again = search(modelRoute, ts.isContinueStatement, "retry continuation");
  const reply = call(retryLoop, "live.blocks");
  const recordReply = search(retryLoop, n => ts.isCallExpression(n)
    && text(n.expression) === "live.settle" && n.getStart(source) > reply.getStart(source)
    && n.arguments[0] && ts.isStringLiteral(n.arguments[0]) && n.arguments[0].text === "assistant/message",
  "assistant response persistence");
  const tokenLimit = condition(retryLoop, "finish.kind === 'max-tokens'");
  const noTools = condition(retryLoop, "toolCalls.length === 0");
  const execute = call(retryLoop, "this.loopCtx.tools.execute");
  const conclude = search(retryLoop, n => ts.isReturnStatement(n) && n.expression
    && ts.isConditionalExpression(n.expression) && text(n.expression.condition) === "result.concluded", "tool result decision");

  const nodes = [];
  const edges = [];
  const diagnostics = [];
  function node(id, kind, label, ...anchors) {
    nodes.push({id, kind, label, evidence: anchors.map(evidence)});
  }
  function edge(start, target, anchor, route = "next", kind = "flow", predicate = null) {
    edges.push({source: start, target, route, kind, condition: predicate, evidence: [evidence(anchor)]});
  }
  function branch(start, yes, no, anchor) {
    const predicate = text(anchor.expression);
    edge(start, yes, anchor, "yes", "conditional", predicate);
    edge(start, no, anchor, "no", "conditional", `!(${predicate})`);
  }
  node("driver", "entry", "Wake driver", drive);
  node("turn_start", "step", "Open a turn", call(turn, "this.session.append", "turn/start"));
  node("pre_step", "step", "Claim input and resolve context/tools", onPreStep,
    call(preStep, "this.dispatch.waterfall", "agent/pre-step"));
  node("blocked", "branch", "Pre-step rejected?", blocked);
  node("drained", "branch", "Ended turn with no admitted input?", drained);
  node("empty", "branch", "First step with no input?", empty);
  node("step", "step", "Open a step", invokeStep, call(turn, "this.session.append", "step/start"));
  node("prepare", "step", "Prepare model request", prepare);
  node("model", "model", "Stream model response", model);
  node("model_error", "branch", "Model finish is error or aborted?", modelRoute);
  node("retry", "branch", "Retry requested by hook?", onRequestError, retry);
  node("persist", "step", "Persist assistant reply", reply, recordReply);
  node("token_limit", "branch", "Output token limit?", tokenLimit);
  node("no_tools", "branch", "No tool calls?", noTools);
  node("tools", "tool", "Execute tool batch and queue context", execute, conclude);
  node("step_done", "step", "Record step end and dispose tools", stepEnd, toolDispose);
  node("stopping", "branch", "End reason set and no next-step input?", next,
    call(stepLoop, "this.dispatch.serial", "agent/turn-stopping"));
  node("turn_error", "branch", "Turn failed or cancelled", turnError);
  node("end_early", "step", "Record blocked or empty first turn", turnEnd);
  node("end_failure", "step", "Record error or cancellation", turnEnd, turnError);
  node("end_normal", "step", "Record normal turn end", turnEnd, turnAgain);
  node("pending", "branch", "Turn queue empty?", turnAgain);
  node("idle", "terminal", "Driver becomes idle", call(kick, "this.setPhase"));

  edge("driver", "turn_start", drive);
  edge("turn_start", "pre_step", stepLoop);
  edge("pre_step", "blocked", blocked);
  branch("blocked", "end_early", "drained", blocked);
  branch("drained", "end_normal", "empty", drained);
  branch("empty", "end_early", "step", empty);
  edge("step", "prepare", prepare);
  edge("prepare", "model", model);
  edge("model", "model_error", modelRoute);
  edge("model", "turn_error", turnError, "thrown", "possible_exception");
  branch("model_error", "retry", "persist", modelRoute);
  edge("retry", "turn_error", retry, "stop", "exception", text(retry.expression));
  edge("retry", "prepare", again, "retry", "loop", `!(${text(retry.expression)})`);
  edge("persist", "token_limit", tokenLimit);
  branch("token_limit", "step_done", "no_tools", tokenLimit);
  branch("no_tools", "step_done", "tools", noTools);
  edge("tools", "step_done", conclude, "concluded", "conditional", "result.concluded");
  edge("tools", "step_done", conclude, "continue", "conditional", "!result.concluded");
  edge("tools", "turn_error", turnError, "thrown", "possible_exception");
  edge("step_done", "stopping", next);
  edge("stopping", "end_normal", next, "stop", "conditional", text(next.expression));
  edge("stopping", "pre_step", next, "next_step", "loop", `!(${text(next.expression)})`);
  edge("pre_step", "turn_error", turnError, "thrown", "possible_exception");
  edge("turn_error", "end_failure", turnEnd);
  edge("end_failure", "idle", kick);
  edge("end_early", "idle", kick);
  edge("end_normal", "pending", turnAgain);
  edge("pending", "idle", turnAgain, "empty", "conditional", text(turnAgain.expression));
  edge("pending", "turn_start", drive, "next_turn", "loop", `!(${text(turnAgain.expression)})`);
  diagnostics.push({message: "Projection covers the core turn, step and request loops. A thrown model/tool error can pass step cleanup before turn error handling. Plugin internals, individual tool scheduling, cleanup errors, and the next driver after a latched wake are collapsed.",
    evidence: evidence(kick), unresolved: false});
  diagnostics.push({message: "Retry, blocking and pending-input edges depend on resolved plugins and input. The graph is the core driver across profiles.",
    evidence: evidence(preStep), unresolved: false});
  const graph = {schema_version: "1.0", extractor: "deepseek-ts", entry: "driver",
    scope: "ReactLoopAgent.kick/turn/preStep/step", graph_kind: "projected_control_flow",
    sources: [{file: path.basename(file), url: sourceUrl, coordinates: "file",
      sha256: crypto.createHash("sha256").update(raw.replace(/\r\n/g, "\n")).digest("hex")}],
    nodes, edges, diagnostics};
  fs.writeFileSync(output, JSON.stringify(graph, null, 2) + "\n", "utf8");
}

try {
  main();
} catch (error) {
  console.error(error.message);
  process.exitCode = 2;
}
