/* Source-backed projection of the OpenCode session loop, its processor, and title task. */
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");

function extract(ts, sessionDir, sourceUrl) {
  function parse(name) {
    const file = path.join(sessionDir, name);
    const raw = fs.readFileSync(file, "utf8");
    const source = ts.createSourceFile(file, raw, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
    if (source.parseDiagnostics.length) throw new Error(`Invalid TypeScript: ${name}`);
    return source;
  }

  const prompt = parse("prompt.ts");
  const processor = parse("processor.ts");
  const compaction = parse("compaction.ts");
  const llm = parse("llm.ts");
  const text = (node) => node.getText().replace(/\s+/g, " ").trim();
  const scan = (root, predicate) => {
    const matches = [];
    function visit(node) {
      if (predicate(node)) matches.push(node);
      ts.forEachChild(node, visit);
    }
    visit(root);
    return matches;
  };
  function one(items, description) {
    if (items.length !== 1) throw new Error(`Expected one ${description}, found ${items.length}`);
    return items[0];
  }
  function moduleBody(source, name) {
    return one(source.statements.filter((n) => ts.isModuleDeclaration(n) && n.name.text === name), name).body;
  }
  function method(module, name) {
    return one(module.statements.filter((n) => ts.isFunctionDeclaration(n) && n.name?.text === name), name);
  }
  function verifyImport(source, symbol, module) {
    one(source.statements.filter((n) => ts.isImportDeclaration(n) && n.moduleSpecifier.text === module &&
      n.importClause?.namedBindings?.elements?.some((s) => s.name.text === symbol &&
        (!s.propertyName || s.propertyName.text === symbol))), `import of ${symbol} from ${module}`);
  }
  function call(root, name) {
    return one(scan(root, (n) => ts.isCallExpression(n) && text(n.expression) === name), `call to ${name}`);
  }
  function awaited(root, name) {
    const expr = call(root, name);
    if (!ts.isAwaitExpression(expr.parent)) throw new Error(`Expected an awaited call to ${name}`);
    return expr;
  }
  function guard(root, predicate, description) {
    return one(scan(root, (n) => ts.isIfStatement(n) && predicate(text(n.expression), n)), description);
  }
  function exactGuard(root, expected) {
    return guard(root, (condition) => condition === expected, `guard ${expected}`);
  }
  function directGuard(block, expected) {
    return one(block.statements.filter((node) => ts.isIfStatement(node) && text(node.expression) === expected),
      `direct guard ${expected}`);
  }
  function checkDirectControl(statement, check, description) {
    const block = ts.isBlock(statement) ? statement.statements : [statement];
    if (!block.some(check)) throw new Error(`Missing ${description}`);
  }
  function returns(value) {
    return (stmt) => ts.isReturnStatement(stmt) && stmt.expression?.text === value &&
      ts.isStringLiteral(stmt.expression);
  }
  function anchors(nodes) {
    const unique = new Map();
    for (const node of nodes) {
      const source = node.getSourceFile();
      const { line, character } = source.getLineAndCharacterOfPosition(node.getStart(source));
      const end = source.getLineAndCharacterOfPosition(node.getEnd()).line;
      const file = path.basename(source.fileName);
      const evidence = { file, line: line + 1, end_line: end + 1, column: character + 1 };
      if (sourceUrl) evidence.url = `${sourceUrl}/${file}#L${line + 1}-L${end + 1}`;
      unique.set(`${file}:${line}:${character}`, evidence);
    }
    return [...unique.values()];
  }

  verifyImport(prompt, "LLM", "./llm");
  verifyImport(processor, "LLM", "./llm");
  verifyImport(prompt, "SessionProcessor", "./processor");
  verifyImport(compaction, "SessionProcessor", "./processor");
  verifyImport(prompt, "SessionCompaction", "./compaction");
  verifyImport(llm, "streamText", "ai");

  const promptModule = moduleBody(prompt, "SessionPrompt");
  const processorModule = moduleBody(processor, "SessionProcessor");
  const compactionModule = moduleBody(compaction, "SessionCompaction");
  const loopDecl = one(scan(promptModule, (n) => ts.isVariableDeclaration(n) && text(n.name) === "loop" &&
    n.initializer && ts.isCallExpression(n.initializer) && text(n.initializer.expression) === "fn"), "loop declaration");
  const loopHandler = one(loopDecl.initializer.arguments.filter(ts.isArrowFunction), "loop handler");
  const loop = one(loopHandler.body.statements.filter((n) => ts.isWhileStatement(n) && text(n.expression) === "true"), "session loop");
  const createProcessor = method(processorModule, "create");
  const processMethod = one(scan(createProcessor, (n) => ts.isMethodDeclaration(n) && text(n.name) === "process"), "processor method");
  const request = awaited(processMethod, "LLM.stream");
  const provider = call(method(moduleBody(llm, "LLM"), "stream"), "streamText");
  const normalProcessor = call(loop, "SessionProcessor.create");
  const turn = awaited(loop, "processor.process");
  const compactionTask = exactGuard(loop, 'task?.type === "compaction"');
  const enterCompaction = awaited(compactionTask, "SessionCompaction.process");
  const compactionHandler = method(compactionModule, "process");
  const compactionProcessor = call(compactionHandler, "SessionProcessor.create");
  const compact = awaited(compactionHandler, "processor.process");
  const title = awaited(method(promptModule, "ensureTitle"), "LLM.stream");
  if (scan(promptModule, (n) => ts.isCallExpression(n) && text(n.expression) === "LLM.stream").length !== 1) {
    throw new Error("Additional direct model calls in SessionPrompt need review");
  }
  const launch = call(loop, "ensureTitle");
  if (!ts.isExpressionStatement(launch.parent)) throw new Error("Title generation is no longer a detached call");
  const titleGuard = launch.parent.parent;
  if (!ts.isIfStatement(titleGuard) || text(titleGuard.expression) !== "step === 1") {
    throw new Error("Title generation is not guarded by the first step");
  }

  const earlyStop = exactGuard(loop, "abort.aborted");
  checkDirectControl(earlyStop.thenStatement, ts.isBreakStatement, "abort loop exit");
  const alreadyFinished = guard(loop, (cond) => cond.startsWith("lastAssistant?.finish &&") &&
    cond.includes("lastUser.id < lastAssistant.id"), "previous answer exit");
  checkDirectControl(alreadyFinished.thenStatement, ts.isBreakStatement, "previous answer loop exit");
  const turnStop = directGuard(loop.statement, 'result === "stop"');
  checkDirectControl(turnStop.thenStatement, ts.isBreakStatement, "processor stop loop exit");
  const compactStop = exactGuard(compactionTask, 'result === "stop"');
  checkDirectControl(compactStop.thenStatement, ts.isBreakStatement, "compaction stop loop exit");
  checkDirectControl(compactionTask.thenStatement, ts.isContinueStatement, "compaction loop continuation");
  const compactFailure = exactGuard(compactionHandler, 'result === "compact"');
  checkDirectControl(compactFailure.thenStatement, returns("stop"), "stop on compaction overflow");
  const compactSuccess = one(compactionHandler.body.statements.filter(returns("continue")), "compaction success return");
  const resumeAfterCompaction = exactGuard(compactionHandler, 'result === "continue" && input.auto');
  const requestCompaction = exactGuard(loop, 'result === "compact"');
  const enqueue = awaited(requestCompaction, "SessionCompaction.create");
  const historyOverflow = guard(loop, (cond) => cond.includes("lastFinished.summary !== true") &&
    cond.includes("SessionCompaction.isOverflow"), "history overflow guard");
  const enqueueFromHistory = awaited(historyOverflow, "SessionCompaction.create");
  const enqueueDefinition = one(scan(compactionModule, (n) => ts.isVariableDeclaration(n) &&
    text(n.name) === "create"), "compaction enqueue definition");
  const continueTurn = one(processMethod.body.statements.flatMap((stmt) => ts.isWhileStatement(stmt)
    ? stmt.statement.statements.filter(returns("continue")) : []), "processor continuation");
  const retry = exactGuard(processMethod, "retry !== undefined");
  const waitRetry = call(retry, "SessionRetry.sleep");
  checkDirectControl(retry.thenStatement, ts.isContinueStatement, "retry loop continuation");
  checkDirectControl(loop.statement, ts.isContinueStatement, "next session iteration");

  const nodes = [
    { id: "__start__", kind: "entry", label: "Start", evidence: anchors([loop]) },
    { id: "turn", kind: "llm_call", label: "Main turn", evidence: anchors([request, turn, normalProcessor, provider]),
      caller: anchors([turn])[0], model_selector: "session user message's provider and model" },
    { id: "compaction", kind: "llm_call", label: "Context compaction",
      evidence: anchors([request, compact, compactionProcessor, enterCompaction, provider]),
      caller: anchors([compact])[0], model_selector: "compaction-agent model or user-selected model" },
    { id: "title", kind: "llm_call", label: "Session title (background)", evidence: anchors([title, launch, provider]),
      caller: anchors([launch])[0], model_selector: "title-agent model, small model, or user-selected model" },
    { id: "__end__", kind: "terminal", label: "Session loop exits", evidence: anchors([earlyStop, alreadyFinished, turnStop, compactStop]) },
  ];
  const edges = [];
  function edge(source, target, witnesses, kind = "next_call") {
    const evidence = anchors(witnesses.flatMap(([, sourceNodes]) => sourceNodes));
    edges.push({
      source, target, kind, route: kind === "async_launch" ? "launch" : "next",
      condition: witnesses.map(([condition]) => condition).join(" OR "),
      evidence, witnesses: witnesses.map(([condition, sourceNodes]) => ({ condition, evidence: anchors(sourceNodes) })),
    });
  }
  edge("__start__", "turn", [["First runnable normal turn", [loop, normalProcessor, turn, request]]]);
  edge("__start__", "compaction", [
    ["A compaction task is pending", [loop, compactionTask, enterCompaction, compact, request]],
    ["Existing history requires compaction", [historyOverflow, enqueueFromHistory, enqueueDefinition, enterCompaction, compact, request]],
  ]);
  edge("__start__", "__end__", [["Already completed or cancelled before a request", [earlyStop, alreadyFinished]]]);
  edge("__start__", "title", [["First step launches title helper and helper guards pass", [titleGuard, launch, title]]], "async_launch");
  edge("turn", "turn", [
    ["Processor permits continuation and the next iteration reaches a normal turn", [request, continueTurn, loop, turn]],
    ["Retriable model failure", [request, retry, waitRetry]],
  ]);
  edge("turn", "compaction", [
    ["Processor requests compaction", [turn, requestCompaction, enqueue, enqueueDefinition, compactionTask, enterCompaction, compact]],
    ["The next loop detects overflowing history", [turn, continueTurn, historyOverflow, enqueueFromHistory, enterCompaction, compact]],
  ]);
  edge("turn", "__end__", [["Processor stops or the next iteration exits", [turnStop, earlyStop, alreadyFinished]]]);
  edge("compaction", "compaction", [["Retriable failure in the compaction processor", [compact, request, retry, waitRetry]]]);
  edge("compaction", "turn", [["Successful automatic compaction resumes the session", [resumeAfterCompaction, compactSuccess, enterCompaction, loop, turn]]]);
  edge("compaction", "__end__", [["Compaction cannot fit the context or exits with an error", [compactFailure, compactStop]]]);

  return {
    schema_version: "1.0", extractor: "opencode-llm", graph_kind: "llm_call_successors", entry: "__start__",
    scope: "OpenCode SessionPrompt.loop, SessionProcessor.process, SessionCompaction.process, and ensureTitle",
    analysis_scope: "AST-anchored projection of direct model paths, with distinct caller context. Background launch is not a first-call ordering guarantee. Provider internals, nested task sessions, and arbitrary plugins are outside scope.",
    sources: [prompt, processor, compaction, llm].map((source) => ({
      file: path.basename(source.fileName), sha256: crypto.createHash("sha256").update(source.text).digest("hex"),
    })),
    nodes, edges, diagnostics: [],
  };
}

function main() {
  const [typescript, sessionDir, sourceUrl, output] = process.argv.slice(2);
  if (!output) throw new Error("Usage: node extract_opencode_llm.cjs TYPESCRIPT SESSION_DIR SOURCE_URL OUTPUT_JSON");
  const ts = require(typescript.startsWith(".") || path.isAbsolute(typescript) ? path.resolve(typescript) : typescript);
  const graph = extract(ts, sessionDir, sourceUrl);
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, JSON.stringify(graph, null, 2) + "\n", "utf8");
}

if (require.main === module) {
  try {
    main();
  } catch (error) {
    console.error(`AGENTEVE OpenCode: ${error.message}`);
    process.exitCode = 2;
  }
}

