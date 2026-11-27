# AGENTEVE 项目背景记忆
> 本次总结截止时间戳：2026-11-27 10:39（UTC+8），首次全量建档

## 一、项目概述

AGENTEVE 是面向 Agent Harness 的实验性评测工具，代码位于 `AGENTEVE/`。它从固定版本源码提取大模型调用图，保留跳转对应的源码位置，再用运行轨迹统计节点完成度和上下游衔接，并包含用原始事件核验模型解释的实验脚本。离线提取使用 Python 3.12+ 标准库。`results/`、`stage2/model-cache/`、`sources/stirrup-runtime/`、`stage2/paired-hotpotqa*.json` 在 `.gitignore` 中。第三方快照在 `AGENTEVE/sources/`，不计入自有函数统计。

对话记录只有本会话：2026-11-26 用户要求分析项目；2026-11-27 用户判断项目写得很乱；同日用户要求按函数功能与调用链做一级、二级、三级统计，以便重构。

## 二、代码架构

自有 Python 入口是 `AGENTEVE/agenteve.py` 的 `main`。子命令为 `extract`（`mini`、`langgraph`、`llm-flow`、`stirrup-llm`）、`import-graph`、`score`、`reachability`。

- `agenteve.py`：源码证据、图对象、LangGraph 声明拓扑、mini-swe-agent 控制流识别、打分、Mermaid、命令行。
- `llm_flow.py`：把单模块正常路径投影到下一次聊天模型调用。`CallFlow` 继承关系上被 `stirrup_flow.py` 的 `StirrupFlow` 扩展。
- `reachability.py`：读取 `llm_call_successors` 图和 JSONL 轨迹，计算可达性与配对完成度。它从 `compare_harness_flows.py` 导入 `next_call_adjacency`、`paths_from`、`provenance`。
- `compare_harness_flows.py`：对照已保存调用图与参考图，并核对轨迹转移。
- `stage2/odr_pilot.py`：Open Deep Research 简报到主管的试点运行、本地/DeepSeek 模型、词面评分。`paired_pilot.py`、`paired_paged.py`、`paired_branches.py`、`stage3/blind_review.py`、`stage3/paired_review.py` 从该文件导入模型、异常、读写和缓存路径。
- `stage3/`：把已保存结果整理成复核提示，并检查模型解释中的引用。
- `extract_opencode_llm.cjs`、`extract_deepseek.cjs`：TypeScript 结构提取，不进入 `agenteve.py` 子命令。
- 测试：`AGENTEVE/tests/` 下 5 个 `unittest` 模块。

## 三、特征提取/核心算法链路

1. `Source` 用 `ast` 解析源码，`evidence` 记录文件、行、列，可选 GitHub `#L` 锚点。
2. `langgraph` 只读 `StateGraph` 建造语句和节点函数里直接返回的 `Command`/`Send`，不执行导入。
3. `CallFlow.extract` 沿正常路径跟踪 `init_chat_model` 及 `bind`、`bind_tools`、`with_config`、`with_retry`、`with_structured_output`，投影为 `graph_kind = llm_call_successors` 的 `next_call` 边。
4. `complete_graph` 计算从入口可达的节点、回路节点，以及是否存在未解析转移。
5. `reachability.analyze` 用 `next_call` 邻接计算最短路径、去掉节点后丢失的终点、逐次完成度。相邻节点同时有验收时，`score_pair` 计算完成个数之比 Q，以及上游通过时下游通过的比例。
6. `score` 把抽出的节点、边、入口、回路与参考 JSON 做集合对齐；参考里有 `call_sites` 时再对调用点坐标。

`llm_flow.py` 的 `analysis_scope` 写明范围是单模块正常路径、同模块已编译子图、同构并行批的一条 worker 路径。异常处理、供应商重试、任意辅助函数内部的模型调用不展开。

## 四、设计决策与动机（D编号，不得修改原内容）

对话记录中没有原始作者的设计讨论。下列条目是代码或 README 里已经写明的约束，动机只记录文本中能直接读到的那一句。

- **D1**：README 规定离线快速开始使用 Python 3.12 或以上，只读取 Open Deep Research 固定快照，使用 Python 标准库。动机原文：「以下命令只读取 Open Deep Research 的固定源码快照，使用 Python 标准库。」
- **D2**：`Graph` 写出的图带 `schema_version = 1.0`。调用投影的 `graph_kind` 为 `llm_call_successors`。`reachability.analyze` 在种类不匹配时抛出 `Reachability requires an AGENTEVE LLM-call graph`。
- **D3**：`llm_flow.CallFlow.extract` 将 `analysis_scope` 写为单模块静态正常路径，并写明异常处理、供应商重试和藏在任意辅助函数中的模型调用在范围外。
- **D4**：`stirrup_flow.StirrupFlow.extract` 写明分支守卫是结构见证，依赖历史的约束不求解；异常处理、客户端重试和工具内嵌 Agent 在范围外。
- **D5**：`reachability.run_analysis` 使用 `args.out.mkdir(parents=True, exist_ok=False)`，输出目录必须是新目录。
- **D6**：`THIRD_PARTY.md` 固定三份上游快照及许可证：Open Deep Research `1b7d2e80`、Stirrup `c092cee8`、OpenCode `c6262f9d`。各快照 `SOURCE.json` 记录文件校验值。

## 五、已记录的问题（P编号，主谓宾+数据格式，不得删改）

- **P1**：`stage3/blind_review.py` 与 `stage3/paired_review.py` 从 `stage2/odr_pilot.py` 导入 `DeepSeekModel`、`LocalModel`、`CACHE` 等符号，阶段 3 复核脚本依赖阶段 2 试点文件。
- **P2**：`compare_harness_flows.py` 第 13–16 行把 `GRAPH_ROOT` 设为 `results/stage1-llm-flow`，把 `TRACE_FILES` 设为 `results/stage2/deepseek-pilot-2048/runs.jsonl` 与 `results/stage2/deepseek-holdout-v1/runs.jsonl`。`.gitignore` 忽略 `/results/`，当前工作区没有该目录。
- **P3**：用户在 2026-11-27 10:32 判定当前项目写得很乱，并在 10:39 要求按函数功能与链路统计一级、二级、三级函数，作为重构前的现状清单。

## 六、当前参数状态快照

来源：`AGENTEVE/stage2/odr_pilot.py` 模块常量。

- `DATE` = `Wed Nov 18, 2026`
- `TOKEN_BUDGET` = 288
- `TEMPERATURE` = 0.5
- `CONTEXT` = 4096
- `THREADS` = 6
- `BATCH_SIZE` = 256
- `MICRO_BATCH_SIZE` = 128
- `MAX_SUPERVISOR_CALLS` = 3
- `MAX_DELEGATIONS` = 2
- `LOCAL_MAX_TOOL_CALLS` = 2
- `DEEPSEEK_URL` = `https://api.deepseek.com/chat/completions`
- `DEEPSEEK_MODEL` = `deepseek-flash`
- 简报节点 `BRIEF` = `write_research_brief:154:22`
- 主管节点 `SUPERVISOR` = `supervisor:214:22`

来源：`AGENTEVE/stage2/paired_pilot.py`。

- `MODEL_BUDGET` = 192
- `MODEL_CALL_LIMIT` = 2
- `SEED` = 101
- `READ_TOOL` = `read_evidence`
- `FINISH_TOOL` = `ResearchComplete`
- `ODR_CALL` = `researcher:415:22`
- `ODR_COMPRESSION` = `compress_research:551:30`
- `STIRRUP_CALL` = `Agent.step:1301:35`

来源：`AGENTEVE/stage3/paired_review.py`。

- `REVIEW_ROWS` = (59, 154)
- `GROUNDED_ROWS` = (59, 99, 118, 154)
- `MODEL_BUDGET` = 640
- `GROUNDED_BUDGET` = 1024
- `LOCAL_SEED` = 17

来源：`AGENTEVE/stage3/blind_review.py`。

- `MODEL_BUDGET` = 1536
- `LOCAL_CONTEXT` = 12288
- `LOCAL_SEED` = 17
- `SOURCE_RUNS` = (`P02-R1`, `P02-R3`)
- `ORDER_RUNS` = (`P01-R1`, `P01-R2`)

来源：`AGENTEVE/requirements-validation.txt`。

- `deepagents==0.7.38`
- `langchain==1.5.0`
- `langchain-core==1.8.0`
- `langgraph==1.2.22`
- `langgraph-prebuilt==1.1.2`
- `langsmith==0.14.17`
- `mini-swe-agent==2.4.6`

## 七、追加记录区

### [2026-11-27 10:39] 本次会话摘要
- **本次截止时间戳**：2026-11-27 10:39（UTC+8）
- **新增问题 P4**：自有代码（不含 `sources/`）共 244 个函数/方法；从各 `main` 出发的最短调用深度为 0 的 13 个、深度 1 的 52 个、深度 2–6 的 117 个；另有 43 个自有函数没有从 `main` 走到的直接调用边，19 个位于 `tests/`。内部直接调用边 391 条。最深链路是 `agenteve.main` → `extract_llm_flow` → `CallFlow.extract` → `CallFlow.build` → `CallFlow.expression` → `CallFlow.invoke` → `CallFlow.model_call`（深度 6，`llm_flow.py:293`）。
- **新增设计决策 D7**：重构用的函数层级按「从进程入口 `main` 的最短静态调用深度」划分。深度 0 为一级，深度 1 为二级，深度大于等于 2 为三级，并保留真实深度。无静态边的函数单独列为运行时入口，不并进三级，也不当成死代码。
- **参数状态变更**：无代码参数改动。调用图统计来自对 `AGENTEVE` 下自有 Python AST 与两个 `.cjs` 的直接调用解析；`sources/` 第三方快照未计入。

### [2026-11-27 12:20] 本次会话摘要
- **本次截止时间戳**：2026-11-27 12:20（UTC+8）
- **新增问题 P5**：重构前 13 个进程入口交叉调用；后评估若只读 `odr-pilot` 的两个布尔，会丢掉配对试验的 `llm_path`。
- **新增设计决策 D8**：交付入口改为 `buildGraph`、`evaluating`、`postEvaluation`、`runBenchmark`。三者不互相 import，用 `callGraph.json` 和 `trials.jsonl` 衔接。模型复核三件套删除。`validate_mini`、`validate_deepagents`、`summarize_dsh_tests` 并入 `confirmRuntimeBranches`，仅在 `--confirm-runtime` 时导入。
- **参数状态变更**：图产物文件名 `flow.json` → `callGraph.json`。试验文件名 `runs.jsonl` → `trials.jsonl`。JSON 键、`schema_version=1.0`、节点 ID、工具名未改。离线复跑仍是 8 节点、12 边，Q=2/2，条件通过率 1/2，13 个测试通过。

### [2026-11-27 13:20] 本次会话摘要
- **本次截止时间戳**：2026-11-27 13:20（UTC+8）
- **新增问题 P6**：`sources/` 在项目里被当成自有源码目录，而这三份上游树是被评测的实例。
- **新增设计决策 D9**：目录 `sources/` 改名为 `instances/`。其下仍是 `odr`、`stirrup`、`opencode` 三个实例。
- **参数状态变更**：路径 `sources/odr/stage2-snapshot` → `instances/odr/stage2-snapshot`。`sources/stirrup`、`sources/stirrup-runtime`、`sources/opencode` 同样改为 `instances/` 下。
