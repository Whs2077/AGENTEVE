# AGENTEVE

评测固定实例上的模型与 Harness 组合。它不给模型打一个孤立分数，而是先从实例源码抽出大模型调用图，再在题单或已有轨迹上记录每次试验，最后按记录分开看结果、步骤数据和轨迹。

走到某个节点只说明程序执行到了这里。节点完成要看这一步的验收是否通过。相邻两步要同时看两个数：完成个数之比 Q，以及上游通过时下游也通过的比例。这两个数不是同一个量。Q 接近或超过 100% 只提示回查配对编号，不表示节点多余。

## 文件结构

```text
AGENTEVE/
├── main.py                          唯一命令行入口。按子命令调用 src 里的步骤
├── config.json                      路径、适配器、预算、模型和节点编号
├── requirements.txt                 在线跑题用的 LangChain 与 LangGraph 版本
├── src/
│   ├── buildGraph.py                按适配器抽出调用图，写出各视图，并与参考图比较
│   ├── evaluating.py                读调用图和题单，写出 trials.jsonl
│   ├── postEvaluation.py            读调用图和轨迹，写出结果、步骤数据、轨迹和报告
│   ├── runBenchmark.py              按建图、记录、后评估顺序执行
│   └── core/
│       ├── graphModel.py            图对象、源码位置、入口可达和回路
│       ├── graphOutput.py           写出 callGraph 的 JSON、真值依据、Mermaid、DOT 和 HTML
│       ├── flowView.py              把流图渲染成可打开的 HTML
│       ├── nodePasses.py            逐次访问和逐次验收的原始记录
│       ├── semanticRecord.py        语义头、格式是否合格，以及一次运行就能确定的指标
│       ├── pyScan.py                Python 语法树查询：点号名、所在函数、导入
│       ├── tsScan.py                按词法读取 TypeScript 函数和调用
│       ├── apiSite.py               API 调用点记录、分桶与协议名、LangChain 调用名
│       ├── locateLlmApi.py          第一、二桶：框架写法与固定协议定位，不产生图节点
│       ├── locateByModel.py         第三桶：模型只给文件、行号和符号，并核对
│       ├── callHierarchy.py         从调用点取 incoming 与 outgoing
│       ├── declaredTopology.py      只读 StateGraph 建造语句
│       ├── projectNextCall.py       把正常路径投影成下一次模型调用
│       ├── projectStirrup.py        沿 Agent.run 投影到 LLMClient.generate
│       ├── projectFromApi.py        定位之后选择路径分析
│       └── planningTrial.py         简报到主管的试点、词面验收和模型客户端
├── common/
│   ├── fileStore.py                 读 JSON、写 JSON、计算文件哈希
│   ├── callPaths.py                 next_call 上的最短路径和删边影响
│   └── runLog.py                    INFO、DEBUG、WARNING、ERROR
├── improve/                         默认关闭。关闭时不调用生成模型
│   ├── settings.py                  校验 repeat、retain、sample_count 和生成模型配置
│   ├── estimates.py                 pass@1、pass@k、pass^k 和首次通过所需次数
│   ├── generator.py                 同义替换与反义替换；没有密钥时不发请求
│   ├── interventions.py             四项测试、位置控制、来源分类和修改建议
│   └── runImprove.py                按 repeat 选择统计方式
├── instances/
│   ├── odr/
│   │   ├── LICENSE
│   │   ├── open_deep_research/      Open Deep Research 1b7d2e80，被测 Harness
│   │   │   ├── deep_researcher.py   建图默认读的源文件
│   │   │   ├── configuration.py
│   │   │   ├── state.py
│   │   │   ├── prompts.py
│   │   │   └── utils.py
│   │   └── check/                   该实例的参考图和回归轨迹
│   │       ├── reference.json
│   │       └── demo-runs.jsonl
│   ├── stirrup/
│   │   ├── LICENSE
│   │   ├── SOURCE.json              固定提交和文件校验值
│   │   ├── stirrup/                 被测 Harness
│   │   │   ├── core/agent.py
│   │   │   ├── core/models.py
│   │   │   └── clients/chat_completions_client.py
│   │   └── check/
│   │       └── reference.json       该实例的参考图
│   └── opencode/
│       ├── LICENSE
│       ├── SOURCE.json
│       ├── session/                 被测 Harness
│       │   ├── prompt.ts
│       │   ├── processor.ts
│       │   ├── compaction.ts
│       │   ├── llm.ts
│       │   ├── retry.ts
│       │   └── summary.ts
│       └── check/
│           ├── reference.json
│           └── buildOpenCodeGraph.cjs   定位到 streamText 之后做 OpenCode 路径分析
├── tasks/
│   └── planningPilot.json           开发划分题单，6 题，每题 3 个种子
├── unitTests/
│   ├── staticRoutesTest.py          声明拓扑
│   ├── llmFlowTest.py               下一次模型调用
│   ├── stirrupFlowTest.py           Stirrup 调用投影
│   ├── reachabilityTest.py          Q 与条件通过率
│   ├── recordMetricsTest.py         旧轨迹、语义头、格式不合格和完成个数之比大于 1
│   ├── improveTest.py               repeat、retain、假生成器和修改建议
│   ├── locateApiTest.py             API 定位、调用层级和参考图
│   └── flowOutputTest.py            流图 HTML、原始通过记录和删边可达性
└── results/                         运行产物。不纳入版本库
```

`src` 里的三步互不 import，只由 `runBenchmark.py` 串起来。依赖单向向下：`main.py` → `src/*.py` → `src/core` → `common`，`src/core` 内部也没有循环或回调。`pyScan`、`tsScan`、`apiSite`、`graphModel` 是最底层，只被上层引用。建图写出 `callGraph.json`、`groundTruth.json`、`callGraph.mmd`、`callGraph.dot` 和 `callGraph.html`，记录写出 `trials.jsonl` 和 `nodePasses.jsonl`，后评估读调用图和轨迹。

## 项目内容

**建图。** `adapter` 为 `locateApi` 时，先定位大模型 API，再做路径分析。定位分三桶：LangChain、LangGraph、DeepAgent 的固定写法；固定协议上的静态符号和调用层级；前两桶没有调用点或互相冲突时，模型只返回文件、行号和符号。图节点在路径分析之后出现，表示从入口走到下一次模型调用的步骤。`projectNextCall`、`projectStirrup` 和 `readDeclaredTopology` 仍可单独调用。建图写出 `callGraph.json` 和 `callGraph.html`。实例的 `check/reference.json` 是参考流图；提供它时再写对照结果。没有参考流图时，`groundTruth.json` 标明这张图只来自本项目分析，没有真值依据。节点、边、入口或回路对不上，或者图中仍有未解析转移时，命令返回非零。

图的范围是单模块正常路径。异常处理、供应商重试，以及藏在任意辅助函数里的模型调用不在图内。

**记录。** `evaluating` 读取调用图和 `tasks`。当前题单是 `tasks/planningPilot.json`，`split` 为 `development-pilot`，只描述这个开发划分。它跑简报到主管，在第一批分派处停下，不执行检索，也不写终稿。每一行试验写入 `trials.jsonl`，其中 `node_completion` 为 `true`、`false` 或 `null`。`null` 表示该步没有验收，不计入失败。

配置里的 `trials` 指向已有轨迹时，`all` 不跑模型，直接把这份轨迹交给后评估。记录步骤另写 `nodePasses.jsonl`：路径上的每一次到达是一行，每一个节点验收也是一行。新跑出的试验会为每个节点写下输入语义头、输出语义头、无输入依据的语义头和格式是否合格。格式不合格沿用现有的 `OutputError`，令牌上限仍是 288。语义头只做别名字符串匹配。

**后评估。** `postEvaluation` 写出这些文件：

- `results.json`：按任务汇总记录里已经写下的结果。演示轨迹没有单独的结果字段，因此这里不另造通过率。
- `stepData.json`：每个节点在已验收次数上的完成率，以及相邻节点的 Q 和条件通过率。
- `trajectories.json`：每次试验的路径、第一个完成度为假的节点、对不上 `next_call` 的转移。
- `report.md`：上述内容的中文摘要。
- `nodePasses.jsonl`：从轨迹展开的原始到达和验收记录。后评估用可达对和删边失去的可达对读这些记录。

报告另有一节一次运行就能确定的指标：到达、完成、格式是否合格、语义保持率、无输入依据的语义所占比例、该次路径上的 Q 和条件通过率、第一个未完成节点、与 `next_call` 不一致的转移。旧轨迹没有语义头时，语义保持率和无输入依据比例写为无定义。格式不合格单独计数，不进入完成率，也不进入必要性。令牌数、调用次数和耗时有记录时单独列出。开发题和留出题没有同时出现时，两类通过率之差为无定义。

运行结束后，日志打印每个节点的完成率，以及它能否从入口到达。满足下列任一条件时再打印逻辑异常：从入口不可达、完成率高于直接上游、直接上游未通过时本节点仍通过、完成个数之比大于 1。这些计算不调用模型。

`improve.enabled` 为 false 时不写 `improve.md`，也不调用生成模型。启用并且 `repeat` 大于 1 时，后评估按轨迹里同一题的多次记录计算 pass@1、pass@k、pass^k 和首次通过所需次数，并给每个节点另存一份只用第一次运行的统计。`repeat` 为负数时做一次语义密集性采样、充分性、必要性、反例与同义代换；负号不是重复次数。反例和同义代换在环境变量里没有密钥时记为未执行。充分性和必要性没有节点执行器时，s-pass@1、s-pass@k、e-pass@1、e-pass@k 为无定义。修改建议只写进报告。

**实例与检查材料。** 每个被测实例的目录里，Harness 源码和它的检查材料放在一起。`check/reference.json` 是该实例的参考图。Open Deep Research 的 `check/demo-runs.jsonl` 是四条合成轨迹，用来核对 Q 和条件通过率。OpenCode 的 `check/buildOpenCodeGraph.cjs` 在 `locateApi` 找到 `streamText` 之后负责该实例的路径分析。

`common/` 是三个实例共用的读写、最短路径和日志，不包含某一份实例的金标或轨迹。

## 调用方法

在 `AGENTEVE/` 目录使用 Python 3.12 或以上。先改 `config.json`，再选择命令：

```text
python -B main.py all --config config.json
python -B main.py buildGraph --config config.json
python -B main.py evaluating --config config.json
python -B main.py postEvaluation --config config.json
python -B -m unittest discover -s unitTests -p "*Test.py"
```

`all` 依次调用建图、记录、后评估。`trials` 有值时跳过记录。`evaluating` 读取配置中的 `graph` 和 `tasks`。`postEvaluation` 读取 `graph` 和 `trials`。`out` 必须是尚不存在或为空的目录。

`config.json` 每次评测都要指定两件事。`instance` 是被测 Harness：相对名从 `instances/` 下找，例如 `odr` 即 `instances/odr`；绝对路径则直接使用该目录。`source`、`reference`、`trials`、`clientProtocol`、`graph` 若是相对路径，则相对这个实例目录，例如 `check/reference.json`。这四项若写成绝对路径，则按绝对路径读取。`out`、`tasks`、`modelCache` 仍相对 `AGENTEVE/`。

新的 Harness 放进 `instances/<名字>/`，检查材料放在该目录的 `check/`。下一次评测改 `instance` 以及它下面的 `source`、`reference`、`trials`，不必改框架代码。

调用模型时，`backend` 为 `api` 表示使用用户指定的模型。每次评测填写 `modelName`、`apiUrl` 和 `apiKeyEnv`。密钥放在 `apiKeyEnv` 所指的环境变量里，不写进配置文件。`backend` 为 `local` 才使用 `modelCache` 里的本地权重，这是单独的本地运行方式。

建图相关的键还有 `adapter`、`builder`、`sourceUrl`、`coordinates`、`clientProtocolUrl`。跑题相关的键还有 `researchModule`、`smoke`、`tokenBudget`、`temperature`、`contextTokens`、`threads`、`batchSize`、`microBatchSize`、`maxSupervisorCalls`、`maxDelegations`、`localMaxToolCalls`、`dateInPrompt`、`briefNode`、`supervisorNode`。`graph` 只在分步的 `evaluating` 和 `postEvaluation` 中使用。

`improve` 默认 `enabled` 为 false。启用时要有整数 `repeat`（不能为 0）、`retain`（0.5 到 1，含两端）和正整数 `sample_count`，以及 `generator` 里的 `modelName`、`apiUrl`、`apiKeyEnv`。密钥放在 `apiKeyEnv` 所指的环境变量里。`retain` 为 1 表示语义密集性采样全部保留。`repeat` 为 1 时只统计一次运行可确定的指标。`repeat` 大于 1 时不另行请求被测模型，只用已经写下的多次轨迹。`repeat` 小于 0 时，生成模型只负责反例和同义代换；没有密钥就不会发请求。

在线跑题把 `backend` 保持为 `api`，填好 `modelName`、`apiUrl`、`apiKeyEnv`，`smoke` 设为 `true`，`trials` 留空，并先安装 `requirements.txt`。该命令会请求配置里的接口。下面的离线数字不是这次在线运行的结果。

当前配置的离线命令已经跑过。`out` 为 `results/instance-path-check` 时，终端给出 8 个节点、12 条边；8 个节点含入口和终点，其中 6 个是模型调用点；节点、边、入口和回路都与参考图一致。后评估报告中，简报到主管的 Q 为 2/2，上游通过时下游也通过的比例为 1/2。这个目录已经有文件，再次运行前要换一个 `out`。

单元测试检查 API 定位、流图输出、原始通过记录、建图投影、后评估计算和 improve 的三种 repeat，不调用模型，也不读取 `results/`。上述发现命令跑过 36 条。本机没有 `node` 时，OpenCode 的路径分析那一条会跳过，其余通过。

## 功能

- 从固定实例抽出可回查源码位置的调用图，并与参考图比较。
- 在开发题单上跑简报到主管的试验，或直接读入已有 `trials.jsonl`。
- 分开报告任务结果、逐步完成率、Q、条件通过率、首个失败节点和未匹配转移。
- 运行结束后在日志里给出节点完成率、入口可达性和逻辑异常。报告在有语义头时给出语义保持率和无输入依据的语义所占比例，没有记录时写无定义。
- `improve` 默认关闭。打开后按 `repeat` 统计多次运行，或用可替换的生成函数做四项测试。建议不写回被测源码。
- 用 `unitTests/` 锁定 API 定位、声明拓扑、调用投影、Q 与条件通过率，以及 improve 的参数和假生成器。
