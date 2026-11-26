# AGENTEVE

AGENTEVE 是一个面向 Agent Harness 的实验性评测工具。它从源码提取大模型调用图，保留跳转对应的源码位置，再利用运行轨迹分析各节点的完成度和上下游衔接。项目内还提供了根据原始事件核验模型解释的实验代码。

## 离线快速开始

在仓库根目录使用 Python 3.12 或以上版本。以下命令只读取 Open Deep Research 的固定源码快照，使用 Python 标准库。

```text
python -B agenteve.py extract llm-flow sources/odr/stage2-snapshot/src/open_deep_research/deep_researcher.py --builder deep_researcher_builder --source-url https://github.com/langchain-ai/open_deep_research/blob/1b7d2e80db9faa586165c60e09096dbbfd483a64/src/open_deep_research/deep_researcher.py --out results/stage1-llm-flow/open-deep-research/extracted
python -B agenteve.py score results/stage1-llm-flow/open-deep-research/extracted/flow.json examples/open-deep-research/reference.json
python -B agenteve.py reachability results/stage1-llm-flow/open-deep-research/extracted/flow.json --runs examples/open-deep-research/demo-runs.jsonl --out results/demo-reachability
python -B -m unittest discover -s tests
```

对固定范围内的人工参考图，自动提取结果匹配 6 个模型调用点和 12 条连线。连线包含起点和终点。提取器输出 `flow.json` 和 Mermaid 图。

可达性报告在 `results/demo-reachability/report.md`。随仓库提供的四条轨迹是合成示例，两节点完成数之比为 2/2，按运行配对后的下游条件通过率为 1/2。每次复跑为 `--out` 使用新的目录。仅传图时可计算结构可达性、最短路径和回路。

## 接入自己的 Agent

- 用 `agenteve.py extract --help` 查看当前的源码适配方式。输出节点和边都可回查源码。动态目标可能产生未解析诊断。
- 以 JSONL 保存逐次运行。每行需要 `run_id`、`task_id`、按执行顺序排列的节点 ID 列表 `path`，以及以节点 ID 为键的 `node_completion`。用 `true` 和 `false` 表示验收结果，未评估的节点可省略或用 `null`。
- 在实验前固定任务、模型、预算和分节点验收条件。重复经过同一节点的事件可保留在 `path` 中，完成度按每条运行记录统计。

## 实验代码

`stage2/odr_pilot.py` 保留简报到首批分派的试验入口和开发题单。完成离线快速开始的源码提取后，安装 `requirements-validation.txt` 的 Python 依赖，可以用自己的 `DEEPSEEK_API_KEY` 运行在线单题。使用真实模型时会产生 API 调用。

```text
python -m pip install -r requirements-validation.txt
python -B stage2/odr_pilot.py --backend deepseek --smoke --max-output-tokens 2048 --out results/pilot-smoke
python -B agenteve.py reachability results/stage1-llm-flow/open-deep-research/extracted/flow.json --runs results/pilot-smoke/runs.jsonl --trace-format odr-pilot --out results/pilot-smoke/reachability
```

`stage2/paired_*.py` 和 `stage3/` 保存其他实验与模型复核实现。复跑这些研究批次时，需要相应的任务清单、固定版本的上游运行环境、逐次轨迹或模型文件。推荐先用离线示例验证工具链，再接入自己的实验配置。

## 上游来源

仓库中的上游源码快照保留原许可证。详细版本见 [第三方来源](THIRD_PARTY.md)。
