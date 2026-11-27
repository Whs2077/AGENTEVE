"""Command line for the three benchmark steps. Paths and parameters are in config.json."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PATH_KEYS = {"out", "tasks", "modelCache"}
INSTANCE_KEYS = {"source", "reference", "clientProtocol", "trials", "graph"}
PLANNING_ATTRS = {
    "dateInPrompt": "DATE",
    "briefNode": "BRIEF",
    "supervisorNode": "SUPERVISOR",
    "tokenBudget": "TOKEN_BUDGET",
    "temperature": "TEMPERATURE",
    "contextTokens": "CONTEXT",
    "threads": "THREADS",
    "batchSize": "BATCH_SIZE",
    "microBatchSize": "MICRO_BATCH_SIZE",
    "maxSupervisorCalls": "MAX_SUPERVISOR_CALLS",
    "maxDelegations": "MAX_DELEGATIONS",
    "localMaxToolCalls": "LOCAL_MAX_TOOL_CALLS",
    "apiUrl": "DEEPSEEK_URL",
    "modelName": "DEEPSEEK_MODEL",
    "apiKeyEnv": "API_KEY_ENV",
}
INSTANCES = ROOT / "instances"


def asPath(value):
    if value in (None, ""):
        return None
    return Path(value)


def resolveInstance(value):
    path = asPath(value)
    if path is None:
        return None
    if path.is_absolute():
        return path
    return INSTANCES / path


def resolveProject(value):
    path = asPath(value)
    if path is None:
        return None
    if path.is_absolute():
        return path
    return ROOT / path


def resolveInside(instance, value):
    path = asPath(value)
    if path is None:
        return None
    if path.is_absolute():
        return path
    if instance is None:
        raise ValueError(f"A relative instance path requires instance: {value}")
    return instance / path


def loadConfig(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    instance = resolveInstance(raw.get("instance"))
    config = {"instance": instance}
    for key, value in raw.items():
        if key == "instance":
            continue
        if key in INSTANCE_KEYS:
            config[key] = resolveInside(instance, value)
        elif key in PATH_KEYS:
            config[key] = resolveProject(value)
        else:
            config[key] = value
    return config


def require(config, key):
    if config.get(key) in (None, ""):
        raise ValueError(f"{key} is missing from the config file")
    return config[key]


def applyPlanning(config):
    import src.core.planningTrial as trial
    for key, attr in PLANNING_ATTRS.items():
        if config.get(key) not in (None, ""):
            setattr(trial, attr, config[key])
    if config.get("instance"):
        trial.SNAPSHOT = config["instance"]
        trial.MODULE = trial.SNAPSHOT / config.get("researchModule", "open_deep_research")
    if config.get("modelCache"):
        trial.CACHE = config["modelCache"]


def commandAll(config):
    from src.runBenchmark import runBenchmark
    return runBenchmark(
        require(config, "source"), require(config, "adapter"), require(config, "out"),
            tasks=config.get("tasks"), trials=config.get("trials"), backend=config.get("backend", "api"),
        reference=config.get("reference"), builder=config.get("builder"), sourceUrl=config.get("sourceUrl"),
        clientProtocol=config.get("clientProtocol"), clientProtocolUrl=config.get("clientProtocolUrl"),
        coordinates=config.get("coordinates", "file"), smoke=bool(config.get("smoke", False)),
        maxOutputTokens=int(require(config, "tokenBudget")),
    )


def commandBuildGraph(config):
    from src.buildGraph import buildGraph
    return buildGraph(
        require(config, "source"), require(config, "adapter"), require(config, "out"),
        builder=config.get("builder"), reference=config.get("reference"), sourceUrl=config.get("sourceUrl"),
        clientProtocol=config.get("clientProtocol"), clientProtocolUrl=config.get("clientProtocolUrl"),
        coordinates=config.get("coordinates", "file"),
    )


def commandEvaluating(config):
    from src.evaluating import runEvaluation
    return runEvaluation(
        require(config, "graph"), require(config, "tasks"), require(config, "out"),
        backend=config.get("backend", "api"), smoke=bool(config.get("smoke", False)),
        maxOutputTokens=int(require(config, "tokenBudget")),
    )


def commandPostEvaluation(config):
    from src.postEvaluation import postEvaluation
    return postEvaluation(require(config, "graph"), require(config, "trials"), require(config, "out"))


COMMANDS = {
    "all": commandAll,
    "buildGraph": commandBuildGraph,
    "evaluating": commandEvaluating,
    "postEvaluation": commandPostEvaluation,
}


def main(argv):
    from common.runLog import useLevel
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    args = parser.parse_args(argv)
    config = loadConfig(args.config)
    useLevel([config.get("logLevel") or "INFO"])
    applyPlanning(config)
    return COMMANDS[args.command](config)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"main: {error}", file=sys.stderr)
        sys.exit(2)
