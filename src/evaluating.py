"""Run a frozen task list on a call graph and write one trial record per attempt."""

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

from common.fileStore import readJson
from common.runLog import useLevel
from src.core.planningTrial import CONTEXT, TOKEN_BUDGET


def runEvaluation(graph, tasks, outDir, *, backend="local", smoke=False, maxOutputTokens=TOKEN_BUDGET):
    """Record planning trials. The task file must contain tasks and seeds."""
    dataset = readJson(tasks)
    if "tasks" not in dataset or "seeds" not in dataset:
        raise ValueError("The task file must contain tasks and seeds.")
    from src.core.planningTrial import recordPlanningTrials
    args = SimpleNamespace(
        graph=Path(graph), tasks=Path(tasks), out=Path(outDir), backend=backend,
        smoke=smoke, max_output_tokens=maxOutputTokens,
    )
    return recordPlanningTrials(args)


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--backend", choices=["local", "deepseek"], default="local")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-output-tokens", type=int, default=TOKEN_BUDGET)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.max_output_tokens <= CONTEXT:
        parser.error(f"--max-output-tokens must be from 1 to {CONTEXT}")
    try:
        return runEvaluation(args.graph, args.tasks, args.out, backend=args.backend, smoke=args.smoke,
                             maxOutputTokens=args.max_output_tokens)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"evaluating: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(useLevel(sys.argv[1:])))
