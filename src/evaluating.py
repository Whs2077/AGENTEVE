"""Run a frozen task list on a call graph and write one trial record per attempt."""

from pathlib import Path
from types import SimpleNamespace

from common.fileStore import readJson
from src.core.planningTrial import TOKEN_BUDGET, recordPlanningTrials


def runEvaluation(graph, tasks, outDir, *, backend="local", smoke=False, maxOutputTokens=TOKEN_BUDGET):
    """Record planning trials. The task file must contain tasks and seeds."""
    dataset = readJson(tasks)
    if "tasks" not in dataset or "seeds" not in dataset:
        raise ValueError("The task file must contain tasks and seeds.")
    args = SimpleNamespace(
        graph=Path(graph), tasks=Path(tasks), out=Path(outDir), backend=backend,
        smoke=smoke, max_output_tokens=maxOutputTokens,
    )
    return recordPlanningTrials(args)
