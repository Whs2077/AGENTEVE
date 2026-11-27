"""Run graph building, trial recording, and post-evaluation in that order."""

from pathlib import Path

from src.buildGraph import buildGraph
from src.core.planningTrial import TOKEN_BUDGET
from src.evaluating import runEvaluation
from src.postEvaluation import postEvaluation


def runBenchmark(source, adapter, out, *, tasks=None, trials=None, backend="local", reference=None,
                 builder=None, sourceUrl=None, clientProtocol=None, clientProtocolUrl=None,
                 coordinates="file", smoke=False, maxOutputTokens=TOKEN_BUDGET):
    """Stop at the first step that returns non-zero. ``trials`` skips recording and reads that file."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose a fresh output directory")
    code = buildGraph(
        source, adapter, out / "graph", builder=builder, reference=reference, sourceUrl=sourceUrl,
        clientProtocol=clientProtocol, clientProtocolUrl=clientProtocolUrl, coordinates=coordinates,
    )
    if code:
        return code
    graph = out / "graph" / "callGraph.json"
    if trials:
        trialPath = Path(trials)
    else:
        if tasks is None:
            raise ValueError("Set tasks or trials in the config file")
        code = runEvaluation(graph, tasks, out / "trials", backend=backend, smoke=smoke,
                             maxOutputTokens=maxOutputTokens)
        if code:
            return code
        trialPath = out / "trials" / "trials.jsonl"
    return postEvaluation(graph, trialPath, out / "post")
