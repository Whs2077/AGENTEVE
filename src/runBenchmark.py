"""Run graph building, trial recording, and post-evaluation in that order."""

import argparse
from pathlib import Path
import sys

from src.buildGraph import buildGraph
from common.runLog import useLevel
from src.evaluating import runEvaluation
from src.core.planningTrial import CONTEXT, TOKEN_BUDGET
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
            raise ValueError("Pass --tasks or --trials")
        code = runEvaluation(graph, tasks, out / "trials", backend=backend, smoke=smoke,
                             maxOutputTokens=maxOutputTokens)
        if code:
            return code
        trialPath = out / "trials" / "trials.jsonl"
    return postEvaluation(graph, trialPath, out / "post")


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--builder")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--source-url")
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--trials", type=Path, help="Skip recording and read this trial file")
    parser.add_argument("--backend", choices=["local", "deepseek"], default="local")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-output-tokens", type=int, default=TOKEN_BUDGET)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.max_output_tokens <= CONTEXT:
        parser.error(f"--max-output-tokens must be from 1 to {CONTEXT}")
    try:
        return runBenchmark(
            args.source, args.adapter, args.out, tasks=args.tasks, trials=args.trials, backend=args.backend,
            reference=args.reference, builder=args.builder, sourceUrl=args.source_url, smoke=args.smoke,
            maxOutputTokens=args.max_output_tokens,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"runBenchmark: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(useLevel(sys.argv[1:])))
