"""Save a portable, compact summary of a DeepSeek Harness Vitest run."""

import argparse
from pathlib import Path

from agenteve import read_json, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("vitest_json", type=Path)
    cli.add_argument("--revision", required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    report = read_json(args.vitest_json)
    suites = []
    for suite in report["testResults"]:
        filename = suite["name"].replace("\\", "/")
        name = "packages/" + filename.split("/packages/", 1)[-1]
        counts = {}
        for test in suite["assertionResults"]:
            status = test["status"]
            counts[status] = counts.get(status, 0) + 1
        suites.append({"file": name, "status": suite["status"], "counts": counts,
                       "passed_case_names": [test["title"] for test in suite["assertionResults"]
                                             if test["status"] == "passed"]})
    write_json(args.out, {"revision": args.revision, "runner": "upstream Vitest with local mock model/events",
                         "success": report["success"], "total": report["numTotalTests"],
                         "passed": report["numPassedTests"], "failed": report["numFailedTests"],
                         "files": suites})
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
