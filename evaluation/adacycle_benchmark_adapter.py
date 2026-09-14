#!/usr/bin/env python
# coding=utf-8
"""External benchmark adapter for AdaCycle-Fork.

The paper uses several third-party benchmarks whose official evaluators and
datasets are not vendored in this repository. This adapter gives every suite
entry an executable interface without fabricating benchmark scores. Replace the
suite command with the official evaluator once the corresponding data/tooling is
installed.
"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--references", default=None)
    parser.add_argument("--delegate-command", default=None)
    args = parser.parse_args()

    metrics_path = Path(os.path.expandvars(args.metrics_json))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "benchmark": args.benchmark,
        "status": "requires_external_evaluator",
        "predictions": args.predictions,
        "references": args.references,
        "delegate_command": args.delegate_command,
        "message": (
            "This repository exposes the benchmark interface, but official "
            "scores require the benchmark's external evaluator and data. "
            "Set this suite entry's command to that evaluator to produce real metrics."
        ),
        "metrics": {},
    }
    with open(metrics_path, "w", encoding="utf-8") as writer:
        json.dump(result, writer, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
