#!/usr/bin/env python
# coding=utf-8
"""Run or validate the AdaCycle-Fork paper benchmark suite.

This runner does not reimplement third-party benchmark evaluators. Instead it
provides a single registry/launcher for MME, POPE, SEED-Bench, VQAv2, GQA,
GenEval, T2I-CompBench, MJHQ, MagicBrush, EditBench, and I2EBench commands.
Each benchmark entry can point to an external command and an optional metrics
JSON produced by that command.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import yaml


def load_suite(path):
    with open(path, "r", encoding="utf-8") as reader:
        return yaml.safe_load(reader)


def load_metrics(path):
    if not path:
        return None
    metrics_path = Path(os.path.expandvars(path))
    if not metrics_path.exists():
        return None
    with open(metrics_path, "r", encoding="utf-8") as reader:
        return json.load(reader)


def run_benchmark(entry, output_dir, run_commands=False, strict=False):
    name = entry["name"]
    metrics_json = entry.get("metrics_json")
    if metrics_json:
        metrics_json = metrics_json.replace("$ADACYCLE_OUTPUT_DIR", str(output_dir))
    result = {
        "name": name,
        "task": entry.get("task"),
        "expected_metrics": entry.get("metrics", []),
        "status": "planned",
        "metrics_json": metrics_json,
        "metrics": load_metrics(metrics_json),
    }

    command = entry.get("command")
    if not command:
        result["status"] = "missing_command"
        if strict:
            raise RuntimeError(f"{name} has no command configured.")
        return result

    if not run_commands:
        result["status"] = "ready"
        return result

    env = os.environ.copy()
    env["ADACYCLE_BENCHMARK"] = name
    env["ADACYCLE_OUTPUT_DIR"] = str(output_dir)
    if metrics_json:
        env["ADACYCLE_METRICS_JSON"] = metrics_json
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command,
        shell=True,
        cwd=entry.get("cwd") or None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    result["returncode"] = completed.returncode
    result["stdout_tail"] = completed.stdout[-4000:]
    result["stderr_tail"] = completed.stderr[-4000:]
    if completed.returncode != 0:
        result["status"] = "failed"
        if strict:
            raise RuntimeError(f"{name} failed with exit code {completed.returncode}.")
    else:
        result["status"] = "completed"
        result["metrics"] = load_metrics(metrics_json)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="configs/adacycle_eval_suite.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--run", action="store_true", help="Actually run configured external commands.")
    parser.add_argument("--strict", action="store_true", help="Fail if a command is missing or exits non-zero.")
    args = parser.parse_args()

    suite = load_suite(args.suite)
    output_dir = Path(suite.get("output_dir", "evaluation/results/adacycle_full_suite"))
    output_path = Path(args.output) if args.output else output_dir / "summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = [
        run_benchmark(entry, output_dir, run_commands=args.run, strict=args.strict)
        for entry in suite.get("benchmarks", [])
        if entry.get("enabled", True)
    ]
    summary = {
        "suite_name": suite.get("suite_name", "adacycle_suite"),
        "num_benchmarks": len(results),
        "num_completed": sum(item["status"] == "completed" for item in results),
        "num_ready": sum(item["status"] == "ready" for item in results),
        "num_missing_command": sum(item["status"] == "missing_command" for item in results),
        "num_failed": sum(item["status"] == "failed" for item in results),
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as writer:
        json.dump(summary, writer, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
