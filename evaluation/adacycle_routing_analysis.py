#!/usr/bin/env python
# coding=utf-8
"""Analyze AdaCycle-Fork routing, active-parameter, latency, and throughput traces.

Expected JSONL rows may contain:
  router_probs: nested list shaped [layers, batch, tokens_or_1, 3] or similar
  alignment_scores: nested list shaped [layers, batch]
  latency_ms: float
  throughput: float
  task: optional task name
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


BRANCHES = ("shared", "understanding", "generation")


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as reader:
        for line in reader:
            if line.strip():
                yield json.loads(line)


def as_layer_arrays(router_probs):
    arr = np.asarray(router_probs, dtype=np.float64)
    if arr.size == 0 or arr.shape[-1] != 3:
        return []
    if arr.ndim == 2:
        arr = arr[None, :, None, :]
    elif arr.ndim == 3:
        arr = arr[:, :, None, :]
    elif arr.ndim > 4:
        arr = arr.reshape(arr.shape[0], -1, 3)
        arr = arr[:, :, None, :]
    return [arr[layer].reshape(-1, 3) for layer in range(arr.shape[0])]


def summarize_router(rows):
    per_task = defaultdict(list)
    all_layers = []
    for row in rows:
        layers = as_layer_arrays(row.get("router_probs", []))
        if not layers:
            continue
        if not all_layers:
            all_layers = [[] for _ in layers]
        for idx, layer_values in enumerate(layers):
            all_layers[idx].append(layer_values)
        per_task[row.get("task", "unknown")].append(layers)

    layer_summaries = []
    for idx, chunks in enumerate(all_layers):
        values = np.concatenate(chunks, axis=0)
        hard = values.argmax(axis=-1)
        layer_summaries.append({
            "layer": idx,
            "soft_shared": float(values[:, 0].mean()),
            "soft_understanding": float(values[:, 1].mean()),
            "soft_generation": float(values[:, 2].mean()),
            "hard_shared": float((hard == 0).mean()),
            "hard_understanding": float((hard == 1).mean()),
            "hard_generation": float((hard == 2).mean()),
        })

    task_summaries = {}
    for task, traces in per_task.items():
        values = np.concatenate(
            [np.concatenate([layer.reshape(-1, 3) for layer in layers], axis=0) for layers in traces],
            axis=0,
        )
        hard = values.argmax(axis=-1)
        task_summaries[task] = {
            "soft": {BRANCHES[i]: float(values[:, i].mean()) for i in range(3)},
            "hard": {BRANCHES[i]: float((hard == i).mean()) for i in range(3)},
        }
    return layer_summaries, task_summaries


def summarize_alignment(rows):
    traces = []
    for row in rows:
        if row.get("alignment_scores") is not None:
            arr = np.asarray(row["alignment_scores"], dtype=np.float64)
            if arr.size:
                traces.append(arr.reshape(arr.shape[0], -1).mean(axis=1))
    if not traces:
        return []
    max_layers = max(len(trace) for trace in traces)
    padded = np.full((len(traces), max_layers), np.nan)
    for idx, trace in enumerate(traces):
        padded[idx, :len(trace)] = trace
    return [
        {"layer": int(layer), "alignment": float(np.nanmean(padded[:, layer]))}
        for layer in range(max_layers)
    ]


def summarize_runtime(rows):
    latency = [float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None]
    throughput = [float(row["throughput"]) for row in rows if row.get("throughput") is not None]
    return {
        "latency_ms_mean": float(np.mean(latency)) if latency else None,
        "latency_ms_p50": float(np.percentile(latency, 50)) if latency else None,
        "latency_ms_p95": float(np.percentile(latency, 95)) if latency else None,
        "throughput_mean": float(np.mean(throughput)) if throughput else None,
    }


def estimate_active_params(layer_summaries, shared_params, understanding_params, generation_params):
    if not layer_summaries:
        return {}
    soft_u = np.mean([item["soft_understanding"] for item in layer_summaries])
    soft_g = np.mean([item["soft_generation"] for item in layer_summaries])
    hard_u = np.mean([item["hard_understanding"] for item in layer_summaries])
    hard_g = np.mean([item["hard_generation"] for item in layer_summaries])
    return {
        "soft_expected_active_params": float(shared_params + soft_u * understanding_params + soft_g * generation_params),
        "hard_expected_active_params": float(shared_params + hard_u * understanding_params + hard_g * generation_params),
        "shared_params": float(shared_params),
        "understanding_branch_params": float(understanding_params),
        "generation_branch_params": float(generation_params),
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as writer:
        csv_writer = csv.DictWriter(writer, fieldnames=list(rows[0].keys()))
        csv_writer.writeheader()
        csv_writer.writerows(rows)


def write_plot(path, layer_summaries, alignment):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not layer_summaries:
        return
    layers = [item["layer"] for item in layer_summaries]
    plt.figure(figsize=(8, 4))
    for branch in BRANCHES:
        plt.plot(layers, [item[f"soft_{branch}"] for item in layer_summaries], label=f"soft {branch}")
    if alignment:
        plt.plot(
            [item["layer"] for item in alignment],
            [item["alignment"] for item in alignment],
            label="alignment",
            linestyle="--",
            color="black",
        )
    plt.xlabel("layer")
    plt.ylabel("score")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="Routing trace JSONL.")
    parser.add_argument("--output-dir", default="evaluation/results/routing")
    parser.add_argument("--shared-params", type=float, default=0.0)
    parser.add_argument("--understanding-params", type=float, default=0.0)
    parser.add_argument("--generation-params", type=float, default=0.0)
    args = parser.parse_args()

    rows = list(load_jsonl(args.jsonl))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_summaries, task_summaries = summarize_router(rows)
    alignment = summarize_alignment(rows)
    runtime = summarize_runtime(rows)
    active_params = estimate_active_params(
        layer_summaries,
        args.shared_params,
        args.understanding_params,
        args.generation_params,
    )
    summary = {
        "num_rows": len(rows),
        "layers": layer_summaries,
        "tasks": task_summaries,
        "alignment": alignment,
        "runtime": runtime,
        "active_params": active_params,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as writer:
        json.dump(summary, writer, indent=2, sort_keys=True)
    write_csv(output_dir / "layer_routing.csv", layer_summaries)
    write_csv(output_dir / "alignment.csv", alignment)
    write_plot(output_dir / "routing_alignment.png", layer_summaries, alignment)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
