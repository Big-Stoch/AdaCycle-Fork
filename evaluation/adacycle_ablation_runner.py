#!/usr/bin/env python
# coding=utf-8
"""Generate and optionally run AdaCycle-Fork ablation configs."""

import argparse
import copy
import json
import subprocess
from pathlib import Path

import yaml


ABLATIONS = {
    "full": {},
    "no_fork": {
        ("model", "showo", "adacycle_fork_enabled"): False,
    },
    "no_cycle_batches": {
        ("training", "adacycle_enable_cycle_batches"): False,
    },
    "no_g2u_semantic_verification": {
        ("training", "adacycle_g2u_verify_cycle_coeff"): 0.0,
        ("training", "adacycle_enable_semantic_verification"): False,
    },
    "caption_plan": {
        ("training", "adacycle_plan_source"): "caption",
    },
    "online_understanding_plan": {
        ("training", "adacycle_plan_source"): "online_understanding",
    },
    "no_edit_batches": {
        ("training", "adacycle_enable_edit_batches"): False,
        ("training", "adacycle_edit_coeff"): 0.0,
        ("training", "adacycle_edit_preserve_coeff"): 0.0,
    },
    "token_routing": {
        ("model", "showo", "adacycle_routing_granularity"): "token",
    },
    "sample_routing": {
        ("model", "showo", "adacycle_routing_granularity"): "layer",
    },
    "hard_routing": {
        ("model", "showo", "adacycle_hard_routing"): True,
        ("model", "showo", "adacycle_hard_routing_st"): True,
    },
}


def set_nested(config, path, value):
    cursor = config
    for key in path[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[path[-1]] = value


def make_variant(base_config, name, overrides, output_dir):
    variant = copy.deepcopy(base_config)
    for path, value in overrides.items():
        set_nested(variant, path, value)
    variant.setdefault("experiment", {})
    variant["experiment"]["name"] = f"{variant['experiment'].get('name', 'adacycle')}-{name}"
    variant["experiment"]["output_dir"] = str(output_dir / name)
    output_path = output_dir / f"{name}.yaml"
    with open(output_path, "w", encoding="utf-8") as writer:
        yaml.safe_dump(variant, writer, sort_keys=False)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/showo_adacycle_fork_stage3.yaml")
    parser.add_argument("--output-dir", default="configs/adacycle_ablations")
    parser.add_argument(
        "--command-template",
        default=None,
        help="Optional command template, e.g. 'accelerate launch training/train.py config={config}'.",
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--only", nargs="*", default=None, help="Subset of ablation names.")
    args = parser.parse_args()

    with open(args.base_config, "r", encoding="utf-8") as reader:
        base_config = yaml.safe_load(reader)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = args.only or list(ABLATIONS.keys())
    generated = []
    for name in selected:
        if name not in ABLATIONS:
            raise KeyError(f"Unknown ablation: {name}")
        config_path = make_variant(base_config, name, ABLATIONS[name], output_dir)
        item = {"name": name, "config": str(config_path)}
        if args.run and args.command_template:
            command = args.command_template.format(config=config_path)
            completed = subprocess.run(command, shell=True)
            item["command"] = command
            item["returncode"] = completed.returncode
        generated.append(item)

    print(json.dumps({"generated": generated}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
