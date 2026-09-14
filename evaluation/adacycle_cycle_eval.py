#!/usr/bin/env python
# coding=utf-8
"""Evaluate AdaCycle-Fork cycle-consistency JSONL outputs.

Expected JSONL fields:
  prompt: original text prompt
  g2u_text: understanding-branch text recovered from the generated image
  plan: optional structured plan extracted by understanding branch
  u2g_text: optional caption/VQA text recovered from reconstructed/edited image

The script reports lightweight automatic consistency metrics for object counts,
attributes/colors, spatial relations, and token overlap. It is intentionally
dependency-free so it can run before installing benchmark-specific packages.
"""

import argparse
import json
import re
from collections import Counter

from training.adacycle import extract_semantic_facts


COLORS = {
    "black", "white", "red", "green", "blue", "yellow", "orange", "purple",
    "pink", "brown", "gray", "grey", "gold", "silver", "cyan", "magenta",
}

SPATIAL = {
    "left", "right", "above", "below", "under", "over", "behind", "front",
    "inside", "outside", "near", "far", "between", "around", "beside",
}

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def normalize(text):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def content_words(text):
    stop = {
        "a", "an", "the", "and", "or", "of", "in", "on", "with", "to",
        "is", "are", "be", "this", "that", "there", "image", "photo",
        "picture", "showing", "shows",
    }
    return [tok for tok in normalize(text) if tok not in stop]


def extract_counts(tokens):
    counts = []
    for tok in tokens:
        if tok.isdigit():
            counts.append(int(tok))
        elif tok in NUMBER_WORDS:
            counts.append(NUMBER_WORDS[tok])
    return counts


def f1_score(reference_items, predicted_items):
    ref = Counter(reference_items)
    pred = Counter(predicted_items)
    overlap = sum((ref & pred).values())
    if not ref and not pred:
        return 1.0
    if not ref or not pred or overlap == 0:
        return 0.0
    precision = overlap / sum(pred.values())
    recall = overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def record_metrics(reference, prediction):
    ref_tokens = normalize(reference)
    pred_tokens = normalize(prediction)
    ref_content = content_words(reference)
    pred_content = content_words(prediction)

    ref_counts = extract_counts(ref_tokens)
    pred_counts = extract_counts(pred_tokens)
    count_acc = 1.0 if ref_counts == pred_counts else 0.0
    if not ref_counts:
        count_acc = 1.0

    ref_colors = [tok for tok in ref_tokens if tok in COLORS]
    pred_colors = [tok for tok in pred_tokens if tok in COLORS]
    ref_spatial = [tok for tok in ref_tokens if tok in SPATIAL]
    pred_spatial = [tok for tok in pred_tokens if tok in SPATIAL]
    ref_facts = extract_semantic_facts(reference)
    pred_facts = extract_semantic_facts(prediction)
    ref_count_facts = [f"{item['count']}:{item['object']}" for item in ref_facts["counts"]]
    pred_count_facts = [f"{item['count']}:{item['object']}" for item in pred_facts["counts"]]
    ref_attr_facts = [
        f"{item['object']}:{item['attribute']}:{item['value']}"
        for item in ref_facts["attributes"]
    ]
    pred_attr_facts = [
        f"{item['object']}:{item['attribute']}:{item['value']}"
        for item in pred_facts["attributes"]
    ]
    ref_relation_facts = [
        f"{item['subject']}:{item['relation']}:{item['object']}"
        for item in ref_facts["relations"]
    ]
    pred_relation_facts = [
        f"{item['subject']}:{item['relation']}:{item['object']}"
        for item in pred_facts["relations"]
    ]

    return {
        "token_f1": f1_score(ref_content, pred_content),
        "count_acc": count_acc,
        "color_f1": f1_score(ref_colors, pred_colors),
        "spatial_f1": f1_score(ref_spatial, pred_spatial),
        "scene_count_f1": f1_score(ref_count_facts, pred_count_facts),
        "scene_attribute_f1": f1_score(ref_attr_facts, pred_attr_facts),
        "scene_relation_f1": f1_score(ref_relation_facts, pred_relation_facts),
    }


def average(metrics):
    if not metrics:
        return {}
    keys = metrics[0].keys()
    return {key: sum(item[key] for item in metrics) / len(metrics) for key in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="Path to AdaCycle cycle JSONL predictions.")
    args = parser.parse_args()

    g2u_metrics = []
    u2g_metrics = []
    with open(args.jsonl, "r", encoding="utf-8") as reader:
        for line in reader:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt", "")
            if row.get("g2u_text"):
                g2u_metrics.append(record_metrics(prompt, row["g2u_text"]))
            if row.get("u2g_text"):
                reference = row.get("plan") or prompt
                u2g_metrics.append(record_metrics(reference, row["u2g_text"]))

    result = {
        "num_g2u": len(g2u_metrics),
        "num_u2g": len(u2g_metrics),
        "g2u": average(g2u_metrics),
        "u2g": average(u2g_metrics),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
