# coding=utf-8
"""AdaCycle-Fork cycle-learning utilities.

This module implements the concrete training loops described by AdaCycle-Fork:

* generation-to-understanding: generated visual tokens are fed back into the
  understanding prompt and trained to recover the original semantic prompt.
* understanding-to-generation: a structured semantic plan prompt reconstructs
  masked visual tokens.
* editing preservation: a contiguous edit mask is reconstructed while the model
  is penalized for changing non-target visual tokens.

The utilities are deliberately independent from dataloader classes so they can
be used by both Show-o training entrypoints.
"""

import re
import time
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from training.prompting_utils import (
    create_attention_mask_for_mmu,
    create_attention_mask_lvg,
    create_attention_mask_predict_next,
)


COLORS = {
    "black", "white", "red", "green", "blue", "yellow", "orange", "purple",
    "pink", "brown", "gray", "grey", "gold", "silver", "cyan", "magenta",
}

SPATIAL_RELATIONS = {
    "left of", "right of", "above", "below", "under", "over", "behind",
    "in front of", "inside", "outside", "near", "next to", "beside",
    "between", "around",
}
SPATIAL_WORDS = {word for relation in SPATIAL_RELATIONS for word in relation.split()}

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def _zero_like_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits.new_zeros(())


def _unwrap_showo_model(model):
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "showo") and hasattr(model.showo, "model"):
        return model.showo.model
    return None


def build_adacycle_trace_record(
    model,
    step: int,
    task: str,
    latency_ms: Optional[float] = None,
    throughput: Optional[float] = None,
) -> Optional[Dict]:
    showo_model = _unwrap_showo_model(model)
    aux_outputs = getattr(showo_model, "_last_adacycle_outputs", None) if showo_model is not None else None
    if aux_outputs is None:
        return None

    router_probs = []
    for probs in aux_outputs.get("router_probs", ()):
        if probs is None:
            continue
        probs = probs.detach().float()
        if probs.dim() == 3:
            probs = probs.mean(dim=1)
        router_probs.append(probs.cpu().tolist())

    alignment_scores = aux_outputs.get("alignment_scores")
    if alignment_scores is not None:
        alignment_scores = alignment_scores.detach().float().cpu().tolist()

    task_ids = aux_outputs.get("task_ids")
    token_type_ids = aux_outputs.get("token_type_ids")
    return {
        "timestamp": time.time(),
        "step": int(step),
        "task": task,
        "latency_ms": latency_ms,
        "throughput": throughput,
        "router_probs": router_probs,
        "alignment_scores": alignment_scores,
        "task_ids": task_ids.detach().cpu().tolist() if task_ids is not None else None,
        "token_type_image_fraction": (
            token_type_ids.detach().float().mean(dim=1).cpu().tolist()
            if token_type_ids is not None
            else None
        ),
    }


def _token_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1].contiguous().view(-1, logits.shape[-1])
    shift_labels = labels[:, 1:].contiguous().view(-1).to(shift_logits.device)
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)


def _masked_token_ce(logits: torch.Tensor, target_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not torch.any(mask):
        return logits.new_zeros(())

    selected_logits = logits[mask]
    selected_targets = target_ids.to(logits.device)[mask].long()
    return F.cross_entropy(selected_logits, selected_targets)


def _visual_slice(sequence: torch.Tensor, num_vq_tokens: int) -> torch.Tensor:
    return sequence[:, -(num_vq_tokens + 1):-1]


def _decode_text_batch(tokenizer, token_ids: torch.Tensor):
    pad_token_id = tokenizer.pad_token_id
    texts = []
    for row in token_ids.detach().cpu().tolist():
        if pad_token_id is not None:
            row = [idx for idx in row if idx != pad_token_id]
        texts.append(tokenizer.decode(row, skip_special_tokens=True).strip())
    return texts


def _as_text_list(text_tokenizer, text_batch, max_samples: Optional[int] = None):
    if isinstance(text_batch, torch.Tensor):
        texts = _decode_text_batch(text_tokenizer, text_batch)
    elif isinstance(text_batch, str):
        texts = [text_batch]
    else:
        texts = list(text_batch)

    if max_samples is not None:
        texts = texts[:max_samples]
    return [str(text) for text in texts]


def _normalize_words(text: str):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _singular(word: str) -> str:
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("ches", "shes", "xes", "zes", "ses")):
        return word[:-2]
    if len(word) > 2 and word.endswith("s"):
        return word[:-1]
    return word


def extract_semantic_facts(text: str) -> Dict[str, list]:
    """Extract lightweight scene-graph/VQA targets from a prompt.

    This deterministic parser is intentionally conservative. It covers the
    count, color-attribute binding, and spatial-relation checks used by common
    T2I composition benchmarks while keeping the training loop dependency-free.
    """

    words = _normalize_words(text)
    facts = {
        "counts": [],
        "attributes": [],
        "relations": [],
        "objects": [],
    }
    stop = {
        "a", "an", "the", "and", "or", "of", "in", "on", "with", "to",
        "is", "are", "photo", "image", "picture", "showing", "shows",
        "high", "quality", "realistic", "beautiful", "detailed",
    }
    stop = stop | SPATIAL_WORDS

    for idx, word in enumerate(words):
        count = None
        if word.isdigit():
            count = int(word)
        elif word in NUMBER_WORDS:
            count = NUMBER_WORDS[word]
        if count is not None and idx + 1 < len(words):
            obj_idx = idx + 1
            while obj_idx < len(words) and words[obj_idx] in COLORS:
                obj_idx += 1
            if obj_idx >= len(words):
                continue
            obj = _singular(words[obj_idx])
            if obj not in stop:
                facts["counts"].append({"object": obj, "count": count})

    for idx, word in enumerate(words[:-1]):
        if word in COLORS:
            obj = _singular(words[idx + 1])
            if obj not in stop and obj not in COLORS:
                facts["attributes"].append({"object": obj, "attribute": "color", "value": word})

    lowered = " ".join(words)
    color_pattern = "|".join(sorted(COLORS))
    for relation in sorted(SPATIAL_RELATIONS, key=len, reverse=True):
        pattern = rf"([a-z0-9]+)\s+{re.escape(relation)}\s+(?:(?:a|an|the)\s+)?(?:(?:{color_pattern})\s+)?([a-z0-9]+)"
        for left, right in re.findall(pattern, lowered):
            left = _singular(left)
            right = _singular(right)
            if left not in stop and right not in stop:
                facts["relations"].append({"subject": left, "relation": relation, "object": right})

    candidates = []
    for idx, word in enumerate(words):
        if word in stop or word in COLORS or word in NUMBER_WORDS or word.isdigit():
            continue
        if idx > 0 and words[idx - 1] in {"of", "in", "on", "with", "to"}:
            continue
        candidates.append(_singular(word))
    facts["objects"] = sorted(set(candidates[:12]))
    return facts


def build_semantic_verification_targets(text_tokenizer, text_batch, prefix: str):
    texts = _as_text_list(text_tokenizer, text_batch)
    targets = []
    for text in texts:
        facts = extract_semantic_facts(text)
        count_lines = [
            f"Q: How many {item['object']} objects should be present? A: {item['count']}."
            for item in facts["counts"]
        ]
        attribute_lines = [
            f"Q: What {item['attribute']} is bound to the {item['object']}? A: {item['value']}."
            for item in facts["attributes"]
        ]
        relation_lines = [
            f"Q: What spatial relation holds? A: {item['subject']} is {item['relation']} {item['object']}."
            for item in facts["relations"]
        ]
        scene_graph = {
            "objects": facts["objects"],
            "counts": facts["counts"],
            "attributes": facts["attributes"],
            "relations": facts["relations"],
        }
        lines = count_lines + attribute_lines + relation_lines
        if not lines:
            lines = ["Q: Is the image semantically consistent with the prompt? A: yes."]
        targets.append(f"{prefix} Scene graph: {scene_graph}. " + " ".join(lines))
    return targets


def build_structured_plan_texts(text_tokenizer, text_batch, prefix: str):
    """Build text semantic plans from captions/questions.

    If raw scene-graph annotations are unavailable, this deterministic plan
    prompt asks the model to interpret the caption as objects, attributes,
    relations, layout, and style. It gives U2G a distinct plan-conditioned
    input instead of simply duplicating the original caption.
    """

    captions = _as_text_list(text_tokenizer, text_batch)
    plan_texts = [
        f"{prefix} {caption}" if caption else prefix
        for caption in captions
    ]
    return plan_texts


@torch.no_grad()
def extract_semantic_plans_with_understanding(
    model,
    image_tokens: torch.Tensor,
    uni_prompting,
    max_new_tokens: int = 96,
    top_k: int = 1,
):
    """Use the understanding branch to extract structured semantic plans.

    This is the literal U(I) step from the paper. It is kept as an explicit
    utility because running autoregressive plan extraction inside every training
    step is usually too expensive; training can use caption-derived plans while
    evaluation or a later fine-tuning stage can call this function.
    """

    prompt = (
        "USER: Describe this image as a structured semantic plan with objects, "
        "attributes, relationships, layout, and style. ASSISTANT:"
    )
    prompts = [prompt for _ in range(image_tokens.shape[0])]
    input_ids, _, _ = uni_prompting((image_tokens, prompts), "mmu")
    input_ids = input_ids.to(image_tokens.device)
    attention_mask = create_attention_mask_for_mmu(
        input_ids,
        eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
    )

    plans = []
    for row_idx in range(input_ids.shape[0]):
        tokens = model.mmu_generate(
            idx=input_ids[row_idx:row_idx + 1],
            attention_mask=attention_mask[row_idx],
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            eot_token=uni_prompting.sptids_dict["<|eot|>"],
        )
        if len(tokens) == 0:
            plans.append("")
        else:
            token_tensor = torch.stack(tokens).reshape(1, -1)
            plans.append(uni_prompting.text_tokenizer.batch_decode(token_tensor, skip_special_tokens=True)[0])
    return plans


def sample_image_mask(
    batch_size: int,
    seq_len: int,
    ratio: float,
    device: torch.device,
    contiguous: bool = False,
) -> torch.Tensor:
    ratio = float(max(0.0, min(1.0, ratio)))
    num_masked = max(1, int(round(seq_len * ratio)))

    if not contiguous:
        ranks = torch.rand(batch_size, seq_len, device=device).argsort(dim=-1)
        return ranks < num_masked

    side = int(seq_len ** 0.5)
    if side * side != seq_len:
        ranks = torch.rand(batch_size, seq_len, device=device).argsort(dim=-1)
        return ranks < num_masked

    mask = torch.zeros(batch_size, side, side, dtype=torch.bool, device=device)
    box_side = max(1, int(round(num_masked ** 0.5)))
    box_h = min(side, box_side)
    box_w = min(side, max(1, int((num_masked + box_h - 1) // box_h)))
    for batch_idx in range(batch_size):
        top = torch.randint(0, side - box_h + 1, (1,), device=device).item()
        left = torch.randint(0, side - box_w + 1, (1,), device=device).item()
        mask[batch_idx, top:top + box_h, left:left + box_w] = True
    return mask.reshape(batch_size, seq_len)


@torch.no_grad()
def generated_tokens_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    config,
    mask_token_id: int,
) -> torch.Tensor:
    """Turn the current T2I logits into detached visual tokens for G2U."""

    num_vq_tokens = int(config.model.showo.num_vq_tokens)
    visual_start = int(config.model.showo.llm_vocab_size + config.model.showo.num_new_special_tokens)
    visual_end = visual_start + int(config.model.showo.codebook_size)

    image_logits = _visual_slice(logits, num_vq_tokens)[..., visual_start:visual_end]
    sampled = image_logits.argmax(dim=-1) + visual_start
    current_image_ids = _visual_slice(input_ids, num_vq_tokens)
    return torch.where(current_image_ids.eq(mask_token_id), sampled, current_image_ids)


def _mmu_text_ce_loss(
    model,
    image_tokens: torch.Tensor,
    target_texts,
    uni_prompting,
    mask_dtype,
) -> torch.Tensor:
    target_texts = _as_text_list(uni_prompting.text_tokenizer, target_texts, image_tokens.shape[0])
    input_ids_mmu, _, labels_mmu = uni_prompting((image_tokens, target_texts), "mmu")
    input_ids_mmu = input_ids_mmu.to(image_tokens.device)
    labels_mmu = labels_mmu.to(image_tokens.device)
    attention_mask = create_attention_mask_for_mmu(
        input_ids_mmu,
        eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
    ).to(mask_dtype)

    logits = model(
        input_ids=input_ids_mmu,
        attention_mask=attention_mask,
        labels=None,
        batch_size_t2i=0,
        batch_size_lm=0,
        batch_size_mmu=0,
        return_adacycle_losses=False,
    )
    return _token_ce(logits, labels_mmu)


def compute_generation_to_understanding_losses(
    model,
    generated_image_tokens: torch.Tensor,
    target_texts,
    uni_prompting,
    config,
    mask_dtype,
) -> Dict[str, torch.Tensor]:
    target_texts = _as_text_list(uni_prompting.text_tokenizer, target_texts, generated_image_tokens.shape[0])
    caption_loss = _mmu_text_ce_loss(
        model,
        generated_image_tokens,
        target_texts,
        uni_prompting,
        mask_dtype,
    )

    verification_loss = caption_loss.new_zeros(())
    if bool(config.training.get("adacycle_enable_semantic_verification", True)):
        verify_prefix = config.training.get(
            "adacycle_semantic_verification_prefix",
            "Verify object counts, attribute bindings, spatial relations, and VQA answers.",
        )
        verification_texts = build_semantic_verification_targets(
            uni_prompting.text_tokenizer,
            target_texts,
            verify_prefix,
        )
        verification_loss = _mmu_text_ce_loss(
            model,
            generated_image_tokens,
            verification_texts,
            uni_prompting,
            mask_dtype,
        )

    return {
        "loss_g2u_cycle": caption_loss,
        "loss_g2u_verify_cycle": verification_loss,
    }


def compute_generation_to_understanding_loss(
    model,
    generated_image_tokens: torch.Tensor,
    target_texts,
    uni_prompting,
    config,
    mask_dtype,
) -> torch.Tensor:
    return compute_generation_to_understanding_losses(
        model,
        generated_image_tokens,
        target_texts,
        uni_prompting,
        config,
        mask_dtype,
    )["loss_g2u_cycle"]


def resize_edit_mask_to_tokens(
    edit_masks: Optional[torch.Tensor],
    num_vq_tokens: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if edit_masks is None or edit_masks.numel() == 0:
        return None
    if edit_masks.dim() == 3:
        edit_masks = edit_masks.unsqueeze(1)
    edit_masks = edit_masks.to(device=device, dtype=torch.float32)
    side = int(num_vq_tokens ** 0.5)
    if side * side != num_vq_tokens:
        return None
    token_masks = F.interpolate(edit_masks, size=(side, side), mode="nearest")
    return token_masks[:, 0].reshape(edit_masks.shape[0], num_vq_tokens) > 0.5


def _masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=pred.device, dtype=pred.dtype)
    while mask.dim() < pred.dim():
        mask = mask.unsqueeze(1)
    denom = mask.sum().clamp_min(1.0) * pred.shape[1]
    return (pred - target.to(pred.device)).abs().mul(mask).sum() / denom


def _simple_ssim_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred = (pred.float() + 1.0) / 2.0
    target = (target.to(pred.device).float() + 1.0) / 2.0
    mask = mask.to(device=pred.device, dtype=pred.dtype)
    while mask.dim() < pred.dim():
        mask = mask.unsqueeze(1)

    kernel_size = 3
    padding = kernel_size // 2
    mu_x = F.avg_pool2d(pred, kernel_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, kernel_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(pred * pred, kernel_size, stride=1, padding=padding) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, kernel_size, stride=1, padding=padding) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, kernel_size, stride=1, padding=padding) - mu_x * mu_y
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    )
    loss_map = (1.0 - ssim).clamp_min(0.0)
    denom = mask.sum().clamp_min(1.0) * pred.shape[1]
    return loss_map.mul(mask).sum() / denom


def _soft_decode_visual_logits(
    vq_model,
    visual_logits: torch.Tensor,
    config,
    temperature: float = 1.0,
) -> Optional[torch.Tensor]:
    if vq_model is None or not hasattr(vq_model, "quantize") or not hasattr(vq_model.quantize, "embedding"):
        return None
    num_vq_tokens = int(config.model.showo.num_vq_tokens)
    side = int(num_vq_tokens ** 0.5)
    if side * side != num_vq_tokens:
        return None

    visual_start = int(config.model.showo.llm_vocab_size + config.model.showo.num_new_special_tokens)
    codebook_size = int(config.model.showo.codebook_size)
    codebook_logits = visual_logits[..., visual_start:visual_start + codebook_size]
    probs = F.softmax(codebook_logits.float() / max(float(temperature), 1e-6), dim=-1)
    embedding = vq_model.quantize.embedding.to(device=visual_logits.device, dtype=probs.dtype)
    z = torch.matmul(probs, embedding)
    z = z.view(visual_logits.shape[0], side, side, -1).permute(0, 3, 1, 2).contiguous()
    decoded = vq_model.decoder(z)["output"]
    return decoded.to(dtype=visual_logits.dtype)


def _token_mask_to_pixel_mask(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    side = int(mask.shape[1] ** 0.5)
    if side * side != mask.shape[1]:
        return torch.ones(mask.shape[0], 1, height, width, device=mask.device, dtype=torch.float32)
    mask = mask.reshape(mask.shape[0], 1, side, side).float()
    return F.interpolate(mask, size=(height, width), mode="nearest")


def _optional_lpips_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    try:
        import lpips  # type: ignore
    except Exception:
        return pred.new_zeros(())

    if not hasattr(_optional_lpips_loss, "_model"):
        model = lpips.LPIPS(net="vgg").to(pred.device).eval()
        for param in model.parameters():
            param.requires_grad_(False)
        _optional_lpips_loss._model = model
    model = _optional_lpips_loss._model.to(pred.device)
    loss = model(pred.float(), target.to(pred.device).float())
    sample_weights = mask.reshape(mask.shape[0], -1).float().mean(dim=1).to(loss.device)
    return (loss.reshape(loss.shape[0]) * sample_weights).mean()


def build_edit_instruction_texts(text_tokenizer, edit_instructions, max_samples: Optional[int] = None):
    instructions = _as_text_list(text_tokenizer, edit_instructions, max_samples)
    return [
        "Edit the source image according to this instruction while preserving all unrelated regions: "
        f"{instruction}"
        for instruction in instructions
    ]


def compute_editing_training_losses(
    model,
    source_image_tokens: torch.Tensor,
    target_image_tokens: torch.Tensor,
    edit_instructions,
    edit_masks: Optional[torch.Tensor],
    uni_prompting,
    config,
    mask_id: int,
    mask_dtype,
    vq_model=None,
    source_pixel_values: Optional[torch.Tensor] = None,
    target_pixel_values: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    if source_image_tokens is None or target_image_tokens is None or source_image_tokens.shape[0] == 0:
        zero = model.showo.lm_head.weight.new_zeros(())
        return {
            "loss_edit": zero,
            "loss_edit_preserve": zero,
            "loss_edit_pixel": zero,
            "loss_preserve_pixel": zero,
            "loss_preserve_ssim": zero,
            "loss_preserve_lpips": zero,
        }

    max_samples = int(config.training.get("adacycle_edit_max_samples", source_image_tokens.shape[0]))
    max_samples = max(1, min(max_samples, source_image_tokens.shape[0], target_image_tokens.shape[0]))
    source_image_tokens = source_image_tokens[:max_samples]
    target_image_tokens = target_image_tokens[:max_samples]
    edit_texts = build_edit_instruction_texts(
        uni_prompting.text_tokenizer,
        edit_instructions,
        max_samples,
    )
    if source_pixel_values is not None:
        source_pixel_values = source_pixel_values[:max_samples]
    if target_pixel_values is not None:
        target_pixel_values = target_pixel_values[:max_samples]
    if edit_masks is not None and edit_masks.numel() > 0:
        edit_masks = edit_masks[:max_samples]

    device = source_image_tokens.device
    num_vq_tokens = int(config.model.showo.num_vq_tokens)
    edit_token_mask = resize_edit_mask_to_tokens(edit_masks, num_vq_tokens, device)
    if edit_token_mask is None:
        edit_token_mask = sample_image_mask(
            max_samples,
            num_vq_tokens,
            float(config.training.get("adacycle_edit_mask_ratio", 0.25)),
            device,
            contiguous=bool(config.training.get("adacycle_edit_contiguous_mask", True)),
        )

    target_input = torch.where(edit_token_mask, torch.full_like(target_image_tokens, mask_id), target_image_tokens)
    target_labels = torch.where(edit_token_mask, target_image_tokens, torch.full_like(target_image_tokens, -100))
    input_ids_edit, _, labels_edit = uni_prompting(
        (edit_texts, source_image_tokens, target_input, target_labels),
        "edit",
    )
    input_ids_edit = input_ids_edit.to(device)
    labels_edit = labels_edit.to(device)
    attention_mask_edit = create_attention_mask_lvg(
        input_ids_edit,
        pad_id=int(uni_prompting.sptids_dict["<|pad|>"]),
        soi_id=int(uni_prompting.sptids_dict["<|soi|>"]),
        eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
        return_inverse_mask=True,
    ).to(mask_dtype)

    logits_edit = model(
        input_ids=input_ids_edit,
        attention_mask=attention_mask_edit,
        labels=None,
        batch_size_t2i=0,
        batch_size_lm=0,
        batch_size_mmu=0,
        return_adacycle_losses=False,
    )
    visual_logits = _visual_slice(logits_edit, num_vq_tokens)
    visual_labels = _visual_slice(labels_edit, num_vq_tokens)
    loss_edit = F.cross_entropy(
        visual_logits.contiguous().view(-1, logits_edit.shape[-1]),
        visual_labels.contiguous().view(-1),
        ignore_index=-100,
    )
    loss_edit_preserve = _masked_token_ce(visual_logits, source_image_tokens, ~edit_token_mask)

    zero = loss_edit.new_zeros(())
    loss_edit_pixel = zero
    loss_preserve_pixel = zero
    loss_preserve_ssim = zero
    loss_preserve_lpips = zero
    pixel_coeff_enabled = any(
        float(config.training.get(key, 0.0)) > 0.0
        for key in (
            "adacycle_edit_pixel_coeff",
            "adacycle_preserve_pixel_coeff",
            "adacycle_preserve_ssim_coeff",
            "adacycle_preserve_lpips_coeff",
        )
    )
    if pixel_coeff_enabled and source_pixel_values is not None and target_pixel_values is not None:
        pred_pixels = _soft_decode_visual_logits(
            vq_model,
            visual_logits,
            config,
            float(config.training.get("adacycle_soft_decode_temperature", 1.0)),
        )
        if pred_pixels is not None:
            source_pixel_values = source_pixel_values.to(pred_pixels.device, dtype=pred_pixels.dtype)
            target_pixel_values = target_pixel_values.to(pred_pixels.device, dtype=pred_pixels.dtype)
            edit_pixel_mask = _token_mask_to_pixel_mask(
                edit_token_mask.to(pred_pixels.device),
                pred_pixels.shape[-2],
                pred_pixels.shape[-1],
            )
            preserve_pixel_mask = 1.0 - edit_pixel_mask
            loss_edit_pixel = _masked_l1(pred_pixels, target_pixel_values, edit_pixel_mask)
            loss_preserve_pixel = _masked_l1(pred_pixels, source_pixel_values, preserve_pixel_mask)
            loss_preserve_ssim = _simple_ssim_loss(pred_pixels, source_pixel_values, preserve_pixel_mask)
            if float(config.training.get("adacycle_preserve_lpips_coeff", 0.0)) > 0.0:
                loss_preserve_lpips = _optional_lpips_loss(pred_pixels, source_pixel_values, preserve_pixel_mask)

    return {
        "loss_edit": loss_edit,
        "loss_edit_preserve": loss_edit_preserve,
        "loss_edit_pixel": loss_edit_pixel,
        "loss_preserve_pixel": loss_preserve_pixel,
        "loss_preserve_ssim": loss_preserve_ssim,
        "loss_preserve_lpips": loss_preserve_lpips,
    }


def compute_understanding_to_generation_losses(
    model,
    image_tokens: torch.Tensor,
    plan_texts,
    uni_prompting,
    config,
    mask_id: int,
    mask_dtype,
    mask_ratio: float,
    edit_mask_ratio: float,
    contiguous_edit_mask: bool,
) -> Dict[str, torch.Tensor]:
    device = image_tokens.device
    num_vq_tokens = int(config.model.showo.num_vq_tokens)

    reconstruction_mask = sample_image_mask(
        image_tokens.shape[0],
        num_vq_tokens,
        mask_ratio,
        device,
        contiguous=False,
    )
    reconstruction_input = torch.where(
        reconstruction_mask,
        torch.full_like(image_tokens, mask_id),
        image_tokens,
    )
    reconstruction_labels = torch.where(reconstruction_mask, image_tokens, torch.full_like(image_tokens, -100))

    plan_texts = _as_text_list(uni_prompting.text_tokenizer, plan_texts, image_tokens.shape[0])
    input_ids_u2g, _, labels_u2g = uni_prompting((plan_texts, reconstruction_input, reconstruction_labels), "t2i")
    input_ids_u2g = input_ids_u2g.to(device)
    labels_u2g = labels_u2g.to(device)
    attention_mask_u2g = create_attention_mask_predict_next(
        input_ids_u2g,
        pad_id=int(uni_prompting.sptids_dict["<|pad|>"]),
        soi_id=int(uni_prompting.sptids_dict["<|soi|>"]),
        eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
        rm_pad_in_image=True,
        return_inverse_mask=True,
    ).to(mask_dtype)

    logits_u2g = model(
        input_ids=input_ids_u2g,
        attention_mask=attention_mask_u2g,
        labels=None,
        batch_size_t2i=0,
        batch_size_lm=0,
        batch_size_mmu=0,
        return_adacycle_losses=False,
    )
    loss_u2g = F.cross_entropy(
        _visual_slice(logits_u2g, num_vq_tokens).contiguous().view(-1, logits_u2g.shape[-1]),
        _visual_slice(labels_u2g, num_vq_tokens).contiguous().view(-1),
        ignore_index=-100,
    )

    edit_mask = sample_image_mask(
        image_tokens.shape[0],
        num_vq_tokens,
        edit_mask_ratio,
        device,
        contiguous=contiguous_edit_mask,
    )
    edit_input = torch.where(edit_mask, torch.full_like(image_tokens, mask_id), image_tokens)
    edit_labels = torch.where(edit_mask, image_tokens, torch.full_like(image_tokens, -100))
    input_ids_edit, _, labels_edit = uni_prompting((plan_texts, edit_input, edit_labels), "t2i")
    input_ids_edit = input_ids_edit.to(device)
    labels_edit = labels_edit.to(device)
    attention_mask_edit = create_attention_mask_predict_next(
        input_ids_edit,
        pad_id=int(uni_prompting.sptids_dict["<|pad|>"]),
        soi_id=int(uni_prompting.sptids_dict["<|soi|>"]),
        eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
        rm_pad_in_image=True,
        return_inverse_mask=True,
    ).to(mask_dtype)

    logits_edit = model(
        input_ids=input_ids_edit,
        attention_mask=attention_mask_edit,
        labels=None,
        batch_size_t2i=0,
        batch_size_lm=0,
        batch_size_mmu=0,
        return_adacycle_losses=False,
    )
    visual_logits_edit = _visual_slice(logits_edit, num_vq_tokens)
    loss_preserve = _masked_token_ce(visual_logits_edit, image_tokens, ~edit_mask)

    return {
        "loss_u2g_cycle": loss_u2g,
        "loss_preserve_cycle": loss_preserve,
    }


def compute_adacycle_cycle_losses(
    model,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    t2i_text_tokens: torch.Tensor,
    t2i_image_tokens: torch.Tensor,
    uni_prompting,
    config,
    mask_id: int,
    mask_dtype,
    global_step: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    if t2i_image_tokens is None or t2i_image_tokens.shape[0] == 0:
        zero = _zero_like_logits(logits)
        return {
            "loss_g2u_cycle": zero,
            "loss_g2u_verify_cycle": zero,
            "loss_u2g_cycle": zero,
            "loss_preserve_cycle": zero,
        }

    max_samples = int(config.training.get("adacycle_cycle_max_samples", t2i_image_tokens.shape[0]))
    max_samples = max(1, min(max_samples, t2i_image_tokens.shape[0]))

    image_tokens = t2i_image_tokens[:max_samples].detach()
    text_samples = _as_text_list(uni_prompting.text_tokenizer, t2i_text_tokens, max_samples)
    input_ids_t2i = input_ids[:max_samples]
    logits_t2i = logits[:max_samples]

    generated_image_tokens = generated_tokens_from_logits(
        logits_t2i.detach(),
        input_ids_t2i.detach(),
        config,
        mask_id,
    )
    g2u_losses = compute_generation_to_understanding_losses(
        model,
        generated_image_tokens,
        text_samples,
        uni_prompting,
        config,
        mask_dtype,
    )

    plan_source = str(config.training.get("adacycle_plan_source", "caption")).lower()
    plan_prefix = config.training.get(
        "adacycle_plan_prefix",
        "Structured semantic plan with objects, attributes, relationships, layout, and style:",
    )
    plan_texts = None
    if plan_source in {"understanding", "online", "online_understanding", "u", "u(i)"}:
        every = int(config.training.get("adacycle_online_plan_every", 1))
        should_extract = global_step is None or every <= 1 or (global_step % every == 0)
        if should_extract:
            was_training = model.training
            model.eval()
            try:
                plan_texts = extract_semantic_plans_with_understanding(
                    model,
                    image_tokens,
                    uni_prompting,
                    max_new_tokens=int(config.training.get("adacycle_online_plan_max_new_tokens", 96)),
                    top_k=int(config.training.get("adacycle_online_plan_top_k", 1)),
                )
            finally:
                if was_training:
                    model.train()
            if not any(plan_texts):
                plan_texts = None
    if plan_texts is None:
        plan_texts = build_structured_plan_texts(
            uni_prompting.text_tokenizer,
            text_samples,
            plan_prefix,
        )
    u2g_losses = compute_understanding_to_generation_losses(
        model,
        image_tokens,
        plan_texts,
        uni_prompting,
        config,
        mask_id,
        mask_dtype,
        float(config.training.get("adacycle_u2g_mask_ratio", 0.50)),
        float(config.training.get("adacycle_edit_mask_ratio", 0.25)),
        bool(config.training.get("adacycle_edit_contiguous_mask", True)),
    )

    return {
        **g2u_losses,
        **u2g_losses,
    }
