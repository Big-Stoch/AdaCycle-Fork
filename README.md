# AdaCycle-Fork: Alignment-Adaptive Branching with Bidirectional Cycle Learning for Unified Multimodal Understanding and Generation

**Research Paper**:

**Project GitHub**:

AdaCycle-Fork is a unified multimodal understanding and generation model built on the Show-o training stack. It represents text and images as discrete tokens in one Transformer and extends the base model with two main additions:

- **Alignment-adaptive forking:** dynamically weights a shared path, understanding branch, and generation branch using task, modality, layer, and hidden-state information.
- **Bidirectional cycle learning:** uses generation-to-understanding and understanding-to-generation auxiliary objectives to improve semantic faithfulness, reconstruction, and editing preservation.

The model supports text-to-image generation, multimodal understanding, visual question answering, image captioning, image inpainting/editing, routing analysis, cycle-consistency evaluation, and ablation experiments. AdaCycle-Fork keeps the Show-o-style unified token interface and MaskGIT-style visual generation, while adding adaptive routing and cycle-consistency supervision during fine-tuning.

## Repository Layout

- `AdaCycle-Fork.tex` - research paper source.
- `models/phi.py` - Phi backbone extensions and AdaCycle-Fork routing modules.
- `models/modeling_showo.py` - Show-o model wrapper with AdaCycle-Fork losses and forward logic.
- `training/train.py` - main training entry point.
- `training/adacycle.py` - cycle-batch, verification, editing, and AdaCycle-specific training utilities.
- `training/prompting_utils.py` - unified prompting and omni-attention masks for T2I and MMU.
- `inference_t2i.py` - text-to-image generation and inpainting inference.
- `inference_mmu.py` - multimodal understanding inference.
- `evaluation/adacycle_benchmark_suite.py` - benchmark suite entry point.
- `evaluation/adacycle_ablation_runner.py` - ablation runner.
- `evaluation/adacycle_cycle_eval.py` - cycle-consistency evaluation.
- `evaluation/adacycle_routing_analysis.py` - routing and alignment-trajectory analysis.
- `configs/showo_pretraining_stage1.yaml` - Stage 1 base pretraining configuration.
- `configs/showo_pretraining_stage2.yaml` - Stage 2 base mixed-task configuration.
- `configs/showo_adacycle_fork_stage3.yaml` - Stage 3 AdaCycle-Fork fine-tuning configuration.
- `configs/adacycle_eval_suite.yaml` - evaluation suite configuration.
- `validation_prompts/`, `mmu_validation/`, and `inpainting_validation/` - sample prompts and validation assets.

## System Requirements

Inference:

- 1 CUDA GPU is recommended.
- Memory depends on resolution, batch size, guidance scale, and whether CLIP-ViT features are enabled.
- The default visual tokenizer setting uses 256 image tokens for 256 by 256 images.

Training:

- Multi-GPU training is recommended.
- The training code uses `accelerate`, `deepspeed`, `bf16`, and optional TF32.
- Stage 3 adds the AdaCycle-Fork router, branch adapters, alignment loss, G2U/U2G cycle batches, and preservation losses, so it is heavier than base Show-o-style training.

## Installation

Create and activate a Python environment, then install the requirements:

```bash
pip install -r requirements.txt
```

The code expects access to the pretrained language model and visual tokenizer paths configured in the YAML files, including:

- `microsoft/phi-1_5`
- `showlab/magvitv2`
- `showlab/show-o`

Update dataset paths, output directories, and checkpoint paths in the relevant config files before running training or inference.

## Quick Start

Text-to-image generation:

```bash
python inference_t2i.py config=configs/showo_demo.yaml mode=t2i prompt="A futuristic eco-friendly city built into steep green cliffs, waterfalls, flying buses, panoramic concept art."
```

Image inpainting:

```bash
python inference_t2i.py config=configs/showo_demo.yaml mode=inpainting prompt="Replace the masked region with a small wooden cabin." image_path=inpainting_validation/alpine_lake.jpg inpainting_mask_path=inpainting_validation/bench_mask.webp
```

Multimodal understanding:

```bash
python inference_mmu.py config=configs/showo_demo.yaml mmu_image_root=mmu_validation question="Describe this image."
```

The inference scripts use OmegaConf command-line overrides, so any value in the YAML config can be overridden from the command line with `key=value`.

## Training Pipeline

AdaCycle-Fork uses three training stages:

1. **Stage 1: unified multimodal pretraining.** Uses the base Show-o-style objectives for text-to-image generation, language modeling, and multimodal understanding.
2. **Stage 2: mixed-task base training.** Continues training on understanding, generation, and editing-style data with the same base objective.
3. **Stage 3: AdaCycle-Fork fine-tuning.** Enables the adaptive router and branch adapters, then adds alignment trajectory, G2U cycle, U2G cycle, semantic verification, and preservation supervision.

Run Stage 3 fine-tuning with:

```bash
accelerate launch training/train.py config=configs/showo_adacycle_fork_stage3.yaml
```

Important Stage 3 controls are defined in `configs/showo_adacycle_fork_stage3.yaml`, including:

- `model.showo.adacycle_fork_enabled`
- `model.showo.adacycle_router_hidden_size`
- `model.showo.adacycle_branch_bottleneck`
- `model.showo.adacycle_routing_granularity`
- `model.showo.adacycle_generation_peak_layer`
- `training.adacycle_align_coeff`
- `training.adacycle_g2u_coeff`
- `training.adacycle_u2g_coeff`
- `training.adacycle_preserve_coeff`
- `training.adacycle_enable_cycle_batches`
- `training.adacycle_g2u_cycle_coeff`
- `training.adacycle_g2u_verify_cycle_coeff`
- `training.adacycle_u2g_cycle_coeff`
- `training.adacycle_preserve_cycle_coeff`

## Evaluation

Run the AdaCycle-Fork evaluation suite with:

```bash
python evaluation/adacycle_benchmark_suite.py config=configs/adacycle_eval_suite.yaml
```

Cycle-consistency evaluation:

```bash
python evaluation/adacycle_cycle_eval.py config=configs/adacycle_eval_suite.yaml
```

Routing and alignment-trajectory analysis:

```bash
python evaluation/adacycle_routing_analysis.py config=configs/adacycle_eval_suite.yaml
```

Ablation experiments:

```bash
python evaluation/adacycle_ablation_runner.py config=configs/adacycle_eval_suite.yaml
```

Before reporting results, make sure the evaluation config points to the correct checkpoint, visual tokenizer, validation prompts, image roots, and output directory.

## Notes on Attention and Generation

AdaCycle-Fork follows Show-o's unified attention design. Text and control tokens use causal autoregressive attention, while visual tokens in T2I generation can attend bidirectionally within the text-image sequence for masked visual-token prediction. This supports MaskGIT-style parallel visual decoding while keeping text generation autoregressive.

## Acknowledgments

This work is based on or uses resources from `open-muse`, Phi-1.5, `muse-maskgit-pytorch`, `maskgit`, `transformers`, `taming-transformers`, `accelerate`, `diffusers`, `webdataset`, MagViT-v2, and Show-o. Thanks to all respective authors for their great work.

## Support

For questions, issues, or missing files, please contact: