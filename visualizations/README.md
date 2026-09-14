# Show-O / Show-O2 / Ours visualization results

These outputs compare the official [Show-O repository](https://github.com/showlab/Show-o) released Hugging Face checkpoints with an added Ours output set.

## Runs

- **Show-O:** `showlab/show-o-512x512`; 20 text-to-image prompts; 512×512; guidance 5; 50 discrete-denoising steps; seed 42.
- **Show-O2:** `showlab/show-o2-7B`; 20 text-to-image prompts; 432×432; guidance 7.5; 50 flow-matching steps; seed 42.
- **Ours:** 20 text-to-image prompts under `ours/t2i/`; generated image tool outputs, overwritten in place per request.
- **Multimodal understanding:** 14 shared input images, one shared detailed-description/counting/OCR question, with Ours responses under `ours/mmu/responses.json`.

The released Show-O / Show-O2 models are unified image-generation and multimodal-understanding models, not LIBERO robot-control policies. Show-O2 currently has no official general text-to-video or image-to-video checkpoint in the repository.

## Files

- `index.html`: prompt-by-prompt generation results and understanding responses for Show-O, Show-O2, and Ours.
- `show_o_contact_sheet.jpg`, `show_o2_contact_sheet.jpg`, `ours_t2i_contact_sheet.jpg`, `ours_mmu_contact_sheet.jpg`: compact visual overviews.
- `manifest.csv`: prompt and image-path mapping.
- `summary.json`: model/checkpoint and output counts.
- `show_o/`, `show_o2/`, and `ours/`: PNG outputs and JSON responses.
- `mmu_inputs/`: shared understanding inputs.

The Show-O2 release's safety-checker call passes PIL images to an API expecting NumPy arrays and can fail with `Image.shape`. All prompts here were curated as benign, so the saved generation images are the model's raw decoded outputs without that broken post-filter. Show-O2 understanding follows the checkpoint config's BF16 weight type on the A800 GPU.
