# coding=utf-8
# Copyright 2024 NUS Show Lab, HuggingFace.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
from transformers import AutoConfig
from .modeling_utils import ConfigMixin, ModelMixin, register_to_config
from .sampling import cosine_schedule, mask_by_random_topk
from .phi import AdaCycleFork, PhiForCausalLM

class Showo(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            w_clip_vit,
            vocab_size,
            llm_vocab_size,
            llm_model_path='',
            codebook_size=8192,
            num_vq_tokens=256,
            load_from_showo=True,
            **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.register_to_config(mask_token_id=vocab_size - 1)
        if load_from_showo:
            config = AutoConfig.from_pretrained(llm_model_path)
            self._apply_adacycle_config(config, kwargs)
            self.showo = PhiForCausalLM(config)
        else:
            config = AutoConfig.from_pretrained(llm_model_path, attn_implementation='sdpa')
            self._apply_adacycle_config(config, kwargs)
            self.showo = PhiForCausalLM.from_pretrained(llm_model_path, config=config)
        self.showo.resize_token_embeddings(self.vocab_size)
        self.output_size = self.vocab_size

        if self.w_clip_vit:
            self.mm_projector = torch.nn.Sequential(
                torch.nn.Linear(1024, 2048),
                torch.nn.GELU(),
                torch.nn.Linear(2048, 2048)
            )

    @staticmethod
    def _apply_adacycle_config(config, kwargs):
        defaults = {
            "adacycle_fork_enabled": False,
            "adacycle_router_hidden_size": None,
            "adacycle_branch_bottleneck": None,
            "adacycle_branch_dropout": 0.0,
            "adacycle_routing_granularity": "layer",
            "adacycle_hard_routing": False,
            "adacycle_hard_routing_st": True,
            "adacycle_generation_peak_layer": None,
        }
        for key, default_value in defaults.items():
            setattr(config, key, kwargs.get(key, default_value))

    def enable_gradient_checkpointing(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing = True
        if hasattr(self.showo, "gradient_checkpointing_enable"):
            try:
                self.showo.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                )
            except TypeError:
                self.showo.gradient_checkpointing_enable()
        elif hasattr(self.showo, "model") and hasattr(self.showo.model, "gradient_checkpointing"):
            self.showo.model.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        if hasattr(self.showo, "gradient_checkpointing_disable"):
            self.showo.gradient_checkpointing_disable()
        elif hasattr(self.showo, "model") and hasattr(self.showo.model, "gradient_checkpointing"):
            self.showo.model.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value
        self.gradient_checkpointing = value

    def _adacycle_task_token_ids(self):
        llm_vocab_size = int(getattr(self.config, "llm_vocab_size", getattr(self.showo.config, "vocab_size", 0)))
        num_new_special_tokens = int(getattr(self.config, "num_new_special_tokens", 10))

        if num_new_special_tokens >= 10:
            # UniversalPrompting adds [PAD] before the nine Show-o task/sentinel tokens.
            return {
                "t2i": llm_vocab_size + 5,
                "mmu": llm_vocab_size + 6,
            }

        return {
            "t2i": llm_vocab_size + 4,
            "mmu": llm_vocab_size + 5,
        }

    def _adacycle_enabled(self):
        return bool(getattr(self.showo.config, "adacycle_fork_enabled", False))

    def _infer_adacycle_task_ids(
            self,
            input_ids,
            batch_size_t2i=0,
            batch_size_lm=0,
            batch_size_mmu=0,
            total_batch_size=None,
    ):
        if total_batch_size is None:
            if input_ids is None:
                return None
            total_batch_size = input_ids.shape[0]

        device = input_ids.device if input_ids is not None else self.showo.lm_head.weight.device
        if input_ids is None and batch_size_t2i == 0 and batch_size_lm == 0 and batch_size_mmu == 0:
            return None

        task_ids = torch.full(
            (total_batch_size,),
            AdaCycleFork.TASK_SHARED,
            dtype=torch.long,
            device=device,
        )
        if batch_size_t2i > 0 or batch_size_lm > 0 or batch_size_mmu > 0:
            if batch_size_t2i > 0:
                task_ids[:batch_size_t2i] = AdaCycleFork.TASK_GENERATION
            mmu_start = batch_size_t2i + batch_size_lm
            if batch_size_mmu > 0:
                task_ids[mmu_start:mmu_start + batch_size_mmu] = AdaCycleFork.TASK_UNDERSTANDING
            return task_ids

        task_token_ids = self._adacycle_task_token_ids()
        t2i_token_id = task_token_ids["t2i"]
        mmu_token_id = task_token_ids["mmu"]
        inferred_rows = min(total_batch_size, input_ids.shape[0])
        inferred_task_ids = task_ids[:inferred_rows]
        inferred_task_ids[input_ids[:inferred_rows].eq(t2i_token_id).any(dim=1)] = AdaCycleFork.TASK_GENERATION
        inferred_task_ids[input_ids[:inferred_rows].eq(mmu_token_id).any(dim=1)] = AdaCycleFork.TASK_UNDERSTANDING
        return task_ids

    def _infer_adacycle_token_type_ids(self, input_ids, total_batch_size=None, seq_length=None):
        if input_ids is None and (total_batch_size is None or seq_length is None):
            return None
        if total_batch_size is None:
            total_batch_size = input_ids.shape[0]
        if seq_length is None:
            seq_length = input_ids.shape[1]
        if input_ids is None:
            return torch.zeros(
                total_batch_size,
                seq_length,
                dtype=torch.long,
                device=self.showo.lm_head.weight.device,
            )

        llm_vocab_size = int(getattr(self.config, "llm_vocab_size", getattr(self.showo.config, "vocab_size", 0)))
        num_new_special_tokens = int(getattr(self.config, "num_new_special_tokens", 0))
        visual_token_start = llm_vocab_size + num_new_special_tokens
        image_mask = input_ids.ge(visual_token_start) | input_ids.eq(self.config.mask_token_id)
        token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
        token_type_ids = torch.where(
            image_mask,
            torch.ones_like(token_type_ids) * AdaCycleFork.MODALITY_IMAGE,
            token_type_ids,
        )
        if token_type_ids.shape[0] != total_batch_size or token_type_ids.shape[1] != seq_length:
            expanded_token_type_ids = torch.zeros(
                total_batch_size,
                seq_length,
                dtype=torch.long,
                device=token_type_ids.device,
            )
            rows = min(total_batch_size, token_type_ids.shape[0])
            cols = min(seq_length, token_type_ids.shape[1])
            expanded_token_type_ids[:rows, :cols] = token_type_ids[:rows, :cols]
            token_type_ids = expanded_token_type_ids
        return token_type_ids

    @staticmethod
    def _masked_mean(hidden_states, mask):
        mask = mask.to(device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden_states * mask).sum(dim=1) / denom

    @staticmethod
    def _select_loss(loss_values, sample_mask, zero):
        if sample_mask is None or not torch.any(sample_mask):
            return zero
        return loss_values[sample_mask].mean()

    def _semantic_cycle_proxy_loss(self, hidden_states, token_type_ids, task_ids, target_task):
        zero = hidden_states.new_zeros(())
        if token_type_ids is None or task_ids is None:
            return zero

        text_mask = token_type_ids.eq(AdaCycleFork.MODALITY_TEXT)
        image_mask = token_type_ids.eq(AdaCycleFork.MODALITY_IMAGE)
        valid = task_ids.eq(target_task) & text_mask.any(dim=1) & image_mask.any(dim=1)
        text_state = self._masked_mean(hidden_states, text_mask)
        image_state = self._masked_mean(hidden_states, image_mask)
        cycle_loss = 1.0 - F.cosine_similarity(text_state.float(), image_state.float(), dim=-1)
        return self._select_loss(cycle_loss.to(hidden_states.dtype), valid, zero)

    def _visual_preserve_proxy_loss(self, input_hidden_states, final_hidden_states, token_type_ids, task_ids):
        zero = final_hidden_states.new_zeros(())
        if input_hidden_states is None or token_type_ids is None or task_ids is None:
            return zero

        image_mask = token_type_ids.eq(AdaCycleFork.MODALITY_IMAGE)
        valid = task_ids.ne(AdaCycleFork.TASK_SHARED) & image_mask.any(dim=1)
        input_image_state = self._masked_mean(input_hidden_states, image_mask)
        final_image_state = self._masked_mean(final_hidden_states, image_mask)
        preserve_loss = F.mse_loss(
            F.normalize(final_image_state.float(), dim=-1),
            F.normalize(input_image_state.detach().float(), dim=-1),
            reduction="none",
        ).mean(dim=-1)
        return self._select_loss(preserve_loss.to(final_hidden_states.dtype), valid, zero)

    def _alignment_trajectory_loss(self, alignment_scores, alignment_valid, task_ids):
        if alignment_scores is None or task_ids is None or alignment_scores.shape[0] < 2:
            device = task_ids.device if task_ids is not None else self.showo.lm_head.weight.device
            return torch.zeros((), device=device)

        zero = alignment_scores.new_zeros(())
        valid = alignment_valid if alignment_valid is not None else torch.ones_like(task_ids, dtype=torch.bool)
        understanding_mask = task_ids.eq(AdaCycleFork.TASK_UNDERSTANDING) & valid
        generation_mask = task_ids.eq(AdaCycleFork.TASK_GENERATION) & valid

        understanding_penalty = F.relu(alignment_scores[:-1] - alignment_scores[1:])
        understanding_loss = self._select_loss(understanding_penalty.transpose(0, 1).mean(dim=1), understanding_mask, zero)

        num_layers = alignment_scores.shape[0]
        peak_layer = getattr(self.showo.config, "adacycle_generation_peak_layer", None)
        if peak_layer is None:
            peak_layer = max(0, (num_layers - 1) // 2)
        peak_layer = int(max(0, min(num_layers - 1, peak_layer)))

        generation_losses = []
        if peak_layer > 0:
            generation_losses.append(F.relu(alignment_scores[:peak_layer] - alignment_scores[1:peak_layer + 1]))
        if peak_layer < num_layers - 1:
            generation_losses.append(F.relu(alignment_scores[peak_layer + 1:] - alignment_scores[peak_layer:-1]))
        if len(generation_losses) == 0:
            generation_loss = zero
        else:
            generation_penalty = torch.cat(generation_losses, dim=0)
            generation_loss = self._select_loss(generation_penalty.transpose(0, 1).mean(dim=1), generation_mask, zero)

        return understanding_loss + generation_loss

    def compute_adacycle_losses(self):
        if not self._adacycle_enabled():
            zero = self.showo.lm_head.weight.new_zeros(())
            return {
                "loss_align": zero,
                "loss_g2u": zero,
                "loss_u2g": zero,
                "loss_preserve": zero,
            }

        aux_outputs = getattr(self.showo.model, "_last_adacycle_outputs", None)
        if aux_outputs is None:
            zero = self.showo.lm_head.weight.new_zeros(())
            return {
                "loss_align": zero,
                "loss_g2u": zero,
                "loss_u2g": zero,
                "loss_preserve": zero,
            }

        final_hidden_states = aux_outputs["final_hidden_states"]
        token_type_ids = aux_outputs["token_type_ids"]
        task_ids = aux_outputs["task_ids"]
        return {
            "loss_align": self._alignment_trajectory_loss(
                aux_outputs["alignment_scores"],
                aux_outputs["alignment_valid"],
                task_ids,
            ),
            "loss_g2u": self._semantic_cycle_proxy_loss(
                final_hidden_states,
                token_type_ids,
                task_ids,
                AdaCycleFork.TASK_GENERATION,
            ),
            "loss_u2g": self._semantic_cycle_proxy_loss(
                final_hidden_states,
                token_type_ids,
                task_ids,
                AdaCycleFork.TASK_UNDERSTANDING,
            ),
            "loss_preserve": self._visual_preserve_proxy_loss(
                aux_outputs["input_hidden_states"],
                final_hidden_states,
                token_type_ids,
                task_ids,
            ),
        }

    def forward(
            self,
            input_ids,
            input_embeddings=None,
            attention_mask=None,
            labels=None,
            label_smoothing=0.0,
            batch_size_t2i=0,
            batch_size_lm=0,
            batch_size_mmu=0,
            max_seq_length=128,
            labels_mask_text=None,
            labels_mask_image=None,
            **kwargs,
    ):
        return_adacycle_losses = kwargs.pop("return_adacycle_losses", False)
        adacycle_task_ids = kwargs.pop("adacycle_task_ids", None)
        adacycle_token_type_ids = kwargs.pop("adacycle_token_type_ids", None)

        if self._adacycle_enabled():
            adacycle_batch_size = input_embeddings.shape[0] if input_embeddings is not None else input_ids.shape[0]
            adacycle_seq_length = input_embeddings.shape[1] if input_embeddings is not None else input_ids.shape[1]
            if adacycle_task_ids is None:
                adacycle_task_ids = self._infer_adacycle_task_ids(
                    input_ids,
                    batch_size_t2i=batch_size_t2i,
                    batch_size_lm=batch_size_lm,
                    batch_size_mmu=batch_size_mmu,
                    total_batch_size=adacycle_batch_size,
                )
            if adacycle_token_type_ids is None:
                adacycle_token_type_ids = self._infer_adacycle_token_type_ids(
                    input_ids,
                    total_batch_size=adacycle_batch_size,
                    seq_length=adacycle_seq_length,
                )

        if input_embeddings is None:
            logits = self.showo(
                input_ids=input_ids,
                attention_mask=attention_mask,
                adacycle_task_ids=adacycle_task_ids,
                adacycle_token_type_ids=adacycle_token_type_ids,
            )['logits']
        else:
            logits = self.showo(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
                adacycle_task_ids=adacycle_task_ids,
                adacycle_token_type_ids=adacycle_token_type_ids,
            )['logits']

        if labels is not None:
            # 1. Mask token prediction (discrete diffusion) for image generation
            # Note that, max_seq_length indicates the maximum number of text tokens, maybe a bit confused.
            loss_t2i = F.cross_entropy(
                logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, self.output_size),
                labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
            )

            # 2. Next token prediction for language modeling
            loss_lm = F.cross_entropy(
                logits[batch_size_t2i:batch_size_t2i + batch_size_lm, :-1].contiguous().view(-1, self.output_size),
                labels[batch_size_t2i:batch_size_t2i + batch_size_lm, 1:].contiguous().view(-1), ignore_index=-100,
            )

            # 3. Next token prediction for captioning/multimodal understanding
            loss_mmu = F.cross_entropy(
                logits[-batch_size_mmu:, :-1].contiguous().view(-1, self.output_size),
                labels[-batch_size_mmu:, 1:].contiguous().view(-1), ignore_index=-100,
            )

            if return_adacycle_losses:
                return logits, loss_t2i, loss_lm, loss_mmu, self.compute_adacycle_losses()

            return logits, loss_t2i, loss_lm, loss_mmu

        return logits

    def t2i_generate(
            self,
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            temperature=1.0,
            timesteps=18,  # ideal number of steps is 18 in maskgit paper
            guidance_scale=0,
            noise_schedule=cosine_schedule,
            generator: torch.Generator = None,
            config=None,
            **kwargs,
    ):
        """
        Generate 1:1 similar to the original MaskGit repo
        https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
        """
        # begin with all image token ids masked
        mask_token_id = self.config.mask_token_id
        num_vq_tokens = config.model.showo.num_vq_tokens
        num_new_special_tokens = config.model.showo.num_new_special_tokens

        input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()
        input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id,
                                                    mask_token_id,
                                                    input_ids_minus_lm_vocab_size - config.model.showo.llm_vocab_size - num_new_special_tokens)

        # for classifier-free guidance
        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, :config.dataset.preprocessing.max_seq_length + 1]

        for step in range(timesteps):
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, input_ids[:, config.dataset.preprocessing.max_seq_length + 1:]], dim=1)
                model_input = torch.cat([input_ids, uncond_input_ids])
                cond_logits, uncond_logits = self(model_input, attention_mask=attention_mask).chunk(2)
                # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                # it seems that muse has a different cfg setting
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[:, -(num_vq_tokens + 1):-1, config.model.showo.llm_vocab_size + num_new_special_tokens:-1]
            else:
                logits = self(input_ids, attention_mask=attention_mask)
                logits = logits[:, -(num_vq_tokens + 1):-1, config.model.showo.llm_vocab_size + num_new_special_tokens:-1]

            probs = logits.softmax(dim=-1)
            sampled = probs.reshape(-1, logits.size(-1))
            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1])

            unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
            # Defines the mask ratio for the next round. The number to mask out is
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))
            # Computes the probabilities of each selected tokens.
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
            selected_probs = selected_probs.squeeze(-1)

            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
            # Gets mask lens for each sample in the batch according to the mask ratio.
            mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            mask_len = torch.max(
                torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )
            # Adds noise for randomness
            temperature = temperature * (1.0 - ratio)
            masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
            # Masks tokens with lower confidence.
            input_ids[:, -(num_vq_tokens + 1):-1] = torch.where(masking, mask_token_id,
                                                          sampled_ids + config.model.showo.llm_vocab_size
                                                          + num_new_special_tokens)
            input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)

        return sampled_ids

    @torch.no_grad()
    def mmu_generate(self, idx=None, input_embeddings=None, attention_mask=None, max_new_tokens=100, temperature=1.0, top_k=None, eot_token=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        try:
            device = idx.device
        except:
            device = input_embeddings.device

        result = []
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            # idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            # logits, _ = self(idx_cond)
            logits = self(idx, input_embeddings=input_embeddings, attention_mask=attention_mask)

            L = attention_mask.shape[-1]
            attention_mask = attention_mask.squeeze()
            attention_mask_a = torch.hstack(
                [
                    attention_mask,  # L, L
                    torch.zeros((L, 1)).to(device) + torch.finfo(logits.dtype).min,
                ]
            )
            attention_mask_b = torch.vstack(
                [
                    attention_mask_a,  # L, L+1
                    torch.hstack([attention_mask[-1, :], torch.tensor([0]).to(device)]).unsqueeze(0),
                ]
            )
            attention_mask = attention_mask_b

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            result.append(idx_next[0][0])
            # append sampled index to the running sequence and continue
            if self.config.w_clip_vit:
                idx_next_embeddings = self.showo.model.embed_tokens(idx_next)
                input_embeddings = torch.cat([input_embeddings, idx_next_embeddings], dim=1)
            else:
                idx = torch.cat((idx, idx_next), dim=1)

            if eot_token is not None and idx_next.cpu() == eot_token:
                break

        return result
