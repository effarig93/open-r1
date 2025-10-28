# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Trainer


class OnPolicyDistillTrainer(Trainer):
    """Trainer implementing on-policy distillation via reverse KL."""

    def __init__(
        self,
        *args,
        teacher_model: nn.Module,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        prompt_column: str = "prompt",
        kl_coef: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if teacher_model is None:
            raise ValueError("`teacher_model` must be provided for OnPolicyDistillTrainer.")

        self.teacher_model = teacher_model.eval()
        self.teacher_model.requires_grad_(False)
        self.prompt_column = prompt_column
        self.kl_coef = kl_coef
        self.generation_kwargs = generation_kwargs or {}
        self._debug_last_completion_mask: Optional[torch.Tensor] = None
        self._debug_last_prompt_lengths: Optional[torch.Tensor] = None

    def _get_generation_kwargs(self) -> Dict[str, Any]:
        kwargs = dict(self.generation_kwargs)
        if "max_new_tokens" not in kwargs:
            kwargs["max_new_tokens"] = getattr(self.args, "generation_max_new_tokens", None)
        if "temperature" not in kwargs:
            kwargs["temperature"] = getattr(self.args, "generation_temperature", 1.0)
        if "top_p" not in kwargs:
            kwargs["top_p"] = getattr(self.args, "generation_top_p", 1.0)
        if "top_k" not in kwargs and getattr(self.args, "generation_top_k", None) is not None:
            kwargs["top_k"] = self.args.generation_top_k
        if "do_sample" not in kwargs:
            kwargs["do_sample"] = getattr(self.args, "generation_do_sample", True)
        kwargs.setdefault("return_dict_in_generate", False)
        kwargs.setdefault("use_cache", True)
        return kwargs

    def compute_loss(
        self, model: nn.Module, inputs: Dict[str, Any], return_outputs: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, Any]]:
        if self.prompt_column not in inputs:
            raise ValueError(f"Missing '{self.prompt_column}' column in batch inputs.")

        prompts: List[str] = inputs[self.prompt_column]
        if isinstance(prompts, str):
            prompts = [prompts]

        tokenizer_kwargs = {
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
        }
        if getattr(self.args, "max_seq_length", None) is not None:
            tokenizer_kwargs["max_length"] = self.args.max_seq_length

        prompt_inputs = self.tokenizer(prompts, **tokenizer_kwargs)
        prompt_inputs = {k: v.to(model.device) for k, v in prompt_inputs.items()}

        generation_kwargs = self._get_generation_kwargs()
        generation_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        generation_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        if generation_kwargs.get("pad_token_id") is None:
            raise ValueError("Tokenizer must define `pad_token_id` for on-policy distillation.")

        was_training = model.training
        model.eval()
        with torch.no_grad():
            generated_sequences = model.generate(**prompt_inputs, **generation_kwargs)
        if was_training:
            model.train()

        attention_mask = (generated_sequences != generation_kwargs["pad_token_id"]).long()
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=-1)

        student_outputs = model(
            input_ids=generated_sequences,
            attention_mask=attention_mask,
            use_cache=False,
        )
        student_logits = student_outputs.logits

        if self.teacher_model.device != model.device:
            self.teacher_model.to(model.device)
        with torch.no_grad():
            teacher_logits = self.teacher_model(
                input_ids=generated_sequences,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits

        student_log_probs = F.log_softmax(student_logits[:, :-1, :], dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits[:, :-1, :], dim=-1)
        token_kl = torch.exp(student_log_probs) * (student_log_probs - teacher_log_probs)
        token_kl = token_kl.sum(dim=-1)

        completion_mask = attention_mask[:, 1:].float()
        prompt_offsets = (prompt_lengths - 1).clamp(min=0).unsqueeze(-1)
        positions = torch.arange(completion_mask.size(-1), device=completion_mask.device).unsqueeze(0)
        completion_mask = completion_mask * (positions >= prompt_offsets)

        kl_sum = (token_kl * completion_mask).sum()
        denom = completion_mask.sum().clamp_min(1.0)
        mean_kl = kl_sum / denom
        loss = mean_kl if self.kl_coef is None else mean_kl * self.kl_coef

        completion_lengths = completion_mask.sum(dim=-1)
        self.log(
            {
                "train/mean_kl": mean_kl.detach().float().item(),
                "train/generation_length": completion_lengths.detach().float().mean().item(),
            }
        )

        self._debug_last_completion_mask = completion_mask.detach().cpu()
        self._debug_last_prompt_lengths = prompt_lengths.detach().cpu()

        outputs = {
            "student_logits": student_logits.detach(),
            "teacher_logits": teacher_logits.detach(),
            "generated_sequences": generated_sequences.detach(),
        }

        return (loss, outputs) if return_outputs else loss
