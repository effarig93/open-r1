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

from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
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
        teacher_device: Optional[torch.device | str] = None,
        vllm_engine: Optional[Any] = None,
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
        self.teacher_device = torch.device(teacher_device) if teacher_device is not None else None
        if self.teacher_device is not None:
            self.teacher_model.to(self.teacher_device)
        self.vllm_engine = vllm_engine

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
        if "eos_token_id" not in kwargs and getattr(self.tokenizer, "eos_token_id", None) is not None:
            kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        return kwargs

    def _prepare_prompt_inputs(self, prompts: Sequence[str], model: nn.Module) -> Dict[str, torch.Tensor]:
        tokenizer_kwargs = {
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
        }
        if getattr(self.args, "max_seq_length", None) is not None:
            tokenizer_kwargs["max_length"] = self.args.max_seq_length

        prompt_inputs = self.tokenizer(prompts, **tokenizer_kwargs)
        return {k: v.to(model.device) for k, v in prompt_inputs.items()}

    def _generate_with_vllm(
        self,
        prompts: Sequence[str],
        prompt_inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        if self.vllm_engine is None:
            raise RuntimeError("vLLM engine is not configured for this trainer instance.")

        try:
            from vllm import SamplingParams
        except ImportError as exc:  # pragma: no cover - defensive, exercised in runtime usage
            raise RuntimeError("vLLM is required for on-policy generation but is not installed.") from exc

        stop_token_ids = generation_kwargs.get("eos_token_id")
        if stop_token_ids is not None and not isinstance(stop_token_ids, (list, tuple)):
            stop_token_ids = [stop_token_ids]

        sampling_params = SamplingParams(
            temperature=generation_kwargs.get("temperature", 1.0),
            top_p=generation_kwargs.get("top_p", 1.0),
            top_k=generation_kwargs.get("top_k"),
            max_tokens=generation_kwargs.get("max_new_tokens"),
            n=1,
            stop_token_ids=stop_token_ids,
        )

        request_outputs = self.vllm_engine.generate(list(prompts), sampling_params=sampling_params)

        pad_token_id = generation_kwargs["pad_token_id"]
        prompt_attention = prompt_inputs["attention_mask"]
        prompt_lengths = prompt_attention.sum(dim=-1)
        input_ids = prompt_inputs["input_ids"]

        sequences: List[torch.Tensor] = []
        for idx, output in enumerate(request_outputs):
            if not getattr(output, "outputs", None):
                completion_ids = torch.empty(0, dtype=torch.long, device=device)
            else:
                token_ids = output.outputs[0].token_ids
                completion_ids = torch.tensor(token_ids, dtype=torch.long, device=device)

            prompt_len = int(prompt_lengths[idx].item())
            prompt_tokens = input_ids[idx, :prompt_len].to(device)
            sequence = torch.cat([prompt_tokens, completion_ids], dim=0)
            sequences.append(sequence)

        if not sequences:
            raise ValueError("vLLM generation produced no sequences.")

        generated_sequences = pad_sequence(sequences, batch_first=True, padding_value=pad_token_id)
        return generated_sequences

    def compute_loss(
        self, model: nn.Module, inputs: Dict[str, Any], return_outputs: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, Any]]:
        if self.prompt_column not in inputs:
            raise ValueError(f"Missing '{self.prompt_column}' column in batch inputs.")

        prompts: List[str] = inputs[self.prompt_column]
        if isinstance(prompts, str):
            prompts = [prompts]

        prompt_inputs = self._prepare_prompt_inputs(prompts, model)
        generation_kwargs = self._get_generation_kwargs()
        generation_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        generation_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        if generation_kwargs.get("pad_token_id") is None:
            raise ValueError("Tokenizer must define `pad_token_id` for on-policy distillation.")

        if self.vllm_engine is not None:
            generated_sequences = self._generate_with_vllm(
                prompts,
                prompt_inputs,
                generation_kwargs,
                model.device,
            )
        else:
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

        teacher_device = self.teacher_device if self.teacher_device is not None else model.device
        teacher_inputs = {
            "input_ids": generated_sequences.to(teacher_device),
            "attention_mask": attention_mask.to(teacher_device),
            "use_cache": False,
        }
        with torch.no_grad():
            teacher_logits = self.teacher_model(**teacher_inputs).logits
        if teacher_device != model.device:
            teacher_logits = teacher_logits.to(model.device)

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
