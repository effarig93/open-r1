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

import logging
import os
import sys
from typing import Dict, List, Optional, Sequence

import datasets
import transformers
from datasets import DatasetDict
from transformers import AutoModelForCausalLM, set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import OnPolicyDistillScriptArguments, SFTConfig
from open_r1.on_policy_distill_trainer import OnPolicyDistillTrainer
from open_r1.utils import get_dataset, get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import ModelConfig, TrlParser

import torch


logger = logging.getLogger(__name__)


def _validate_dataset_columns(dataset: DatasetDict, prompt_column: str) -> None:
    for split, split_dataset in dataset.items():
        if prompt_column not in split_dataset.column_names:
            raise ValueError(
                f"Dataset split '{split}' does not contain the prompt column '{prompt_column}'."
            )


def _build_data_collator(prompt_column: str):
    def collate(features: List[Dict[str, str]]) -> Dict[str, List[str]]:
        return {prompt_column: [feature[prompt_column] for feature in features]}

    return collate


def _normalize_device_spec(device: str) -> str:
    normalized = str(device).strip()
    if not normalized:
        raise ValueError("Empty device string provided in `student_devices`.")

    if normalized.isdigit():
        return f"cuda:{normalized}"

    if ":" not in normalized and not normalized.startswith("cpu"):
        return f"cuda:{normalized}"

    return normalized


def _place_student_model(
    model: torch.nn.Module,
    devices: Sequence[str],
    logger: logging.Logger,
) -> torch.nn.Module:
    normalized_devices = [_normalize_device_spec(device) for device in devices]

    if len(normalized_devices) == 1:
        student_device = torch.device(normalized_devices[0])
        logger.info("Placing student model on %s", student_device)
        model.to(student_device)
        return model

    try:
        from accelerate import dispatch_model
        from accelerate.utils import infer_auto_device_map
    except ImportError as exc:  # pragma: no cover - accelerate is a training dependency
        raise ImportError(
            "Accelerate is required for multi-GPU student placement. "
            "Please install accelerate or provide a single `--student_device`."
        ) from exc

    device_set = list(dict.fromkeys(normalized_devices))
    logger.info("Dispatching student model across devices: %s", ", ".join(device_set))

    try:
        dtype = next(model.parameters()).dtype
    except StopIteration:
        dtype = None

    no_split = getattr(model, "_no_split_modules", None) or getattr(
        getattr(model, "config", None), "no_split_module_classes", None
    )

    try:
        device_map = infer_auto_device_map(
            model,
            max_memory={device: "auto" for device in device_set},
            dtype=dtype,
            no_split_module_classes=no_split,
        )
    except Exception as exc:  # pragma: no cover - relies on accelerate internals
        raise RuntimeError(
            "Failed to infer a device map for the requested student devices."
        ) from exc

    model = dispatch_model(model, device_map=device_map)
    logger.info("Student device map: %s", device_map)
    return model


def main(script_args, training_args, model_args):
    set_seed(training_args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    training_args.remove_unused_columns = False

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    dataset = get_dataset(script_args)
    _validate_dataset_columns(dataset, script_args.dataset_prompt_column)

    tokenizer = get_tokenizer(model_args, training_args)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    student_model = get_model(model_args, training_args)
    if script_args.student_devices:
        student_model = _place_student_model(student_model, script_args.student_devices, logger)
    elif script_args.student_device is not None:
        student_model = _place_student_model(student_model, [script_args.student_device], logger)

    teacher_kwargs = {
        "revision": script_args.teacher_revision,
        "trust_remote_code": model_args.trust_remote_code,
        "attn_implementation": model_args.attn_implementation,
    }
    try:
        student_param_dtype = next(student_model.parameters()).dtype
    except StopIteration:
        student_param_dtype = None
    if student_param_dtype is not None and getattr(student_param_dtype, "is_floating_point", False):
        teacher_kwargs["torch_dtype"] = student_param_dtype

    logger.info("*** Loading teacher model ***")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        script_args.teacher_model_name_or_path,
        **teacher_kwargs,
    ).eval()
    teacher_model.requires_grad_(False)

    teacher_device: Optional[torch.device] = None
    if script_args.teacher_device is not None:
        teacher_device = torch.device(script_args.teacher_device)
        logger.info("Placing teacher model on %s", teacher_device)
        teacher_model.to(teacher_device)

    generation_kwargs = {
        "max_new_tokens": script_args.generation_max_new_tokens,
        "temperature": script_args.generation_temperature,
        "top_p": script_args.generation_top_p,
        "do_sample": script_args.generation_do_sample,
    }
    if script_args.generation_top_k is not None:
        generation_kwargs["top_k"] = script_args.generation_top_k

    data_collator = _build_data_collator(script_args.dataset_prompt_column)

    vllm_engine = None
    if script_args.use_vllm_generation:
        try:
            from vllm import LLM
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "vLLM must be installed to enable --use_vllm_generation."
            ) from exc

        vllm_model_name = script_args.vllm_model_name_or_path or model_args.model_name_or_path
        vllm_kwargs = {
            "revision": script_args.vllm_revision or model_args.model_revision,
            "trust_remote_code": model_args.trust_remote_code,
            "tensor_parallel_size": script_args.vllm_tensor_parallel_size,
            "gpu_memory_utilization": script_args.vllm_gpu_memory_utilization,
        }
        if script_args.vllm_max_model_len is not None:
            vllm_kwargs["max_model_len"] = script_args.vllm_max_model_len
        if script_args.vllm_dtype is not None:
            vllm_kwargs["dtype"] = script_args.vllm_dtype

        logger.info("Initializing vLLM engine with model %s", vllm_model_name)
        vllm_engine = LLM(model=vllm_model_name, **vllm_kwargs)

    trainer = OnPolicyDistillTrainer(
        model=student_model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None),
        tokenizer=tokenizer,
        callbacks=get_callbacks(training_args, model_args),
        data_collator=data_collator,
        teacher_model=teacher_model,
        generation_kwargs=generation_kwargs,
        prompt_column=script_args.dataset_prompt_column,
        kl_coef=script_args.kl_coef,
        teacher_device=teacher_device,
        vllm_engine=vllm_engine,
    )

    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Save model ***")
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((OnPolicyDistillScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
