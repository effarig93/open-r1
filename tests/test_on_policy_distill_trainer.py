from types import SimpleNamespace

import pytest

datasets = pytest.importorskip("datasets")
tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, TrainingArguments

from open_r1.on_policy_distill_trainer import OnPolicyDistillTrainer


torch = pytest.importorskip("torch")


def build_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {
        "<pad>": 0,
        "<eos>": 1,
        "<unk>": 2,
        "Hello": 3,
        "world": 4,
        "Test": 5,
        "prompt": 6,
        "Another": 7,
    }
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<eos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    tokenizer.model_max_length = 32
    return tokenizer


def build_model(vocab_size: int) -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=64,
        n_layer=2,
        n_head=2,
        n_embd=16,
        bos_token_id=1,
        eos_token_id=1,
        pad_token_id=0,
    )
    return GPT2LMHeadModel(config)


def prompt_collator(features):
    return {"prompt": [feature["prompt"] for feature in features]}


def test_training_step_runs_end_to_end(tmp_path):
    torch.manual_seed(0)
    tokenizer = build_tokenizer()
    student_model = build_model(len(tokenizer))
    teacher_model = build_model(len(tokenizer))
    teacher_model.requires_grad_(False)
    teacher_model.eval()

    dataset = Dataset.from_dict({"prompt": ["Hello world", "Test prompt", "Another prompt"]})

    training_args = TrainingArguments(
        output_dir=str(tmp_path / "outputs"),
        per_device_train_batch_size=2,
        num_train_epochs=1,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = OnPolicyDistillTrainer(
        model=student_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=prompt_collator,
        teacher_model=teacher_model,
        generation_kwargs={
            "max_new_tokens": 4,
            "temperature": 1.0,
            "top_p": 1.0,
            "do_sample": True,
        },
        prompt_column="prompt",
    )

    batch = {"prompt": ["Hello world", "Test prompt"]}
    loss, outputs = trainer.compute_loss(trainer.model, batch, return_outputs=True)

    assert torch.isfinite(loss.detach()).item() == 1
    assert trainer._debug_last_completion_mask is not None
    assert trainer._debug_last_prompt_lengths is not None

    for mask_row, prompt_len in zip(trainer._debug_last_completion_mask, trainer._debug_last_prompt_lengths):
        cutoff = max(int(prompt_len.item()) - 1, 0)
        if cutoff > 0:
            assert mask_row[:cutoff].sum().item() == 0

    assert outputs["student_logits"].shape[0] == len(batch["prompt"])
    assert outputs["teacher_logits"].shape == outputs["student_logits"].shape


class DummyVLLMEngine:
    def __init__(self, completions):
        self._completions = completions

    def generate(self, prompts, sampling_params):  # noqa: D401 - vLLM compatibility shim
        outputs = []
        for idx, token_ids in enumerate(self._completions):
            completion = SimpleNamespace(token_ids=token_ids, text="")
            outputs.append(SimpleNamespace(outputs=[completion], request_id=str(idx)))
        return outputs


def test_training_step_with_vllm_engine(tmp_path):
    torch.manual_seed(0)
    tokenizer = build_tokenizer()
    student_model = build_model(len(tokenizer))
    teacher_model = build_model(len(tokenizer))
    teacher_model.requires_grad_(False)
    teacher_model.eval()

    dataset = Dataset.from_dict({"prompt": ["Hello world"]})

    training_args = TrainingArguments(
        output_dir=str(tmp_path / "outputs_vllm"),
        per_device_train_batch_size=1,
        num_train_epochs=1,
        report_to=[],
        remove_unused_columns=False,
    )

    dummy_engine = DummyVLLMEngine([[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("world")]])

    trainer = OnPolicyDistillTrainer(
        model=student_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=prompt_collator,
        teacher_model=teacher_model,
        generation_kwargs={
            "max_new_tokens": 4,
            "temperature": 1.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        },
        prompt_column="prompt",
        teacher_device=torch.device("cpu"),
        vllm_engine=dummy_engine,
    )

    batch = {"prompt": ["Hello world"]}
    loss, outputs = trainer.compute_loss(trainer.model, batch, return_outputs=True)

    assert torch.isfinite(loss.detach()).item() == 1
    generated_sequences = outputs["generated_sequences"]
    assert generated_sequences.shape[0] == 1
    prompt_len = trainer._debug_last_prompt_lengths[0].int().item()
    assert prompt_len > 0
    # Ensure prompt tokens remain unchanged at the beginning of the sequence
    prompt_tokens = trainer.tokenizer(batch["prompt"], return_tensors="pt")["input_ids"][0, :prompt_len]
    torch.testing.assert_close(generated_sequences[0, :prompt_len].cpu(), prompt_tokens)
