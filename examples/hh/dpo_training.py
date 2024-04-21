# flake8: noqa
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""
# peft:
python examples/scripts/dpo.py \
    --dataset_name=trl-internal-testing/hh-rlhf-trl-style \
    --model_name_or_path=gpt2 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="dpo_anthropic_hh" \
    --optim rmsprop \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=16
"""
import logging
import multiprocessing
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import sys
sys.path.append('/home/lijiang/trl')
from contextlib import nullcontext
import wandb
import pandas as pd

TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)

from trl.commands.cli_utils import DpoScriptArguments, init_zero_verbose, TrlParser

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasets import load_dataset
from dataclasses import dataclass, field

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainingArguments,
)
from trl import (
    DPOTrainer,
    ModelConfig,
    RichProgressCallback,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import pad_to_length


from examples.datasets.anthropic_hh import extract_dialogue


if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)

@dataclass
class PromptCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_prompt_length: Optional[int] = None
    prompt_field: str = "prompt"
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompts = [feat[self.prompt_field] for feat in features]

        original_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        tokenized_batch = self.tokenizer(
            prompts,
            truncation=True,
            padding=True,
            max_length=self.max_prompt_length,
            return_tensors=self.return_tensors,
        )
        tokenized_batch["prompt"] = prompts

        self.tokenizer.padding_side = original_side

        return tokenized_batch


class GoldModelRewardCallback(TrainerCallback):
    def __init__(
        self,
        args,
        gold_model,
        gold_eval_dataset,
        tokenizer,
        accelerator,
        max_length,
        max_prompt_length,
        log_n_samples_during_eval=0,
        generation_config=None,
    ):
        self.max_length = max_length
        self.log_n_samples_during_eval = log_n_samples_during_eval
        self.generation_config = generation_config

        # data_collator = DataCollatorWithPadding(tokenizer)
        data_collator = PromptCollator(
            tokenizer,
            max_prompt_length=max_prompt_length,
            prompt_field="prompt",
        )
        dataloader_params = {
            "batch_size": args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": args.dataloader_num_workers,
            "pin_memory": args.dataloader_pin_memory,
        }
        dataloader = DataLoader(gold_eval_dataset, **dataloader_params)
        self.gold_model, self.dataloader = accelerator.prepare(gold_model, dataloader)
        self.accelerator = accelerator

    def on_evaluate(self, args, state, control, model, tokenizer, metrics, **kwargs):
        samples_to_log = []
        gold_reward_sum = 0.0
        total_samples = 0

        for inputs in tqdm(
            self.dataloader, desc="Gold Eval", dynamic_ncols=True, disable=not state.is_local_process_zero
        ):
            policy_output_decoded, ref_output_decoded, policy_output_ids = self.get_batch_samples(
                model,
                tokenizer,
                inputs["input_ids"],
                inputs["attention_mask"],
                return_ids=True,
            )

            # gold reward
            policy_output_attention_mask = (policy_output_ids != tokenizer.pad_token_id).to(torch.int64)
            with torch.no_grad():
                gold_rewards = self.gold_model(
                    input_ids=policy_output_ids, attention_mask=policy_output_attention_mask
                )[0]

            gold_rewards = self.accelerator.gather_for_metrics(gold_rewards)

            if state.is_local_process_zero:
                gold_reward_sum += gold_rewards.sum().item()
                total_samples += gold_rewards.size(0)

                # Sample and save to game log if requested (for one batch to save time)
                for i, (prompt, pol, ref) in enumerate(
                    zip(inputs["prompt"], policy_output_decoded, ref_output_decoded)
                ):
                    if len(samples_to_log) < self.log_n_samples_during_eval:
                        samples_to_log.append([prompt, pol[len(prompt) :], ref[len(prompt) :]])
                    else:
                        break

        if state.is_world_process_zero:
            print(f"gold reward mean {gold_reward_sum / total_samples}")
            gold_log = {
                "eval/gold_rewards_mean": gold_reward_sum / total_samples,
            }
            if state.epoch:
                gold_log["epoch"] = round(state.epoch, 2)
                gold_log["step"] = state.global_step
            if samples_to_log:
                gold_log["eval/game_log"] = (
                    wandb.Table(
                        columns=["Prompt", "Policy", "Ref Model"],
                        rows=samples_to_log,
                    ),
                )
                # if isinstance(state.epoch, int) or (isinstance(state.epoch, float) and state.epoch.is_integer()):
                df = pd.DataFrame(samples_to_log, columns=["Prompt", "Policy", "Ref Model"])
                csv_file_path = f"results/dpo/game_log_epoch_{state.epoch}.csv"
                if not os.path.exists(os.path.dirname(csv_file_path)):
                    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
                df.to_csv(csv_file_path, index=False)
                print(f"save log to {csv_file_path}")

                wandb.log(gold_log)

    def get_batch_samples(self, model, tokenizer, input_ids, attention_mask, return_ids=False) -> Tuple[str, str]:
        """Reduce inputs to unseen prompts, and maximum batch size if necessary
        Generate samples from the model and reference model for the given batch of inputs."""
        policy_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=self.generation_config,
        )

        # if self.ref_model is None:
        try:
            with self.accelerator.unwrap_model(model).disable_adapter():
                reference_output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=self.generation_config,
                )
        except:
            print("on eval: ref_model == model")
            reference_output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=self.generation_config,
                )
        # else:
        #     reference_output = self.ref_model.generate(
        #         **inputs,
        #         generation_config=self.generation_config,
        #     )

        policy_output = pad_to_length(policy_output, self.max_length, tokenizer.pad_token_id)
        policy_output_decoded = tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        reference_output = pad_to_length(reference_output, self.max_length, tokenizer.pad_token_id)
        reference_output_decoded = tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        if return_ids:
            return policy_output_decoded, reference_output_decoded, policy_output
        else:
            return policy_output_decoded, reference_output_decoded


if __name__ == "__main__":
    parser = TrlParser((DpoScriptArguments, TrainingArguments, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()

    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    # TODO
    # gold_model = AutoModelForCausalLM.from_pretrained(model_config.gold_model_name_or_path, **model_kwargs)
    peft_config = get_peft_config(model_config)
    if peft_config is None:
        model_ref = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    else:
        model_ref = None
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n\n'}}{% endfor %}{{ eos_token }}"
    if args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the DPOTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Dataset
    ################
    ds = load_dataset(args.dataset_name)
    if args.sanity_check:
        for key in ds:
            ds[key] = ds[key].select(range(50))

    def process(row):
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        return row

    ds = ds.map(
        process,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )
    train_dataset = ds["train"]
    eval_dataset = ds["test"]

    ################
    # Training
    ################
    with init_context:
        trainer = DPOTrainer(
            model,
            model_ref,
            args=training_args,
            beta=args.beta,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            max_length=args.max_length,
            max_target_length=args.max_target_length,
            max_prompt_length=args.max_prompt_length,
            generate_during_eval=args.generate_during_eval,
            peft_config=get_peft_config(model_config),
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
        )

    # Gold Eval
    """
    gold_eval_dataset = load_dataset(
        args.gold_dataset_name,
        split=args.gold_eval_split,
    )

    def strip_prompt(examples):
        examples["prompt"] = [prompt.strip() for prompt in examples["prompt"]]

        return examples

    gold_eval_dataset = gold_eval_dataset.map(strip_prompt, batched=True)

    generation_config = GenerationConfig(
            max_new_tokens=args.max_target_length,
            do_sample=False,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    gold_eval_callback = GoldModelRewardCallback(
        training_args,
        gold_model,
        gold_eval_dataset,
        tokenizer,
        trainer.accelerator,
        args.max_length,
        args.max_prompt_length,
        script_args.log_n_samples_during_eval,
        generation_config,
    )

    trainer.add_callback(gold_eval_callback)
    """

    trainer.train()

    with save_context:
        trainer.save_model(training_args.output_dir)
