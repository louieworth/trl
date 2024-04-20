# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass, field
from typing import List, Optional
import bitsandbytes as bnb
import wandb
import pandas as pd
import os
from datetime import datetime

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from peft.tuners.lora import LoraLayer
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    GenerationConfig,
    pipeline,
)

# from transformers.trainer_utils import get_last_checkpoint
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, CPOBaseTrainer, set_seed
from trl.core import LengthSampler
from trl.models.modeling_value_adapter import AutoModelForCausalLMWithValueAdapter
from trl.trainer.utils import CPODataCollatorWithPadding

# import copy
# from torch_ema import ExponentialMovingAverage
# from transforers import pipeline

tqdm.pandas()


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """
    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    reward_adapter_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    # tokenizer_name: Optional[str] = field(default=None, metadata={"help": "the tokenizer name"})
    dataset_name: Optional[str] = field(
        default="CarperAI/openai_summarize_tldr", metadata={"help": "the dataset name"}
    )
    train_split: Optional[str] = field(
        default="train", metadata={"help": "the dataset split to evaluate on; default to 'none' (no evaluation)"}
    )
    eval_split: Optional[str] = field(
        default="test[:5000]", metadata={"help": "the dataset split to evaluate on; default to 'none' (no evaluation)"}
    )
    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    log_interval: Optional[int] = field(default=500, metadata={"help": "log interval"})
    
    # training parameters
    optimizer_type: Optional[str] = field(default="adamw_torch", metadata={"help": "the optimizer type"})
    warmup_steps: Optional[int] = field(default=150)
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    learning_rate: Optional[float] = field(default=1e-3, metadata={"help": "optimizer learning rate"})
    # per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "batch size per device"})
    max_length: Optional[int] = field(default=560, metadata={"help": "max length of each sample"})
    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "max length of each sample's prompt"})
    max_target_length: Optional[int] = field(
        default=48, metadata={"help": "Only used for encoder decoder model. Max target of each sample's prompt"}
    )
    log_n_samples_during_eval: Optional[int] = field(default=100)
    # num_train_epochs: Optional[int] = field(default=4, metadata={"help": "the number of training epochs"})
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "whether to use gradient checkpointing"}
    )

    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=4, metadata={"help": "the batch size"})
    epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})

    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    # reward_baseline: Optional[float] = field(
    #     default=0.0,
    #     metadata={"help": "a baseline value that is subtracted from the reward"},
    # )

    # batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    # save_epochs: Optional[int] = field(default=1, metadata={"help": "the number of steps to save at"})
    save_strategy: Optional[str] = field(default="final")
    output_dir: Optional[str] = field(default="runs/", metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})
    steps: Optional[int] = field(default=20000, metadata={"help": "number of steps"})
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )
    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})
    use_reward_model: Optional[bool] = field(default=False, metadata={"help": "Use reward model"})
    value_adapter: Optional[bool] = field(default=False)
    # separate_reward_model: Optional[str] = field(default=None, metadata={"help": "the reward model name"})

    # Generation
    output_min_length: Optional[int] = field(default=24, metadata={"help": "the batch size"})
    output_max_length: Optional[int] = field(default=48, metadata={"help": "the batch size"})
    input_max_length: Optional[int] = field(default=512, metadata={"help": "maximum length for generation"})

    # Quantization
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    bf16: Optional[bool] = field(
        default=False,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )

    # LoRA
    use_peft: Optional[bool] = field(
        default=True,
    )
    lora_alpha: Optional[float] = field(default=32, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})
    lora_all_linear: Optional[bool] = field(default=False, metadata={"help": "lora adapter on all linear layers"})

    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    # cpo
    rep_approach: Optional[str] = field(default="mean", metadata={"help": "rep approach"})
    response_only: Optional[bool] = field(default=True, metadata={"help": "n steps to save the model"})
    similarity: Optional[str] = field(default="dot", metadata={"help": "how to compute the similarty"})
    loss_type: Optional[str] = field(default="sigmoid", metadata={"help": "how to compute the loss function"})
    # Gold Model
    eval_steps: Optional[int] = field(default=None)
    gold_model_name: Optional[str] = field(default=None, metadata={"help": "the reward model name"})
    gold_in_8bit: Optional[bool] = field(default=False, metadata={"help": "gold the model in 8 bits precision"})
    gold_in_4bit: Optional[bool] = field(default=False, metadata={"help": "gold the model in 4 bits precision"})
    gold_bf16: Optional[bool] = field(default=False,)
    gold_fp16: Optional[bool] = field(default=False,)
    gold_eval_greedy: Optional[bool] = field(default=True) # generate_greedy
    # # EMA stuff
    # ema_decay: Optional[float] = field(default=0.995, metadata={"help": "the ema decay rate"})
    # reset_freq: Optional[int] = field(default=None, metadata={"help": "reset every n epochs"})

    input_ids_input: Optional[bool] = field(default=False,)
    strip_prompt: Optional[bool] = field(default=False,)
    gold_dataset_name: Optional[str] = field(
        default="CarperAI/openai_summarize_tldr", metadata={"help": "the dataset name"}
    )
    gold_eval_split: Optional[str] = field(default="valid")
    just_eval: Optional[bool] = field(default=False)


def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.load_in_4bit else (bnb.nn.Linear8bitLt if args.load_in_8bit else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")

    if "score" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("score")

    return list(lora_module_names)


def create_and_prepare_model(args):
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif args.load_in_8bit or args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit)
        device_map = {"": Accelerator().local_process_index}
    else:
        device_map = None
        quantization_config = None

    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    # TODO: future work, AutoModelForCausalLMWithRepHead, ValueHead (reward model)
    if script_args.use_reward_model:
        if script_args.value_adapter:
            model_cls = AutoModelForCausalLMWithValueAdapter
        else:
            model_cls = AutoModelForCausalLMWithValueHead
    else:
        model_cls = AutoModelForCausalLM

    model = model_cls.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=dtype,
    )

    rep_model = model_cls.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=dtype,
    )

    model.config.torch_dtype = dtype
    rep_model.config.torch_dtype = dtype
    model.config.use_cache = not script_args.gradient_checkpointing 
    rep_model.config.use_cache = not script_args.gradient_checkpointing
    # if script_args.ignore_bias_buffers:
    # torch distributed hack
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=script_args.gradient_checkpointing)
        rep_model = prepare_model_for_kbit_training(rep_model, use_gradient_checkpointing=script_args.gradient_checkpointing)
    # we add `score` to the list of modules to save to
    # correctly save the score head.
    # set target modules to be query_key_value for Pythia
    if args.lora_all_linear:
        modules = find_all_linear_names(args, model)
        rep_modules = find_all_linear_names(args, rep_model)
    else:
        modules = None
        rep_modules = None

    if args.use_peft:
        modules_to_save = ["lm_head"]
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            target_modules=modules,
            modules_to_save=modules_to_save,
        )

        rep_peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            target_modules=rep_modules,
            modules_to_save=modules_to_save,
        )

        model = get_peft_model(model, peft_config)
        rep_model = get_peft_model(rep_model, rep_peft_config)

        for key, _ in model.named_modules():
            target_module_found = any(key.endswith(target_key) for target_key in modules_to_save)
            if target_module_found:
                model.get_submodule(key + ".original_module").requires_grad_(False)
        
        for key, _ in rep_model.named_modules():
            target_module_found = any(key.endswith(target_key) for target_key in modules_to_save)
            if target_module_found:
                model.get_submodule(key + ".original_module").requires_grad_(False)

    if args.bf16:
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "score" in name or "embed_tokens" in name:
                if hasattr(module, "weight") and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
        
        for name, module in rep_model.named_modules():
            if isinstance(module, LoraLayer):
                module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "score" in name or "embed_tokens" in name:
                if hasattr(module, "weight") and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    # tokenizer_name = script_args.model_name if script_args.tokenizer_name is None else script_args.tokenizer_name
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
    # tokenizer.truncation_side = "left"
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = model.config.eos_token_id

    if script_args.gold_in_8bit or script_args.gold_in_4bit:
        gold_quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.gold_in_8bit, load_in_4bit=script_args.gold_in_4bit
        )
        device_map = {"": Accelerator().local_process_index}
    else:
        gold_device_map = None
        gold_quantization_config = None

    if script_args.gold_bf16:
        torch_dtype = torch.bfloat16
    elif script_args.gold_fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    gold_model = AutoModelForSequenceClassification.from_pretrained(
        script_args.gold_model_name,
        quantization_config=gold_quantization_config,
        torch_dtype=torch_dtype,
        device_map=gold_device_map,
    )

    if getattr(gold_model.config, "pad_token_id", None) is None:
        gold_model.config.pad_token_id = gold_model.config.eos_token_id

    if args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]
        rep_model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in rep_model.named_buffers() if buffer.dtype == torch.bool
        ]

    return model, rep_model, tokenizer, gold_model

parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
# set seed before initializing value head for deterministic eval
model, rep_model, tokenizer, gold_model = create_and_prepare_model(script_args)
config = PPOConfig(
    steps=script_args.steps,
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    ppo_epochs=script_args.epochs,
    seed=script_args.seed,
    dtype=model.config.torch_dtype,
    max_length=script_args.max_length,
    max_prompt_length=script_args.max_prompt_length,
    max_target_length=script_args.max_target_length,
    init_kl_coef=script_args.init_kl_coef,
    adap_kl_ctrl=script_args.adap_kl_ctrl,
    rep_approach=script_args.rep_approach,
    response_only=script_args.response_only,
    similarity=script_args.similarity,
    loss_type=script_args.loss_type,
    accelerator_kwargs={"kwargs_handlers": [DistributedDataParallelKwargs(find_unused_parameters=False)]},
)
set_seed(config.seed)
current_datetime = datetime.now()
current_datetime = current_datetime.strftime("%Y%m%d_%H")

dataloader = CPODataCollatorWithPadding(tokenizer, 
                                        label_pad_token_id=config.label_pad_token_id,
                                        max_length=config.max_length,
                                        max_prompt_length=config.max_prompt_length)
train_dataset = load_dataset(script_args.dataset_name, split=script_args.train_split)
train_dataset = train_dataset[:19]
eval_dataset = load_dataset(script_args.dataset_name, split=script_args.eval_split)
# train_dataset, eval_dataset = create_and_prepare_dataset(script_args)

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
cpo_trainer = CPOBaseTrainer(
    config,
    model,
    rep_model,
    ref_model=None,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataloader=dataloader)
# data_collator=collator,

gold_model = cpo_trainer.accelerator.prepare(gold_model)
gold_model.eval()

if script_args.gold_eval_greedy:
    generation_config = GenerationConfig(
        max_new_tokens=script_args.max_target_length,
        do_sample=False,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
else:
    generation_config = GenerationConfig(
        max_new_tokens=script_args.max_target_length,
        min_length=-1,
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

for epoch in range(script_args.epochs):  # 外层循环遍历每个epoch
    for step, inputs in tqdm(
        enumerate(cpo_trainer.data_collator),
        total=(len(train_dataset) // config.batch_size),
        disable=not cpo_trainer.accelerator.is_local_main_process,
    ):
        if epoch >= config.total_ppo_epochs:
            break
        # Run CPO step
        stats = cpo_trainer.cpo_step(inputs)

        # CPO gold model eval
        if step % script_args.log_interval == 0 and stats is not None:
            cpo_trainer.accelerator.log(stats, cpo_trainer.current_step)
        if step % script_args.eval_steps == 0:
            samples_to_log = []
            gold_reward_sum = 0.0
            total_samples = 0
            return_dict = cpo_trainer.get_batch_samples(
                inputs,
                generate_ref_response=True,
                generation_config=generation_config,
                on_evaluate=True
                )
            with torch.no_grad():
                gold_rewards = gold_model(
                    input_ids=return_dict['policy_output'], 
                    attention_mask=return_dict['policy_output_attention_mask']
                )[0]
            gold_rewards = cpo_trainer.accelerator.gather_for_metrics(gold_rewards)
            if cpo_trainer.accelerator.is_local_main_process:
                gold_reward_sum += gold_rewards.sum().item()
                total_samples += gold_rewards.size(0)

                # Sample and save to game log if requested (for one batch to save time)
                for i, (prompt, pol, ref) in enumerate(
                    zip(inputs["prompt"], return_dict["policy_output_decoded"], return_dict["ref_output_decoded"])
                ):
                    if len(samples_to_log) < script_args.log_n_samples_during_eval:
                        samples_to_log.append([prompt, pol[len(prompt) :], ref[len(prompt) :]])
                    else:
                        break
            
            if cpo_trainer.accelerator.is_local_main_process:
                print(f"gold reward mean {gold_reward_sum / total_samples}")
                gold_log = {
                    "eval/gold_rewards_mean": gold_reward_sum / total_samples,
                }
                if samples_to_log:
                    # cpo_trainer.accelerator.log({"eval/game_log": (
                    #     wandb.Table(
                    #         columns=["Prompt", "Policy", "Ref Model"],
                    #         data=samples_to_log,
                    #     ),
                    # )})
                    df = pd.DataFrame(samples_to_log, columns=["Prompt", "Policy", "Ref Model"])
                    csv_file_path = f"results/cpo_{current_datetime}/game_log.csv"
                    if not os.path.exists(os.path.dirname(csv_file_path)):
                        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
                    if os.path.exists(csv_file_path):
                        df.to_csv(csv_file_path, mode='a', header=False, index=False)
                    else:
                        df.to_csv(csv_file_path, index=False)

                    cpo_trainer.accelerator.log(gold_log, cpo_trainer.current_step)
                    # cpo_trainer.accelerator.log(gold_log_table)


        # stats["epoch"] = epoch
        # cpo_trainer.log_stats(stats, batch, rewards, gold_rewards)
        # cpo_trainer.accelerator.print(stats)

    if epoch < script_args.epoch - 1 and epoch > 0 and script_args.save_strategy != "final":
        cpo_trainer.save_pretrained(script_args.output_dir + f"epoch_{epoch}")

cpo_trainer.save_pretrained(script_args.output_dir + f"epoch_{script_args.epoch - 1}")
        
