# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import inspect
import math
import os
import time
from datetime import datetime
import typing
import warnings
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from torch.utils.data import DataLoader

import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, is_deepspeed_available
from datasets import Dataset
from huggingface_hub import whoami
from packaging import version
from torch.optim import Adam

from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

from ..core import (
    WANDB_PADDING,
    PPODecorators,
    clip_by_value,
    convert_to_scalar,
    entropy_from_logits,
    flatten_dict,
    logprobs_from_logits,
    masked_mean,
    masked_var,
    masked_whiten,
    set_seed,
    stack_dicts,
    stats_to_np,
)
from ..import_utils import is_torch_greater_2_0, is_xpu_available
from ..models import SUPPORTED_ARCHITECTURES, PreTrainedModelWrapper, create_reference_model
from . import AdaptiveKLController, BaseTrainer, FixedKLController, RunningMoments
from .ppo_config import PPOConfig
from ..import_utils import is_peft_available, is_wandb_available
from ..models import PreTrainedModelWrapper, create_reference_model
from .utils import pad_to_length, disable_dropout_in_model




if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


if is_wandb_available():
    import wandb


if is_deepspeed_available():
    import deepspeed

MODEL_CARD_TEMPLATE = """---
license: apache-2.0
tags:
- trl
- transformers
- reinforcement-learning
---

# {model_name}

This is a [TRL language model](https://github.com/huggingface/trl) that has been fine-tuned with reinforcement learning to
 guide the model outputs according to a value, function, or human feedback. The model can be used for text generation.

## Usage

To use this model for inference, first install the TRL library:

```bash
python -m pip install trl
```

You can then generate text as follows:

```python
from transformers import pipeline

generator = pipeline("text-generation", model="{model_id}")
outputs = generator("Hello, my llama is cute")
```

If you want to use the model for training or to obtain the outputs from the value head, load the model as follows:

```python
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

tokenizer = AutoTokenizer.from_pretrained("{model_id}")
model = AutoModelForCausalLMWithValueHead.from_pretrained("{model_id}")

inputs = tokenizer("Hello, my llama is cute", return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"])
```
"""


class CPOBaseTrainer(BaseTrainer):
    """
    The PPOTrainer uses Proximal Policy Optimization to optimise language models.
    Note, this trainer is heavily inspired by the original OpenAI learning to summarize work here:
    https://github.com/openai/summarize-from-feedback

    Attributes:
        **config** (`PPOConfig`) -- Configuration object for PPOTrainer. Check the documentation of `PPOConfig` for more
            details.
        **model** (`PreTrainedModelWrapper`) -- Model to be optimized, Hugging Face transformer model with a value head.
            Check the documentation of `PreTrainedModelWrapper` for more details.
        **ref_model** (`PreTrainedModelWrapper`, *optional*) -- Reference model to be used for KL penalty, Hugging Face
            transformer model with a casual language modelling head. Check the documentation of `PreTrainedModelWrapper`
            for more details. If no reference model is provided, the trainer will create a reference model with the same
             architecture as the model to be optimized with shared layers.
        **tokenizer** (`PreTrainedTokenizerBase`) -- Tokenizer to be used for encoding the
            data. Check the documentation of `transformers.PreTrainedTokenizer` and
            `transformers.PreTrainedTokenizerFast` for more details.
        **dataset** (Union[`torch.utils.data.Dataset`, `datasets.Dataset`], *optional*) -- PyTorch dataset or Hugging
            Face dataset. This is used to create a PyTorch dataloader. If no dataset is provided, the dataloader must be
             created outside the trainer users needs to design their own dataloader and make sure the batch
            size that is used is the same as the one specified in the configuration object.
        **optimizer** (`torch.optim.Optimizer`, *optional*) -- Optimizer to be used for training. If no optimizer is
            provided, the trainer will create an Adam optimizer with the learning rate specified in the configuration
            object.
        **data_collator** (DataCollatorForLanguageModeling, *optional*) -- Data collator to be used for training and
            passed along the dataloader
        **num_shared_layers** (int, *optional*) -- Number of layers to be shared between the model and the reference
            model, if no reference model is passed. If no number is provided, all the layers will be shared.
        **lr_scheduler** (`torch.optim.lr_scheduler`, *optional*) -- Learning rate scheduler to be used for training.
    """

    def __init__(
        self,
        config: PPOConfig = None,
        model: Union[PreTrainedModel, nn.Module, str] = None,
        rep_model: Union[PreTrainedModel, nn.Module, str] = None,
        ref_model: Optional[PreTrainedModelWrapper] = None,
        tokenizer: PreTrainedTokenizerBase = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        utd_ratio: Optional[float] = 1,
        dataloader: Optional[DataLoader] = None,
        data_collator: Optional[typing.Callable] = None,
        num_shared_layers: Optional[int] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        """
        Initialize PPOTrainer.

        Args:
            config (`PPOConfig`):
                Configuration object for PPOTrainer. Check the documentation of `PPOConfig` for more details.
            model (`PreTrainedModelWrapper`):
                Hugging Face transformer model with a value head.
            ref_model (`PreTrainedModelWrapper`):
                Hugging Face transformer model with a casual language modelling head. Used for KL penalty
            tokenizer (`transformers.PreTrainedTokenizerBase`):
                Hugging Face tokenizer
            dataset (Optional[Union[`torch.utils.data.Dataset`, `datasets.Dataset`]]):
                PyTorch dataset or Hugging Face dataset. If a Hugging Face dataset is passed, the dataset
                will be preprocessed by removing the columns that are not used by the model. If none is passed,
                a warning will be raised in a multi-GPU setting.
            optimizer (Optional[`torch.optim.Optimizer`]):
                Optimizer used for training. If `None`, the `Adam` is used as default.
            data_collator (Optional[function]):
                Data collator function.
            num_shared_layers (Optional[int]):
                Number of shared layers between the model and the reference model. If `None`, all layers are shared.
                used only if `ref_model` is `None`.
            lr_scheduler (Optional[`torch.optim.lr_scheduler`]):
                Learning rate scheduler used for training.
        """
        super().__init__(config)

        # initial seed for reproducible experiments
        set_seed(config.seed)

        # Step 0: check positional arguments validity
        if not isinstance(config, PPOConfig):
            raise ValueError(f"config must be a PPOConfig, got {type(config)}")
        if not isinstance(tokenizer, (PreTrainedTokenizerBase)):
            raise ValueError(
                f"tokenizer must be a PreTrainedTokenizerBase like a PreTrainedTokenizer or a PreTrainedTokenizerFast, got {type(tokenizer)}"
            )
        # Step 1: Initialize Accelerator
        self.accelerator = Accelerator(
            log_with=config.log_with,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            project_config=ProjectConfiguration(**config.project_kwargs),
            **config.accelerator_kwargs,
        )

        # Step 1.1 Runtime variables filled by the accelerator
        config.world_size = self.accelerator.num_processes
        config.global_backward_batch_size = config.backward_batch_size * config.world_size
        config.global_batch_size = config.batch_size * config.world_size
        self.label_pad_token_id = config.label_pad_token_id
        self.padding_value = config.padding_value
        self.utd_ratio = utd_ratio

        self.policy_model = model
        self.rep_model = rep_model
        self.policy_gen_dict = None
        self.policy_model_params = filter(lambda p: p.requires_grad, self.policy_model.parameters())
        self.rep_model_params = filter(lambda p: p.requires_grad, self.rep_model.parameters())
        self.is_encoder_decoder = hasattr(self.policy_model, "is_encoder_decoder")
        self.is_peft_model = is_peft_available() and isinstance(self.policy_model, PeftModel)
        config.is_encoder_decoder = self.is_encoder_decoder
        config.is_peft_model = self.is_peft_model
        if os.environ["WANDB_MODE"] == "online":
            project_name = "huggingface"
            tracker_kwargs={"wandb": {"entity": "louis_t0",  "name": os.environ["WANDB_NAME"]}}

            is_using_tensorboard = config.log_with is not None and config.log_with == "tensorboard"
            self.accelerator.init_trackers(
                project_name=project_name,
                config=dict(trl_ppo_trainer_config=config.to_dict()) if not is_using_tensorboard else config.to_dict(),
                init_kwargs=tracker_kwargs,
            )
        is_using_tensorboard = config.log_with is not None and config.log_with == "tensorboard"
        self.accelerator.init_trackers(
            config.tracker_project_name,
            config=dict(trl_ppo_trainer_config=config.to_dict()) if not is_using_tensorboard else config.to_dict(),
            init_kwargs=config.tracker_kwargs,
        )
        self.is_using_text_environment = getattr(config, "use_text_environment", False)

        if isinstance(ref_model, SUPPORTED_ARCHITECTURES):
            self.ref_model = ref_model
            if num_shared_layers is not None:
                warnings.warn(
                    "num_shared_layers is ignored when ref_model is provided. Two different models are used for the "
                    "model and the reference model and no layers are shared.",
                    UserWarning,
                )
        elif ref_model is None and not self.is_peft_model:
            self.ref_model = create_reference_model(self.policy_model, num_shared_layers=num_shared_layers)
        elif self.is_peft_model:
            self.ref_model = None
        else:
            raise ValueError(
                f"ref_model must be a PreTrainedModelWrapper or `None`, got {type(ref_model)} - supported "
                f"architectures are: {SUPPORTED_ARCHITECTURES} "
            )
        self.optional_peft_ctx = (
            self.accelerator.unwrap_model(self.policy_model).disable_adapter
            if self.is_peft_model
            else nullcontext
        )

        if not (isinstance(tokenizer, PreTrainedTokenizer) or isinstance(tokenizer, PreTrainedTokenizerFast)):
            raise ValueError(
                "tokenizer must be a transformers.PreTrainedTokenizer or transformers.PreTrainedTokenizerFast"
            )
        self.tokenizer = tokenizer
        self.max_length = config.max_length
        if self.max_length is None:
            warnings.warn(
                "When using DPODataCollatorWithPadding, you should set `max_length` in the DPOTrainer's init"
                " it will be set to `512` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            self.max_length = 512
        
        self.max_prompt_length = config.max_prompt_length
        if config.max_prompt_length is None:
            warnings.warn(
                "When using DPODataCollatorWithPadding, you should set `max_prompt_length` in the DPOTrainer's init"
                " it will be set to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            self.max_prompt_length = 128

        self.max_target_length = config.max_target_length
        if config.max_target_length is None and self.is_encoder_decoder:
            warnings.warn(
                "When using DPODataCollatorWithPadding with an encoder decoder architecture, you should set `max_target_length` in the DPOTrainer's init"
                " it will be set to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            self.max_target_length = 128

        # Step 3: Initialize optimizer and data collator
        data_collator = dataloader.generate_batches(train_dataset, config.batch_size)
        self.dataloader = dataloader
        self.data_collator = data_collator

        # if optimizer is None:
        self.policy_optimizer = Adam(
                filter(lambda p: p.requires_grad, self.policy_model.parameters()),
                lr=self.config.learning_rate,
            )

        self.rep_optimizer = Adam(
                filter(lambda p: p.requires_grad, self.rep_model.parameters()),
                lr=self.config.learning_rate,
        )

        self.lr_scheduler = lr_scheduler
        # self.global_step = 0

        if self.config.adap_kl_ctrl:
            self.kl_ctl = AdaptiveKLController(self.config.init_kl_coef, self.config.target, self.config.horizon)
        else:
            self.kl_ctl = FixedKLController(self.config.init_kl_coef)

        # Safety checkers for DS integration
        is_deepspeed_used = self.accelerator.distributed_type == "DEEPSPEED" and hasattr(
            self.accelerator.state, "deepspeed_plugin"
        )

        (
            self.policy_model,
            self.rep_model,
            self.policy_optimizer,
            self.rep_optimizer,
            self.dataloader,
            self.data_collator,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.policy_model,
            self.rep_model,
            self.policy_optimizer,
            self.rep_optimizer,
            self.dataloader,
            self.data_collator,
            self.lr_scheduler,
        )

        # self.data_collator = self.accelerator.prepare_data_loader(self.data_collator)
        if is_deepspeed_used:
            # Quantized models are already set on the correct device
            if not self.is_peft_model and not (
                getattr(self.ref_model.pretrained_model, "is_loaded_in_8bit", False)
                or getattr(self.ref_model.pretrained_model, "is_loaded_in_4bit", False)
            ):
                self.ref_model = self._prepare_deepspeed(self.ref_model)
        else:
            self.ref_model = self.accelerator.prepare(self.ref_model)

        # In a distributed setup, only logging needs to be performed on the main process
        # check: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
        # or: https://discuss.pytorch.org/t/use-distributed-data-parallel-correctly/82500/11
        self.is_distributed = self.accelerator.distributed_type == "MULTI_GPU"

        # init the current step
        self.current_step = 0

        # init variables for pushing model to hub
        if config.push_to_hub_if_best_kwargs:
            if "repo_id" not in config.push_to_hub_if_best_kwargs:
                raise ValueError("You have to specify repo_id in order to push the model to the hub!")
            self.push_to_hub_kwargs = config.push_to_hub_if_best_kwargs
            self.compare_step = 0
            self.highest_reward = torch.tensor(-float("inf"))

        # post process for PP
        if not getattr(self.policy_model, "is_sequential_parallel", False):
            self.current_device = self.accelerator.device
        else:
            if is_xpu_available():
                self.current_device = torch.device("xpu:0")
            else:
                self.current_device = torch.device("cuda:0")

        PPODecorators.optimize_device_cache = self.config.optimize_device_cache

        self.running = RunningMoments(self.accelerator)

    def _filter_kwargs(self, kwargs, target_func):
        """
        filter the keyword arguments that are supported by the target function.

        Args:
            kwargs (dict):
                Keyword arguments
            target_func (function):
                Target function
        """
        return {k: v for k, v in kwargs.items() if k in inspect.signature(target_func).parameters.keys()}

    # def _get_collator_with_removed_columns(
    #     self, data_collator: Callable, description: Optional[str] = None
    # ) -> Callable:
    #     """Wrap the data collator in a callable removing unused columns."""
    #     if not self.args.remove_unused_columns:
    #         return data_collator
    #     self._set_signature_columns_if_needed()
    #     signature_columns = self._signature_columns

    #     remove_columns_collator = RemoveColumnsCollator(
    #         data_collator=data_collator,
    #         signature_columns=signature_columns,
    #         logger=logger,
    #         description=description,
    #         model_name=self.policy_model.__class__.__name__,
    #     )
    #     return remove_columns_collator
    
    # def prepare_dataloader(self, dataset: Union[torch.utils.data.Dataset, Dataset], data_collator=None):
    #     """
    #     Prepare the dataloader for training.

    #     Args:
    #         dataset (Union[`torch.utils.data.Dataset`, `datasets.Dataset`]):
    #             PyTorch dataset or Hugging Face dataset. If a Hugging Face dataset is passed, the dataset
    #             will be preprocessed by removing the columns that are not used by the model.
    #         data_collator (Optional[function]):
    #             Data collator function.

    #     Returns:
    #         `torch.utils.data.DataLoader`: PyTorch dataloader
    #     """
    #     if isinstance(dataset, Dataset):
    #         dataset = self._remove_unused_columns(dataset)
    #     dataloader = torch.utils.data.DataLoader(
    #         dataset,
    #         batch_size=self.config.batch_size,
    #         collate_fn=data_collator,
    #         shuffle=True,
    #         drop_last=True,
    #     )
    #     return dataloader


    # Adapted from transformers.Trainer._set_signature_columns_if_needed
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.policy_model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # label => sentiment | we need query and response for logging purpose
            self._signature_columns += ["label", "query", "response"]
    
    def get_batch_samples(
        self,
        batch: Dict[str, torch.LongTensor],
        generate_ref_response: bool = False,
        generation_config: Optional[Dict[str, Any]] = None,
        on_evaluate: bool = False,
    ) -> Dict[Any, Any]:
        """
        Generate response with the model given the query tensor.

        Args:
            batch: A batch of data. Must contain the keys 'prompt_input_ids' and 'prompt_attention_mask', which are tensors of shape (batch_size, sequence_length).
            generate_ref_response: If True, generate a reference response as well.
        Returns:
            `List[str]`: A list of length `batch_size` containing the tokens and decoded responses."""
        return_dict = {}
        if not on_evaluate:
            policy_output = self.accelerator.unwrap_model(self.policy_model).generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.max_length,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            ).to(self.current_device)
        else:
            policy_output = self.accelerator.unwrap_model(self.policy_model).generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                generation_config=generation_config,
            )

        # policy_output = pad_to_length(policy_output, self.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)
        return_dict['policy_output'] = policy_output
        return_dict['policy_output_decoded'] = policy_output_decoded
        policy_output_attention_mask = (policy_output != 0).to(torch.int64)
        return_dict["policy_output_attention_mask"] = policy_output_attention_mask.to(self.current_device)

        policy_labels = policy_output.clone()  # Create a tensor filled with padding_value
        length = policy_labels.shape[1]
        prompt_total_length = batch["prompt_input_ids"].shape[1]
        for i in range(policy_labels.shape[0]):
            prompt_length = sum(batch["prompt_input_ids"][i]!=self.padding_value)
            front_padding_length = prompt_total_length - prompt_length
            # if front_padding_length == 0:
            #     continue
            if front_padding_length > 0:
                policy_labels[i, :front_padding_length]= torch.tensor([self.padding_value] * front_padding_length).to(self.current_device)
                padding_temp = policy_labels[i, :front_padding_length].clone()
                valid_temp = policy_labels[i, front_padding_length:].clone()
                policy_labels[i, :length - front_padding_length] = valid_temp
                policy_labels[i, -front_padding_length:] = padding_temp

                policy_output[i, :length - front_padding_length] = valid_temp
                policy_output[i, -front_padding_length:] = padding_temp
            policy_labels[i, :prompt_length] = torch.tensor([self.padding_value] * prompt_length).to(self.current_device)

        return_dict["policy_labels"] = policy_labels

        
        # for row in policy_output_attention_mask:
        #     zero_indices = (row == 0).nonzero(as_tuple=True)[0]
        #     if len(zero_indices) > 0:
        #         first_zero_index = zero_indices[0]
        #         row[first_zero_index] = 1


        if generate_ref_response:
            model = self.accelerator.unwrap_model(self.policy_model)
            with self.optional_peft_ctx():
                if not on_evaluate:
                        ref_output = model.generate(
                            batch["prompt_input_ids"],
                            attention_mask=batch["prompt_attention_mask"],
                            max_length=self.max_length,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                        ).to(self.current_device)
                else:
                        ref_output = model.generate(
                            batch["prompt_input_ids"],
                            attention_mask=batch["prompt_attention_mask"],
                            generation_config=generation_config,
                        )

            ref_output_decoded = self.tokenizer.batch_decode(ref_output, skip_special_tokens=True)
            return_dict['ref_output'] = ref_output
            return_dict['ref_output_decoded'] = ref_output_decoded

            # create ref labels
            ref_labels = ref_output.clone()  # Create a tensor filled with padding_value
            length = ref_labels.shape[1]
            prompt_total_length = batch["prompt_input_ids"].shape[1]

            for i in range(ref_labels.shape[0]):
                prompt_length = sum(batch["prompt_input_ids"][i]!=self.padding_value)
                front_padding_length = prompt_total_length - prompt_length
                if front_padding_length > 0:
                    ref_labels[i, :front_padding_length]= torch.tensor([self.padding_value] * front_padding_length).to(self.current_device)
                    padding_temp = ref_labels[i, :front_padding_length].clone()
                    valid_temp = ref_labels[i, front_padding_length:].clone()
                    ref_labels[i, :length - front_padding_length] = valid_temp
                    ref_labels[i, -front_padding_length:] = padding_temp

                    ref_output[i, :length - front_padding_length] = valid_temp
                    ref_output[i, -front_padding_length:] = padding_temp
                ref_labels[i, :prompt_length] = torch.tensor([self.padding_value] * prompt_length).to(self.current_device)

            return_dict["ref_labels"] = ref_labels
            ref_output_attention_mask = (ref_output != 0).to(torch.int64)
            return_dict["policy_output_attention_mask"] = ref_output_attention_mask.to(self.current_device)
        return return_dict

    

    def _step_safety_checker(
        self,
        inputs: Dict[str, Union[torch.Tensor, Any]],
    ):
        """
        Check if the input data is valid for training.

        Args:
            batch_size (int):
                Batch size from the config file.
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            scores (List[`torch.FloatTensor`]):
                List of tensors containing the scores.
            masks (List[`torch.LongTensor`], *optional*):
                list of optional tensors containing the masks of shape (`query_length` + `response_length`)
        Returns:
            `tuple`: The input processed data.
        """
        # for name, tensor_list in zip(["queries", "responses", "scores"], [queries, responses, scores]):
        #     if not isinstance(tensor_list, list):
        #         raise ValueError(f"{name} must be a list of tensors - got {type(tensor_list)}")
        #     if not isinstance(tensor_list[0], torch.Tensor):
        #         raise ValueError(f"Elements in {name} must be tensors - got {type(tensor_list[0])}")
        #     if batch_size is not None and len(tensor_list) != batch_size:
        #         raise ValueError(
        #             f"Batch size ({batch_size}) does not match number of examples - but got {len(tensor_list)} for: {name}"
        #         )
                #     for k, v in logs.items():
                # if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
                #     logs[k] = v.float()
        
        # q = a // b
        # if a != q * b:
        #     raise ValueError(f"{custom_error_message}, {a_str}={a}, {b_str}={b}, inexact division: {a} / {b} = {a / b}")
        
        update_period = self.config.gradient_accumulation_steps * (1 + self.utd_ratio)
        update_period = update_period // self.accelerator.num_processes
        update_period_remain = update_period % self.accelerator.num_processes
        if update_period_remain != 0:
            raise ValueError("update_period_remain must be zero")
        seperate_step_in_period_remain = self.config.gradient_accumulation_steps % self.accelerator.num_processes
        if seperate_step_in_period_remain != 0:
            raise ValueError("seperate_step_in_period_remain must be zero")

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.accelerator.device) 
        
        return inputs

    def concatenated_inputs(
        self,
        batch: Dict[str, Union[List, torch.LongTensor]],
        policy_gen_dict: Dict[str, Union[List, torch.LongTensor]],
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
            policy_output: The output of the policy model. Shape: (batch_size, self.max_length)

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        policy_output_tokens = policy_gen_dict['policy_output']
        policy_output_decoded = policy_gen_dict['policy_output_decoded']
        policy_output_attention_mask = policy_gen_dict['policy_output_attention_mask']
        concatenated_batch = {}

        if self.is_encoder_decoder:
            max_length = max(
                batch["chosen_labels"].shape[1],
                batch["rejected_labels"].shape[1],
                policy_output_decoded.shape[1]
            )
        else:
            max_length = max(
                batch["chosen_input_ids"].shape[1], 
                batch["rejected_input_ids"].shape[1], 
                policy_output_tokens.shape[1]
            )

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k or self.is_encoder_decoder else self.padding_value
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k or self.is_encoder_decoder else self.padding_value
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(self.accelerator.device)

        concatenated_batch["concatenated_input_ids"] = torch.cat(
            (
                concatenated_batch["concatenated_input_ids"],
                pad_to_length(policy_output_tokens, max_length, pad_value=self.padding_value),
            ),
            dim=0,
        ).to(self.accelerator.device)

        concatenated_batch["concatenated_attention_mask"] = torch.cat(
            (
                concatenated_batch["concatenated_attention_mask"],
                pad_to_length(policy_output_attention_mask, max_length, pad_value=0),
            ),
            dim=0,
        ).to(self.accelerator.device)

        concatenated_batch["concatenated_labels"] = torch.cat(
            (
                concatenated_batch["concatenated_labels"],
                pad_to_length(policy_gen_dict["policy_labels"], max_length, pad_value=self.label_pad_token_id),
            ),
        )

        if self.is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1)
            concatenated_batch["concatenated_attention_mask"] = batch["prompt_attention_mask"].repeat(2, 1)

        return concatenated_batch

    def concatenated_rep_forward(
        self, 
        concatenated_batch: Dict[str, Union[List, torch.LongTensor]],
        len_chosen: int,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        rep_model_kwargs = {"output_hidden_states": True}
        if self.is_encoder_decoder:
            rep_model_kwargs.update(
                {
                    "labels": concatenated_batch["concatenated_labels"],
                    "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
                }
            )

        last_hidden_states = self.rep_model(
                concatenated_batch["concatenated_input_ids"],
                attention_mask=concatenated_batch["concatenated_attention_mask"],
                **rep_model_kwargs,
            ).hidden_states[-1].to(self.config.dtype)

        chosen_last_hidden_state = last_hidden_states[:len_chosen, :, :]
        rejected_last_hidden_state = last_hidden_states[len_chosen:-len_chosen, :, :]
        policy_last_hidden_state = last_hidden_states[-len_chosen:, :, :]
        

        return (chosen_last_hidden_state, rejected_last_hidden_state, policy_last_hidden_state)

    def _get_response_rep(
        self,
        positive_sim: torch.FloatTensor,
        negative_sim: torch.LongTensor,
        labels: torch.LongTensor,
        average_sim: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Compute the response representations loss.

        Args:
            positive_sim (torch.FloatTensor): Positive similarity scores. Shape: (batch_size, sequence_length)
            negative_sim (torch.LongTensor): Negative similarity scores. Shape: (batch_size, sequence_length)
            labels (torch.LongTensor): Labels for adjusting the similarities. Labels with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)

        Returns:
            Two tensors of shape (batch_size,) containing either the average or the sum of the adjusted positive and negative similarities for each item in the batch.

        Raises:
            ValueError: If the shapes of positive_sim, negative_sim, and labels do not match.
        """        
        combine_sim = torch.cat([positive_sim, negative_sim], dim=0)
        if positive_sim.shape[0] < self.config.batch_size:
            indice = positive_sim.shape[0]
        else:
            indice = self.config.batch_size
        combine_labels = labels[:indice * 2]
        # TODO, here we do not count more information from the policy output which has greater length than positive or negative responses
        if combine_sim.shape[:-1] != combine_labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not self.is_encoder_decoder:
            combine_labels = combine_labels[:, 1:].clone()
            combine_sim = combine_sim[:, :-1]
    
        loss_mask = combine_labels != self.label_pad_token_id
        loss_mask = loss_mask.unsqueeze(-1).expand_as(combine_sim)
        
        if self.config.response_only:
            combine_sim *= loss_mask
        
        if self.config.rep_approach == "mean":
            combine_sim = combine_sim.mean(dim=-1)
        elif self.config.rep_approach == 'max_pooling':
            combine_sim = combine_sim.max(dim=-1)
        elif self.config.rep_approach == 'mean_max_pooling':
            combine_sim_mean = combine_sim.mean(dim=-1)
            combine_sim_max = combine_sim.max(dim=-1)[0]

            combine_sim = (combine_sim_mean + combine_sim_max) / 2
        combine_sim = combine_sim.mean(dim=-1)
        positive_sim = combine_sim[:self.config.batch_size]
        negative_sim = combine_sim[self.config.batch_size:]
        if average_sim:
            return (positive_sim / loss_mask.sum(-1), negative_sim / loss_mask.sum(-1))
        else:
            return (positive_sim, negative_sim)

    def cpo_loss(
        self,
        chosen_last_hidden_state: torch.FloatTensor,
        rejected_last_hidden_state: torch.FloatTensor,
        policy_last_hidden_state: torch.FloatTensor,
        labels: torch.LongTensor,
        use_other_negative_samples: bool,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the CPO loss for a batch of policy and reference model log probabilities.

        Args:
            chosen_last_hidden_state: Last hidden state of the policy model for the chosen responses. Shape: (batch_size, hidden_size)
            rejected_last_hidden_state: Last hidden state of the policy model for the rejected responses. Shape: (batch_size, hidden_size)
            policy_last_hidden_state: Last hidden state of the policy model for the chosen responses. Shape: (batch_size, hidden_size)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the CPO loss for each example in the batch.
        """
        positive_dot_product = torch.einsum('bik,bik->bik', policy_last_hidden_state, chosen_last_hidden_state)
        negative_dot_product = torch.einsum('bik,bik->bik', policy_last_hidden_state, rejected_last_hidden_state)

        if self.config.similarity == "dot":
            norm = torch.sqrt(torch.tensor(chosen_last_hidden_state.shape[-1], dtype=torch.float32)).to(self.accelerator.device)
            positive_sim = positive_dot_product
            negative_sim = positive_dot_product
            if use_other_negative_samples:
                raise NotImplementedError("use_other_negative_samples is not implemented yet")
        elif self.config.similarity == "cosine":
            policy_norm = torch.linalg.norm(policy_last_hidden_state, dim=-1, keepdim=True)
            chosen_norm = torch.linalg.norm(chosen_last_hidden_state, dim=-1, keepdim=True)
            rejected_norm = torch.linalg.norm(rejected_last_hidden_state, dim=-1, keepdim=True)

            positive_sim = positive_dot_product / (policy_norm * chosen_norm + 1e-8)
            negative_sim = negative_dot_product / (policy_norm * rejected_norm + 1e-8)

            if use_other_negative_samples:
                raise NotImplementedError("use_other_negative_samples is not implemented yet")
        else:
            raise ValueError(f"Unknown similarity function: {self.config.similarity}. Should be one of ['dot', 'cosine']")


        (positive_sim, negative_sim) = self._get_response_rep(positive_sim,
                                                              negative_sim,
                                                              labels)

        if self.config.loss_type == "sigmoid":
            loss = -F.logsigmoid(positive_sim - negative_sim).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}. Should be one of ['sigmoid', 'InfoNCE']")

        # To avoid RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one.
        # see: https://github.com/pytorch/pytorch/issues/43259
        loss = loss + 0. * sum(p.sum() for p in self.policy_model.parameters())
        loss = loss + 0. * sum(p.sum() for p in self.rep_model.parameters())

        return (loss, positive_sim, negative_sim)


    @PPODecorators.empty_device_cache()
    def cpo_step(
            self,
            inputs: Dict[str, Union[torch.Tensor, Any]],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        self._step_safety_checker(inputs)
        # TODO
        update_period = self.config.gradient_accumulation_steps * (1 + self.utd_ratio) 
        update_period = update_period // self.accelerator.num_processes
        step_in_period = self.current_step % update_period
        seperate_step_in_period = self.config.gradient_accumulation_steps // self.accelerator.num_processes
        
        # only generate new y for policy update; use prior y for rep update
        # if step_in_period <= seperate_step_in_period:
        #     with self.accelerator.accumulate(self.policy_model):
        #         policy_gen_dict = self.get_batch_samples(inputs)
        #         self.policy_gen_dict = policy_gen_dict
        # else:
        #     policy_gen_dict = self.policy_gen_dict 
        warnings.warn("check the self.gradient_accumulation_steps \
                      requires to seperate the policy training and rep training loop")
        with self.accelerator.accumulate(self.policy_model, self.rep_model):
            policy_gen_dict = self.get_batch_samples(inputs)

            concatenated_inputs = self.concatenated_inputs(inputs, policy_gen_dict)

            len_chosen = inputs['chosen_input_ids'].shape[0]
            (
                chosen_last_hidden_state,
                rejected_last_hidden_state,
                policy_last_hidden_state,
            ) = self.concatenated_rep_forward(concatenated_inputs, len_chosen)

            (loss, positive_sim, negative_sim) = self.cpo_loss(chosen_last_hidden_state,
                                                            rejected_last_hidden_state,
                                                            policy_last_hidden_state,
                                                            concatenated_inputs["concatenated_labels"],
                                                            use_other_negative_samples=False)
            
            if step_in_period < seperate_step_in_period:
                self.update_policy_model(loss)
            else:
                self.update_rep_model(loss)
            self.current_step += 1

        stats = self.record_step_stats(
            kl_coef=None,
            loss=loss,
            positive_sim=positive_sim,
            negative_sim=negative_sim,
            )
        # Gather/Reduce stats from all processes
        if self.is_distributed:
            stats = self.gather_stats(stats)

        return stats

    # @PPODecorators.empty_device_cache()
    # def step(
    #     self,
    #     queries: List[torch.LongTensor],
    #     responses: List[torch.LongTensor],
    #     scores: List[torch.FloatTensor],
    #     response_masks: Optional[List[torch.LongTensor]] = None,
    # ):
    #     """
    #     Run a PPO optimisation step given a list of queries, model responses, and rewards.

    #     Args:
    #         queries (List[`torch.LongTensor`]):
    #             List of tensors containing the encoded queries of shape (`query_length`)
    #         responses (List[`torch.LongTensor`]):
    #             List of tensors containing the encoded responses of shape (`response_length`)
    #         scores (List[`torch.FloatTensor`]):
    #             List of tensors containing the scores.
    #         response_masks (List[`torch.FloatTensor`], *optional*)):
    #             List of tensors containing masks of the response tokens.

    #     Returns:
    #         `dict[str, Any]`: A summary of the training statistics
    #     """
        
    #     bs = self.config.batch_size

    #     queries, responses, scores, response_masks = self._step_safety_checker(
    #         bs, queries, responses, scores, response_masks
    #     )
    #     scores = torch.tensor(scores, device=self.current_device)
    #     if self.config.use_score_scaling:
    #         # Score scaling
    #         scores_mean, scores_std = self.running.update(scores)
    #         tensor_to_kwargs = dict(dtype=scores.dtype, device=scores.device)
    #         score_scaling_factor = self.running.std.to(**tensor_to_kwargs) + torch.finfo(scores.dtype).eps
    #         if self.config.use_score_norm:
    #             scores = (scores - self.running.mean.to(**tensor_to_kwargs)) / score_scaling_factor
    #         else:
    #             scores /= score_scaling_factor

    #     if self.config.score_clip is not None:
    #         # Score clipping
    #         scores_dtype = scores.dtype
    #         scores = torch.clip(scores.float(), -self.config.score_clip, self.config.score_clip).to(dtype=scores_dtype)

    #     # if we want to push best model to the hub
    #     if hasattr(self, "highest_reward"):
    #         if self.compare_step % self.config.compare_steps == 0:
    #             curr_mean_reward = scores.mean()
    #             # if the best reward ever seen
    #             if curr_mean_reward > self.highest_reward:
    #                 self.highest_reward = curr_mean_reward
    #                 # push model to hub
    #                 self.push_to_hub(**self.push_to_hub_kwargs)
    #         self.compare_step += 1

    #     timing = dict()
    #     t0 = time.time()

    #     t = time.time()

    #     # TODO 
    #     model_inputs = self.prepare_model_inputs(queries, responses)

    #     if self.is_distributed:
    #         pad_first = self.tokenizer.padding_side == "left"

    #         model_inputs["input_ids"] = self.accelerator.pad_across_processes(
    #             model_inputs["input_ids"],
    #             dim=1,
    #             pad_index=self.tokenizer.pad_token_id,
    #             pad_first=pad_first,
    #         )
    #         model_inputs["attention_mask"] = self.accelerator.pad_across_processes(
    #             model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
    #         )
    #         if self.is_encoder_decoder:
    #             model_inputs["decoder_input_ids"] = self.accelerator.pad_across_processes(
    #                 model_inputs["decoder_input_ids"],
    #                 dim=1,
    #                 pad_index=self.tokenizer.pad_token_id,
    #                 pad_first=pad_first,
    #             )
    #             model_inputs["decoder_attention_mask"] = self.accelerator.pad_across_processes(
    #                 model_inputs["decoder_attention_mask"],
    #                 dim=1,
    #                 pad_index=0,
    #                 pad_first=pad_first,
    #             )

    #     model_inputs_names = list(model_inputs.keys())

    #     full_kl_penalty = self.config.kl_penalty == "full"

    #     with torch.no_grad():
    #         all_logprobs, logits_or_none, values, masks = self.batched_forward_pass(
    #             self.policy_model,
    #             queries,
    #             responses,
    #             model_inputs,
    #             response_masks=response_masks,
    #             return_logits=full_kl_penalty,
    #         )
    #         with self.optional_peft_ctx():
    #             ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(
    #                 self.policy_model if self.is_peft_model else self.ref_model,
    #                 queries,
    #                 responses,
    #                 model_inputs,
    #                 return_logits=full_kl_penalty,
    #             )

    #     timing["time/ppo/forward_pass"] = time.time() - t

    #     with torch.no_grad():
    #         t = time.time()
    #         if full_kl_penalty:
    #             active_full_logprobs = logprobs_from_logits(logits_or_none, None, gather=False)
    #             ref_full_logprobs = logprobs_from_logits(ref_logits_or_none, None, gather=False)

    #             rewards, non_score_reward = self.compute_rewards(
    #                 scores, active_full_logprobs, ref_full_logprobs, masks
    #             )
    #         else:
    #             rewards, non_score_reward = self.compute_rewards(scores, all_logprobs, ref_logprobs, masks)
    #         timing["time/ppo/compute_rewards"] = time.time() - t

    #         t = time.time()
    #         values, advantages, returns = self.compute_advantages(values, rewards, masks)
    #         timing["time/ppo/compute_advantages"] = time.time() - t

    #     # upcast to float32 to avoid dataset issues
    #     batch_dict = {
    #         "queries": queries,
    #         "responses": responses,
    #         "logprobs": all_logprobs.to(torch.float32),
    #         "values": values.to(torch.float32),
    #         "masks": masks,
    #         "advantages": advantages,
    #         "returns": returns,
    #     }
    #     batch_dict.update(model_inputs)

    #     t = time.time()
    #     all_stats = []
    #     early_stop = False
    #     for _ in range(self.config.ppo_epochs):
    #         if early_stop:
    #             break
    #         b_inds = np.random.permutation(bs)
    #         for backward_batch_start in range(0, bs, self.config.backward_batch_size):
    #             backward_batch_end = backward_batch_start + self.config.backward_batch_size
    #             backward_batch_inds = b_inds[backward_batch_start:backward_batch_end]

                # for mini_batch_start in range(0, self.config.backward_batch_size, self.config.mini_batch_size):
                #     mini_batch_end = mini_batch_start + self.config.mini_batch_size
                #     mini_batch_inds = backward_batch_inds[mini_batch_start:mini_batch_end]
                #     mini_batch_dict = {
                #         "logprobs": batch_dict["logprobs"][mini_batch_inds],
                #         "values": batch_dict["values"][mini_batch_inds],
                #         "masks": batch_dict["masks"][mini_batch_inds],
                #         # hacks: the queries and responses are ragged.
                #         "queries": [batch_dict["queries"][i] for i in mini_batch_inds],
                #         "responses": [batch_dict["responses"][i] for i in mini_batch_inds],
                #         "advantages": batch_dict["advantages"][mini_batch_inds],
                #         "returns": batch_dict["returns"][mini_batch_inds],
                #     }
                    # for k in model_inputs_names:
                    #     mini_batch_dict[k] = batch_dict[k][mini_batch_inds]
                    # with self.accelerator.accumulate(self.policy_model):
                    #     model_inputs = {k: mini_batch_dict[k] for k in model_inputs_names}

                        # logprobs, logits, vpreds, _ = self.batched_forward_pass(
                        #     self.policy_model,
                        #     mini_batch_dict["queries"],
                        #     mini_batch_dict["responses"],
                        #     model_inputs,
                        #     return_logits=True,
                        # )
                        # train_stats = self.train_minibatch(
                        #     mini_batch_dict["logprobs"],
                        #     mini_batch_dict["values"],
                        #     logprobs,
                        #     logits,
                        #     vpreds,
                        #     mini_batch_dict["masks"],
                        #     mini_batch_dict["advantages"],
                        #     mini_batch_dict["returns"],
                        # )
                        # all_stats.append(train_stats)

            # typically, early stopping is done at the epoch level
            # if self.config.early_stopping:
            #     policykl = train_stats["policy/policykl"]
            #     early_stop = self._early_stop(policykl)
            #     if early_stop:
            #         break

        # timing["time/ppo/optimize_step"] = time.time() - t

        # t = time.time()
        # train_stats = stack_dicts(all_stats)

        # # reshape advantages/ratios such that they are not averaged.
        # train_stats["policy/advantages"] = torch.flatten(train_stats["policy/advantages"]).unsqueeze(0)
        # train_stats["policy/advantages"] = torch.nan_to_num(train_stats["policy/advantages"], WANDB_PADDING)
        # train_stats["policy/ratio"] = torch.flatten(train_stats["policy/ratio"]).unsqueeze(0)

        # stats = self.record_step_stats(
        #     scores=scores,
        #     logprobs=all_logprobs,
        #     ref_logprobs=ref_logprobs,
        #     non_score_reward=non_score_reward,
        #     train_stats=train_stats,
        #     kl_coef=self.kl_ctl.value,
        #     masks=masks,
        #     queries=queries,
        #     responses=responses,
        # )
        # # Gather/Reduce stats from all processes
        # if self.is_distributed:
        #     stats = self.gather_stats(stats)
        # stats = stats_to_np(stats)
        # timing["time/ppo/calc_stats"] = time.time() - t
        # stats["ppo/learning_rate"] = self.policy_optimizer.param_groups[0]["lr"]

        # # Update the KL control - multiply the batch_size by the number of processes
        # self.kl_ctl.update(
        #     stats["objective/kl"],
        #     self.config.batch_size * self.accelerator.num_processes,
        # )

        # # Log the total ppo time
        # timing["time/ppo/total"] = time.time() - t0
        # stats.update(timing)

        # # post-process stats for tensorboard and other loggers
        # if self.config.log_with != "wandb":
        #     stats = convert_to_scalar(stats)

        # return stats

    def _early_stop(self, policykl):
        r"""
        Handles the early stopping logic. If the policy KL is greater than the target KL, then the gradient is zeroed and
        the optimization step is skipped.
        This also handles the multi-gpu case where the policy KL is averaged across all processes.

        Args:
            policy_kl (torch.Tensor):
                the policy KL

        Returns:
            `bool`: whether to early stop or not
        """
        early_stop = False
        if not self.config.early_stopping:
            return early_stop

        if not self.is_distributed and policykl > 1.5 * self.config.target_kl:
            self.policy_optimizer.zero_grad()
            early_stop = True
        elif self.is_distributed:
            import torch.distributed as dist

            # Wait for all processes to finish
            dist.barrier()

            # all gather the policykl
            dist.all_reduce(policykl, dist.ReduceOp.SUM)
            policykl /= self.accelerator.num_processes

            if policykl > 1.5 * self.config.target_kl:
                self.policy_optimizer.zero_grad()
                early_stop = True
        return early_stop

    def gather_stats(self, stats):
        """
        Gather stats from all processes. Useful in the context of distributed training.

        Args:
            stats (dict[str, Any]):
            a dictionary of stats to be gathered. The stats should contain torch tensors.

        Returns:
            `dict[str, Any]`: A dictionary of stats with the tensors gathered.
        """
        import torch.distributed as dist

        # Wait for all processes to finish
        dist.barrier()

        for k, v in stats.items():
            if isinstance(v, torch.Tensor):
                dist.all_reduce(v.to(self.accelerator.device), dist.ReduceOp.SUM)
                v /= self.accelerator.num_processes
            stats[k] = v
        return stats

    # def prepare_model_inputs(self, queries: torch.Tensor, responses: torch.Tensor):
    #     if self.is_encoder_decoder:
    #         input_data = self.data_collator(
    #             [{"input_ids": q, "attention_mask": torch.ones_like(q)} for q in queries]
    #         ).to(self.current_device)

    #         decoder_inputs = self.data_collator(
    #             [{"input_ids": r, "attention_mask": torch.ones_like(r)} for r in responses]
    #         ).to(self.current_device)

    #         input_data["decoder_input_ids"] = decoder_inputs["input_ids"]
    #         input_data["decoder_attention_mask"] = decoder_inputs["attention_mask"]
    #     else:
    #         input_ids = [torch.cat([q, r]) for q, r in zip(queries, responses)]
    #         input_data = self.data_collator(
    #             [{"input_ids": ids, "attention_mask": torch.ones_like(ids)} for ids in input_ids]
    #         ).to(self.current_device)

    #     input_data.pop("labels", None)  # we don't want to compute LM losses
    #     return input_data

    @PPODecorators.empty_device_cache()
    def update_policy_model(
        self,
        loss: torch.FloatTensor,
    ):
        # for p in self.rep_model.parameters():
        #     p.requires_grad = False

        self.policy_model.train()
        self.accelerator.backward(loss)
        if self.config.max_grad_norm is not None:
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.policy_model_params, self.config.max_grad_norm)

        self.policy_optimizer.step()
        self.policy_optimizer.zero_grad()
        # for p in self.rep_model.parameters():
        #     p.requires_grad = True

    @PPODecorators.empty_device_cache()
    def update_rep_model(
        self,
        loss: torch.FloatTensor,
    ):
        # for p in self.policy_model.parameters():
        #     p.requires_grad = False
        self.rep_model.train()
        self.accelerator.backward(loss)
        if self.config.max_grad_norm is not None:
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.policy_model_params, self.config.max_grad_norm)

        self.rep_optimizer.step()
        self.rep_optimizer.zero_grad()
        # for p in self.policy_model.parameters():
        #     p.requires_grad = True
        
    @PPODecorators.empty_device_cache()
    def train_minibatch(
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        mask: torch.LongTensor,
        advantages: torch.FloatTensor,
        returns: torch.FloatTensor,
    ):
        """
        Train one PPO minibatch

        Args:
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape [batch_size, response_length]
            values (`torch.FloatTensor`):
                Values of the value head, shape [batch_size, response_length]
            query (`torch.LongTensor`):
                Encoded queries, shape [batch_size, query_length]
            response (`torch.LongTensor`):
                Encoded responses, shape [batch_size, response_length]
            model_input (`torch.LongTensor`):
                Concatenated queries and responses, shape [batch_size, query_length+response_length]

        Returns:
            train_stats (dict[str, `torch.Tensor`]):
                Dictionary of training statistics
        """
        self.policy_model.train()
        loss_p, loss_v, train_stats = self.loss(
            old_logprobs, values, logits, vpreds, logprobs, mask, advantages, returns
        )
        loss = loss_p + loss_v
        self.accelerator.backward(loss)
        if self.config.max_grad_norm is not None:
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model_params, self.config.max_grad_norm)
        self.policy_optimizer.step()
        # we call optimizer.zero_grad() every time and let `accelerator` handle accumulation
        # see https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation#the-finished-code
        self.policy_optimizer.zero_grad()
        return train_stats

    @PPODecorators.empty_device_cache()
    def compute_reward_model_score(
        self, input_ids: torch.FloatTensor, attention_mask: torch.FloatTensor = None, **kwargs
    ):
        r"""
        Computes the reward score for a given input for a model with a reward modelling adapter. The method has first to enable the adapter
        and then compute the reward score. After that the model disables the reward modeling
        adapter and enables the default ppo adapter again.
        """
        unwrap_model = self.accelerator.unwrap_model(self.policy_model)
        if not unwrap_model.supports_rm_adapter:
            raise ValueError(
                "This model does not support reward modeling adapter, you must compute reward scores with another method."
            )

        # enable rm adapter
        unwrap_model.pretrained_model.set_adapter(unwrap_model.rm_adapter_name)
        # TODO check
        # self.policy_model.eval()

        with torch.no_grad():
            _, _, scores = self.policy_model(
                use_score=True,
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )

        unwrap_model.pretrained_model.set_adapter(unwrap_model.policy_adapter_name)
        # self.policy_model.train()

        return scores

    def compute_rewards(
        self,
        scores: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        ref_logprobs: torch.FloatTensor,
        masks: torch.LongTensor,
    ):
        """
        Compute per token rewards from scores and KL-penalty.

        Args:
            scores (`torch.FloatTensor`):
                Scores from the reward model, shape (`batch_size`)
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
            ref_logprobs (`torch.FloatTensor`):
                Log probabilities of the reference model, shape (`batch_size`, `response_length`)
        """
        rewards, non_score_rewards = [], []
        for score, logprob, ref_logprob, mask in zip(scores, logprobs, ref_logprobs, masks):
            # compute KL penalty (from difference in logprobs)
            kl = self._kl_penalty(logprob, ref_logprob)
            non_score_reward = -self.kl_ctl.value * kl
            non_score_rewards.append(non_score_reward)
            reward = non_score_reward.clone()
            last_non_masked_index = mask.nonzero()[-1]

            # reward is preference model score + KL penalty
            reward[last_non_masked_index] += score
            rewards.append(reward)
        return torch.stack(rewards), torch.stack(non_score_rewards)

    def _kl_penalty(self, logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor) -> torch.FloatTensor:
        if self.config.kl_penalty == "kl":
            return logprob - ref_logprob

        if self.config.kl_penalty == "abs":
            return (logprob - ref_logprob).abs()

        if self.config.kl_penalty == "mse":
            return 0.5 * (logprob - ref_logprob).square()

        if self.config.kl_penalty == "full":
            # Flip is required due to this issue? :https://github.com/pytorch/pytorch/issues/57459
            return F.kl_div(ref_logprob, logprob, log_target=True, reduction="none").sum(-1)

        raise NotImplementedError

    def record_step_stats(self, **data):
        """
        Record training step statistics.


        Args:
            kl_coef (`float`):
                KL coefficient
            data (`dict`):
                Dictionary of training step data

        Returns:
            stats (`dict`):
                Dictionary of training step statistics
        """
        # if mean_kl.item() < -1.0:
        #     # warn users
        #     warnings.warn(
        #         f"KL divergence is starting to become negative: {mean_kl.item():.2f} - this might be a precursor for failed training."
        #         " sometimes this happens because the generation kwargs are not correctly set. Please make sure"
        #         " that the generation kwargs are set correctly, or review your training hyperparameters."
        #     )
        stats = {
            "training/loss": data['loss'],
            "training/positive_sim": data['positive_sim'].sum(),
            "training/negative_sim": data['negative_sim'].sum(),
            "training/acc":(data['positive_sim'] > data['negative_sim']).float().mean()
        }
        return stats

    def create_model_card(self, path: str, model_name: Optional[str] = "TRL Model") -> None:
        """Creates and saves a model card for a TRL model.

        Args:
            path (`str`): The path to save the model card to.
            model_name (`str`, *optional*): The name of the model, defaults to `TRL Model`.
        """
        try:
            user = whoami()["name"]
        # handle the offline case
        except:  # noqa
            warnings.warn("Cannot retrieve user information assuming you are running in offline mode.")
            return

        if not os.path.exists(path):
            os.makedirs(path)

        model_card_content = MODEL_CARD_TEMPLATE.format(model_name=model_name, model_id=f"{user}/{path}")
        with open(os.path.join(path, "README.md"), "w", encoding="utf-8") as f:
            f.write(model_card_content)

    def _save_pretrained(self, save_directory: str) -> None:
        self.accelerator.unwrap_model(self.policy_model).save_pretrained(save_directory)
        self.accelerator.unwrap_model(self.rep_model).save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        self.create_model_card(save_directory)

    def _show_tokens(self, tokens, masks):
        from rich import print
        from rich.text import Text

        text = Text()

        for i, (token, mask) in enumerate(zip(tokens, masks)):
            if mask == 1:
                text.append(self.tokenizer.decode(token.item()), style="black on deep_sky_blue1")
                text.append(" ")
            else:
                text.append(self.tokenizer.decode(token.item()), style="black on cyan3")
                text.append(" ")
        print(text)

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepspeed_plugin.deepspeed_config
        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model
