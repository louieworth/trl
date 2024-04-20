# DPO Authors: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn 2023
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
# See the Licxense for the specific language governing permissions and
# limitations under the License.
import inspect
import random
import warnings
from collections import OrderedDict
from copy import deepcopy
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import is_deepspeed_available
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction

from ..import_utils import is_peft_available, is_wandb_available
from ..models import PreTrainedModelWrapper, create_reference_model
from .utils import DPODataCollatorWithPadding, disable_dropout_in_model, pad_to_length


if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


if is_wandb_available():
    import wandb


if is_deepspeed_available():
    import deepspeed


class CPOTrainer(Trainer):
    r"""
    Initialize CPOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss. If no
            reference model is provided, the trainer will create a reference model with the same architecture as the model to be optimized.
        beta (`float`, defaults to 0.1):
            The beta factor in CPO loss. Higher beta means less divergence from the initial policy.
        loss_type (`str`, defaults to `"sigmoid"`):
            The type of DPO loss to use. Either `"sigmoid"` the default DPO loss or `"hinge"` loss from SLiC paper.
        args (`transformers.TrainingArguments`):
            The arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        label_pad_token_id (`int`, defaults to `-100`):
            The label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`int`, defaults to `0`):
            The padding value. This argument is required if you want to use the default data collator.
        truncation_mode (`str`, defaults to `keep_end`):
            The truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use the default data collator.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        max_length (`int`, defaults to `None`):
            The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
        max_prompt_length (`int`, defaults to `None`):
            The maximum length of the prompt. This argument is required if you want to use the default data collator.
        max_target_length (`int`, defaults to `None`):
            The maximum length of the target. This argument is required if you want to use the default data collator and your model is an encoder-decoder.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            If no model is provided, we need to know if the model_init returns an encoder-decoder.
        disable_dropout (`bool`, defaults to `True`):
            Whether or not to disable dropouts in `model` and `ref_model`.
        generate_during_eval (`bool`, defaults to `False`):
            Whether to sample and log generations during evaluation step.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
        model_init_kwargs: (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when instantiating the model from a string
        ref_model_init_kwargs: (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when instantiating the ref model from a string

    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, str] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        beta: float = 0.1,
        begin_utd: int = 10,
        end_utd: int = 2,
        loss_type: Literal["sigmoid", "hinge"] = "sigmoid",
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        model_init_kwargs: Optional[Dict] = None,
        adapters: Optional[List] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
    ):
        self.adapters = adapters
        if model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_kwargs to the CPOTrainer. But your model is already instantiated.")

        if ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model, str):
            raise ValueError(
                "You passed ref_model_kwargs to the CPOTrainer. But your ref_model is already instantiated."
            )

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the CPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if isinstance(ref_model, str):
            warnings.warn(
                "You passed a ref model_id to the DPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM`"
            )
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model, **ref_model_init_kwargs)

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_config, we merge and unload it first
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                preprare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                if _support_gc_kwargs:
                    preprare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                model = prepare_model_for_kbit_training(model, **preprare_model_kwargs)
            elif getattr(args, "gradient_checkpointing", False):
                # For backward compatibility with older versions of transformers
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            # get peft model with the given config
            model = get_peft_model(model, peft_config)

        # For models that use gradient_checkpoiting, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        elif getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if generate_during_eval and not is_wandb_available():
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install `wandb` to resolve."
            )

        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder  # False
        elif is_encoder_decoder is None:
            raise ValueError("When no model is provided, you need to pass the parameter is_encoder_decoder.")
        else:
            self.is_encoder_decoder = is_encoder_decoder

        self.is_peft_model = is_peft_available() and self.adapters is not None

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model:
            # The `model` with adapters turned off will be used as the reference model
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)

        if data_collator is None:
            if tokenizer is None:
                raise ValueError(
                    "max_length or a tokenizer must be specified when using the default DPODataCollatorWithPadding"
                )
            if max_length is None:
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `max_length` in the DPOTrainer's init"
                    " it will be set to `512` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_length = 512
            if max_prompt_length is None:
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `max_prompt_length` in the DPOTrainer's init"
                    " it will be set to `128` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_prompt_length = 128

            if max_target_length is None and self.is_encoder_decoder:
                warnings.warn(
                    "When using DPODataCollatorWithPadding with an encoder decoder architecture, you should set `max_target_length` in the DPOTrainer's init"
                    " it will be set to `128` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_target_length = 128

            data_collator = DPODataCollatorWithPadding(
                tokenizer,
                max_length=max_length,
                max_prompt_length=max_prompt_length,
                label_pad_token_id=label_pad_token_id,
                padding_value=padding_value,
                truncation_mode=truncation_mode,
                is_encoder_decoder=self.is_encoder_decoder,
                max_target_length=max_target_length,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True

            args.label_names = ["chosen_labels", "rejected_labels"]
        else:
            self.use_dpo_data_collator = False

        if disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.max_length = max_length
        self.generate_during_eval = generate_during_eval
        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value

        self.begin_utd = begin_utd
        self.end_utd = end_utd
        self.beta = beta
        self.loss_type = loss_type

        if compute_metrics is None:
            compute_metrics = compute_dpo_metrics

        # args.beta = beta

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        if self.ref_model is None:
            if not hasattr(self.accelerator.unwrap_model(self.model), "disable_adapters"):
                raise ValueError(
                    "You are using a `peft` version that does not support `disable_adapters`. Please update your `peft` version to the latest version."
                )
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        self.optional_peft_ctx = (
            self.accelerator.unwrap_model(self.model).disable_adapters if self.is_peft_model else nullcontext
        )

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

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

    def concatenated_inputs(
        self,
        batch: Dict[str, Union[List, torch.LongTensor]],
        policy_output_tokens: torch.LongTensor,
        policy_output_decoded: List[str],
        policy_output_attention_mask: torch.LongTensor,
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
            policy_output: The output of the policy model. Shape: (batch_size, self.max_length)

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
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

        if self.is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1)
            concatenated_batch["concatenated_attention_mask"] = batch["prompt_attention_mask"].repeat(2, 1)

        return concatenated_batch

    def cpo_loss(
        self,
        chosen_last_hidden_state: torch.FloatTensor,
        rejected_last_hidden_state: torch.FloatTensor,
        policy_last_hidden_state: torch.FloatTensor,
        similarity: str,
        loss_type: str,
        use_other_negative_samples: str,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the CPO loss for a batch of policy and reference model log probabilities.

        Args:
            chosen_last_hidden_state: Last hidden state of the policy model for the chosen responses. Shape: (batch_size, hidden_size)
            rejected_last_hidden_state: Last hidden state of the policy model for the rejected responses. Shape: (batch_size, hidden_size)
            policy_last_hidden_state: Last hidden state of the policy model for the chosen responses. Shape: (batch_size, hidden_size)
            similarity: The similarity function to use. Either "dot" or "cosine".
            loss_type: The type of CPO loss to use. Either "sigmoid" the default CPO loss or "InfoNCE" loss from SLiC paper.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the CPO loss for each example in the batch.
        """
        if similarity == "dot":
            norm = torch.sqrt(torch.tensor(chosen_last_hidden_state.shape[-1], dtype=torch.float32)).to(self.accelerator.device)
            chosen_last_hidden_state = chosen_last_hidden_state.unsqueeze(1)
            rejected_last_hidden_state = rejected_last_hidden_state.unsqueeze(1)
            policy_last_hidden_state = policy_last_hidden_state.unsqueeze(1)
            positive_sim = torch.einsum('bij,bij->b', policy_last_hidden_state, chosen_last_hidden_state) / norm
            negative_sim = torch.einsum('bij,bij->b', policy_last_hidden_state, rejected_last_hidden_state) / norm
            if use_other_negative_samples:
                raise NotImplementedError("use_other_negative_samples is not implemented yet")
        elif similarity == "cosine":
            positive_sim = F.cosine_similarity(policy_last_hidden_state, chosen_last_hidden_state, dim=1)
            negative_sim = F.cosine_similarity(policy_last_hidden_state, rejected_last_hidden_state, dim=1)
            if use_other_negative_samples:
                raise NotImplementedError("use_other_negative_samples is not implemented yet")
        else:
            raise ValueError(f"Unknown similarity function: {similarity}. Should be one of ['dot', 'cosine']")

        if loss_type == "sigmoid":
            loss = -F.logsigmoid(positive_sim - negative_sim).mean()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Should be one of ['sigmoid', 'InfoNCE']")

        return loss, positive_sim.mean(), negative_sim.mean()

    def concatenated_rep_forward(
        self, 
        model: nn.Module, 
        concatenated_batch: Dict[str, Union[List, torch.LongTensor]],
        len_chosen: int,
        rep_approach: str,
        response_only: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """

        # TODO: check for a lot of tokens = 0 at the beginning 
        # TODO: add function for ref model
        rep_model_kwargs = {"output_hidden_states": True}
        if self.is_encoder_decoder:
            rep_model_kwargs.update(
                {
                    "labels": concatenated_batch["concatenated_labels"],
                    "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
                }
            )
        # TODO
        last_hidden_states = self.model(
                concatenated_batch["concatenated_input_ids"],
                attention_mask=concatenated_batch["concatenated_attention_mask"],
                **rep_model_kwargs,
            ).hidden_states[-1].to(torch.float32)

        chosen_last_hidden_state = last_hidden_states[:len_chosen, :, :]
        rejected_last_hidden_state = last_hidden_states[len_chosen:-len_chosen, :, :]
        policy_last_hidden_state = last_hidden_states[-len_chosen:, :, :]
        
        if rep_approach == "mean":
            chosen_last_hidden_state = chosen_last_hidden_state.mean(dim=1)
            rejected_last_hidden_state = rejected_last_hidden_state.mean(dim=1)
            policy_last_hidden_state = policy_last_hidden_state.mean(dim=1)
        elif rep_approach == 'max_pooling':
            chosen_last_hidden_state = chosen_last_hidden_state.max(dim=1).values
            rejected_last_hidden_state = rejected_last_hidden_state.max(dim=1).values
            policy_last_hidden_state = policy_last_hidden_state.max(dim=1).values
        elif rep_approach == 'mean_max_pooling':
            chosen_mean = chosen_last_hidden_state = chosen_last_hidden_state.mean(dim=1)
            chosen_max = chosen_last_hidden_state = chosen_last_hidden_state.max(dim=1).values
            chosen_last_hidden_state = torch.cat([chosen_mean, chosen_max], dim=1)

            rejected_mean = rejected_last_hidden_state = rejected_last_hidden_state.mean(dim=1)
            rejected_max = rejected_last_hidden_state = rejected_last_hidden_state.max(dim=1).values
            rejected_last_hidden_state = torch.cat([rejected_mean, rejected_max], dim=1)

            policy_mean = policy_last_hidden_state = policy_last_hidden_state.mean(dim=1)
            policy_max = policy_last_hidden_state = policy_last_hidden_state.max(dim=1).values
            policy_last_hidden_state = torch.cat([policy_mean, policy_max], dim=1)
        
        if response_only:
            raise NotImplementedError("response_only is not implemented yet")

        return (chosen_last_hidden_state, rejected_last_hidden_state, policy_last_hidden_state)

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        rep_approach: str = "mean",
        response_only: bool = False,
        similarity: str = "dot",
        loss_type: str = "sigmoid",
        use_other_negative_samples: str = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        TODO how to save the forward training time for training rep moel
        1. set a adapter for the training
        Approach 1:
            a. at the beginning we are focusing on training rep model (utd ratio is high)
            b. at the end of training we are focusing on training the policy model (utd ratio is low)
        Approach 2:
            a. clip the rep_distance: if the rep distance is far enough 
               go for policy training 
        """
        utd_ratio = int(self.begin_utd - (self.begin_utd - self.end_utd) * self.state.global_step / self.state.num_train_epochs)

        self.model.set_adapter("policy")  
        policy_output_tokens, policy_output_decoded = self.get_batch_samples(inputs)
        policy_output_attention_mask = (policy_output_tokens != 0).to(torch.int64)
            
        concatenated_inputs = self.concatenated_inputs(
                inputs, policy_output_tokens, policy_output_decoded, policy_output_attention_mask
            )
            
        self.model.set_adapter("rep")
        
        len_chosen = inputs['chosen_input_ids'].shape[0]
        (
            chosen_last_hidden_state,
            rejected_last_hidden_state,
            policy_last_hidden_state,
        ) = self.concatenated_rep_forward(model, 
                                        concatenated_inputs,
                                        len_chosen,
                                        rep_approach, 
                                        response_only)

        loss, chosen_sim, rejected_sim = self.cpo_loss(
            chosen_last_hidden_state,
            rejected_last_hidden_state,
            policy_last_hidden_state,
            similarity,
            loss_type,
            use_other_negative_samples
        )
        outputs = OrderedDict(
            {
                "loss": loss,
                "chosen_sim": chosen_sim,
                "rejected_sim": rejected_sim,
                "distance_sim": chosen_sim - rejected_sim,
            }
        )
        
        # if self.state.global_step % utd_ratio == 0:
            # training the policy model
        self.model.set_adapter(['policy'])
        # else:
            # training the rep model
            # self.model.set_adapter("rep")

        return (loss, outputs) if return_outputs else loss

    def get_batch_samples(
        self,
        batch: Dict[str, torch.LongTensor],
        generate_ref_response: bool = False,
    ) -> Tuple[str, str]:
        """
        Generate response with the model given the query tensor.

        Args:
            batch: A batch of data. Must contain the keys 'prompt_input_ids' and 'prompt_attention_mask', which are tensors of shape (batch_size, sequence_length).
            generate_ref_response: If True, generate a reference response as well.
        Returns:
            `List[str]`: A list of length `batch_size` containing the tokens and decoded responses."""

        policy_output = self.model.generate(
            input_ids=batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            max_length=self.max_length,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if generate_ref_response:
            with self.optional_peft_ctx():
                reference_output = self.model.generate(
                    batch["prompt_input_ids"],
                    attention_mask=batch["prompt_attention_mask"],
                    max_length=self.max_length,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # reference_output = pad_to_length(reference_output, self.max_length, self.tokenizer.pad_token_id)
            reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        # policy_output = pad_to_length(policy_output, self.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        if generate_ref_response:
            return policy_output, policy_output_decoded, reference_output, reference_output_decoded
        return policy_output, policy_output_decoded

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )

        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """

        # Sample and save to game log if requested (for one batch to save time)
        if self.generate_during_eval:
            # Generate random indices within the range of the total number of samples
            num_samples = len(dataloader.dataset)
            random_indices = random.sample(range(num_samples), k=self.args.eval_batch_size)

            # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
            random_batch_dataset = dataloader.dataset.select(random_indices)
            random_batch = self.data_collator(random_batch_dataset)
            random_batch = self._prepare_inputs(random_batch)
            
            self.model.set_adapter("policy")
            _, policy_output_decoded, _, ref_output_decoded = self.get_batch_samples(
                random_batch, generate_ref_response=True
            )

            self.log(
                {
                    "game_log": wandb.Table(
                        columns=["Prompt", "Policy", "Ref Model"],
                        rows=[
                            [prompt, pol[len(prompt) :], ref[len(prompt) :]]
                            for prompt, pol, ref in zip(
                                random_batch["prompt"], policy_output_decoded, ref_output_decoded
                            )
                        ],
                    )
                }
            )
            self.state.log_history.pop()

        # Base evaluation
        initial_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        return initial_output


def compute_dpo_metrics(eval_preds: EvalPrediction):
    (
        chosen_rewards,
        rejected_rewards,
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
    ) = eval_preds.predictions

    reward_accuracies = (chosen_rewards > rejected_rewards).mean()

    metrics = {}
    metrics["rewards/chosen"] = chosen_rewards.mean()
    metrics["rewards/rejected"] = rejected_rewards.mean()
    metrics["rewards/accuracies"] = reward_accuracies.mean()
    metrics["rewards/margins"] = (chosen_rewards - rejected_rewards).mean()
    metrics["logps/rejected"] = policy_rejected_logps.mean()
    metrics["logps/chosen"] = policy_chosen_logps.mean()

    return metrics
