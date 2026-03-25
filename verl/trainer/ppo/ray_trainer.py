# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import re
import time
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from string import Template
from typing import Any, Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.model import compute_position_id_with_mask
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.torch_functional import postprocess_data
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, using max_colocate_count=3: actor_critic_ref, rollout, reward model (optional)
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=3, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # Check if sum_pi_squared is available
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            # Get pre-computed rollout IS weights if available
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = config.actor_rollout_ref.actor.get("self_distillation", {}).get("reprompt_truncation", "error")
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)
        # legacy reward model implementation
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_reward_loop = self.config.reward_model.use_reward_loop

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        # BEACON: blindspot map (EMA across steps) for adaptive calibration weights
        # Maps error_type_id → B-quadrant fraction (how often model predicts Pass but actually fails)
        self._blindspot_ema = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        self._blindspot_ema_alpha = 0.3  # EMA smoothing factor

        # Binary diagnosis for small models (<= 4B params), 5-class for larger
        model_path = config.actor_rollout_ref.model.path.lower()
        self._use_binary_diagnosis = any(s in model_path for s in ["1.7b", "1.5b", "0.5b", "0.6b", "3b", "4b"])
        if self._use_binary_diagnosis:
            print(f"[BEACON] Using BINARY diagnosis (Pass/Fail) for small model: {config.actor_rollout_ref.model.path}")
        else:
            print(f"[BEACON] Using 5-CLASS diagnosis (Pass/CE/RE/WA/TLE) for model: {config.actor_rollout_ref.model.path}")

        # Pre-load CoderJudge actor on a dedicated GPU (async, overlaps with worker init)
        self._init_code_judge_actor()

    def _init_code_judge_actor(self):
        """Initialize the CoderJudge Ray actor (lazy, once per training)."""
        if hasattr(self, "_judge_actor") and self._judge_actor is not None:
            return

        judge_config = getattr(self.config.reward_model, "coder_judge", None)
        if judge_config is None or not getattr(judge_config, "enable", False):
            self._judge_actor = None
            return

        from verl.utils.reward_score.sandbox_fusion.code_judge import CoderJudgeActor

        self._judge_actor = CoderJudgeActor.options(num_gpus=1).remote(
            model_name=getattr(judge_config, "model_name", "Qwen/Qwen2.5-Coder-14B-Instruct"),
            tensor_parallel_size=getattr(judge_config, "tensor_parallel_size", 1),
            gpu_memory_utilization=getattr(judge_config, "gpu_memory_utilization", 0.85),
            max_model_len=getattr(judge_config, "max_model_len", 4096),
            max_num_seqs=getattr(judge_config, "max_num_seqs", 256),
            judge_n=getattr(judge_config, "judge_n", 3),
            judge_temperature=getattr(judge_config, "judge_temperature", 0.3),
            confidence_threshold=getattr(judge_config, "confidence_threshold", 0.5),
        )
        print("[CoderJudge] Ray actor created, model loading on dedicated GPU...")

    def _run_self_critique(self, batch: "DataProto") -> "DataProto":
        """BEACON Error Self-Diagnosis: student predicts the error TYPE of its own code.

        Predicts one of 5 categories: Pass, Compile Error, Runtime Error, Wrong Answer, Timeout.
        After sandbox, compare predictions with actual error types for:
          - 4-quadrant BEACON analysis (A/B/C/D)
          - 5×5 confusion matrix (predicted vs actual error type)
          - Blindspot topology profiling per error type
          - Code Calibration Score (CCS)
        """
        prompt_ids = batch.batch["prompts"]
        response_ids = batch.batch["responses"]
        attention_mask = batch.batch["attention_mask"]
        prompt_len = prompt_ids.shape[-1]
        batch_size = len(batch)

        if "extra_info" not in batch.non_tensor_batch:
            batch.non_tensor_batch["extra_info"] = [{} for _ in range(batch_size)]

        # Build critique prompts for each sample
        critique_messages_list = []
        critique_indices = []  # track which batch indices we're critiquing

        for i in range(batch_size):
            if batch.non_tensor_batch["extra_info"][i] is None:
                batch.non_tensor_batch["extra_info"][i] = {}

            data_source = batch[i].non_tensor_batch.get(
                self.config.data.reward_fn_key, ""
            )
            if data_source not in ("codecontests", "apps", "codeforces", "taco", "livecodebench"):
                continue

            valid_response_length = attention_mask[i, prompt_len:].sum()
            valid_response_ids = response_ids[i][:valid_response_length]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            # Extract code
            solution = response_str
            if "```python" in response_str:
                solution = response_str.split("```python")[-1].split("```")[0]
            elif "```" in response_str:
                parts = response_str.split("```")
                if len(parts) >= 2:
                    solution = parts[1]
                    if "\n" in solution:
                        first_line, rest = solution.split("\n", 1)
                        if first_line.strip().isalpha():
                            solution = rest

            # Truncate to stay under max_prompt_length=4096 tokens
            # problem + solution + template ≈ need each under 1500 chars to stay safe
            if len(solution) > 1500:
                solution = solution[:1500] + "\n# ... (truncated)"

            # Get original problem from raw_prompt
            problem = batch.non_tensor_batch["raw_prompt"][i][-1]["content"]
            if len(problem) > 1500:
                problem = problem[:1500] + "\n... (truncated)"

            # Binary diagnosis for small models (<= 4B), 5-class for larger
            if getattr(self, '_use_binary_diagnosis', False):
                critique_prompt = (
                    f"Given the following programming problem and a candidate solution, "
                    f"predict whether it will pass all test cases.\n\n"
                    f"## Problem\n{problem}\n\n"
                    f"## Solution\n```python\n{solution}\n```\n\n"
                    f"Will this solution pass all test cases? Choose exactly one:\n"
                    f"A) Pass\n"
                    f"B) Fail\n\n"
                    f"Put your answer letter in \\boxed{{}}."
                )
            else:
                critique_prompt = (
                    f"Given the following programming problem and a candidate solution, "
                    f"predict the execution result.\n\n"
                    f"## Problem\n{problem}\n\n"
                    f"## Solution\n```python\n{solution}\n```\n\n"
                    f"Predict the execution result. Choose exactly one from:\n"
                    f"A) Pass\n"
                    f"B) Compile Error\n"
                    f"C) Runtime Error\n"
                    f"D) Wrong Answer\n"
                    f"E) Timeout\n\n"
                    f"Put your answer letter in \\boxed{{}}."
                )

            critique_messages_list.append([{"role": "user", "content": critique_prompt}])
            critique_indices.append(i)

        if not critique_indices:
            return batch

        import numpy as np

        # Build DataProto with raw_prompt in non_tensor_batch for async_rollout_manager
        # The AgentLoopWorker handles tokenization internally via raw_prompt
        n_critique = len(critique_indices)
        raw_prompts = np.array(critique_messages_list, dtype=object)

        # Need agent_name for dispatch; use the same default agent loop
        agent_config = self.config.actor_rollout_ref.rollout.get("agent", {})
        default_agent = agent_config.get("default_agent_loop", "default")
        agent_names = np.array([default_agent] * n_critique, dtype=object)

        # Dummy tensors — async_rollout_manager only uses non_tensor_batch
        dummy_ids = torch.zeros(n_critique, 1, dtype=torch.long)

        critique_batch = DataProto.from_dict(
            tensors={
                "input_ids": dummy_ids,
                "attention_mask": torch.ones_like(dummy_ids),
                "position_ids": torch.zeros_like(dummy_ids),
            },
            non_tensors={
                "raw_prompt": raw_prompts,
                "agent_name": agent_names,
            },
        )
        # Set generation params: short output, low temperature for deterministic judgment
        critique_batch.meta_info["temperature"] = 0.1
        # Binary diagnosis needs fewer tokens (just Pass/Fail), 5-class needs more reasoning
        critique_batch.meta_info["max_new_tokens"] = 64 if getattr(self, '_use_binary_diagnosis', False) else 512
        critique_batch.meta_info["top_p"] = 0.9

        # Generate critiques using async rollout manager (same as main rollout)
        critique_output = self.async_rollout_manager.generate_sequences(critique_batch)
        critique_response_ids = critique_output.batch["responses"]
        critique_response_mask = critique_output.batch.get("response_mask", critique_output.batch["attention_mask"])

        # Parse critique outputs
        n_predicted_fail = 0
        pred_counts = {}  # count predictions per error type
        prompt_len = critique_output.batch["prompts"].shape[-1] if "prompts" in critique_output.batch else 0
        for idx_in_critique, batch_idx in enumerate(critique_indices):
            # Get response tokens (after prompt)
            if "responses" in critique_output.batch:
                resp_ids = critique_response_ids[idx_in_critique]
                if "response_mask" in critique_output.batch:
                    valid_len = critique_output.batch["response_mask"][idx_in_critique].sum().int().item()
                else:
                    valid_len = (resp_ids != 0).sum().int().item()
                valid_ids = resp_ids[:valid_len]
            else:
                # Fallback: use full input_ids after prompt
                full_ids = critique_output.batch["input_ids"][idx_in_critique]
                valid_ids = full_ids[prompt_len:]
                valid_ids = valid_ids[valid_ids != 0]

            critique_text = self.tokenizer.decode(valid_ids, skip_special_tokens=True).strip()

            import re
            if getattr(self, '_use_binary_diagnosis', False):
                # Binary mode: A=Pass, B=Fail
                predicted_pass = True  # default to Pass if ambiguous
                boxed_matches = re.findall(r'\\boxed\{([AB])\}', critique_text)
                if boxed_matches:
                    predicted_pass = (boxed_matches[-1] == "A")
                else:
                    matches = re.findall(r'\b([AB])\b', critique_text.strip())
                    if matches:
                        predicted_pass = (matches[-1] == "A")
                predicted_error_id = 0 if predicted_pass else 1  # 0=Pass, 1=Fail
            else:
                # 5-class mode: A=Pass, B=CE, C=RE, D=WA, E=TLE
                _LETTER_TO_ERROR_ID = {"A": 0, "B": 4, "C": 1, "D": 2, "E": 3}
                predicted_error_id = None
                boxed_matches = re.findall(r'\\boxed\{([A-E])\}', critique_text)
                if boxed_matches:
                    predicted_error_id = _LETTER_TO_ERROR_ID.get(boxed_matches[-1])
                else:
                    matches = re.findall(r'\b([A-E])\b', critique_text.strip())
                    if matches:
                        predicted_error_id = _LETTER_TO_ERROR_ID.get(matches[-1])
                if predicted_error_id is None:
                    predicted_error_id = 0
                predicted_pass = (predicted_error_id == 0)

            if not predicted_pass:
                n_predicted_fail += 1

            _ERROR_NAMES = {0: "Pass", 1: "RE", 2: "WA", 3: "TLE", 4: "CE"} if not getattr(self, '_use_binary_diagnosis', False) else {0: "Pass", 1: "Fail"}
            critique = {
                "predicted_pass": predicted_pass,
                "predicted_error_id": predicted_error_id,
                "predicted_error_name": _ERROR_NAMES.get(predicted_error_id, "Fail"),
                "critique_text": critique_text[:500],
                "critique_messages": critique_messages_list[idx_in_critique],  # store for auxiliary critique loss
            }
            batch.non_tensor_batch["extra_info"][batch_idx]["self_critique"] = critique
            pred_counts[predicted_error_id] = pred_counts.get(predicted_error_id, 0) + 1

        n_analyzed = len(critique_indices)
        _ERROR_NAMES = {0: "Pass", 1: "RE", 2: "WA", 3: "TLE", 4: "CE"} if not getattr(self, '_use_binary_diagnosis', False) else {0: "Pass", 1: "Fail"}
        pred_dist = " ".join(f"{_ERROR_NAMES.get(k, '?')}={v}" for k, v in sorted(pred_counts.items()))
        print(
            f"[ErrorDiagnosis] analyzed={n_analyzed} | predicted_fail={n_predicted_fail} "
            f"({100*n_predicted_fail/n_analyzed:.1f}%) | dist: {pred_dist}",
            flush=True,
        )

        return batch

    def _compute_critique_accuracy(
        self,
        batch: "DataProto",
        reward_tensor: torch.Tensor,
        reward_extra_infos_dict: Optional[dict[str, list]] = None,
    ) -> dict[str, float]:
        """BEACON: Blindspot-aware Execution-Aligned Calibration with On-policy Normalization.

        Compares error self-diagnosis predictions with actual sandbox results.
        Builds 4-quadrant analysis:
          A (Pass + predicted Pass):  model truly understands → weight 0.8 (light distillation)
          B (Fail + predicted Pass):  overconfident blindspot → weight 2.0 (strongest distillation)
          C (Pass + predicted Fail):  underconfident → weight 1.2 (calibration correction)
          D (Fail + predicted Fail):  knows it doesn't know → weight 1.0 (standard)

        Also tracks:
          - 5-class error type diagnosis accuracy (exact match)
          - Blindspot topology map: per error type, fraction in B quadrant
          - CCS (Code Calibration Score): 1 - mean|predicted_pass - actual_pass|
        """
        _ERROR_NAMES = {0: "Pass", 1: "RE", 2: "WA", 3: "TLE", 4: "CE"}

        batch_size = len(batch)
        extra_infos = batch.non_tensor_batch.get("extra_info", None)
        if extra_infos is None:
            return {}

        # Get per-sample scores from reward_tensor (sum over seq dim)
        seq_scores = reward_tensor.sum(dim=-1).detach().cpu().numpy()

        # Get actual error types from sandbox
        actual_error_types = None
        if reward_extra_infos_dict is not None:
            raw_et = reward_extra_infos_dict.get("error_type", [])
            if len(raw_et) >= batch_size:
                actual_error_types = [int(raw_et[i]) for i in range(batch_size)]

        # --- 4-quadrant counters ---
        quad_A = 0  # Pass + predicted Pass (truly understands)
        quad_B = 0  # Fail + predicted Pass (overconfident blindspot)
        quad_C = 0  # Pass + predicted Fail (underconfident)
        quad_D = 0  # Fail + predicted Fail (knows it doesn't know)
        n_critiqued = 0

        # --- 5-class diagnosis accuracy ---
        n_exact_match = 0
        confusion_matrix = {}  # (predicted_name, actual_name) → count

        # --- Blindspot topology: per actual error type, count B quadrant ---
        blindspot_B_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # B quadrant per error type
        blindspot_total_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # total per error type

        # --- CCS accumulator ---
        ccs_diffs = []

        # Base quadrant weights
        W_A = 0.8   # truly understands → light distillation
        W_B_base = 2.0   # overconfident blindspot → strongest distillation (modulated by blindspot map)
        W_C = 1.2   # underconfident → calibration correction
        W_D = 1.0   # knows it doesn't know → standard

        weights = torch.ones(batch_size, dtype=torch.float32, device=reward_tensor.device)

        for i in range(batch_size):
            if extra_infos[i] is None:
                continue
            critique = extra_infos[i].get("self_critique", None)
            if critique is None:
                continue

            n_critiqued += 1
            predicted_pass = critique["predicted_pass"]
            predicted_error_id = critique["predicted_error_id"]
            actual_pass = float(seq_scores[i]) > 0

            # Actual error type from sandbox
            actual_error_id = 0 if actual_pass else (actual_error_types[i] if actual_error_types else -1)

            # Diagnosis accuracy: binary (pass/fail match) or 5-class exact match
            _ERROR_NAMES = {0: "Pass", 1: "RE", 2: "WA", 3: "TLE", 4: "CE"} if not getattr(self, '_use_binary_diagnosis', False) else {0: "Pass", 1: "Fail"}
            if getattr(self, '_use_binary_diagnosis', False):
                # Binary mode: just check pass/fail match
                if predicted_pass == actual_pass:
                    n_exact_match += 1
                confusion_key = ("Pass" if predicted_pass else "Fail", "Pass" if actual_pass else "Fail")
                confusion_matrix[confusion_key] = confusion_matrix.get(confusion_key, 0) + 1
            elif actual_error_types and actual_error_id >= 0:
                if predicted_error_id == actual_error_id:
                    n_exact_match += 1
                confusion_key = (_ERROR_NAMES.get(predicted_error_id, "?"), _ERROR_NAMES.get(actual_error_id, "?"))
                confusion_matrix[confusion_key] = confusion_matrix.get(confusion_key, 0) + 1

            # CCS: |predicted_pass(0/1) - actual_pass(0/1)|
            ccs_diffs.append(abs(float(predicted_pass) - float(actual_pass)))

            # Blindspot topology tracking
            if getattr(self, '_use_binary_diagnosis', False):
                # Binary mode: use id=1 for all failures
                if not actual_pass:
                    blindspot_total_counts[1] = blindspot_total_counts.get(1, 0) + 1
            elif actual_error_types and actual_error_id >= 0:
                blindspot_total_counts[actual_error_id] = blindspot_total_counts.get(actual_error_id, 0) + 1

            # 4-quadrant assignment — store in extra_info for Metacognitive Distillation
            if actual_pass and predicted_pass:
                quad_A += 1
                weights[i] = W_A
                quadrant = "A"
            elif not actual_pass and predicted_pass:
                quad_B += 1
                # Asymmetric Calibration: W_B modulated by blindspot severity
                if getattr(self, '_use_binary_diagnosis', False):
                    blindspot_severity = self._blindspot_ema.get(1, 0.0)  # single Fail type
                else:
                    blindspot_severity = self._blindspot_ema.get(actual_error_id, 0.0)
                weights[i] = W_B_base * (1.0 + blindspot_severity)
                quadrant = "B"
                # Track blindspot
                if getattr(self, '_use_binary_diagnosis', False):
                    blindspot_B_counts[1] = blindspot_B_counts.get(1, 0) + 1
                elif actual_error_types and actual_error_id >= 0:
                    blindspot_B_counts[actual_error_id] = blindspot_B_counts.get(actual_error_id, 0) + 1
            elif actual_pass and not predicted_pass:
                quad_C += 1
                weights[i] = W_C
                quadrant = "C"
            else:  # not actual_pass and not predicted_pass
                quad_D += 1
                weights[i] = W_D
                quadrant = "D"

            # Store quadrant label for use in Metacognitive Distillation reprompt
            extra_infos[i]["beacon_quadrant"] = quadrant
            extra_infos[i]["actual_error_id"] = actual_error_id

        if n_critiqued == 0:
            return {}

        # Store weights in batch for use by distillation
        batch.batch["critique_weight"] = weights

        # --- Build auxiliary critique training data ---
        # Tokenize critique prompts + correct answers for auxiliary cross-entropy loss
        critique_loss_beta = self.config.actor_rollout_ref.actor.self_distillation.get("critique_loss_beta", 0.0)
        if critique_loss_beta > 0:
            self._build_critique_training_data(batch, extra_infos, batch_size)

        # --- Compute metrics ---
        # Binary accuracy (same as before, for comparison)
        binary_acc = (quad_A + quad_D) / n_critiqued

        # Diagnosis accuracy
        _is_binary = getattr(self, '_use_binary_diagnosis', False)
        diag_acc = n_exact_match / n_critiqued if (_is_binary or actual_error_types) else 0.0

        # CCS: Code Calibration Score = 1 - mean|pred - actual|
        ccs = 1.0 - (sum(ccs_diffs) / len(ccs_diffs)) if ccs_diffs else 0.0

        # Blindspot map
        _fail_eids = [1] if _is_binary else [1, 2, 3, 4]
        _ERROR_NAMES = {0: "Pass", 1: "Fail"} if _is_binary else {0: "Pass", 1: "RE", 2: "WA", 3: "TLE", 4: "CE"}
        blindspot_str_parts = []
        for eid in _fail_eids:
            total = blindspot_total_counts.get(eid, 0)
            b_count = blindspot_B_counts.get(eid, 0)
            if total > 0:
                b_frac = b_count / total
                blindspot_str_parts.append(f"{_ERROR_NAMES[eid]}={b_frac:.2f}({b_count}/{total})")

        metrics = {
            "beacon/quad_A": quad_A,
            "beacon/quad_B": quad_B,
            "beacon/quad_C": quad_C,
            "beacon/quad_D": quad_D,
            "beacon/quad_A_frac": quad_A / n_critiqued,
            "beacon/quad_B_frac": quad_B / n_critiqued,
            "beacon/quad_C_frac": quad_C / n_critiqued,
            "beacon/quad_D_frac": quad_D / n_critiqued,
            "beacon/binary_accuracy": binary_acc,
            "beacon/diagnosis_accuracy": diag_acc,
            "beacon/ccs": ccs,
            "beacon/n_critiqued": n_critiqued,
            # Blindspot per error type (B fraction)
            **{f"beacon/blindspot_B_{_ERROR_NAMES[eid]}": (blindspot_B_counts.get(eid, 0) / blindspot_total_counts.get(eid, 1)) if blindspot_total_counts.get(eid, 0) > 0 else 0.0 for eid in _fail_eids},
            # EMA blindspot values
            **{f"beacon/blindspot_ema_{_ERROR_NAMES[eid]}": self._blindspot_ema[eid] for eid in _fail_eids},
            # Effective B-quadrant weights
            **{f"beacon/W_B_eff_{_ERROR_NAMES[eid]}": W_B_base * (1.0 + self._blindspot_ema[eid]) for eid in _fail_eids},
        }

        # Update blindspot EMA
        alpha = self._blindspot_ema_alpha
        for eid in _fail_eids:
            total = blindspot_total_counts.get(eid, 0)
            if total > 0:
                b_frac = blindspot_B_counts.get(eid, 0) / total
                self._blindspot_ema[eid] = alpha * b_frac + (1 - alpha) * self._blindspot_ema[eid]

        blindspot_str = " | ".join(blindspot_str_parts) if blindspot_str_parts else "N/A"

        confusion_str = ""
        if confusion_matrix:
            sorted_cm = sorted(confusion_matrix.items(), key=lambda x: -x[1])
            top_entries = sorted_cm[:8]
            confusion_str = " ".join(f"{p}->{a}:{c}" for (p, a), c in top_entries)
            # Add confusion matrix to metrics for logging
            for (pred, actual), count in confusion_matrix.items():
                metrics[f"beacon/cm/{pred}_to_{actual}"] = count

        print(
            f"[BEACON] n={n_critiqued} | A={quad_A} B={quad_B} C={quad_C} D={quad_D} | "
            f"bin_acc={binary_acc:.3f} diag_acc={diag_acc:.3f} CCS={ccs:.3f} | "
            f"blindspot(B): {blindspot_str}",
            flush=True,
        )
        if confusion_str:
            print(f"[BEACON-CM] {confusion_str}", flush=True)

        eff_w_str = " ".join(f"{_ERROR_NAMES[eid]}={W_B_base*(1+self._blindspot_ema[eid]):.2f}" for eid in _fail_eids)
        print(f"[BEACON-CalibW] effective W_B: {eff_w_str}", flush=True)

        return metrics

    def _build_critique_training_data(self, batch: "DataProto", extra_infos: list, batch_size: int):
        """Build tokenized critique training sequences for auxiliary critique loss.

        For each critiqued sample, creates a sequence: [critique_prompt + correct_answer]
        with response_mask on the answer tokens only. Non-critiqued samples get zero masks.
        """
        _is_binary = getattr(self, '_use_binary_diagnosis', False)
        # Reverse mapping: error_id → correct letter answer
        if _is_binary:
            _ERROR_ID_TO_LETTER = {0: "A", 1: "B"}
        else:
            _ERROR_ID_TO_LETTER = {0: "A", 4: "B", 1: "C", 2: "D", 3: "E"}

        critique_train_messages = []
        critique_train_indices = []

        for i in range(batch_size):
            if extra_infos[i] is None:
                continue
            critique = extra_infos[i].get("self_critique", None)
            if critique is None or "critique_messages" not in critique:
                continue

            actual_error_id = extra_infos[i].get("actual_error_id", 0)
            actual_pass = (actual_error_id == 0)

            # Build enriched correct answer with reasoning
            if actual_pass:
                correct_answer = (
                    "This solution correctly handles all test cases and produces the expected output. "
                    "\\boxed{A}"
                )
            else:
                _ERROR_DESCRIPTIONS = {
                    1: "a Runtime Error, likely due to edge cases such as empty input, index out of range, or division by zero",
                    2: "a Wrong Answer, indicating the algorithm logic is incorrect or misses certain boundary conditions",
                    3: "a Timeout, meaning the time complexity is too high and needs optimization",
                    4: "a Compile Error, likely due to syntax errors, missing imports, or undefined variables",
                }
                err_desc = _ERROR_DESCRIPTIONS.get(actual_error_id, "a failure during execution")
                correct_answer = (
                    f"This solution will fail with {err_desc}. "
                    "\\boxed{B}"
                )

            # Build training messages: user critique prompt + assistant correct answer
            user_msg = critique["critique_messages"][0]  # {"role": "user", "content": ...}
            train_msgs = [
                user_msg,
                {"role": "assistant", "content": correct_answer},
            ]
            critique_train_messages.append(train_msgs)
            critique_train_indices.append(i)

        if not critique_train_indices:
            return

        # Tokenize all critique training sequences
        max_critique_len = 512  # cap total length for efficiency
        try:
            all_tokenized = []
            all_prompt_lens = []
            for msgs in critique_train_messages:
                # Full sequence: prompt + correct answer
                full_tok = self.tokenizer.apply_chat_template(
                    msgs, tokenize=True, max_length=max_critique_len,
                    padding=False, truncation=True, add_generation_prompt=False,
                    enable_thinking=False,
                )
                if isinstance(full_tok, torch.Tensor):
                    full_tok = full_tok.squeeze(0).tolist()
                elif isinstance(full_tok, list) and len(full_tok) > 0 and isinstance(full_tok[0], list):
                    full_tok = full_tok[0]
                all_tokenized.append(full_tok)

                # Prompt only (to find where answer starts)
                prompt_tok = self.tokenizer.apply_chat_template(
                    [msgs[0]], tokenize=True, max_length=max_critique_len,
                    padding=False, truncation=True, add_generation_prompt=True,
                    enable_thinking=False,
                )
                if isinstance(prompt_tok, torch.Tensor):
                    prompt_tok = prompt_tok.squeeze(0).tolist()
                elif isinstance(prompt_tok, list) and len(prompt_tok) > 0 and isinstance(prompt_tok[0], list):
                    prompt_tok = prompt_tok[0]
                all_prompt_lens.append(len(prompt_tok))
        except Exception as e:
            print(f"[BEACON-CritiqueLoss] tokenization failed: {e}", flush=True)
            return

        # Pad to same length
        max_len = max(len(t) for t in all_tokenized)
        critique_input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        critique_attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        critique_response_mask = torch.zeros(batch_size, max_len, dtype=torch.float32)

        for idx_in_list, batch_idx in enumerate(critique_train_indices):
            seq = all_tokenized[idx_in_list]
            seq_len = len(seq)
            prompt_len = all_prompt_lens[idx_in_list]
            critique_input_ids[batch_idx, :seq_len] = torch.tensor(seq, dtype=torch.long)
            critique_attention_mask[batch_idx, :seq_len] = 1
            # Response mask: only on the answer tokens (after prompt)
            if prompt_len < seq_len:
                critique_response_mask[batch_idx, prompt_len:seq_len] = 1.0

        # Compute position_ids from attention_mask
        critique_position_ids = critique_attention_mask.long().cumsum(-1) - 1
        critique_position_ids.masked_fill_(critique_attention_mask == 0, 0)

        batch.batch["critique_input_ids"] = critique_input_ids
        batch.batch["critique_attention_mask"] = critique_attention_mask
        batch.batch["critique_position_ids"] = critique_position_ids
        batch.batch["critique_response_mask"] = critique_response_mask

        n_with_data = (critique_response_mask.sum(dim=-1) > 0).sum().item()
        print(f"[BEACON-CritiqueLoss] built training data: {n_with_data}/{batch_size} samples", flush=True)

    def _run_code_judge(self, batch: "DataProto") -> "DataProto":
        """
        Run CoderJudge on generated code before sandbox execution.

        Extracts code in driver (no GPU needed),
        sends to CoderJudge Ray actor for batch inference (has GPU).
        Stores judge_results in extra_info for each sample.
        """
        judge_config = getattr(self.config.reward_model, "coder_judge", None)
        if judge_config is None or not getattr(judge_config, "enable", False):
            return batch

        self._init_code_judge_actor()
        if self._judge_actor is None:
            return batch

        import ast as _ast
        import json

        # Extract codes (CPU-only, in driver)
        prompt_ids = batch.batch["prompts"]
        response_ids = batch.batch["responses"]
        attention_mask = batch.batch["attention_mask"]
        prompt_len = prompt_ids.shape[-1]

        codes = []
        code_sample_indices = []
        ast_fail_count = 0

        for i in range(len(batch)):
            data_item = batch[i]
            data_source = data_item.non_tensor_batch.get(self.config.data.reward_fn_key, "")

            if data_source not in ("codecontests", "apps", "codeforces", "taco", "livecodebench"):
                continue

            valid_response_length = attention_mask[i, prompt_len:].sum()
            valid_response_ids = response_ids[i][:valid_response_length]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            # Extract code from markdown
            solution = response_str
            if "```python" in response_str:
                solution = response_str.split("```python")[-1].split("```")[0]
            elif "```" in response_str:
                parts = response_str.split("```")
                if len(parts) >= 2:
                    solution = parts[1]
                    if "\n" in solution:
                        first_line, rest = solution.split("\n", 1)
                        if first_line.strip().isalpha():
                            solution = rest
            else:
                continue

            # Skip obviously too-long code
            MAX_CODE_CHARS = 16000
            if len(solution) > MAX_CODE_CHARS:
                continue

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            try:
                tc = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
                if not isinstance(tc, dict) or ("inputs" not in tc and "assert_case" not in tc):
                    continue
            except Exception:
                continue

            # AST check: syntax errors are 100% certain failures, skip LLM
            try:
                _ast.parse(solution)
            except SyntaxError as _se:
                if "extra_info" not in batch.non_tensor_batch:
                    batch.non_tensor_batch["extra_info"] = [{} for _ in range(len(batch))]
                if batch.non_tensor_batch["extra_info"][i] is None:
                    batch.non_tensor_batch["extra_info"][i] = {}
                batch.non_tensor_batch["extra_info"][i]["judge_results"] = {
                    "passed": False,
                    "error_type": "syntax_error",
                    "reason": f"SyntaxError: {_se.msg} (line {_se.lineno})",
                }
                ast_fail_count += 1
                continue

            codes.append(solution)
            code_sample_indices.append(i)

        if ast_fail_count > 0:
            print(f"[CoderJudge] AST syntax check: {ast_fail_count} failed (skipped LLM)", flush=True)

        if not codes:
            batch.meta_info["judge_total"] = ast_fail_count
            batch.meta_info["judge_skip"] = ast_fail_count
            batch.meta_info["judge_sandbox"] = 0
            return batch

        print(f"[CoderJudge] Sending {len(codes)} samples to LLM judge (ast_fail={ast_fail_count})", flush=True)

        # Send to Ray actor for GPU inference
        all_judge_results = ray.get(self._judge_actor.batch_judge.remote(codes))

        # Store results in extra_info
        if "extra_info" not in batch.non_tensor_batch:
            batch.non_tensor_batch["extra_info"] = [{} for _ in range(len(batch))]
        extra_info = batch.non_tensor_batch["extra_info"]
        for idx, sample_i in enumerate(code_sample_indices):
            if extra_info[sample_i] is None:
                extra_info[sample_i] = {}
            extra_info[sample_i]["judge_results"] = all_judge_results[idx]

        llm_skip = sum(1 for r in all_judge_results if not r.get("passed", True))
        total_skip = ast_fail_count + llm_skip
        total_sandbox = len(all_judge_results) - llm_skip

        print(f"[CoderJudge] total={len(batch)} | ast_fail={ast_fail_count} | llm_skip={llm_skip} | sandbox={total_sandbox}", flush=True)

        batch.meta_info["judge_total"] = len(batch)
        batch.meta_info["judge_skip"] = total_skip
        batch.meta_info["judge_sandbox"] = total_sandbox

        return batch

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _compute_or_extract_reward(
        self,
        batch: DataProto,
        reward_fn=None,
        return_dict: bool = False,
        sum_reward: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor | dict[str, Any]:
        """
        Compute or extract reward from batch.

        When use_reward_loop=True, rewards are already computed during generate_sequences
        and stored in rm_scores. This method directly extracts them instead of calling
        reward functions which would only perform format conversion.

        Args:
            batch: DataProto containing the batch data
            reward_fn: Reward function to use if rm_scores doesn't exist (for training/validation)
            return_dict: Whether to return dict format with reward_extra_info (for validation)
            sum_reward: Whether to sum reward tensor along last dimension (for REMAX baseline)

        Returns:
            If return_dict=True: dict with "reward_tensor" and "reward_extra_info"
            If return_dict=False and sum_reward=True: summed reward_tensor (1D tensor)
            If return_dict=False and sum_reward=False: reward_tensor (2D tensor)
        """
        # When rm_scores already exists, extract it directly (format conversion only)
        if "rm_scores" in batch.batch.keys():
            reward_tensor = batch.batch["rm_scores"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)

            if return_dict:
                # Extract reward_extra_info if available
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_info = (
                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
                )
                return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
            else:
                # If sum_reward=True, only return tensor (for REMAX baseline)
                if sum_reward:
                    return reward_tensor
                # Otherwise, return tuple with reward_extra_info (for training loop)
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_infos_dict = (
                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
                )
                return reward_tensor, reward_extra_infos_dict

        # Otherwise, compute reward using reward_fn
        if reward_fn is None:
            raise ValueError("reward_fn must be provided when rm_scores is not available.")

        if return_dict:
            result = reward_fn(batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            reward_extra_info = result.get("reward_extra_info", {})
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            return reward_tensor, reward_extra_infos_dict

    @staticmethod
    def _collect_feedback(
        include_environment_feedback: bool,
        reward_extra_infos_dict: Optional[dict[str, Any]],
        batch_size: int
    ) -> list[Any]:
        """
        Collect environment feedback from reward_extra_infos_dict.

        Args:
            include_environment_feedback: Whether to include environment feedback
            reward_extra_infos_dict: Dictionary containing reward extra information
            batch_size: Size of the batch

        Returns:
            List of feedback strings (or None for entries without feedback)
        """
        feedback_list: list[Any] = [None] * batch_size
        if include_environment_feedback and reward_extra_infos_dict is not None:
            raw_feedback = reward_extra_infos_dict.get("feedback", [])
            for i in range(min(len(raw_feedback), batch_size)):
                # Only include non-empty feedback strings
                if raw_feedback[i] and isinstance(raw_feedback[i], str) and raw_feedback[i].strip():
                    feedback_list[i] = raw_feedback[i]
        return feedback_list

    def _collect_solutions_by_uid(self, batch: DataProto, reward_tensor: torch.Tensor, success_reward_threshold: float) -> dict[Any, list[int]]:
        seq_scores = reward_tensor.sum(dim=-1).detach().cpu().numpy()
        uids = batch.non_tensor_batch["uid"]
        success_by_uid: dict[Any, list[int]] = defaultdict(list)
        for idx, uid in enumerate(uids):
            if seq_scores[idx] >= success_reward_threshold:
                success_by_uid[uid].append(idx)
        return success_by_uid

    @staticmethod
    def _remove_thinking_trace(text: str) -> str:
        """Remove <think>...</think> tags and their content from text."""
        return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)

    def _get_solution(
        self,
        idx: int,
        success_by_uid: dict[Any, list[int]],
        uids: list[Any],
        response_texts: list[str],
        dont_reprompt_on_self_success: bool = False,
        remove_thinking_from_demonstration: bool = False,
    ) -> Optional[str]:
        uid = uids[idx]
        solution_idxs = success_by_uid[uid]
        if dont_reprompt_on_self_success:
            solution_idxs = [j for j in solution_idxs if j != idx]
        if len(solution_idxs) == 0:
            return None
        solution_idx = solution_idxs[0]  # taking the first successful demonstration effectively selects a random one
        solution_str = response_texts[solution_idx]
        if remove_thinking_from_demonstration:
            solution_str = self._remove_thinking_trace(solution_str)
        return solution_str


    def _maybe_build_self_distillation_batch(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        reward_extra_infos_dict: Optional[dict[str, list]] = None,
    ) -> Optional[tuple[DataProto, dict[str, float]]]:
        self_distillation_cfg = self.config.actor_rollout_ref.actor.get("self_distillation", None)
        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        if self_distillation_cfg is None or loss_mode not in ("score", "decode"):
            return None

        device = batch.batch["input_ids"].device
        response_mask = batch.batch["response_mask"]
        responses = batch.batch["responses"]
        response_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in responses]
        prompt_texts = [msgs[-1]["content"] for msgs in batch.non_tensor_batch["raw_prompt"]]
        batch_size = batch.batch.batch_size[0]

        # Extract feedback if available and include_environment_feedback is enabled
        feedback_list = self._collect_feedback(
            include_environment_feedback=self_distillation_cfg.include_environment_feedback,
            reward_extra_infos_dict=reward_extra_infos_dict,
            batch_size=batch_size,
        )

        success_by_uid = self._collect_solutions_by_uid(batch, reward_tensor, success_reward_threshold=self_distillation_cfg.success_reward_threshold)
        solution_strs = [
            self._get_solution(
                i,
                success_by_uid,
                batch.non_tensor_batch["uid"],
                response_texts,
                self_distillation_cfg.dont_reprompt_on_self_success,
                self_distillation_cfg.get("remove_thinking_from_demonstration", False),
            )
            for i in range(batch_size)
        ]

        # Extract error_type info for distillation reweighting
        error_type_list = None
        error_label_list = ["unknown"] * batch_size
        if loss_mode in ("score", "decode") and reward_extra_infos_dict is not None:
            raw_error_types = reward_extra_infos_dict.get("error_type", [])
            if len(raw_error_types) >= batch_size:
                error_type_list = [int(raw_error_types[i]) for i in range(batch_size)]
            raw_error_labels = reward_extra_infos_dict.get("error_type_label", [])
            if len(raw_error_labels) >= batch_size:
                error_label_list = [str(raw_error_labels[i]) for i in range(batch_size)]

        def _build_teacher_message(i: int) -> list[dict]:
            system_messages = batch.non_tensor_batch["raw_prompt"][i][:-1]
            has_solution = solution_strs[i] is not None
            has_feedback = feedback_list[i] is not None
            feedback_only_without_solution = self_distillation_cfg.get("environment_feedback_only_without_solution", False)

            # If feedback_only_without_solution is True, only use feedback when no solution exists
            use_feedback = has_feedback and (not feedback_only_without_solution or not has_solution)

            # build solution section
            solution_section = ""
            if has_solution:
                solution_section = self_distillation_cfg.solution_template.format(
                    successful_previous_attempt=solution_strs[i]
                )

            # build feedback section
            feedback_section = ""
            if use_feedback:
                feedback_section = self_distillation_cfg.feedback_template.format(
                    feedback_raw=feedback_list[i]
                )

            # Enriched reprompt: show student's incorrect code vs correct solution
            # BEACON Metacognitive Distillation: B quadrant gets overconfidence warning
            if (use_feedback or has_solution) and self_distillation_cfg.get("enriched_reprompt_template", None):
                student_code = response_texts[i]
                if len(student_code) > 3000:
                    student_code = student_code[:3000] + "\n... (truncated)"
                error_label = error_label_list[i] if error_type_list is not None else "unknown"

                # Check BEACON quadrant for metacognitive distillation
                extra_info_i = batch.non_tensor_batch.get("extra_info", [None] * batch_size)[i]
                beacon_quadrant = extra_info_i.get("beacon_quadrant", None) if extra_info_i else None

                metacog_prefix = ""
                _ERROR_NAMES_MC = {0: "Pass", 1: "Runtime Error", 2: "Wrong Answer", 3: "Timeout", 4: "Compile Error"}
                predicted_err_name = ""
                if extra_info_i and extra_info_i.get("self_critique"):
                    pred_eid = extra_info_i["self_critique"].get("predicted_error_id", None)
                    predicted_err_name = _ERROR_NAMES_MC.get(pred_eid, "unknown") if pred_eid is not None else ""

                if beacon_quadrant == "B":
                    # Overconfident blindspot: model predicted PASS but actually failed
                    # 方案3: Error-specific repair guidance based on blindspot severity
                    actual_eid_b = extra_info_i.get("actual_error_id", -1) if extra_info_i else -1
                    severity = self._blindspot_ema.get(actual_eid_b, 0.0)

                    _ERROR_REPAIR_HINTS = {
                        4: "Check for syntax errors, missing imports, undefined variables, and incorrect indentation.",
                        1: "Check for edge cases (empty input, large values), type mismatches, index out of range, and division by zero.",
                        2: "Re-read the problem statement carefully. Check algorithm correctness, boundary conditions, off-by-one errors, and integer overflow.",
                        3: "Optimize time complexity. Consider using better data structures (dict/set instead of list), avoid nested loops, and use efficient algorithms.",
                    }
                    repair_hint = _ERROR_REPAIR_HINTS.get(actual_eid_b, "Carefully review the logic and fix the issue.")

                    if severity > 0.2:
                        # High blindspot: more detailed guidance
                        metacog_prefix = (
                            f"[METACOGNITIVE WARNING - HIGH BLINDSPOT] You predicted this code would Pass, "
                            f"but it actually failed with: {error_label}. "
                            f"This is a recurring blindspot (severity={severity:.2f}). "
                            f"Common causes for {error_label}: {repair_hint}\n\n"
                        )
                    else:
                        metacog_prefix = (
                            f"[METACOGNITIVE WARNING] You predicted this code would Pass, "
                            f"but it actually failed with: {error_label}. "
                            f"{repair_hint}\n\n"
                        )
                elif beacon_quadrant == "C":
                    # Underconfident: model predicted FAIL but code actually passed
                    metacog_prefix = (
                        f"[CALIBRATION NOTE] You predicted this code would get {predicted_err_name}, "
                        "but it actually passed all test cases. "
                        "Your solution was correct — study why it works to improve your confidence.\n\n"
                    )
                elif beacon_quadrant == "D" and predicted_err_name:
                    # Knows it doesn't know — check if error type prediction was exact
                    actual_eid = extra_info_i.get("actual_error_id", -1) if extra_info_i else -1
                    pred_eid = extra_info_i["self_critique"].get("predicted_error_id", -1) if extra_info_i and extra_info_i.get("self_critique") else -1
                    if pred_eid >= 0 and actual_eid >= 0 and pred_eid != actual_eid:
                        actual_err_name = _ERROR_NAMES_MC.get(actual_eid, "unknown")
                        metacog_prefix = (
                            f"[DIAGNOSIS CORRECTION] You predicted {predicted_err_name}, "
                            f"but the actual error was {actual_err_name}. "
                            "Generate a corrected solution.\n\n"
                        )

                enriched_template = self_distillation_cfg.enriched_reprompt_template
                reprompt_text = metacog_prefix + enriched_template.format(
                    prompt=prompt_texts[i],
                    solution=solution_section,
                    feedback=feedback_section,
                    error_type=error_label,
                    student_code=student_code,
                )
            elif use_feedback or has_solution:
                reprompt_text = self_distillation_cfg.reprompt_template.format(
                    prompt=prompt_texts[i],
                    solution=solution_section,
                    feedback=feedback_section,
                )
            else:
                reprompt_text = prompt_texts[i]

            return system_messages + [
                {"role": "user", "content": reprompt_text},
            ]


        messages = [_build_teacher_message(i) for i in range(batch_size)]
        enable_thinking = self.config.data.apply_chat_template_kwargs.get("enable_thinking", True) if self.config.data.apply_chat_template_kwargs else True
        teacher_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            continue_final_message=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            max_length=self_distillation_cfg.max_reprompt_len,
            padding=True,
            truncation=True,
        )
        teacher_input_ids = torch.cat([teacher_prompt["input_ids"].to(device), responses], dim=1)
        teacher_attention_mask = torch.cat([teacher_prompt["attention_mask"].to(device), response_mask], dim=1)
        teacher_position_ids = compute_position_id_with_mask(teacher_attention_mask)

        # Compute which samples actually use feedback (accounting for environment_feedback_only_without_solution)
        feedback_only_without_solution = self_distillation_cfg.get("environment_feedback_only_without_solution", False)
        feedback_used = [
            feedback_list[i] is not None and (not feedback_only_without_solution or solution_strs[i] is None)
            for i in range(batch_size)
        ]

        # self_distillation_mask is True if sample has a solution OR feedback is used (i.e., will get a reprompted message)
        self_distillation_mask = torch.tensor(
            [solution_strs[i] is not None or feedback_used[i] for i in range(batch_size)],
            dtype=torch.float32,
            device=device
        )

        # Apply critique weights to distillation mask (samples with correct self-critique get boosted)
        if "critique_weight" in batch.batch:
            critique_weight = batch.batch["critique_weight"].to(device)
            self_distillation_mask = self_distillation_mask * critique_weight

        uids = set(batch.non_tensor_batch["uid"])
        num_with_feedback_available = sum(1 for f in feedback_list if f is not None)
        num_with_feedback_used = sum(1 for f in feedback_used if f)
        num_with_solution = sum(1 for s in solution_strs if s is not None)
        metrics = {
            "self_distillation/success_group_fraction": len([uid for uid in uids if len(success_by_uid[uid]) > 0]) / len(uids),
            "self_distillation/success_sample_fraction": num_with_solution / batch_size,
            "self_distillation/feedback_available_fraction": num_with_feedback_available / batch_size,
            "self_distillation/feedback_used_fraction": num_with_feedback_used / batch_size,
            "self_distillation/reprompt_sample_fraction": self_distillation_mask.float().mean().item(),
        }
        tensors = {
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_attention_mask,
            "teacher_position_ids": teacher_position_ids,
            "self_distillation_mask": self_distillation_mask,
        }
        if loss_mode in ("score", "decode") and error_type_list is not None:
            tensors["error_types"] = torch.tensor(error_type_list, dtype=torch.long, device=device)
        return DataProto.from_dict(tensors=tensors), metrics

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid", "raw_prompt"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        val_pbar = tqdm(self.val_dataloader, desc="Validation", leave=False)
        for test_data in val_pbar:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            val_pbar.set_postfix_str("generating...")
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            val_pbar.set_postfix_str("gen done")

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            val_pbar.set_postfix_str("computing reward...")
            result = self._compute_or_extract_reward(test_batch, reward_fn=self.val_reward_fn, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            reward_extra_info = result.get("reward_extra_info", {})
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}
        if result_b is None:
            result_b = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            if self.use_legacy_worker_impl == "disable":
                # convert critic_cfg into TrainingWorkerConfig
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        # for legacy discriminative reward model, we create a reward model worker here
        # for reward loop discriminative reward model, we create a reward loop manager here
        if not self.use_reward_loop:
            # legacy reward model only handle reward-model based scenario
            if self.use_rm:
                # we create a RM here
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                rm_cls = RayClassWithInitArgs(
                    self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model
                )
                self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls
        else:
            # reward loop handle hybrid reward scenario (rule, disrm, genrm, ...)
            # Note: mode is always "async" since sync mode is deprecated
            can_reward_loop_parallelize = not self.use_rm or self.config.reward_model.enable_resource_pool
            # judge if we can asynchronously parallelize reward model with actor rollout
            # two condition that we can parallelize reward model with actor rollout:
            # 1. reward model is not enabled (rule-based reward can parallelize)
            # 2. reward model is enabled but extra resource pool is enabled
            # If we cannot parallelize, we should enable synchronous mode here, and launch a reward loop manager here
            # else for parallelize mode, we launch a reward worker for each rollout worker (in agent loop, not here)
            if not can_reward_loop_parallelize:
                from verl.experimental.reward_loop import RewardLoopManager

                self.config.reward_model.n_gpus_per_node = self.config.trainer.n_gpus_per_node
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                self.reward_loop_manager = RewardLoopManager(
                    config=self.config,
                    rm_resource_pool=resource_pool,
                )

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm and not self.use_reward_loop:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
        else:
            rm_resource_pool = None

        self.async_rollout_manager = AgentLoopManager(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rm_resource_pool=rm_resource_pool,
        )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, compute_loss=False)
            output = self.critic_wg.infer_batch(batch_td)
            output = output.get()
            values = tu.get(output, "values")
            values = no_padding_2_padding(values, batch_td)
            values = tu.get_tensordict({"values": values.float()})
            values = DataProto.from_tensordict(values)
        else:
            values = self.critic_wg.compute_values(batch)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            metadata = {"calculate_entropy": False, "compute_loss": False}
            if self.ref_in_actor:
                metadata["no_lora_adapter"] = True
            tu.assign_non_tensor(batch_td, **metadata)
            if self.ref_in_actor:
                output = self.actor_rollout_wg.compute_log_prob(batch_td)
            else:
                output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
            # gather output
            log_probs = tu.get(output, "log_probs")
            # step 4. No padding to padding
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
            ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        else:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        if self.use_legacy_worker_impl == "disable":
            # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            # gather output
            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
            # step 4. No padding to padding
            entropy = no_padding_2_padding(entropy, batch_td)
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
            old_log_prob = DataProto.from_tensordict(old_log_prob)
        else:
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            old_log_prob_mfu = 0
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=calculate_entropy,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_output = tu.get(actor_output, "metrics")
            actor_output = rename_dict(actor_output, "actor/")
            # modify key name
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        else:
            actor_output = self.actor_rollout_wg.update_actor(batch)
        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            tu.assign_non_tensor(
                batch_td,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            output = self.critic_wg.train_mini_batch(batch_td)
            output = output.get()
            output = tu.get(output, "metrics")
            output = rename_dict(output, "critic/")
            # modify key name
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = self.critic_wg.update_critic(batch)
        return critic_output

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
            group_name=self.config.trainer.get("group_name", None),
        )

        self.global_steps = 0

        # ---- Metrics file: organized per-step metrics written to project dir ----
        import datetime as _dt
        _score_dir = os.environ.get("SCORE_DIR", os.getcwd())
        _experiment = self.config.trainer.experiment_name
        _metrics_dir = os.path.join(_score_dir, "metrics")
        os.makedirs(_metrics_dir, exist_ok=True)
        _metrics_txt_path = os.path.join(_metrics_dir, f"metrics_{_experiment}.txt")
        with open(_metrics_txt_path, "w") as _f:
            _f.write(f"SDPO Training Log - {_dt.datetime.now().isoformat()}\n")
            _f.write(f"Model: {self.config.actor_rollout_ref.model.path}\n")
            _f.write(f"Experiment: {_experiment}\n")
            _f.write("=" * 80 + "\n")

        _TRAIN_KEYS = [
            ("actor/pg_loss", "pg_loss"),
            ("actor/entropy", "entropy"),
            ("actor/grad_norm", "grad_norm"),
            ("actor/lr", "lr"),
            ("critic/score/mean", "reward_mean"),
            ("response_length/mean", "resp_len"),
            ("self_distillation/success_group_fraction", "success_group_frac"),
            ("self_distillation/success_sample_fraction", "success_sample_frac"),
            ("judge/total", "judge_total"),
            ("judge/skip", "judge_skip"),
            ("judge/sandbox", "judge_sandbox"),
            # BEACON metacognitive metrics
            ("beacon/ccs", "ccs"),
            ("beacon/diagnosis_accuracy", "diag_acc"),
            ("beacon/binary_accuracy", "binary_acc"),
            ("beacon/quad_A_frac", "quad_A_frac"),
            ("beacon/quad_B_frac", "quad_B_frac"),
            ("beacon/quad_C_frac", "quad_C_frac"),
            ("beacon/quad_D_frac", "quad_D_frac"),
            # 5-class blindspot B-fraction per error type
            ("beacon/blindspot_B_RE", "bspot_RE"),
            ("beacon/blindspot_B_WA", "bspot_WA"),
            ("beacon/blindspot_B_TLE", "bspot_TLE"),
            ("beacon/blindspot_B_CE", "bspot_CE"),
            # Binary blindspot
            ("beacon/blindspot_B_Fail", "bspot_Fail"),
            # EMA blindspot
            ("beacon/blindspot_ema_RE", "bspot_ema_RE"),
            ("beacon/blindspot_ema_WA", "bspot_ema_WA"),
            ("beacon/blindspot_ema_TLE", "bspot_ema_TLE"),
            ("beacon/blindspot_ema_CE", "bspot_ema_CE"),
            ("beacon/blindspot_ema_Fail", "bspot_ema_Fail"),
            # Effective W_B weights
            ("beacon/W_B_eff_RE", "W_B_eff_RE"),
            ("beacon/W_B_eff_WA", "W_B_eff_WA"),
            ("beacon/W_B_eff_TLE", "W_B_eff_TLE"),
            ("beacon/W_B_eff_CE", "W_B_eff_CE"),
            ("beacon/W_B_eff_Fail", "W_B_eff_Fail"),
            # Auxiliary critique loss
            ("actor/critique_loss", "critique_loss"),
        ]
        _VAL_KEYS = [
            ("val-core/livecodebench/acc/mean@1", "pass@1"),
            ("val-aux/livecodebench/reward/mean@1", "reward_mean@1"),
            ("val-core/livecodebench/acc/mean@4", "pass@4"),
            ("val-aux/livecodebench/reward/mean@4", "reward_mean@4"),
            ("val-aux/livecodebench/acc/best@4/mean", "best@4"),
            ("val-aux/livecodebench/acc/worst@4/mean", "worst@4"),
            ("val-aux/livecodebench/acc/maj@4/mean", "maj@4"),
        ]

        def _append_metrics_csv(m: dict, step: int):
            ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            has_val = any(m.get(k) is not None for k, _ in _VAL_KEYS)
            epoch = m.get("training/epoch", "")
            # Write train line
            parts = [f"step={step}"]
            if epoch != "":
                parts.append(f"epoch={epoch}")
            for full_key, short_key in _TRAIN_KEYS:
                v = m.get(full_key)
                if v is not None:
                    parts.append(f"{short_key}={v:.6f}" if isinstance(v, float) else f"{short_key}={v}")
            with open(_metrics_txt_path, "a") as _f:
                _f.write(f"[{ts}] train {' | '.join(parts)}\n")
            # Write confusion matrix if present
            cm_keys = [(k, v) for k, v in m.items() if k.startswith("beacon/cm/")]
            if cm_keys:
                cm_parts = [f"step={step}"]
                for k, v in sorted(cm_keys):
                    short = k.replace("beacon/cm/", "")
                    cm_parts.append(f"{short}={v}")
                with open(_metrics_txt_path, "a") as _f:
                    _f.write(f"[{ts}] cm    {' | '.join(cm_parts)}\n")
            # Write eval line if val metrics present
            if has_val:
                val_parts = [f"step={step}"]
                for full_key, short_key in _VAL_KEYS:
                    v = m.get(full_key)
                    if v is not None:
                        val_parts.append(f"{short_key}={v:.6f}" if isinstance(v, float) else f"{short_key}={v}")
                with open(_metrics_txt_path, "a") as _f:
                    _f.write(f"[{ts}] eval  {' | '.join(val_parts)}\n")

        # load checkpoint before doing anything
        self._load_checkpoint()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            _append_metrics_csv(val_metrics, self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training")
        _stage_start = None

        def _set_stage(name):
            """Update progress bar to show current stage."""
            import time as _t
            nonlocal _stage_start
            _stage_start = _t.perf_counter()
            progress_bar.set_description(f"Step {self.global_steps} | {name}")
            progress_bar.refresh()

        def _end_stage(name):
            """Mark stage as done and show elapsed time."""
            import time as _t
            nonlocal _stage_start
            elapsed = _t.perf_counter() - _stage_start if _stage_start else 0
            progress_bar.set_postfix_str(f"{name} {elapsed:.1f}s", refresh=True)

        # ---- Inline SFT setup ----
        _sft_every_n_steps = self.config.trainer.get("sft_every_n_steps", 0)
        _sft_buffer = []  # buffer of successful (input_ids, attention_mask, position_ids, response_mask)
        if _sft_every_n_steps > 0:
            print(f"[InlineSFT] Enabled: SFT every {_sft_every_n_steps} steps", flush=True)

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        _set_stage("Rollout (vLLM)")
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        _end_stage("rollout")

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                if not self.use_reward_loop:
                                    rm_scores = self.rm_wg.compute_rm_score(batch)
                                else:
                                    assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                    rm_scores = self.reward_loop_manager.compute_rm_score(batch)
                                batch = batch.union(rm_scores)

                            # Compute or extract reward for REMAX baseline
                            reward_baseline_tensor = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, sum_reward=True
                            )

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all

                    # === Code Judge: pre-filter rollouts before sandbox ===
                    with marked_timer("code_judge", timing_raw, color="magenta"):
                        _set_stage("CoderJudge")
                        batch = self._run_code_judge(batch)
                        _end_stage("judge")

                    # Set judge_gate_active on all samples so reward manager knows to skip
                    _judge_enabled = getattr(self.config.reward_model, "coder_judge", None) is not None and getattr(self.config.reward_model.coder_judge, "enable", False)
                    if _judge_enabled and "extra_info" in batch.non_tensor_batch:
                        for i in range(len(batch)):
                            if batch.non_tensor_batch["extra_info"][i] is None:
                                batch.non_tensor_batch["extra_info"][i] = {}
                            batch.non_tensor_batch["extra_info"][i]["judge_gate_active"] = True

                    with marked_timer("reward", timing_raw, color="yellow"):
                        _set_stage("Reward")
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            if not self.use_reward_loop:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                            else:
                                assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # Compute or extract reward for training
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, return_dict=False
                            )
                        _end_stage("reward")

                    # === Self-Critique: use student model to predict pass/fail ===
                    # Runs AFTER sandbox is launched async (future_reward), so
                    # GPU critique generation runs in parallel with CPU sandbox.
                    # Requires free_cache_engine=False to avoid CUDA wake_up errors.
                    with marked_timer("self_critique", timing_raw, color="cyan"):
                        _set_stage("Self-Critique")
                        batch = self._run_self_critique(batch)
                        _end_stage("critique")

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))
                            _end_stage("old_logprob")

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            _set_stage("Ref LogProb")
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)
                            _end_stage("ref_logprob")

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            _set_stage("Critic Values")
                            values = self._compute_values(batch)
                            batch = batch.union(values)
                            _end_stage("values")

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            _set_stage("Waiting Async Reward")
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                            _end_stage("async_reward")
                        batch.batch["token_level_scores"] = reward_tensor

                        # ---- Collect successful solutions for inline SFT ----
                        if _sft_every_n_steps > 0 and reward_tensor is not None:
                            seq_scores = reward_tensor.sum(dim=-1).detach().cpu().numpy()
                            n_collected = 0
                            for _si in range(len(seq_scores)):
                                if seq_scores[_si] >= 1.0:
                                    _sft_buffer.append({
                                        "input_ids": batch.batch["input_ids"][_si].detach().cpu(),
                                        "attention_mask": batch.batch["attention_mask"][_si].detach().cpu(),
                                        "position_ids": batch.batch["position_ids"][_si].detach().cpu(),
                                        "response_mask": batch.batch["response_mask"][_si].detach().cpu(),
                                        "responses": batch.batch["responses"][_si].detach().cpu(),
                                    })
                                    n_collected += 1
                            if n_collected > 0:
                                print(f"  [InlineSFT] step={self.global_steps} collected={n_collected} buffer={len(_sft_buffer)}", flush=True)

                        # === Self-Critique accuracy: compare predictions with sandbox results ===
                        critique_metrics = self._compute_critique_accuracy(batch, reward_tensor, reward_extra_infos_dict)
                        if critique_metrics:
                            metrics.update(critique_metrics)

                        self_distillation_data = self._maybe_build_self_distillation_batch(batch, reward_tensor, reward_extra_infos_dict)
                        if self_distillation_data is not None:
                            self_distillation_batch, self_distillation_metrics = self_distillation_data
                            batch = batch.union(self_distillation_batch)
                            metrics.update(self_distillation_metrics)

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                        # === BEACON 方案4: Blindspot-driven Advantage Amplification ===
                        # For B-quadrant samples (overconfident blindspot), amplify advantage
                        # based on blindspot severity for that error type.
                        # This makes RL gradients stronger on the model's blindspots.
                        extra_infos = batch.non_tensor_batch.get("extra_info", None)
                        if extra_infos is not None and reward_extra_infos_dict is not None:
                            raw_et = reward_extra_infos_dict.get("error_type", [])
                            adv = batch.batch["advantages"]
                            n_amplified = 0
                            for i in range(len(batch)):
                                if extra_infos[i] is None:
                                    continue
                                quadrant = extra_infos[i].get("beacon_quadrant", None)
                                if quadrant != "B":
                                    continue
                                # Get actual error type for this sample
                                if i < len(raw_et):
                                    eid = int(raw_et[i])
                                    severity = self._blindspot_ema.get(eid, 0.0)
                                    # Amplify advantage: stronger gradient for severe blindspots
                                    amp_factor = 1.0 + severity  # range [1.0, ~1.4]
                                    adv[i] = adv[i] * amp_factor
                                    n_amplified += 1
                            if n_amplified > 0:
                                batch.batch["advantages"] = adv
                                print(f"[BEACON-AdvAmp] amplified={n_amplified} samples in B-quadrant", flush=True)

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            _set_stage("Update Critic")
                            critic_output = self._update_critic(batch)
                            _end_stage("update_critic")
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            _set_stage("Update Actor")
                            actor_output = self._update_actor(batch)
                            _end_stage("update_actor")
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # ---- Inline SFT: periodically train on successful solutions ----
                _sft_stop_step = self.config.trainer.get("sft_stop_step", 40)
                _sft_epoch_interval = self.config.trainer.get("sft_epoch_interval", 4)
                if _sft_epoch_interval > 0:
                    _sft_step_interval = len(self.train_dataloader) * _sft_epoch_interval
                else:
                    _sft_step_interval = 1  # every step
                if (
                    _sft_every_n_steps > 0
                    and self.global_steps <= _sft_stop_step
                    and self.global_steps > 0
                    and self.global_steps % _sft_step_interval == 0
                    and len(_sft_buffer) >= 8  # need enough samples
                ):
                    with marked_timer("inline_sft", timing_raw, color="blue"):
                        import random as _rnd
                        _set_stage("Inline SFT")
                        # Sample up to 64 solutions from buffer (deduplicate by keeping recent)
                        _sft_samples = _sft_buffer[-256:] if len(_sft_buffer) > 256 else _sft_buffer
                        _rnd.shuffle(_sft_samples)
                        _sft_samples = _sft_samples[:64]

                        # Stack into tensors
                        _sft_tensors = {
                            k: torch.stack([s[k] for s in _sft_samples])
                            for k in _sft_samples[0].keys()
                        }
                        _sft_batch = DataProto.from_dict(tensors=_sft_tensors)
                        _sft_batch.meta_info["sft_mode"] = True
                        _sft_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                        _sft_batch.meta_info["sft_steps"] = self.config.trainer.get("sft_steps", 2)

                        print(f"  [SFT] Training on {len(_sft_samples)} samples (buffer={len(_sft_buffer)})...", flush=True)
                        _sft_output = self.actor_rollout_wg.update_actor(_sft_batch)
                        _sft_metrics = reduce_metrics(_sft_output.meta_info.get("metrics", {}))
                        _end_stage("sft")
                        _sft_t = timing_raw.get("inline_sft", 0)
                        print(f"\n{'='*60}", flush=True)
                        print(f"  [InlineSFT] step={self.global_steps} samples={len(_sft_samples)} buffer={len(_sft_buffer)} ({_sft_t:.1f}s)", flush=True)
                        for _sk, _sv in _sft_metrics.items():
                            print(f"    {_sk}: {_sv:.4f}", flush=True)
                        print(f"{'='*60}\n", flush=True)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        _set_stage("Validation")
                        val_metrics: dict = self._validate()
                        _end_stage("validation")
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        _set_stage("Saving Checkpoint")
                        self._save_checkpoint()
                        _end_stage("checkpoint")

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # CoderJudge metrics
                if batch.meta_info.get("judge_total", 0) > 0:
                    metrics.update({
                        "judge/total": batch.meta_info.get("judge_total", 0),
                        "judge/skip": batch.meta_info.get("judge_skip", 0),
                        "judge/sandbox": batch.meta_info.get("judge_sandbox", 0),
                    })

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                _append_metrics_csv(metrics, self.global_steps)

                progress_bar.set_description("Training")
                progress_bar.set_postfix_str("")
                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
