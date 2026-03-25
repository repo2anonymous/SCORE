# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig

from .model import HFModelConfig
from .rollout import RolloutConfig

__all__ = ["CoderJudgeConfig", "SandboxFusionConfig", "RewardModelConfig"]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class CoderJudgeConfig(BaseConfig):
    """Configuration for the LLM-based code judge pre-filter.

    The judge runs on a dedicated GPU via a Ray actor (num_gpus=1).
    It predicts whether code will pass before sending to sandbox,
    avoiding wasting sandbox time on obviously broken code.
    """

    enable: bool = False
    model_name: str = "Qwen/Qwen2.5-Coder-14B-Instruct"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 4096
    max_num_seqs: int = 256
    judge_n: int = 3
    judge_temperature: float = 0.3
    confidence_threshold: float = 0.5


@dataclass
class SandboxFusionConfig(BaseConfig):
    """Configuration for cloud/local sandbox fusion.

    Args:
        url (Optional[str]): Cloud/local function URL for sandbox execution.
        max_concurrent (int): Max concurrent requests allowed to sandbox.
        memory_limit_mb (int): Max memory limit for each sandbox process in MB.
    """

    url: Optional[str] = None
    max_concurrent: int = 64
    memory_limit_mb: int = 1024


@dataclass
class RewardModelConfig(BaseConfig):
    _mutable_fields = BaseConfig._mutable_fields

    reward_manager: Optional[str] = None

    enable: bool = False
    enable_resource_pool: bool = False
    n_gpus_per_node: int = 0
    nnodes: int = 0

    # reward model args
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    model: HFModelConfig = field(default_factory=HFModelConfig)
    sandbox_fusion: SandboxFusionConfig = field(default_factory=SandboxFusionConfig)
    coder_judge: CoderJudgeConfig = field(default_factory=CoderJudgeConfig)

    def __post_init__(self):
        super().__post_init__()
        if self.reward_manager is not None:
            logger.warning(
                f"`reward_model.reward_manager` is deprecated, but got value {self.reward_manager}. "
                "Please use `reward_manager.name instead. "
                "See `verl/trainer/config/config.py:RewardManagerConfig` for more details."
            )
