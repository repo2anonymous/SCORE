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

import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

# Force unbuffered stdout in Ray worker processes
if not os.environ.get("PYTHONUNBUFFERED"):
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

# Number of parallel threads for reward computation.
_REWARD_WORKERS = int(os.environ.get("VERL_REWARD_WORKERS", 8))


@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key

    @staticmethod
    def _classify_judge_error_type(judge_r: dict) -> int:
        """Map CoderJudge error_type string to TPDD integer constant."""
        from verl.utils.reward_score.feedback.code import (
            ERROR_TYPE_COMPILE, ERROR_TYPE_RUNTIME, ERROR_TYPE_TIMEOUT, ERROR_TYPE_WRONG_ANSWER,
        )
        et = judge_r.get("error_type", "")
        if et in ("syntax_error", "compile_error"):
            return ERROR_TYPE_COMPILE
        if et == "runtime_error":
            return ERROR_TYPE_RUNTIME
        if et == "timeout":
            return ERROR_TYPE_TIMEOUT
        return ERROR_TYPE_WRONG_ANSWER

    def __call__(self, data: DataProto, return_dict: bool = True) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # Pre-extract all inputs (tokenizer decode must happen in main thread)
        items = []
        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores
            extra_info["truncated"] = not (valid_response_ids == self.tokenizer.eos_token_id).any().item()

            items.append({
                "idx": i,
                "prompt_str": prompt_str,
                "response_str": response_str,
                "ground_truth": ground_truth,
                "data_source": data_source,
                "extra_info": extra_info,
                "valid_response_length": valid_response_length,
            })

        # Separate judge-skipped items (instant score=0) from sandbox items
        sandbox_items = []
        judge_skipped_items = []
        for item in items:
            judge_gate_active = item["extra_info"].get("judge_gate_active", False)
            judge_results = item["extra_info"].get("judge_results")
            if judge_gate_active and judge_results and not judge_results.get("passed", True):
                judge_skipped_items.append(item)
            else:
                sandbox_items.append(item)

        if judge_skipped_items:
            print(f"[CoderJudge] Reward: total={len(items)} | judge_skipped={len(judge_skipped_items)} | sandbox={len(sandbox_items)}", flush=True)

        # Score function
        def _score_one(item):
            return item["idx"], self.compute_score(
                data_source=item["data_source"],
                solution_str=item["response_str"],
                ground_truth=item["ground_truth"],
                extra_info=item["extra_info"],
            )

        scores = [None] * len(data)

        # Score judge-skipped items: assign score=0 without sandbox
        for item in judge_skipped_items:
            idx = item["idx"]
            judge_r = item["extra_info"].get("judge_results", {})
            scores[idx] = {
                "score": 0.0,
                "acc": 0.0,
                "pred": f"[judge_skip] {judge_r.get('error_type', 'unknown')}: {judge_r.get('reason', '')}",
                "incorrect_format": 0,
                "error_in_test_cases": 1 if judge_r.get("error_type") in ("runtime_error", "syntax_error") else 0,
                "timed_out": 1 if judge_r.get("error_type") == "timeout" else 0,
                "truncated": 1 if item["extra_info"].get("truncated", False) else 0,
                "truncated_and_missing_answer": 0,
                "error_type": self._classify_judge_error_type(judge_r),
                "error_type_label": judge_r.get("error_type", "unknown"),
                "feedback": judge_r.get("reason", ""),
            }

        # Score sandbox items with thread pool
        from tqdm import tqdm
        if sandbox_items:
            with ThreadPoolExecutor(max_workers=_REWARD_WORKERS) as pool:
                futures = {pool.submit(_score_one, item): item for item in sandbox_items}
                pbar = tqdm(as_completed(futures), total=len(sandbox_items), desc="[Sandbox] compute_score", leave=False)
                for future in pbar:
                    idx, score = future.result()
                    scores[idx] = score
                pbar.close()

        # === Soft reward for all-fail prompt groups ===
        # When ALL rollouts for a prompt score 0, use judge pass-vote ratio
        # to give small positive rewards, so GRPO has gradient signal.
        _SOFT_REWARD_ALPHA = float(os.environ.get("VERL_SOFT_REWARD_ALPHA", "0.1"))

        if _SOFT_REWARD_ALPHA > 0:
            # Group items by prompt
            prompt_groups = defaultdict(list)
            for item in items:
                prompt_groups[item["prompt_str"]].append(item)

            soft_reward_count = 0
            all_fail_groups = 0
            for _, group_items in prompt_groups.items():
                # Check if ALL items in this group scored 0
                all_zero = True
                for item in group_items:
                    s = scores[item["idx"]]
                    sc = s.get("score", 0) if isinstance(s, dict) else (s or 0)
                    if sc != 0:
                        all_zero = False
                        break

                if not all_zero:
                    continue

                all_fail_groups += 1

                # Compute soft reward from judge pass-vote ratio
                for item in group_items:
                    judge_r = item["extra_info"].get("judge_results", {})
                    votes_str = judge_r.get("votes", "")
                    if "/" in votes_str:
                        n_reject, n_total = map(int, votes_str.split("/"))
                        pass_ratio = 1.0 - (n_reject / n_total) if n_total > 0 else 0.0
                    elif judge_r.get("passed", False):
                        pass_ratio = 1.0
                    else:
                        pass_ratio = 0.0

                    soft_reward = _SOFT_REWARD_ALPHA * pass_ratio
                    if soft_reward > 0:
                        idx = item["idx"]
                        if isinstance(scores[idx], dict):
                            scores[idx]["score"] = soft_reward
                        else:
                            scores[idx] = soft_reward
                        soft_reward_count += 1

            if all_fail_groups > 0:
                print(
                    f"[SoftReward] all_fail_groups={all_fail_groups}/{len(prompt_groups)} | "
                    f"soft_rewards_applied={soft_reward_count} (alpha={_SOFT_REWARD_ALPHA})",
                    flush=True,
                )

        # Collect results
        already_print_data_sources = {}
        for i, item in enumerate(items):
            score = scores[item["idx"]]
            valid_response_length = item["valid_response_length"]
            data_source = item["data_source"]

            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[item["idx"], valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", item["prompt_str"])
                print("[response]", item["response_str"])
                print("[ground_truth]", item["ground_truth"])
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
