# Copyright 2025 Individual Contributor
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
LLM-based code judge for pre-filtering rollout code before sandbox execution.

Flow:
  1. Student vLLM generates code rollouts
  2. CoderJudge (on a dedicated GPU via Ray actor) batch-reviews all samples
  3. Samples predicted to fail are skipped from sandbox (score=0)
  4. Only likely-correct samples go through expensive sandbox execution

Error types (aligned with sandbox result codes):
  - "correct"          -> True   (predicted to pass)
  - "syntax_error"     -> -4     (compile/parse error)
  - "runtime_error"    -> -2     (crash, exception)
  - "wrong_answer"     -> False  (runs but wrong output)
  - "timeout"          -> -3     (infinite loop, TLE)
  - "logic_error"      -> False  (algorithm flaw)
"""

import gc
import json
import logging
import math
import os
import time
from typing import Any, Optional

import ray
import torch

logger = logging.getLogger(__name__)

# Maps judge error types to sandbox-compatible result codes
ERROR_TYPE_TO_RESULT_CODE = {
    "correct": True,
    "syntax_error": -4,
    "runtime_error": -2,
    "wrong_answer": False,
    "timeout": -3,
    "logic_error": False,
}

JUDGE_PROMPT_TEMPLATE = """\
You are a code reviewer for competitive programming. This code already passed syntax checking. Only reject it if you find a CLEAR, OBVIOUS bug.

## Solution Code
```python
{code}
```

## Instructions
Only mark as FAILED if you find a CLEAR bug:
- **Runtime error**: obvious division by zero, definite index out of bounds, undefined variable
- **Wrong answer**: clearly wrong algorithm or obviously wrong formula
- **Timeout**: clearly O(n^3) or worse brute force on large inputs

Do NOT fail for: minor edge cases, suboptimal but possibly correct algorithms, style issues.
When in doubt, mark as PASSED — let the sandbox verify uncertain cases.

Output a JSON object:
- "passed": bool (true unless CLEAR bug found)
- "error_type": one of ["correct", "runtime_error", "wrong_answer", "timeout", "logic_error"]
- "reason": string (MAX 15 words)

Output ONLY the JSON object.

## Output
"""


def _parse_single_judge_response(response_text: str) -> dict[str, Any]:
    """Parse a single JSON object response from the judge.

    If JSON is truncated but we can detect "passed": false, we treat it as failed
    instead of defaulting to pass-through.
    """
    import re

    text = response_text.strip()

    # Handle markdown code blocks
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break

    start = text.find("{")
    end = text.rfind("}")

    # Try normal JSON parse first
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(text[start : end + 1])
            if isinstance(result, dict):
                error_type = result.get("error_type", "correct")
                if error_type not in ERROR_TYPE_TO_RESULT_CODE:
                    error_type = "wrong_answer"
                return {
                    "passed": result.get("passed", error_type == "correct"),
                    "error_type": error_type,
                    "reason": result.get("reason", ""),
                }
        except (json.JSONDecodeError, ValueError):
            pass

    # JSON parse failed (likely truncated response). Try regex extraction.
    fragment = text[start:] if start != -1 else text

    passed_match = re.search(r'"passed"\s*:\s*(true|false)', fragment, re.IGNORECASE)
    error_match = re.search(r'"error_type"\s*:\s*"([^"]*)"', fragment)
    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)', fragment)

    if passed_match:
        passed = passed_match.group(1).lower() == "true"
        error_type = error_match.group(1) if error_match else ("correct" if passed else "wrong_answer")
        if error_type not in ERROR_TYPE_TO_RESULT_CODE:
            error_type = "wrong_answer"
        reason = reason_match.group(1) if reason_match else "truncated response"
        return {
            "passed": passed,
            "error_type": error_type,
            "reason": reason[:200],
        }

    # Truly unparseable — default to pass-through
    logger.warning(f"Judge response unparseable: {text[:200]}")
    return {"passed": True, "error_type": "correct", "reason": "Failed to parse (pass-through)"}


def _extract_passed_token_confidence(completion, tokenizer) -> float:
    """Extract the probability the model assigned to 'true'/'false' for the 'passed' field.

    Searches through the generated tokens for the literal 'true' or 'false' token
    and returns exp(logprob) as the confidence. Falls back to 1.0 if unavailable.
    """
    if not hasattr(completion, "logprobs") or not completion.logprobs:
        return 1.0

    # Pre-compute token IDs for "true" and "false"
    try:
        false_ids = set(tokenizer.encode("false", add_special_tokens=False))
        true_ids = set(tokenizer.encode("true", add_special_tokens=False))
        target_ids = false_ids | true_ids
    except Exception:
        return 1.0

    # Search through generated tokens for the true/false token
    token_ids = completion.token_ids if hasattr(completion, "token_ids") else []
    logprobs_list = completion.logprobs  # list of dicts, one per token

    for i, token_id in enumerate(token_ids):
        if token_id in target_ids and i < len(logprobs_list):
            lp_dict = logprobs_list[i]
            if lp_dict and token_id in lp_dict:
                lp_obj = lp_dict[token_id]
                logprob_val = lp_obj.logprob if hasattr(lp_obj, "logprob") else lp_obj
                return math.exp(logprob_val)

    return 1.0


class CoderJudge:
    """
    Manages loading Qwen-Coder on GPU for batch judging via vLLM.

    Uses n-way self-consistency voting + logprob confidence gating to minimize
    false negatives. A sample is only rejected when ALL n votes agree on failure
    AND each vote's confidence exceeds the threshold.

    Usage:
        judge = CoderJudge(model_name="Qwen/Qwen2.5-Coder-14B-Instruct", ...)
        judge.load()
        results = judge.batch_judge(all_codes)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-14B-Instruct",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        max_num_seqs: int = 256,
        judge_n: int = 3,
        judge_temperature: float = 0.3,
        confidence_threshold: float = 0.5,
        dtype: str = "auto",
    ):
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.judge_n = judge_n
        self.judge_temperature = judge_temperature
        self.confidence_threshold = confidence_threshold
        self.dtype = dtype
        self.llm = None
        self.tokenizer = None

    def load(self):
        """Load the judge model into GPU using vLLM offline inference."""
        if self.llm is not None:
            logger.warning("CoderJudge already loaded, skipping")
            return

        logger.info(f"Loading CoderJudge model: {self.model_name}")
        start = time.perf_counter()

        from vllm import LLM

        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            dtype=self.dtype,
            trust_remote_code=True,
            enforce_eager=True,
        )
        self.tokenizer = self.llm.get_tokenizer()

        elapsed = time.perf_counter() - start
        logger.info(f"CoderJudge loaded in {elapsed:.1f}s")

    def unload(self):
        """Unload the judge model and free all GPU memory."""
        if self.llm is None:
            return

        logger.info("Unloading CoderJudge model...")
        start = time.perf_counter()

        try:
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
        except (ImportError, Exception) as e:
            logger.warning(f"destroy_model_parallel failed (non-fatal): {e}")
        del self.llm
        self.llm = None
        self.tokenizer = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        logger.info(f"CoderJudge unloaded in {elapsed:.1f}s")

    def batch_judge(
        self,
        codes: list[str],
        max_tokens: int = 512,
    ) -> list[dict[str, Any]]:
        """
        Batch judge all code samples using n-way voting + logprob confidence.

        A sample is ONLY rejected when:
          1. ALL n votes unanimously say passed=False
          2. Every vote's logprob confidence for "false" exceeds confidence_threshold

        This dual gating minimizes false negatives (judge wrongly rejecting good code).

        Args:
            codes: List of extracted code strings (one per sample).
            max_tokens: Max tokens for each judge response.

        Returns:
            List of per-sample dicts with keys: passed, error_type, reason, confidence, votes.
        """
        assert self.llm is not None, "Must call load() before batch_judge()"

        from vllm import SamplingParams

        max_prompt_tokens = self.max_model_len - max_tokens

        # Build prompts, skip overlong ones
        prompts = []
        valid_indices = []
        for sample_idx, code in enumerate(codes):
            prompt = JUDGE_PROMPT_TEMPLATE.format(code=code)

            if self.tokenizer and hasattr(self.tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": prompt}]
                try:
                    prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    pass

            # Quick char-length check
            if len(prompt) > max_prompt_tokens * 4:
                continue

            prompts.append(prompt)
            valid_indices.append(sample_idx)

        # Default: pass-through to sandbox
        default_result = {"passed": True, "error_type": "correct", "reason": "Skipped (too long)"}
        all_results = [default_result.copy() for _ in codes]

        if not prompts:
            logger.info("CoderJudge: all samples too long, skipping judge")
            return all_results

        # Use n > 1 with temperature for self-consistency voting, request logprobs
        n = self.judge_n
        temp = self.judge_temperature if n > 1 else 0.0
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temp,
            logprobs=1,  # get logprob for each generated token
        )

        logger.info(
            f"CoderJudge: running batch inference on {len(prompts)}/{len(codes)} samples "
            f"(n={n}, temp={temp}, conf_threshold={self.confidence_threshold})"
        )
        start = time.perf_counter()

        outputs = self.llm.generate(prompts, sampling_params)

        elapsed = time.perf_counter() - start
        logger.info(f"CoderJudge: batch inference done in {elapsed:.1f}s ({len(prompts)/elapsed:.1f} samples/s)")

        # Aggregate votes per sample
        unanimous_reject_count = 0
        low_conf_passthrough_count = 0

        for i, output in enumerate(outputs):
            completions = output.outputs  # list of n CompletionOutput
            votes = []
            confidences = []

            for comp in completions:
                parsed = _parse_single_judge_response(comp.text)
                votes.append(parsed)
                conf = _extract_passed_token_confidence(comp, self.tokenizer)
                confidences.append(conf)

            # Count reject votes
            n_reject = sum(1 for v in votes if not v["passed"])
            all_reject = n_reject == len(votes)
            min_confidence = min(confidences) if confidences else 0.0

            if all_reject and min_confidence >= self.confidence_threshold:
                # Unanimous reject with high confidence → skip sandbox
                unanimous_reject_count += 1
                best_vote = votes[0]
                all_results[valid_indices[i]] = {
                    "passed": False,
                    "error_type": best_vote["error_type"],
                    "reason": best_vote["reason"],
                    "confidence": min_confidence,
                    "votes": f"{n_reject}/{len(votes)}",
                }
            elif all_reject and min_confidence < self.confidence_threshold:
                # Unanimous reject but LOW confidence → pass through to sandbox
                low_conf_passthrough_count += 1
                all_results[valid_indices[i]] = {
                    "passed": True,
                    "error_type": "correct",
                    "reason": f"low_conf_passthrough (conf={min_confidence:.2f}, votes={n_reject}/{len(votes)})",
                    "confidence": min_confidence,
                    "votes": f"{n_reject}/{len(votes)}",
                }
            else:
                # Vote split → pass through to sandbox (not unanimous)
                pass_count = len(votes) - n_reject
                all_results[valid_indices[i]] = {
                    "passed": True,
                    "error_type": "correct",
                    "reason": f"vote_split (pass={pass_count}/{len(votes)})",
                    "confidence": min_confidence,
                    "votes": f"{n_reject}/{len(votes)}",
                }

        logger.info(
            f"CoderJudge: reject={unanimous_reject_count} | "
            f"low_conf_passthrough={low_conf_passthrough_count} | "
            f"pass/split={len(prompts) - unanimous_reject_count - low_conf_passthrough_count}"
        )

        return all_results


@ray.remote
class CoderJudgeActor:
    """
    Ray actor that wraps CoderJudge on a dedicated GPU.

    With num_gpus=1, Ray assigns a separate GPU to this actor.
    The model stays loaded permanently — no load/unload overhead.

    Uses n-way self-consistency voting + logprob confidence gating.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-14B-Instruct",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        max_num_seqs: int = 256,
        judge_n: int = 3,
        judge_temperature: float = 0.3,
        confidence_threshold: float = 0.5,
    ):
        self.judge = CoderJudge(
            model_name=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            judge_n=judge_n,
            judge_temperature=judge_temperature,
            confidence_threshold=confidence_threshold,
        )
        self.judge.load()
        logger.info(
            f"CoderJudgeActor: model loaded (n={judge_n}, temp={judge_temperature}, "
            f"conf_threshold={confidence_threshold})"
        )

    def batch_judge(
        self,
        codes: list[str],
        max_tokens: int = 512,
    ) -> list[dict[str, Any]]:
        """Run batch judge (model already loaded, no load/unload)."""
        return self.judge.batch_judge(codes, max_tokens=max_tokens)
