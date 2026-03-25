"""
Standalone evaluation script for SDPO checkpoints.
Usage:
    python eval_checkpoint.py \
        --checkpoint_dir /path/to/global_step_X/actor \
        --test_file /path/to/test.parquet \
        --output_file eval_results.txt \
        --n 4 --temperature 0.6 --top_p 0.95 \
        --gpu_memory_utilization 0.85 \
        --max_model_len 18944
"""

import argparse
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Add project root to path
SCORE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCORE_DIR)

from verl.utils.reward_score.feedback import compute_score


def merge_fsdp_checkpoint(checkpoint_dir, target_dir):
    """Merge FSDP sharded checkpoint into HF format."""
    print(f"Merging FSDP checkpoint: {checkpoint_dir} -> {target_dir}")
    os.makedirs(target_dir, exist_ok=True)
    from verl.model_merger.base_model_merger import ModelMergerConfig
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=checkpoint_dir,
        target_dir=target_dir,
        trust_remote_code=True,
        hf_model_config_path=os.path.join(checkpoint_dir, "huggingface"),
    )
    merger = FSDPModelMerger(config)
    merger.merge_and_save()
    merger.cleanup()
    print(f"Merge complete: {target_dir}")


def load_test_data(test_file, tokenizer):
    """Load test parquet and prepare prompts."""
    df = pd.read_parquet(test_file)
    items = []
    for idx, row in df.iterrows():
        prompt_messages = row["prompt"]  # list of dicts with role/content
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        ground_truth = row["reward_model"]["ground_truth"]
        data_source = row["data_source"]
        extra_info = row.get("extra_info", {})
        if extra_info is None:
            extra_info = {}
        # Keep raw problem text for diagnosis prompts
        problem_text = prompt_messages[-1]["content"] if prompt_messages else ""
        items.append({
            "idx": idx,
            "prompt_text": prompt_text,
            "ground_truth": ground_truth,
            "data_source": data_source,
            "extra_info": extra_info,
            "problem_text": problem_text,
        })
    return items


def _extract_code(response_text):
    """Extract code block from model response."""
    if "```python" in response_text:
        return response_text.split("```python")[-1].split("```")[0]
    elif "```" in response_text:
        parts = response_text.split("```")
        if len(parts) >= 2:
            code = parts[1]
            if "\n" in code:
                first_line, rest = code.split("\n", 1)
                if first_line.strip().isalpha():
                    return rest
            return code
    return response_text


def _build_diagnosis_prompt(problem_text, solution_code):
    """Build diagnosis prompt matching training format."""
    # Truncate to stay within reasonable length
    if len(problem_text) > 1500:
        problem_text = problem_text[:1500] + "\n... (truncated)"
    if len(solution_code) > 1500:
        solution_code = solution_code[:1500] + "\n# ... (truncated)"
    return (
        f"You are a code reviewer. Given the following programming problem and a candidate solution, "
        f"predict the execution result.\n\n"
        f"## Problem\n{problem_text}\n\n"
        f"## Candidate Solution\n```python\n{solution_code}\n```\n\n"
        f"Analyze the solution briefly, then answer with exactly ONE letter on the last line:\n"
        f"A - Pass (correct solution)\n"
        f"B - Compile Error (syntax errors, missing imports, undefined variables)\n"
        f"C - Runtime Error (crashes during execution, type errors, index out of range)\n"
        f"D - Wrong Answer (runs but produces incorrect output)\n"
        f"E - Timeout (correct logic but too slow, TLE)\n\n"
        f"Your answer (one letter):"
    )


def _parse_diagnosis(diag_text):
    """Parse A/B/C/D/E from diagnosis output -> error_type_id."""
    # error_type_id: 0=Pass, 1=RE, 2=WA, 3=TLE, 4=CE
    _LETTER_TO_ERROR_ID = {"A": 0, "B": 4, "C": 1, "D": 2, "E": 3}
    last_line = diag_text.strip().split("\n")[-1].strip().upper()
    for letter in ("A", "B", "C", "D", "E"):
        if letter in last_line:
            return _LETTER_TO_ERROR_ID[letter]
    return 0  # default to Pass if ambiguous


def evaluate(model_path, test_items, n, temperature, top_p,
             gpu_memory_utilization, max_model_len, tensor_parallel_size=1,
             max_num_seqs=32, skip_ccs=False, tokenizer=None):
    """Generate and evaluate using vLLM."""
    print(f"Loading model from {model_path} (tp={tensor_parallel_size})")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        tensor_parallel_size=tensor_parallel_size,
    )

    sampling_params = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        max_tokens=16896,  # match training max_response_length
    )

    # Generate
    prompts = [item["prompt_text"] for item in test_items]
    print(f"Generating {len(prompts)} prompts x {n} samples...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    gen_time = time.time() - t0
    print(f"Generation done in {gen_time:.1f}s")

    # Evaluate
    print("Evaluating...")
    t0 = time.time()
    all_scores = []  # per-item list of n scores
    all_accs = []
    all_error_types = []  # per-item list of n actual error_type_ids
    all_responses = []  # per-item list of n response texts
    all_truncated = []  # per-item list of n truncation flags
    error_type_counts = defaultdict(int)

    for item, output in zip(test_items, outputs):
        item_scores = []
        item_accs = []
        item_error_types = []
        item_responses = []
        item_truncated = []
        for completion in output.outputs:
            response_text = completion.text
            extra = dict(item["extra_info"])
            extra["split"] = "test"
            truncated = completion.finish_reason == "length"
            extra["truncated"] = truncated
            result = compute_score(
                data_source=item["data_source"],
                solution_str=response_text,
                ground_truth=item["ground_truth"],
                extra_info=extra,
            )
            item_scores.append(result["score"])
            item_accs.append(result.get("acc", result["score"]))
            item_error_types.append(result.get("error_type", 0))
            item_responses.append(response_text)
            item_truncated.append(truncated)
            if "error_type_label" in result:
                error_type_counts[result["error_type_label"]] += 1

        all_scores.append(item_scores)
        all_accs.append(item_accs)
        all_error_types.append(item_error_types)
        all_responses.append(item_responses)
        all_truncated.append(item_truncated)

    eval_time = time.time() - t0
    print(f"Evaluation done in {eval_time:.1f}s")

    # Compute metrics
    all_scores = np.array(all_scores)  # (num_items, n)
    all_accs = np.array(all_accs)

    metrics = {}

    # Binary pass/fail: 1 if ALL test cases pass (score == 1.0), else 0
    binary = (all_scores == 1.0).astype(float)  # (num_items, n)

    # avg@k: average accuracy over k rollouts (each rollout binary pass/fail)
    # Per-item: mean of binary results across k rollouts, then average over all items
    metrics[f"avg@{n}"] = float(np.mean(binary))

    # pass@1: for each item, probability that a single random rollout passes
    # = mean of per-item pass rate
    metrics["pass@1"] = float(np.mean(np.mean(binary, axis=1)))

    # pass@k: for each item, whether at least 1 of k rollouts passes
    per_item_any_pass = (np.sum(binary, axis=1) > 0).astype(float)
    metrics[f"pass@{n}"] = float(np.mean(per_item_any_pass))

    # maj@k: majority vote (pass if >50% of rollouts pass)
    per_item_pass_rate = np.mean(binary, axis=1)
    metrics[f"maj@{n}"] = float(np.mean(per_item_pass_rate >= 0.5))

    # Also report partial score stats for reference
    metrics["partial_score/mean"] = float(np.mean(all_scores))
    metrics["partial_score/std"] = float(np.std(np.mean(all_scores, axis=1)))

    # Truncation rate
    total_truncated = sum(sum(t) for t in all_truncated)
    total_samples = len(test_items) * n
    metrics["truncation_rate"] = float(total_truncated / total_samples)

    # --- CCS: Code Calibration Score (self-diagnosis) ---
    if not skip_ccs:
        print("Running self-diagnosis for CCS...")
        t0 = time.time()

        # Build diagnosis prompts for all (item, sample) pairs
        diag_prompts = []
        diag_indices = []  # (item_idx, sample_idx)
        for i, item in enumerate(test_items):
            for j in range(n):
                code = _extract_code(all_responses[i][j])
                prompt = _build_diagnosis_prompt(item["problem_text"], code)
                if tokenizer is not None:
                    diag_text = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False, add_generation_prompt=True,
                        enable_thinking=False,
                    )
                else:
                    diag_text = prompt
                diag_prompts.append(diag_text)
                diag_indices.append((i, j))

        diag_sampling = SamplingParams(
            n=1, temperature=0.1, top_p=0.9, max_tokens=64,
        )
        print(f"  Diagnosing {len(diag_prompts)} samples...")
        diag_outputs = llm.generate(diag_prompts, diag_sampling)
        diag_time = time.time() - t0
        print(f"  Diagnosis done in {diag_time:.1f}s")

        # Parse and compute CCS metrics
        _ERROR_NAMES = {0: "Pass", 1: "RE", 2: "WA", 3: "TLE", 4: "CE"}
        ccs_diffs = []
        quad_A, quad_B, quad_C, quad_D = 0, 0, 0, 0
        n_exact_match = 0
        n_total = 0

        for (i, j), diag_out in zip(diag_indices, diag_outputs):
            diag_text = diag_out.outputs[0].text
            predicted_error_id = _parse_diagnosis(diag_text)
            predicted_pass = (predicted_error_id == 0)
            actual_pass = (all_scores[i][j] == 1.0)
            actual_error_id = all_error_types[i][j]

            # CCS diff: |predicted_pass - actual_pass|
            ccs_diffs.append(abs(float(predicted_pass) - float(actual_pass)))

            # 5-class exact match
            actual_id_for_match = 0 if actual_pass else actual_error_id
            if predicted_error_id == actual_id_for_match:
                n_exact_match += 1

            # 4-quadrant
            if actual_pass and predicted_pass:
                quad_A += 1
            elif not actual_pass and predicted_pass:
                quad_B += 1
            elif actual_pass and not predicted_pass:
                quad_C += 1
            else:
                quad_D += 1

            n_total += 1

        if n_total > 0:
            metrics["CCS"] = float(1.0 - np.mean(ccs_diffs))
            metrics["diag_acc_5class"] = float(n_exact_match / n_total)
            metrics["quad_A(pass+pred_pass)"] = quad_A
            metrics["quad_B(fail+pred_pass)"] = quad_B
            metrics["quad_C(pass+pred_fail)"] = quad_C
            metrics["quad_D(fail+pred_fail)"] = quad_D
            metrics["diag_binary_acc"] = float((quad_A + quad_D) / n_total)
            metrics["diag_time"] = diag_time

    metrics["gen_time"] = gen_time
    metrics["eval_time"] = eval_time
    metrics["total_items"] = len(test_items)
    metrics["n"] = n

    # Error type distribution
    for etype, count in sorted(error_type_counts.items()):
        metrics[f"error_type/{etype}"] = count

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate SDPO checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to global_step_X/actor or a merged HF model dir")
    parser.add_argument("--test_file", type=str,
                        default=os.path.join(SCORE_DIR, "datasets/lcb_v6/test.parquet"))
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output txt file for results")
    parser.add_argument("--n", type=int, default=4, help="Number of samples per prompt")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=18944)
    parser.add_argument("--skip_merge", action="store_true",
                        help="Skip FSDP merge (if checkpoint_dir is already HF format)")
    parser.add_argument("--merged_dir", type=str, default=None,
                        help="Directory to save merged HF model (default: checkpoint_dir/merged_hf)")
    parser.add_argument("--tensor_parallel_size", type=int, default=2,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--skip_ccs", action="store_true",
                        help="Skip CCS (Code Calibration Score) self-diagnosis evaluation")
    args = parser.parse_args()

    # Determine model path
    if args.skip_merge:
        model_path = args.checkpoint_dir
    else:
        # Check if this is an FSDP checkpoint (has model_world_size_*.pt files)
        has_fsdp = any(f.startswith("model_world_size_") for f in os.listdir(args.checkpoint_dir))
        if has_fsdp:
            if args.merged_dir:
                merged_dir = args.merged_dir
            else:
                merged_dir = os.path.join(args.checkpoint_dir, "merged_hf")
            if not os.path.exists(os.path.join(merged_dir, "config.json")):
                merge_fsdp_checkpoint(args.checkpoint_dir, merged_dir)
            else:
                print(f"Using existing merged model: {merged_dir}")
            model_path = merged_dir
        else:
            model_path = args.checkpoint_dir

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load test data
    test_items = load_test_data(args.test_file, tokenizer)
    print(f"Loaded {len(test_items)} test items")

    # Run evaluation
    metrics = evaluate(
        model_path=model_path,
        test_items=test_items,
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        skip_ccs=args.skip_ccs,
        tokenizer=tokenizer,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    # Save to file
    if args.output_file:
        output_file = args.output_file
    else:
        # Default: save next to checkpoint
        step_dir = os.path.dirname(args.checkpoint_dir.rstrip("/"))
        step_name = os.path.basename(step_dir)
        output_file = os.path.join(SCORE_DIR, f"eval_{step_name}.txt")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(output_file, "a") as f:
        parts = [f"[{timestamp}] eval checkpoint={args.checkpoint_dir}"]
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.6f}")
            else:
                parts.append(f"{k}={v}")
        f.write(" | ".join(parts) + "\n")

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
