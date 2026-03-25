#!/bin/bash
# Usage: bash eval_checkpoint.sh [checkpoint_dir]
# Example: bash eval_checkpoint.sh /path/to/scratch/checkpoint/baseline-qwen3-1.7b/global_step_10/actor

set -x

# ---- Activate environment ----
source /path/to/scratch/train-env/bin/activate

# ---- Storage ----
SCRATCH=/path/to/scratch
export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export TORCH_HOME=$SCRATCH/torch_cache

# ---- Fix GLIBC ----
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
unset ROCR_VISIBLE_DEVICES

# ---- vLLM ----
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
ulimit -c 0

# ---- Checkpoint paths ----
SCORE_DIR=/path/to/SCORE
export PYTHONPATH=$SCORE_DIR:$PYTHONPATH

# ---- Checkpoint to evaluate ----
CHECKPOINT_DIR=${1:-./checkpoints/score-qwen3-1.7b/merged_hf}
TEST_FILE=$SCORE_DIR/datasets/lcb_v6/test.parquet
OUTPUT_FILE=$SCORE_DIR/eval_results.txt

cd $SCORE_DIR
python eval_checkpoint.py \
    --checkpoint_dir $CHECKPOINT_DIR \
    --test_file $TEST_FILE \
    --output_file $OUTPUT_FILE \
    --n 4 \
    --temperature 0.6 \
    --top_p 0.95 \
    --gpu_memory_utilization 0.85 \
    --max_model_len 18944 \
    --tensor_parallel_size 1
