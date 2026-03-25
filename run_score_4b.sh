#!/bin/bash
# SCORE try3: Qwen3-4B with RFT warmup + binary self-diagnosis (4B <= 4B threshold)
# Usage: sbatch submit_score_4b.sh
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
export NCCL_SOCKET_IFNAME=lo
unset ROCR_VISIBLE_DEVICES
export VERL_REWARD_WORKERS=6

# ---- Ray (unique ports for 4B experiment) ----
export RAY_TMPDIR=/tmp/ray_${USER}_try3_4b_$$
mkdir -p $RAY_TMPDIR
export RAY_PORT=6404
export RAY_DASHBOARD_PORT=8271
export RAY_OBJECT_STORE_MEMORY=1000000000

# ---- vLLM ----
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore::FutureWarning"
ulimit -c 0

# ---- Paths ----
export SCORE_DIR=/path/to/SCORE
export PYTHONPATH=$SCORE_DIR:$PYTHONPATH
export TASK=datasets/lcb_v6
export EXPERIMENT=score_try3_sft-qwen3-4b-lcb-v6_2


RFT_MODEL="Qwen/Qwen3-4B"
echo "Using model: $RFT_MODEL"

# ---- Output log ----
LOG_TXT="$SCORE_DIR/logs_${EXPERIMENT}.txt"

# ---- Run SCORE training ----
cd $SCORE_DIR
stdbuf -oL -eL python -u -m verl.trainer.main_ppo --config-name score \
    data.train_batch_size=32 \
    data.train_files="['$SCORE_DIR/$TASK/train.parquet']" \
    data.val_files="['$SCORE_DIR/$TASK/test.parquet']" \
    data.max_prompt_length=4096 \
    data.max_response_length=8192  \
    data.filter_overlong_prompts=True \
    data.trust_remote_code=True \
    "data.apply_chat_template_kwargs={enable_thinking: false}" \
    actor_rollout_ref.model.path=$RFT_MODEL \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=18944 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.self_distillation.distillation_topk=20 \
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=10240 \
    actor_rollout_ref.actor.self_distillation.is_clip=2.0 \
    actor_rollout_ref.actor.self_distillation.alpha=1.0 \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.01 \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=18944 \
    actor_rollout_ref.rollout.max_model_len=18944 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    custom_reward_function.path=$SCORE_DIR/verl/utils/reward_score/feedback/__init__.py \
    reward_model.use_reward_loop=False \
    reward_model.launch_reward_fn_async=True \
    reward_model.coder_judge.enable=False \
    trainer.project_name=score_try3_sft_4b-local \
    trainer.group_name=score_try3_sft_4b-lcb-v6 \
    trainer.experiment_name=$EXPERIMENT \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_epochs=30 \
    trainer.save_freq=5 \
    trainer.test_freq=10 \
    trainer.val_before_train=False \
    trainer.resume_mode=disable \
    trainer.total_training_steps=135 \
    trainer.default_local_dir=$SCRATCH/checkpoint/score_try3_sft-qwen3-4b_2 \
    max_model_len=18944 2>&1 | tee "$LOG_TXT"
