#!/usr/bin/env bash
set -Eeuo pipefail

experiment_dir="$(cd "$(dirname "$0")" && pwd)"
project_root="$(cd "$experiment_dir/../.." && pwd)"
smoke_root="${SMOKE_ROOT:-/kaggle/working/vision-opd-t4-smoke}"
data_dir="${SMOKE_DATA_DIR:-$smoke_root/data}"
output_dir="${SMOKE_OUTPUT_DIR:-$smoke_root/output}"
log_path="${SMOKE_LOG_PATH:-$smoke_root/smoke.log}"
model_path="${SMOKE_MODEL_PATH:-katuni4ka/tiny-random-qwen2.5-vl}"
ray_temp_dir="${RAY_TMPDIR:-$smoke_root/ray}"

mkdir -p "$smoke_root" "$data_dir" "$output_dir" "$ray_temp_dir"
exec > >(tee -a "$log_path") 2>&1

printf '[t4-smoke] started=%s\n' "$(date --iso-8601=seconds)"
printf '[t4-smoke] project=%s\n' "$project_root"
printf '[t4-smoke] model=%s\n' "$model_path"
/opt/bin/nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader

cd "$project_root"
python experiments/kaggle_t4_compat/verify_vllm_registration.py
python experiments/kaggle_t4_compat/prepare_smoke_data.py --output-dir "$data_dir"

python - <<PY
from pathlib import Path
from transformers import AutoProcessor
from verl.utils.dataset.rl_dataset import RLHFDataset

model_path = ${model_path@Q}
data_path = ${data_dir@Q} + "/train.verl.jsonl"
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
dataset = RLHFDataset(
    data_files=data_path,
    tokenizer=processor.tokenizer,
    processor=processor,
    config={
        "max_prompt_length": 256,
        "truncation": "error",
        "return_multi_modal_inputs": True,
        "image_key": "images",
    },
)
sample = dataset[0]
assert sample["vision_selector_budget"] == 0.2
assert len(sample["multi_modal_data"]["image"]) == 1
assert "image_grid_thw" in sample["multi_modal_inputs"]
print("verl_dataset_load=PASS", len(dataset), Path(data_path))
PY

export HF_HOME="${HF_HOME:-$smoke_root/huggingface}"
export RAY_TMPDIR="$ray_temp_dir"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

MODEL_PATH="$model_path" \
TEACHER_MODEL_PATH="$model_path" \
TASK_TRAIN_FILE="$data_dir/train.verl.jsonl" \
TRAINER_DEFAULT_LOCAL_DIR="$output_dir/checkpoints" \
TRAINER_ROLLOUT_DATA_DIR="$output_dir/rollouts" \
TRAINER_LOGGER='["console"]' \
bash scripts/run_vision_opd.sh \
    actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.vision_selector.hidden_dim=16 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
    actor_rollout_ref.rollout.max_model_len=384 \
    actor_rollout_ref.rollout.max_num_batched_tokens=384 \
    actor_rollout_ref.rollout.response_length=16 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=384 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    data.train_batch_size=2 \
    data.dataloader_num_workers=0 \
    data.max_prompt_length=256 \
    data.max_response_length=16 \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.save_freq=1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.logger='["console"]' \
    trainer.default_local_dir="$output_dir/checkpoints" \
    trainer.rollout_data_dir="$output_dir/rollouts"

printf '[t4-smoke] completed=%s\n' "$(date --iso-8601=seconds)"
printf '[t4-smoke] log=%s\n' "$log_path"

