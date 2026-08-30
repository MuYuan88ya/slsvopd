#!/usr/bin/env bash
set -Eeuo pipefail

conda_root="${CONDA_ROOT:-/root/miniconda3}"
conda_env="${CONDA_ENV:-vision-opd}"
source "$conda_root/etc/profile.d/conda.sh"
conda activate "$conda_env"

experiment_dir="$(cd "$(dirname "$0")" && pwd)"
project_root="$(cd "$experiment_dir/../.." && pwd)"
smoke_root="${SMOKE_ROOT:-/kaggle/working/vision-opd-t4-smoke}"
data_dir="${SMOKE_DATA_DIR:-$smoke_root/data}"
output_dir="${SMOKE_OUTPUT_DIR:-$smoke_root/output}"
log_path="${SMOKE_LOG_PATH:-$smoke_root/smoke.log}"
total_steps="${SMOKE_TOTAL_STEPS:-1}"
total_epochs="${SMOKE_TOTAL_EPOCHS:-1}"
warmup_steps="${SMOKE_WARMUP_STEPS:-0}"
selector_init_scale="${SMOKE_SELECTOR_INIT_SCALE:-0.0001}"
selector_hidden_dim="${SMOKE_SELECTOR_HIDDEN_DIM:-16}"
actor_lr="${SMOKE_ACTOR_LR:-0.000002}"
optimizer_eps="${SMOKE_OPTIMIZER_EPS:-0.000001}"
fsdp_dtype="${SMOKE_FSDP_DTYPE:-float32}"
model_id="${SMOKE_MODEL_ID:-katuni4ka/tiny-random-qwen2.5-vl}"
base_model_path="${SMOKE_BASE_MODEL_PATH:-$smoke_root/model}"
model_path="${SMOKE_MODEL_PATH:-$smoke_root/model-hd16}"
prepare_tiny_model="${SMOKE_PREPARE_TINY_MODEL:-1}"
ray_temp_dir="${RAY_TMPDIR:-$smoke_root/ray}"

mkdir -p "$smoke_root" "$data_dir" "$output_dir" "$ray_temp_dir"
exec > >(tee -a "$log_path") 2>&1

printf '[t4-smoke] started=%s\n' "$(date --iso-8601=seconds)"
printf '[t4-smoke] project=%s\n' "$project_root"
printf '[t4-smoke] model=%s\n' "$model_path"
printf '[t4-smoke] steps=%s epochs=%s warmup=%s selector_hidden_dim=%s selector_init_scale=%s actor_lr=%s optimizer_eps=%s fsdp_dtype=%s\n' \
    "$total_steps" "$total_epochs" "$warmup_steps" "$selector_hidden_dim" "$selector_init_scale" "$actor_lr" \
    "$optimizer_eps" "$fsdp_dtype"
/opt/bin/nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader

cd "$project_root"
python experiments/kaggle_t4_compat/verify_vllm_registration.py
python experiments/kaggle_t4_compat/prepare_smoke_data.py --output-dir "$data_dir"

if [[ ! -f "$base_model_path/config.json" ]]; then
    printf '[t4-smoke] downloading_model=%s\n' "$model_id"
    mkdir -p "$base_model_path"
    HF_HUB_DISABLE_XET=1 hf download "$model_id" --local-dir "$base_model_path"
fi
if [[ "$prepare_tiny_model" == "1" ]]; then
    python experiments/kaggle_t4_compat/prepare_tiny_model.py \
        --base-model "$base_model_path" \
        --output-dir "$model_path"
elif [[ ! -f "$model_path/config.json" ]]; then
    printf 'SMOKE_PREPARE_TINY_MODEL=0 requires a model at %s\n' "$model_path" >&2
    exit 1
fi
export HF_HUB_OFFLINE=1

python - <<PY
import asyncio
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
assert sample["raw_prompt"][0]["content"][0]["type"] == "image"
images, videos = asyncio.run(
    RLHFDataset.process_vision_info(sample["raw_prompt"], image_patch_size=14, config={})
)
assert len(images) == 1 and not videos
prompt = processor.apply_chat_template(
    sample["raw_prompt"], add_generation_prompt=True, tokenize=False
)
model_inputs = processor(text=[prompt], images=images, return_tensors="pt")
assert "image_grid_thw" in model_inputs
print("verl_dataset_load=PASS", len(dataset), Path(data_path))
PY

export HF_HOME="${HF_HOME:-$smoke_root/huggingface}"
export RAY_TMPDIR="$ray_temp_dir"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="$experiment_dir:$project_root${PYTHONPATH:+:$PYTHONPATH}"

MODEL_PATH="$model_path" \
TEACHER_MODEL_PATH="$model_path" \
TASK_TRAIN_FILE="$data_dir/train.verl.jsonl" \
TRAINER_DEFAULT_LOCAL_DIR="$output_dir/checkpoints" \
TRAINER_ROLLOUT_DATA_DIR="$output_dir/rollouts" \
TRAINER_LOGGER='["console"]' \
bash scripts/run_vision_opd.sh \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.vision_selector.hidden_dim="$selector_hidden_dim" \
    actor_rollout_ref.model.vision_selector.init_scale="$selector_init_scale" \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
    actor_rollout_ref.rollout.max_model_len=384 \
    actor_rollout_ref.rollout.max_num_batched_tokens=384 \
    actor_rollout_ref.rollout.max_num_seqs=4 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len=384 \
    actor_rollout_ref.rollout.response_length=16 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=384 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.optim.lr="$actor_lr" \
    actor_rollout_ref.actor.optim.lr_warmup_steps="$warmup_steps" \
    actor_rollout_ref.actor.optim.override_optimizer_config="{eps: $optimizer_eps}" \
    actor_rollout_ref.actor.fsdp_config.dtype="$fsdp_dtype" \
    actor_rollout_ref.actor.fsdp_config.model_dtype="$fsdp_dtype" \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.dtype="$fsdp_dtype" \
    actor_rollout_ref.ref.fsdp_config.model_dtype="$fsdp_dtype" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    data.train_batch_size=2 \
    data.dataloader_num_workers=0 \
    data.max_prompt_length=256 \
    data.max_response_length=16 \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_epochs="$total_epochs" \
    trainer.total_training_steps="$total_steps" \
    trainer.save_freq=1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.logger='["console"]' \
    trainer.default_local_dir="$output_dir/checkpoints" \
    trainer.rollout_data_dir="$output_dir/rollouts" \
    ray_kwargs.ray_init.num_cpus=8

printf '[t4-smoke] completed=%s\n' "$(date --iso-8601=seconds)"
printf '[t4-smoke] log=%s\n' "$log_path"
