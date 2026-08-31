#!/usr/bin/env bash
set -Eeuo pipefail

conda_root="${CONDA_ROOT:-/root/miniconda3}"
conda_env="${CONDA_ENV:-vision-opd}"
source "$conda_root/etc/profile.d/conda.sh"
conda activate "$conda_env"

# The full 3B phase-residency probe briefly sits above Ray's conservative 95%
# node threshold while vLLM starts, although the kernel still reports >1 GiB
# available and the teacher moves from CPU to GPU before the actor update.
export RAY_memory_usage_threshold="${SMOKE_RAY_MEMORY_USAGE_THRESHOLD:-0.99}"

# Kaggle exposes the driver library outside CUDA_HOME. FlashInfer's JIT linker
# needs this directory for full-size Qwen2.5-VL attention heads.
if [[ -d /usr/local/nvidia/lib64 ]]; then
    export LD_LIBRARY_PATH="/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LIBRARY_PATH="/usr/local/nvidia/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
fi

experiment_dir="$(cd "$(dirname "$0")" && pwd)"
project_root="$(cd "$experiment_dir/../.." && pwd)"
smoke_root="${SMOKE_ROOT:-/kaggle/working/vision-opd-t4-smoke}"
data_dir="${SMOKE_DATA_DIR:-$smoke_root/data}"
training_file="${SMOKE_TRAIN_FILE:-$data_dir/train.verl.jsonl}"
output_dir="${SMOKE_OUTPUT_DIR:-$smoke_root/output}"
log_path="${SMOKE_LOG_PATH:-$smoke_root/smoke.log}"
total_steps="${SMOKE_TOTAL_STEPS:-1}"
total_epochs="${SMOKE_TOTAL_EPOCHS:-1}"
save_freq="${SMOKE_SAVE_FREQ:-1}"
warmup_steps="${SMOKE_WARMUP_STEPS:-0}"
selector_init_scale="${SMOKE_SELECTOR_INIT_SCALE:-0.0001}"
selector_hidden_dim="${SMOKE_SELECTOR_HIDDEN_DIM:-16}"
actor_lr="${SMOKE_ACTOR_LR:-0.000002}"
optimizer_eps="${SMOKE_OPTIMIZER_EPS:-0.000001}"
fsdp_dtype="${SMOKE_FSDP_DTYPE:-float32}"
actor_param_offload="${SMOKE_ACTOR_PARAM_OFFLOAD:-true}"
actor_optimizer_offload="${SMOKE_ACTOR_OPTIMIZER_OFFLOAD:-true}"
ref_param_offload="${SMOKE_REF_PARAM_OFFLOAD:-true}"
rollout_tp="${SMOKE_ROLLOUT_TP:-1}"
rollout_gpu_memory_utilization="${SMOKE_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.25}"
rollout_mm_processor_cache_gb="${SMOKE_ROLLOUT_MM_PROCESSOR_CACHE_GB:-0}"
ppo_micro_batch_size_per_gpu="${SMOKE_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
teacher_gpu_resident_during_update="${SMOKE_TEACHER_GPU_RESIDENT_DURING_UPDATE:-false}"
rollout_budgets="${SMOKE_ROLLOUT_BUDGETS:-}"
selector_checkpoint_only="${SMOKE_SELECTOR_CHECKPOINT_ONLY:-false}"
max_prompt_length="${SMOKE_MAX_PROMPT_LENGTH:-256}"
max_model_len="${SMOKE_MAX_MODEL_LEN:-384}"
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
printf '[t4-smoke] training_file=%s\n' "$training_file"
printf '[t4-smoke] steps=%s epochs=%s save_freq=%s warmup=%s selector_hidden_dim=%s selector_init_scale=%s actor_lr=%s optimizer_eps=%s fsdp_dtype=%s\n' \
    "$total_steps" "$total_epochs" "$save_freq" "$warmup_steps" "$selector_hidden_dim" "$selector_init_scale" "$actor_lr" \
    "$optimizer_eps" "$fsdp_dtype"
printf '[t4-smoke] actor_param_offload=%s actor_optimizer_offload=%s ref_param_offload=%s rollout_tp=%s rollout_gpu_memory_utilization=%s ppo_micro_batch_size_per_gpu=%s\n' \
    "$actor_param_offload" "$actor_optimizer_offload" "$ref_param_offload" "$rollout_tp" \
    "$rollout_gpu_memory_utilization" "$ppo_micro_batch_size_per_gpu"
printf '[t4-smoke] teacher_gpu_resident_during_update=%s\n' "$teacher_gpu_resident_during_update"
printf '[t4-smoke] rollout_budgets=%s selector_checkpoint_only=%s\n' \
    "${rollout_budgets:-fixed}" "$selector_checkpoint_only"
printf '[t4-smoke] rollout_mm_processor_cache_gb=%s\n' "$rollout_mm_processor_cache_gb"
/opt/bin/nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader

cd "$project_root"
python experiments/kaggle_t4_compat/verify_vllm_registration.py
if [[ -z "${SMOKE_TRAIN_FILE:-}" ]]; then
    python experiments/kaggle_t4_compat/prepare_smoke_data.py --output-dir "$data_dir"
elif [[ ! -f "$training_file" ]]; then
    printf 'SMOKE_TRAIN_FILE does not exist: %s\n' "$training_file" >&2
    exit 1
fi

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
data_path = ${training_file@Q}
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
dataset = RLHFDataset(
    data_files=data_path,
    tokenizer=processor.tokenizer,
    processor=processor,
    config={
        "max_prompt_length": int(${max_prompt_length@Q}),
        "truncation": "error",
        "return_multi_modal_inputs": True,
        "image_key": "images",
    },
)
sample = dataset[0]
assert sample.get("vision_selector_budget", 0.2) == 0.2
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
TASK_TRAIN_FILE="$training_file" \
TRAINER_DEFAULT_LOCAL_DIR="$output_dir/checkpoints" \
TRAINER_ROLLOUT_DATA_DIR="$output_dir/rollouts" \
TRAINER_LOGGER='["console"]' \
VISION_SELECTOR_ROLLOUT_BUDGETS="$rollout_budgets" \
bash scripts/run_vision_opd.sh \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.vision_selector.hidden_dim="$selector_hidden_dim" \
    actor_rollout_ref.model.vision_selector.init_scale="$selector_init_scale" \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$rollout_tp" \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.gpu_memory_utilization="$rollout_gpu_memory_utilization" \
    actor_rollout_ref.rollout.max_model_len="$max_model_len" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$max_model_len" \
    actor_rollout_ref.rollout.max_num_seqs=4 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len="$max_model_len" \
    actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb="$rollout_mm_processor_cache_gb" \
    actor_rollout_ref.rollout.response_length=16 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$ppo_micro_batch_size_per_gpu" \
    actor_rollout_ref.actor.self_distillation.teacher_gpu_resident_during_update="$teacher_gpu_resident_during_update" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$max_model_len" \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.optim.lr="$actor_lr" \
    actor_rollout_ref.actor.optim.lr_warmup_steps="$warmup_steps" \
    actor_rollout_ref.actor.optim.override_optimizer_config="{eps: $optimizer_eps}" \
    actor_rollout_ref.actor.fsdp_config.dtype="$fsdp_dtype" \
    actor_rollout_ref.actor.fsdp_config.model_dtype="$fsdp_dtype" \
    actor_rollout_ref.actor.fsdp_config.param_offload="$actor_param_offload" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="$actor_optimizer_offload" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.dtype="$fsdp_dtype" \
    actor_rollout_ref.ref.fsdp_config.model_dtype="$fsdp_dtype" \
    actor_rollout_ref.ref.fsdp_config.param_offload="$ref_param_offload" \
    data.train_batch_size=2 \
    data.dataloader_num_workers=0 \
    data.max_prompt_length="$max_prompt_length" \
    data.max_response_length=16 \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_epochs="$total_epochs" \
    trainer.total_training_steps="$total_steps" \
    trainer.save_freq="$save_freq" \
    trainer.vision_selector_checkpoint_only="$selector_checkpoint_only" \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.logger='["console"]' \
    trainer.default_local_dir="$output_dir/checkpoints" \
    trainer.rollout_data_dir="$output_dir/rollouts" \
    ray_kwargs.ray_init.num_cpus=8

printf '[t4-smoke] completed=%s\n' "$(date --iso-8601=seconds)"
printf '[t4-smoke] log=%s\n' "$log_path"
