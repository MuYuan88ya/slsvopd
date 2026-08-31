# Kaggle dual-T4 compatibility smoke test

This directory contains an opt-in compatibility test for the ephemeral Kaggle
machine with two Tesla T4 GPUs. It does not change the normal Vision-OPD
training entry point or its Flash-Attention defaults.

The smoke test checks three contracts:

1. the editable install exposes the `vllm.general_plugins` entry point and
   resolves the native VisionSelector Qwen2.5-VL architecture;
2. Vero-style JSONL and image records pass through the real data converter and
   VeRL dataset loader;
3. one small FSDP + vLLM training step runs with the Hugging Face actor using
   SDPA and vLLM choosing a T4-compatible backend.

VeRL's colocated FSDP resource pool reserves three logical CPUs per GPU. The
Kaggle host has only four physical vCPUs, so the smoke command registers eight
logical CPUs with Ray. This is deliberate oversubscription for a two-sample
control-flow test, not a recommended throughput configuration.

The default smoke model is `katuni4ka/tiny-random-qwen2.5-vl`. It is only for
control-flow validation. It is not evidence that the 3B training configuration
fits on two 15 GiB GPUs.

Run on the remote machine after activating the `vision-opd` Conda environment:

```bash
experiments/kaggle_t4_compat/sync_to_kaggle.sh
ssh kaggle
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vision-opd
cd /root/Vision-OPD
bash experiments/kaggle_t4_compat/run_smoke.sh
```

The smoke length and numerically sensitive tiny-model settings can be
overridden without changing the production launcher. The following variant is
useful when the test must demonstrate a measurable optimizer update rather
than only validate control flow:

```bash
SMOKE_TOTAL_STEPS=2 SMOKE_TOTAL_EPOCHS=2 \
SMOKE_SELECTOR_HIDDEN_DIM=16 SMOKE_SELECTOR_INIT_SCALE=0.01 SMOKE_ACTOR_LR=0.001 \
SMOKE_OPTIMIZER_EPS=0.000000000001 \
bash experiments/kaggle_t4_compat/run_smoke.sh
```

The tiny FSDP actor/ref default to FP32 because this random model produces
selector gradient norms around `1e-18`; FP16 turns that artificial smoke signal
into zero. Native vLLM rollout remains FP16. Set `SMOKE_FSDP_DTYPE=float16`
only for the separate full-model host-capacity probe.

For the real 3B two-T4 residency experiment, keep the actor and its selector
optimizer on GPU, shard native-vLLM rollout across both GPUs, and phase the
fixed teacher between host and device memory. The teacher stays on CPU during
rollout, then moves to GPU after vLLM enters level-2 sleep and releases its
weights:

```bash
SMOKE_PREPARE_TINY_MODEL=0 \
SMOKE_BASE_MODEL_PATH=/kaggle/working/vision-opd-t4-smoke/qwen2.5-vl-3b \
SMOKE_MODEL_PATH=/kaggle/working/vision-opd-t4-smoke/qwen2.5-vl-3b \
SMOKE_PREPARE_TINY_MODEL=0 \
SMOKE_SELECTOR_HIDDEN_DIM=1024 SMOKE_FSDP_DTYPE=float16 \
SMOKE_ACTOR_PARAM_OFFLOAD=false SMOKE_ACTOR_OPTIMIZER_OFFLOAD=false \
SMOKE_REF_PARAM_OFFLOAD=false SMOKE_TEACHER_GPU_RESIDENT_DURING_UPDATE=true \
SMOKE_ROLLOUT_TP=2 SMOKE_ROLLOUT_GPU_MEMORY_UTILIZATION=0.45 \
SMOKE_ROLLOUT_MM_PROCESSOR_CACHE_GB=0 SMOKE_SAVE_FREQ=-1 \
SMOKE_TOTAL_STEPS=2 SMOKE_TOTAL_EPOCHS=1 \
bash experiments/kaggle_t4_compat/run_smoke.sh
```

The training micro-batch already defaults to one sample per GPU, the minimum
possible value. Larger normalized mini-batches are split into these micro
batches and therefore use gradient accumulation without increasing the
per-forward activation peak.

The frozen 3B backbone and selector forward still use FP16 on T4, but the four
trainable selector tensors are kept as a separate FP32 FSDP unit. This costs
only about 16 MiB of unsharded parameter storage and prevents their small
unscaled gradients from underflowing in FP16 optimizer parameters. With the
default `1e-4` selector initialization, the real 3B/Vero gate reports a finite,
nonzero selector gradient norm and completes the optimizer step.

To exercise actual Vero images instead of the generated fixture, materialize
the dataset and point the smoke launcher at a small JSONL slice:

```bash
python scripts/prepare_vero35k.py materialize --data-dir /root/vero35k
python scripts/prepare_vero35k.py build --data-dir /root/vero35k \
  --dataset-root /root/vero35k --output /root/vero35k/train.smoke2.verl.jsonl --limit 2
SMOKE_TRAIN_FILE=/root/vero35k/train.smoke2.verl.jsonl \
SMOKE_MAX_PROMPT_LENGTH=512 SMOKE_MAX_MODEL_LEN=640 \
bash experiments/kaggle_t4_compat/run_smoke.sh
```

`SMOKE_TOTAL_STEPS=2` with the one-batch/one-epoch fixture executes one update
without marking it as the final planned step. This matters because VeRL always
saves a checkpoint at the final planned step even when `save_freq=-1`. A
standard FSDP checkpoint materializes one CPU state-dict shard on every rank;
the two 3B FP16 shards need about 7.5 GiB in addition to the live Ray and model
state, which does not fit this 31 GiB host reliably. Tiny-model checkpoint and
resume are still tested end to end. Use a host with more RAM for the combined
3B train-and-save gate; the production checkpoint implementation is unchanged.
The smoke-only Ray threshold override is not applied by the production
launcher.

Set `SMOKE_PREPARE_TINY_MODEL=0` and point both `SMOKE_BASE_MODEL_PATH` and
`SMOKE_MODEL_PATH` at an existing full Qwen2.5-VL model to skip synthetic model
generation.

The model is explicitly downloaded before Ray starts, then the training process
runs with `HF_HUB_OFFLINE=1`. This avoids an implicit Hub download inside the
TaskRunner being mistaken for a worker-initialization hang. Override the source
model ID or local model location with `SMOKE_MODEL_ID` and `SMOKE_MODEL_PATH`.
Override output locations with
`SMOKE_DATA_DIR`, `SMOKE_OUTPUT_DIR`, and `SMOKE_LOG_PATH`. All generated data,
checkpoints, Ray state, and command output stay outside the source tree by
default.

The sync script anchors generated-directory exclusions at the project root.
Do not replace `/data/` with an unanchored `data/`: the latter also excludes
`verl/trainer/config/data/` and leaves Hydra unable to compose the training
configuration.

For a stalled remote Python or Ray worker, send `SIGUSR1` to its PID. The
experiment-only `sitecustomize.py` registers this signal as a non-destructive
all-thread Python stack dump and does not affect normal project runs.
