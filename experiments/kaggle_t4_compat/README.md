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

The smoke length and numerically sensitive T4/FP16 settings can be overridden
without changing the production launcher, for example:

```bash
SMOKE_TOTAL_STEPS=2 SMOKE_TOTAL_EPOCHS=2 \
SMOKE_SELECTOR_INIT_SCALE=0.01 SMOKE_ACTOR_LR=0.001 \
bash experiments/kaggle_t4_compat/run_smoke.sh
```

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
