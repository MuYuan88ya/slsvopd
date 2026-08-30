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

The default smoke model is `katuni4ka/tiny-random-qwen2.5-vl`. It is only for
control-flow validation. It is not evidence that the 3B training configuration
fits on two 15 GiB GPUs.

Run on the remote machine after activating the `vision-opd` Conda environment:

```bash
cd /root/Vision-OPD
bash experiments/kaggle_t4_compat/run_smoke.sh
```

Override the model or output locations with `SMOKE_MODEL_PATH`,
`SMOKE_DATA_DIR`, `SMOKE_OUTPUT_DIR`, and `SMOKE_LOG_PATH`. All generated data,
checkpoints, Ray state, and command output stay outside the source tree by
default.

