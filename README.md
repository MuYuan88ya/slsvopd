# Vision-OPD

**Vision-OPD: Learning to See Fine-Grained Details for Multimodal LLMs via On-Policy Self-Distillation**

<p align="center">
📃 <a href="https://arxiv.org/pdf/2605.18740" target="_blank">Paper</a> | 
🤗 <a href="https://huggingface.co/collections/yuanqianhao/vision-opd">Hugging Face</a> 
</p>

## Overview

Vision-OPD is a regional-to-global on-policy self-distillation framework that transfers a model's own privileged regional perception to its full-image policy, enabling fine-grained visual understanding in a single forward pass — without external teachers, ground-truth labels, reward verifiers, or inference-time tool use.

<p align="center">
  <img src="figures/average_bar_chart.png" alt="Vision-OPD Average Scores" width="60%"/>
</p>

<p align="center"><i>Average scores across fine-grained visual understanding benchmarks, including V* Bench, ZoomBench,  HR-Bench 4K, HR-Bench 8K, MME-RealWorld-EN and MME-RealWorld-CN.</i></p>

## Environment Setup

```bash
conda create -n vision-opd python=3.12 -y
conda activate vision-opd
pip install --upgrade pip
pip install --no-deps -r requirements.txt
pip install -e . --no-deps
pip install flash-attn --no-build-isolation
pip install causal-conv1d==1.6.1 --no-build-isolation
```

## Quick Start

### 1. Deployment

Serve the merged checkpoint with vLLM, for example:

```bash
vllm serve yuanqianhao/Vision-OPD-9B \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 8 \
    --served-model-name Vision-OPD-9B \
    --trust-remote-code
```

The server listens on port 8000 by default. You can then query the model via the OpenAI-compatible API at `http://localhost:8000/v1/chat/completions`.

Please check out our [Hugging Face Collection](https://huggingface.co/collections/yuanqianhao/vision-opd) for all public Vision-OPD checkpoints.

### 2. Evaluation

Evaluate the deployed model on fine-grained visual benchmarks:

```bash
API_BASE="http://localhost:8000/v1/" \
OPENAI_MODEL_ID="Vision-OPD-9B" \
JUDGE_API_BASE="YOUR_JUDGE_API_BASE" \
JUDGE_MODEL="YOUR_JUDGE_MODEL_NAME" \
BENCHMARK="vstar,zoombench,hrbench-4k,hrbench-8k,mme-realworld,mme-realworld-cn" \
bash eval/run_eval.sh
```

Supported benchmarks: `vstar`, `zoombench`, `hrbench-4k`, `hrbench-8k`, `mme-realworld`, `mme-realworld-cn`, `mme-realworld-lite`, `visualprobe`, `mmvp`, `cv-bench`, `mmstar`, `pope`.

The evaluation script runs inference via the OpenAI-compatible API. Judge configuration can be set via `JUDGE_API_BASE` / `JUDGE_MODEL_PATH` environment variables. We use `openai/gpt-oss-120b` as the judge model. Other powerful models like Qwen3.5 or closed-source models are also recommended.

To evaluate Qwen3.5 models as baselines, set `ENABLE_THINKING=False` to run in non-thinking mode, for example:

```bash
API_BASE="http://localhost:8000/v1/" \
OPENAI_MODEL_ID="Qwen3.5-9B" \
ENABLE_THINKING=False \
JUDGE_API_BASE="YOUR_JUDGE_API_BASE" \
JUDGE_MODEL="YOUR_JUDGE_MODEL_NAME" \
BENCHMARK="vstar,zoombench,hrbench-4k,hrbench-8k,mme-realworld,mme-realworld-cn" \
bash eval/run_eval.sh
```

## Training

### 1. Prepare Training Data

Vision-OPD trains on the VeRL annotations from
[Vero-35K-VisionSelector](https://huggingface.co/datasets/muyuho/Vero-35k-VisionSelector). On a laptop or login node,
download only the annotation JSONL and small information files:

```bash
python scripts/prepare_vero35k.py metadata --data-dir ./data/vero35k
```

This command never downloads the image archives. On the training machine, explicitly materialize the approximately
23.6 GB compressed image archive and build the training JSONL:

```bash
python scripts/prepare_vero35k.py materialize --data-dir ./data/vero35k
```

The generated `train.verl.jsonl` keeps Vero's original VeRL fields and only changes each image entry into an absolute
`{"path": ...}` reference valid on that machine. If the images were copied or extracted separately, run only the
conversion step:

```bash
python scripts/prepare_vero35k.py build \
    --data-dir ./data/vero35k \
    --dataset-root /path/containing/the/images/directory
```

### 2. Training

The default training entry now runs the Qwen2.5-VL-3B VisionSelector variant. The actor applies a differentiable
per-image token budget after the vision merger, while a fixed teacher receives the identical unmasked multimodal
input. The LIS scorer is always trainable, the vision encoder and merger are always frozen, and the language model
can either remain frozen or be fully fine-tuned. LoRA is intentionally unsupported in this configuration. The pinned
Torch 2.10, Transformers 5.5, and vLLM 0.18 versions in `requirements.txt` are required because the rollout
implementation subclasses vLLM's native
Qwen2.5-VL model. Keep the editable install (`pip install -e . --no-deps`) so every vLLM subprocess can discover the
`vllm.general_plugins` entry point.

Launch training:

```bash
bash scripts/run_vision_opd.sh
```

The default is LIS-only training. Set `LLM_TRAIN_MODE=full` to fully fine-tune the language model while keeping the
entire vision tower frozen:

```bash
LLM_TRAIN_MODE=full bash scripts/run_vision_opd.sh
```

The defaults use `Qwen/Qwen2.5-VL-3B-Instruct`, retain 20% of the visual tokens for each image, and use a fixed
same-input teacher. During training this is a differentiable soft mask: token count remains unchanged, but its mask
weights sum to the target retained count. AngelSlim evaluation performs hard pruning, so a 20% retention budget maps
to AngelSlim's `ratio=0.8` (prune 80%). To initialize
from AngelSlim-compatible selector weights, set `SELECTOR_PATH` to a Hub repository, local directory, or supported
state-dict file:

```bash
SELECTOR_PATH=AngelSlim/Qwen2.5-VL-3B-Instruct-Selector \
VISION_SELECTOR_BUDGET=0.2 \
bash scripts/run_vision_opd.sh
```

When LIS is initialized from scratch, the actor also uses the original VisionSelector mask regularizer: a detached
hard Top-K mask supervises the differentiable soft mask with binary cross-entropy. Its weight increases linearly from
`0.1` on the first actor update to `2.0` on the final update. This sharpens token decisions while the VOPD loss supplies
the task signal. Override it with `SELECTOR_MASK_LOSS_WEIGHT_START`, `SELECTOR_MASK_LOSS_WEIGHT_END`, or disable it with
`SELECTOR_MASK_LOSS_ENABLED=false`. The selector projections and their mask calculations stay in FP32 even when the
rest of the actor uses FP16.

Videos are passed through without gating in this first version. Hard token deletion and inference acceleration are not
implemented; rollout uses the same differentiable soft gate as actor training.

Every rollout carries its resolved `vision_selector_budget`. By default all members of every group use `0.2`. To use
different budgets inside an eight-response group, provide exactly eight values:

```bash
VISION_SELECTOR_ROLLOUT_BUDGETS='[0.2,0.2,0.2,0.2,0.15,0.15,0.1,0.1]' \
bash scripts/run_vision_opd.sh
```

Validation remains fixed at `VISION_SELECTOR_BUDGET` unless `VISION_SELECTOR_EVAL_BUDGET` is set. A dataset row may
override the configured training budgets with a top-level `vision_selector_budget` column. A scalar applies to every
rollout for that row; a list must contain exactly `rollout.n` values. The resolved budget is passed to vLLM through
`mm_processor_kwargs`, retained with the generated trajectory, and reused by the FSDP student forward. The fixed
teacher receives the same image tensors but no selector budget and therefore always uses the complete visual feature.

To save only the four LIS tensors at checkpoint events, without model,
optimizer, scheduler, RNG, or dataloader state, enable selector-only mode:

```bash
VISION_SELECTOR_CHECKPOINT_ONLY=true bash scripts/run_vision_opd.sh
```

This writes `global_step_<N>/selector/{model.safetensors,config.json,selector_scorer.py}`. It is intentionally not a
resumable trainer checkpoint; pass that directory back through `SELECTOR_PATH` to initialize LIS in a new run. The
default remains a complete resumable VeRL checkpoint plus the same selector export.

### 3. Evaluate the trained selector with AngelSlim

The AngelSlim codebase is vendored under `third_party/AngelSlim` at the revision recorded in `THIRD_PARTY.md`. Install
its token-compressor and `lmms-eval` dependencies in a dedicated GPU evaluation environment following
`third_party/AngelSlim/docs/source/features/token_compressor/installation.md`.

Evaluate only the trained VOPD-VisionSelector at 20% retained visual tokens:

```bash
python scripts/run_angelslim_selector_eval.py \
    --model-path Qwen/Qwen2.5-VL-3B-Instruct \
    --selector-path ./checkpoints/Vision-OPD-Qwen2.5-VL-3B-Instruct/global_step_65/selector
```

The default tasks are `mmmu_val`, `docvqa_val`, and `chartqa`. The same four-tensor selector export is also accepted by
AngelSlim IDPruner:

```bash
python scripts/run_angelslim_selector_eval.py \
    --strategy idpruner \
    --selector-path /path/to/checkpoint/selector
```

For language-model full fine-tuning, pass the merged actor directory as `--model-path`. Original AngelSlim weights or
other methods remain available through the vendored AngelSlim commands and are not run by this helper by default.

### 4. Merge Checkpoints

After training, merge the FSDP-sharded checkpoint into a standard HuggingFace model:

```bash
bash scripts/merge_checkpoint.sh <path_to_checkpoint>
```

For example:

```bash
bash scripts/merge_checkpoint.sh ./checkpoints/Vision-OPD-Qwen2.5-VL-3B-Instruct/global_step_65/
```

The standard checkpoint contains model, optimizer, scheduler, RNG, and training-step state. Each save also writes a
strict four-tensor AngelSlim-compatible export under `selector/`. The merged actor preserves the scorer tensors, but
this first version intentionally does not produce a standalone hard-pruning Hugging Face deployment repository;
the standard VeRL entrypoint reads the custom rollout architecture from `vopd.yaml`, while vLLM discovers its native
implementation through the installed `vllm.general_plugins` entry point.

## Citation

If you find Vision-OPD useful for your research, please consider citing:

```bibtex
@article{yuan2026vision,
  title={Vision-OPD: Learning to See Fine Details for Multimodal LLMs via On-Policy Self-Distillation},
  author={Yuan, Qianhao and Lou, Jie and Yu, Xing and Lin, Hongyu and Sun, Le and Han, Xianpei and Lu, Yaojie},
  journal={arXiv preprint arXiv:2605.18740},
  year={2026}
}
```

## License

Apache-2.0 License
