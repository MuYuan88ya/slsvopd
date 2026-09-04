# Vision-OPD + VisionSelector

本仓库在 [Vision-OPD](https://arxiv.org/pdf/2605.18740) 的 on-policy self-distillation 训练流程中接入了
VisionSelector/LIS（Learned Importance Scorer），用于训练 Qwen2.5-VL 的视觉 token 选择器。

训练时，学生和固定教师读取完全相同的文本与图片：

- 学生在 Qwen2.5-VL vision merger 后使用 LIS 产生可微 soft Top-K mask；
- 教师不创建 LIS，始终使用完整视觉特征，并且所有参数固定；
- 学生通过 VOPD 蒸馏损失学习任务信息，同时通过原版 VisionSelector mask BCE 逐渐锐化选择结果；
- LIS 始终训练，vision encoder 和 vision merger 始终冻结；
- language model 可以选择冻结（默认）或全参数训练，不支持 LoRA；
- rollout 使用项目注册的 native vLLM Qwen2.5-VL architecture，训练侧使用 FSDP1 pre-hook 注入 LIS。

当前默认模型为 `Qwen/Qwen2.5-VL-3B-Instruct`，默认每张图片保留 20% 视觉 token。

## 目录

- [当前支持范围](#当前支持范围)
- [安装](#安装)
- [构建 Vero-35K 训练数据](#构建-vero-35k-训练数据)
- [开始训练](#开始训练)
- [视觉 token 预算](#视觉-token-预算)
- [Checkpoint 保存、恢复与导出](#checkpoint-保存恢复与导出)
- [使用 AngelSlim 评估](#使用-angelslim-评估)
- [通用 Vision-OPD Benchmark 评估](#通用-vision-opd-benchmark-评估)
- [测试](#测试)
- [实现结构](#实现结构)

## 当前支持范围

| 项目 | 当前实现 |
| --- | --- |
| 基础模型 | Qwen2.5-VL-3B-Instruct |
| 训练后端 | VeRL + FSDP1 |
| Rollout 后端 | vLLM 0.18 native Qwen2.5-VL |
| 学生输入 | 原始文本和图片 |
| 教师输入 | 与学生完全相同的文本和图片 |
| 教师 | fixed、全冻结、无 LIS |
| 视觉模块 | encoder 和 merger 全冻结，LIS 始终训练 |
| Language model | `frozen` 或 `full` |
| 视频 | 不经过 selector |
| 训练时选择 | soft gating，不减少 token 数量 |
| AngelSlim 评估 | hard pruning，真实删除视觉 token |

第一版不支持 LoRA、视频 token 选择、Transformers 通用 vLLM fallback，以及训练期间的硬 token 删除。

## 安装

### 运行环境

完整训练需要 Linux、NVIDIA GPU 和 CUDA。推荐使用 Python 3.12，并保持以下关键版本与
[`requirements.txt`](requirements.txt) 一致：

- PyTorch 2.10.0
- Transformers 5.5.0
- vLLM 0.18.0
- Ray 2.53.0

macOS 可以构建元数据、检查 JSONL 和运行不依赖真实 vLLM/CUDA 的单元测试，但不能运行本仓库的 vLLM
rollout 训练。

### 创建环境

```bash
git clone --branch experiment/kaggle-t4-residency https://github.com/MuYuan88ya/slsvopd.git
cd slsvopd

conda create -n vision-opd python=3.12 -y
conda activate vision-opd

python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-deps -r requirements.txt
python -m pip install -e . --no-deps
python -m pip install flash-attn --no-build-isolation
```

必须保留 `pip install -e . --no-deps`。它不只是设置 `PYTHONPATH`，还会安装
`vllm.general_plugins` entry point，使 Ray actor、vLLM EngineCore 和 GPU worker 都能找到自定义 architecture：

```text
Qwen2_5_VLForConditionalGenerationWithVisionSelector
```

可以用下面的命令确认插件已经安装：

```bash
python -c "from importlib.metadata import entry_points; print([(e.name, e.value) for e in entry_points(group='vllm.general_plugins') if e.name == 'vision_opd_selector'])"
```

预期输出包含：

```text
('vision_opd_selector', 'vision_opd_selector.vllm_plugin:register')
```

如果更新了 `vision_opd_selector/`、切换了仓库或重建了环境，请再次执行 editable install。

## 构建 Vero-35K 训练数据

默认训练数据为
[muyuho/Vero-35k-VisionSelector](https://huggingface.co/datasets/muyuho/Vero-35k-VisionSelector)。项目保留原始
VeRL JSONL 格式，不把图片重新编码进 JSONL，只将 `images` 中的相对路径转换为训练机器上的绝对路径。

### 1. 只下载元数据，不下载图片

在本地电脑或登录节点执行：

```bash
python scripts/prepare_vero35k.py metadata \
    --data-dir ./data/vero35k
```

该命令的 allowlist 只有以下小文件，绝不会下载 `images.tar.gz.part_*`：

```text
data/vero35k/
├── vero_train_all_35k.verl.jsonl
├── README.md
└── summary.json
```

### 2. 在训练机器下载并解压图片

只有明确执行 `materialize` 时才会下载大型图片分卷：

```bash
python scripts/prepare_vero35k.py materialize \
    --data-dir ./data/vero35k
```

它会依次完成：

1. 下载元数据；
2. 下载 `images.tar.gz.part_*`；
3. 流式拼接并解压图片，不额外生成一份完整 tar 包；
4. 检查每一个图片路径；
5. 生成 `data/vero35k/train.verl.jsonl`；
6. 默认删除下载的压缩分卷。

如需保留压缩分卷，增加 `--keep-archives`。第一次只想验证少量样本时，可以增加 `--limit`：

```bash
python scripts/prepare_vero35k.py materialize \
    --data-dir ./data/vero35k \
    --limit 100
```

`--limit` 只限制输出 JSONL 的记录数，不会减少需要下载的图片分卷体积。

### 3. 已经有图片时只构建 JSONL

如果图片由共享存储、对象存储或另一台机器提供，不需要再次下载：

```bash
python scripts/prepare_vero35k.py build \
    --data-dir ./data/vero35k \
    --dataset-root /mnt/datasets/Vero-35k-VisionSelector
```

也可以显式指定输入和输出：

```bash
python scripts/prepare_vero35k.py build \
    --source ./data/vero35k/vero_train_all_35k.verl.jsonl \
    --dataset-root /mnt/datasets/Vero-35k-VisionSelector \
    --output ./data/vero35k/train.verl.jsonl
```

生成文件中的图片路径是绝对路径。数据移动到另一台机器或挂载点变化后，应重新运行 `build`。

### 训练记录格式

转换脚本保留源数据中的其他字段，仅规范化 `images`：

```json
{
  "data_source": "...",
  "prompt": [
    {"role": "user", "content": "<image>\nQuestion ..."}
  ],
  "images": [
    {"path": "/absolute/path/to/image.jpg"}
  ],
  "reward_model": {
    "style": "...",
    "ground_truth": "..."
  },
  "extra_info": {
    "answer": "..."
  }
}
```

可选字段 `vision_selector_budget` 用于覆盖某条数据的采样预算：

```json
{"vision_selector_budget": 0.2}
```

标量会应用到该问题的所有 rollout。也可以提供长度恰好等于 `rollout.n` 的列表：

```json
{"vision_selector_budget": [0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05]}
```

## 开始训练

### 默认训练

准备好 `data/vero35k/train.verl.jsonl` 后执行：

```bash
bash scripts/run_vision_opd.sh
```

默认训练入口使用：

| 参数 | 默认值 |
| --- | --- |
| 模型 | `Qwen/Qwen2.5-VL-3B-Instruct` |
| 训练文件 | `data/vero35k/train.verl.jsonl` |
| GPU 数 | 8 |
| train batch size | 96 |
| rollout.n | 8 |
| 视觉 token 保留率 | 0.2 |
| 学生 | LIS 可训练，vision 与 LLM 冻结 |
| 教师 | 相同输入、固定、全冻结、完整视觉特征 |
| actor/ref param offload | 开启 |
| optimizer offload | 开启 |
| 保存频率 | 只在最后一步保存 |

训练输出默认写入：

```text
checkpoints/Vision-OPD-Qwen2.5-VL-3B-Instruct/
rollouts/Vision-OPD-Qwen2.5-VL-3B-Instruct/
```

### 常用环境变量

```bash
DATA_DIR=/mnt/datasets/vero35k \
TASK_TRAIN_FILE=/mnt/datasets/vero35k/train.verl.jsonl \
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct \
VISION_SELECTOR_BUDGET=0.2 \
bash scripts/run_vision_opd.sh
```

| 环境变量 | 含义 | 默认值 |
| --- | --- | --- |
| `DATA_DIR` | 数据目录 | `./data/vero35k` |
| `TASK_TRAIN_FILE` | 训练 JSONL | `$DATA_DIR/train.verl.jsonl` |
| `MODEL_PATH` | 学生基础模型 | `Qwen/Qwen2.5-VL-3B-Instruct` |
| `TEACHER_MODEL_PATH` | 固定教师模型 | 与 `MODEL_PATH` 相同 |
| `LLM_TRAIN_MODE` | `frozen` 或 `full` | `frozen` |
| `VISION_SELECTOR_BUDGET` | 默认视觉 token 保留率 | `0.2` |
| `VISION_SELECTOR_ROLLOUT_BUDGETS` | 同一 group 内的预算列表 | 空，使用固定预算 |
| `VISION_SELECTOR_EVAL_BUDGET` | validation 预算 | 空，使用默认预算 |
| `SELECTOR_PATH` | 初始化 LIS 的权重目录、Hub repo 或文件 | 空，从头初始化 |
| `VISION_SELECTOR_CHECKPOINT_ONLY` | 是否只保存 LIS | `false` |
| `SELECTOR_MASK_LOSS_ENABLED` | 是否启用 mask BCE | `true` |
| `SELECTOR_MASK_LOSS_WEIGHT_START` | mask BCE 初始权重 | `0.1` |
| `SELECTOR_MASK_LOSS_WEIGHT_END` | mask BCE 最终权重 | `2.0` |
| `CUSTOM_CHAT_TEMPLATE_FILE` | 自定义 chat template 文件 | 空 |
| `MAX_MODEL_LEN` | vLLM 最大序列长度 | prompt + response 长度 |

脚本末尾会继续接收 Hydra override，因此不常用参数无需修改脚本。例如在 4 张 GPU 上训练、每 10 步保存一次：

```bash
bash scripts/run_vision_opd.sh \
    trainer.n_gpus_per_node=4 \
    data.train_batch_size=32 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    trainer.save_freq=10 \
    trainer.max_actor_ckpt_to_keep=2
```

显存足够时，可以关闭原生 FSDP offload 来减少 CPU 内存传输：

```bash
bash scripts/run_vision_opd.sh \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=False
```

### LIS-only 与 LLM 全参数训练

默认只训练 LIS：

```bash
LLM_TRAIN_MODE=frozen bash scripts/run_vision_opd.sh
```

训练 LIS 和全部 language model 参数，同时继续冻结整个 vision tower：

```bash
LLM_TRAIN_MODE=full bash scripts/run_vision_opd.sh
```

无论哪种模式，教师都不会更新。当前配置明确禁止 LoRA。

### 从已有 LIS 权重开始

`SELECTOR_PATH` 支持：

- 本仓库导出的 `selector/` 目录；
- AngelSlim-compatible selector 目录；
- Hugging Face selector repo；
- `.safetensors`、`.pt` 或 `.pth` state dict 文件。

目录中必须包含四个参数：

```text
k_proj.weight
k_proj.bias
q_proj.weight
q_proj.bias
```

示例：

```bash
SELECTOR_PATH=/path/to/global_step_50/actor/selector \
bash scripts/run_vision_opd.sh
```

如果同时恢复完整 checkpoint，checkpoint 内的 actor 状态会覆盖初始化用的 `SELECTOR_PATH`。

### Actor loss

当前学生优化目标为：

```text
actor_loss = VOPD_loss + mask_loss_weight × BCE(soft_mask, stop_gradient(hard_topk_mask))
```

mask BCE 来自原版 VisionSelector。它主要将 soft mask 锐化；哪些视觉 token 对任务有用，仍由 VOPD
student/teacher 蒸馏信号决定。权重默认从第一次 actor update 的 0.1 线性增长到最后一次 update 的 2.0。

LIS 的 q/k projection、可微 Top-K 和 mask loss 始终以 FP32 计算，即使其余 actor 使用 FP16。mask 与 merger
输出相乘时才转换回视觉特征 dtype。

## 视觉 token 预算

`VISION_SELECTOR_BUDGET` 表示保留率，不是删除率：

| budget | 含义 |
| --- | --- |
| `0.2` | 保留 20%，删除率为 80% |
| `0.15` | 保留 15%，删除率为 85% |
| `0.1` | 保留 10%，删除率为 90% |

每张图片保留的 merger token 数为：

```text
max(1, floor(image_token_count × budget))
```

训练和 rollout 的 soft mask 不改变 token 数量、placeholder、attention mask 或 M-RoPE。只有 AngelSlim 评估阶段
才进行硬 token 删除。

### 同一 GRPO group 使用不同预算

默认一个 group 的 8 条 rollout 都使用 0.2。下面的设置会按顺序给同一个问题的 8 条 rollout 分配不同预算：

```bash
VISION_SELECTOR_ROLLOUT_BUDGETS='[0.2,0.2,0.2,0.2,0.15,0.15,0.1,0.1]' \
bash scripts/run_vision_opd.sh
```

列表长度必须恰好等于 `actor_rollout_ref.rollout.n`。对于多图片样本，同一条 rollout 的预算会展开到该样本的
每张图片，但不同图片仍分别执行 Top-K，不会跨图片竞争。

预算通过 vLLM `mm_processor_kwargs` 进入 rollout，保存在生成的 trajectory 中，并在 FSDP 学生 forward 时复用。
教师复用同一份 `pixel_values`、attention mask 和 position ids，但不接收 selector budget，因此始终使用全部视觉特征。

## Checkpoint 保存、恢复与导出

### 完整、可恢复 checkpoint（默认）

默认 `VISION_SELECTOR_CHECKPOINT_ONLY=false`。训练最后一步一定保存；如需周期保存，设置正数
`trainer.save_freq`：

```bash
bash scripts/run_vision_opd.sh \
    trainer.save_freq=10 \
    trainer.max_actor_ckpt_to_keep=3
```

完整 checkpoint 的结构为：

```text
checkpoints/Vision-OPD-Qwen2.5-VL-3B-Instruct/
├── latest_checkpointed_iteration.txt
└── global_step_N/
    ├── actor/
    │   ├── model_world_size_*_rank_*.pt
    │   ├── optim_world_size_*_rank_*.pt
    │   ├── extra_state_world_size_*_rank_*.pt
    │   ├── huggingface/
    │   ├── fsdp_config.json
    │   └── selector/
    │       ├── model.safetensors
    │       ├── config.json
    │       └── selector_scorer.py
    └── data.pt
```

其中包含 actor 模型分片、optimizer、scheduler、RNG、训练 step 和 dataloader 状态。`actor/selector/` 是保存完整
checkpoint 成功后额外导出的 AngelSlim-compatible 四参数 LIS。

### 自动恢复

默认 `trainer.resume_mode=auto`。使用同一个 `trainer.default_local_dir` 再次启动时，会读取
`latest_checkpointed_iteration.txt` 并恢复最新完整 checkpoint：

```bash
bash scripts/run_vision_opd.sh
```

显式恢复某一步：

```bash
bash scripts/run_vision_opd.sh \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path=/absolute/path/to/global_step_50
```

强制从头开始：

```bash
bash scripts/run_vision_opd.sh trainer.resume_mode=disable
```

### 只保存 LIS

如果只需要训练后的 selector，不需要恢复 optimizer 和训练进度：

```bash
VISION_SELECTOR_CHECKPOINT_ONLY=true \
bash scripts/run_vision_opd.sh trainer.save_freq=10
```

此模式写入：

```text
global_step_N/
└── selector/
    ├── model.safetensors
    ├── config.json
    └── selector_scorer.py
```

注意：selector-only checkpoint 不能恢复原训练，也不能传给 `merge_checkpoint.sh`。它只能通过
`SELECTOR_PATH` 初始化新的训练，或直接传给 AngelSlim 评估。

### 合并完整 FSDP checkpoint

需要完整 Hugging Face actor 时，传入 `global_step_N`，不能传入 `actor/` 或 selector-only 目录：

```bash
bash scripts/merge_checkpoint.sh \
    ./checkpoints/Vision-OPD-Qwen2.5-VL-3B-Instruct/global_step_50
```

合并结果写到同一个 `global_step_50/` 顶层，同时保留原来的 `actor/` 分片。对于默认 LIS-only 训练，评估 selector
时通常不需要合并，直接让 AngelSlim 加载基础 Qwen2.5-VL 和 `actor/selector/` 即可。只有
`LLM_TRAIN_MODE=full` 时，才需要把合并目录作为评估的 `--model-path`。

合并后的普通 Hugging Face 模型本身不会自动执行 AngelSlim 硬剪枝；硬剪枝评估仍应使用下一节的 helper。

## 使用 AngelSlim 评估

仓库在 [`third_party/AngelSlim`](third_party/AngelSlim) 固定了一份 AngelSlim 源码，版本记录在
[`THIRD_PARTY.md`](THIRD_PARTY.md)。建议在独立 GPU 环境中安装，避免 `lmms-eval` 与训练依赖互相影响：

```bash
python -m pip install -e third_party/AngelSlim
python -m pip install -r third_party/AngelSlim/requirements/requirements_multimodal.txt

git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
python -m pip install -e './lmms-eval[all]'
```

不要使用 `pip install lmms_eval` 代替官方仓库的 editable install。

### VisionSelector 评估

完整 checkpoint 的 selector 路径为 `global_step_N/actor/selector`：

```bash
python scripts/run_angelslim_selector_eval.py \
    --model-path Qwen/Qwen2.5-VL-3B-Instruct \
    --selector-path ./checkpoints/Vision-OPD-Qwen2.5-VL-3B-Instruct/global_step_50/actor/selector \
    --strategy vision_selector \
    --retention-budget 0.2
```

selector-only checkpoint 的路径少一层 `actor/`：

```bash
python scripts/run_angelslim_selector_eval.py \
    --model-path Qwen/Qwen2.5-VL-3B-Instruct \
    --selector-path ./checkpoints/Vision-OPD-Qwen2.5-VL-3B-Instruct/global_step_50/selector \
    --retention-budget 0.2
```

默认任务为：

```text
mmmu_val docvqa_val chartqa
```

自定义任务和输出目录：

```bash
python scripts/run_angelslim_selector_eval.py \
    --model-path Qwen/Qwen2.5-VL-3B-Instruct \
    --selector-path /path/to/selector \
    --retention-budget 0.15 \
    --tasks mmmu_val docvqa_val chartqa textvqa_val \
    --batch-size 1 \
    --output-dir ./eval_results/selector-r015 \
    --log-samples
```

helper 接收的是保留率，并自动转换为 AngelSlim 的删除率：

```text
AngelSlim ratio = 1 - retention_budget
```

因此 `--retention-budget 0.2` 会生成 `ratio=0.8` 的 AngelSlim 配置。

### IDPruner 评估

同一份四参数 LIS 也可以作为 AngelSlim IDPruner 的 importance scorer：

```bash
python scripts/run_angelslim_selector_eval.py \
    --model-path Qwen/Qwen2.5-VL-3B-Instruct \
    --selector-path /path/to/selector \
    --strategy idpruner \
    --retention-budget 0.2 \
    --mmr-lambda 0.5 \
    --parallel-k 1
```

如果训练时使用 `LLM_TRAIN_MODE=full`，先合并完整 checkpoint，再把合并后的 `global_step_N/` 作为
`--model-path`。

## 通用 Vision-OPD Benchmark 评估

[`eval/run_eval.sh`](eval/run_eval.sh) 用于评估已经部署为 OpenAI-compatible API 的模型。它与上一节的
AngelSlim 本地硬剪枝评估是两条不同流程；这个脚本本身不会加载 LIS 或执行 token pruning。

```bash
API_BASE=http://localhost:8000/v1/ \
OPENAI_MODEL_ID=your-served-model-name \
JUDGE_API_BASE=your-judge-api-base \
JUDGE_MODEL=your-judge-model \
BENCHMARK='vstar,zoombench,hrbench-4k,hrbench-8k,mme-realworld,mme-realworld-cn' \
bash eval/run_eval.sh
```

支持的 benchmark 包括：`vstar`、`zoombench`、`hrbench-4k`、`hrbench-8k`、`mme-realworld`、
`mme-realworld-cn`、`mme-realworld-lite`、`visualprobe`、`mmvp`、`cv-bench`、`mmstar` 和 `pope`。

## 测试

本地单元测试不要求安装真实 vLLM；vLLM 注册和权重同步测试使用受控的 fake upstream：

```bash
python -m pytest -q tests
```

训练前建议至少检查：

```bash
bash -n scripts/run_vision_opd.sh
python scripts/prepare_vero35k.py --help
python scripts/run_angelslim_selector_eval.py --help
```

完整 GPU 验证还应覆盖一次真实 Qwen2.5-VL-3B rollout、teacher forward、actor backward、optimizer step 和
checkpoint resume。

## 实现结构

| 路径 | 作用 |
| --- | --- |
| [`vision_opd_selector/core.py`](vision_opd_selector/core.py) | FP32 LIS、可微 Top-K、逐图片 mask 和 mask BCE |
| [`vision_opd_selector/fsdp_hook.py`](vision_opd_selector/fsdp_hook.py) | 在 actor `from_pretrained` 后、FSDP wrapping 前注入 LIS |
| [`vision_opd_selector/vllm_plugin.py`](vision_opd_selector/vllm_plugin.py) | 注册自定义 vLLM architecture |
| [`vision_opd_selector/vllm_model.py`](vision_opd_selector/vllm_model.py) | native Qwen2.5-VL rollout 模型与多模态 processor |
| [`vision_opd_selector/budget.py`](vision_opd_selector/budget.py) | 固定、group 动态和逐样本预算展开 |
| [`vision_opd_selector/checkpoint.py`](vision_opd_selector/checkpoint.py) | LIS 权重加载、校验和 AngelSlim-compatible 导出 |
| [`verl/trainer/config/vopd.yaml`](verl/trainer/config/vopd.yaml) | 当前训练预设 |
| [`scripts/prepare_vero35k.py`](scripts/prepare_vero35k.py) | 元数据下载、图片 materialize 和 JSONL 构建 |
| [`scripts/run_vision_opd.sh`](scripts/run_vision_opd.sh) | 主训练入口 |
| [`scripts/run_angelslim_selector_eval.py`](scripts/run_angelslim_selector_eval.py) | AngelSlim VisionSelector/IDPruner 评估入口 |
| [`scripts/merge_checkpoint.sh`](scripts/merge_checkpoint.sh) | FSDP actor checkpoint 合并 |

## Citation

```bibtex
@article{yuan2026vision,
  title={Vision-OPD: Learning to See Fine Details for Multimodal LLMs via On-Policy Self-Distillation},
  author={Yuan, Qianhao and Lou, Jie and Yu, Xing and Lin, Hongyu and Sun, Le and Han, Xianpei and Lu, Yaojie},
  journal={arXiv preprint arXiv:2605.18740},
  year={2026}
}
```

## License

本仓库主体使用 Apache-2.0 License。`third_party/AngelSlim` 遵循其自身的 license 与 notice，详见
[`THIRD_PARTY.md`](THIRD_PARTY.md)。
