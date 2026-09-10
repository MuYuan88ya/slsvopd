# SLSOPD 操作指南

本指南中的命令均从 `Vision-OPD/` 仓库根目录执行。项目目标、训练约束和代码流程见 [AGENTS.md](AGENTS.md)。

## 1. 准备环境

完整训练需要 Linux、支持 BF16 的 NVIDIA GPU 和兼容 CUDA 环境。以 `requirements.txt` 为依赖依据：Python 3.12、PyTorch 2.10.0、Transformers 5.5.0、vLLM 0.18.0、Ray 2.53.0。macOS 可做数据元信息处理和本地测试，不能执行这里的 vLLM GPU 训练。

```bash
conda create -n vision-opd python=3.12 -y
conda activate vision-opd
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-deps -r requirements.txt
python -m pip install -e . --no-deps
python -m pip install flash-attn --no-build-isolation
```

Editable install 会安装 `vllm.general_plugins` entry point，仅设置 `PYTHONPATH` 不够。切换代码目录或重建环境后重新安装，并检查：

```bash
python -c "from importlib.metadata import entry_points; print([(e.name, e.value) for e in entry_points(group='vllm.general_plugins') if e.name == 'vision_opd_selector'])"
```

输出应包含 `vision_opd_selector.vllm_plugin:register`。GPU/attention backend 的可用性需在训练机器上确认。

## 2. 准备当前实验数据

数据源为 `muyuho/Vero-35k-VisionSelector`。最终输入是 VeRL JSONL，图片保存在文件系统，不重新编码进 JSONL。

只下载小型元数据：

```bash
python scripts/prepare_vero35k.py metadata --data-dir ./data/vero35k
```

输出含 `vero_train_all_35k.verl.jsonl`、数据集 README 和 `summary.json`，此步骤不下载图片分卷。

在训练机器下载图片并构建可训练数据：

```bash
python scripts/prepare_vero35k.py materialize --data-dir ./data/vero35k
```

该命令下载图片分卷、流式解压、检查图片路径，生成 `data/vero35k/train.verl.jsonl`，默认删除已下载的压缩分卷。`--keep-archives` 保留分卷；`--limit 100` 只限制输出样本数，不减少图片下载量。

已有图片时，先准备元数据，再仅构建 JSONL：

```bash
python scripts/prepare_vero35k.py build \
    --source ./data/vero35k/vero_train_all_35k.verl.jsonl \
    --dataset-root /mnt/datasets/Vero-35k-VisionSelector \
    --output ./data/vero35k/train.verl.jsonl
```

`--dataset-root` 应能与源 JSONL 中的相对图片路径拼成真实文件路径。构建后 `images` 为 `[{"path": "/absolute/path/to/image.jpg"}]`，其他字段保留，包括 `prompt`、`reward_model`、`extra_info`。换机器或挂载点后重新 build；多节点需所有节点都能访问相同路径。

当前实验固定使用同输入教师，不需要原版 `bbox_images`。为了保持所有样本预算 0.2，不要在数据中加入其他值的 `vision_selector_budget`：该字段能覆盖默认预算。

## 3. 启动 LIS 训练

```bash
bash scripts/slsopd.sh
```

使用外部模型和数据路径：

```bash
MODEL_PATH=/mnt/models/Qwen2.5-VL-3B-Instruct \
TASK_TRAIN_FILE=/mnt/datasets/vero35k/train.verl.jsonl \
bash scripts/slsopd.sh
```

### 默认训练参数

以下是脚本实际传入的值；单独执行 `--config-name slsopd` 的 YAML 默认值不一定与脚本相同。

| 参数 | 脚本默认值 / 含义 |
| --- | --- |
| 配置 | `slsopd.yaml` |
| 模型 | `Qwen/Qwen2.5-VL-3B-Instruct` |
| group size / `actor_rollout_ref.rollout.n` | 8，每个 prompt 生成 8 个回答 |
| 视觉预算 | 0.2，每张图片保留率 20%；训练仍是 soft gating |
| 辅助 BCE | 关闭，仅 VOPD 任务 loss 训练 LIS |
| 学生可训练参数 | 默认只有 LIS；vision/merger 和 LLM 冻结 |
| 教师 | fixed、same_input、完整视觉特征、更新率 0 |
| `data.train_batch_size` | 96 |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | 96，VeRL 按 rollout 配置处理实际样本数 |
| 学习率 / warmup | `2e-6` / 10 steps |
| prompt / response 上限 | 8192 / 1024 |
| rollout 最大模型长度 | 9216，默认两者之和 |
| 动态 batch | 开启，actor token 上限每 GPU 9216 |
| GPU / 节点 | 每节点 8 GPU；节点数读取 `WORLD_SIZE`，默认 1 |
| rollout tensor parallel | 1 |
| vLLM 显存比例 | 0.7 |
| actor 参数与 optimizer offload | 开启 |
| ref 参数 offload | 开启 |
| FSDP `use_orig_params` | true，由实验 YAML 配置 |
| 训练周期 | 1 epoch |
| 保存频率 | `-1`，无周期保存，当前 trainer 最后一步保存 |
| 验证 | `val_before_train=False`、`test_freq=-1`、空验证集 |
| 日志 | console、tensorboard |

这里 `WORLD_SIZE` 是脚本约定的节点数，不是 GPU 进程总数；设置它本身不会完成多节点 Ray 集群部署。

### 可直接设置的环境变量

| 变量 | 默认值 / 用途 |
| --- | --- |
| `MODEL_PATH` | 默认 Qwen2.5-VL-3B，学生模型 |
| `TEACHER_MODEL_PATH` | 跟随 `MODEL_PATH`，固定教师模型 |
| `DATA_DIR` | 仓库下 `data/vero35k` |
| `TASK_TRAIN_FILE` | `$DATA_DIR/train.verl.jsonl` |
| `LLM_TRAIN_MODE` | `frozen`；可选 `full`，vision/merger 仍冻结 |
| `VISION_SELECTOR_BUDGET` | `0.2` |
| `VISION_SELECTOR_ROLLOUT_BUDGETS` | 空，同 group 使用默认预算；本实验保持空 |
| `VISION_SELECTOR_EVAL_BUDGET` | 空，跟随默认预算 |
| `VISION_SELECTOR_HIDDEN_DIM` | `1024` |
| `VISION_SELECTOR_INIT_SCALE` | `0.0001` |
| `SELECTOR_MASK_LOSS_ENABLED` | `false` |
| `SELECTOR_MASK_LOSS_WEIGHT_START` | `0.1`，仅 BCE 开启时生效 |
| `SELECTOR_MASK_LOSS_WEIGHT_END` | `2.0`，仅 BCE 开启时生效 |
| `SELECTOR_PATH` | 空；已有 selector 的目录、文件或 Hub repo |
| `VISION_SELECTOR_CHECKPOINT_ONLY` | `false`，默认保存完整可恢复 checkpoint |
| `CUSTOM_CHAT_TEMPLATE_FILE` | 空，默认使用模型模板 |
| `MAX_MODEL_LEN` | `9216`，调整 rollout 长度上限 |
| `WORLD_SIZE` | `1`，节点数 |

`ROLLOUT_N`、`LR`、GPU 数等脚本内固定赋值不接受同名环境变量覆盖，请通过末尾 Hydra overrides 调整：

```bash
bash scripts/slsopd.sh \
    trainer.n_gpus_per_node=4 \
    data.train_batch_size=32 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    trainer.save_freq=10
```

修改序列长度时需同时核对 data、rollout 和 actor token 上限，不能只调整 `MAX_MODEL_LEN`。

仅在需要辅助正则时开启 BCE：

```bash
SELECTOR_MASK_LOSS_ENABLED=true bash scripts/slsopd.sh
```

开启后 BCE 权重按 actor update 进度从 0.1 线性变化到 2.0；任务路径仍用 soft mask，不改为 hard target 训练。

## 4. 保存、续训与 selector 初始化

默认输出位于 `checkpoints/SLSOPD-<模型目录名>/` 和 `rollouts/SLSOPD-<模型目录名>/`。默认 `resume_mode=auto`，同输出目录再次启动会寻找最新完整 checkpoint。

显式恢复完整训练：

```bash
bash scripts/slsopd.sh \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path=/absolute/path/to/global_step_50
```

完整 checkpoint 包含 actor 模型/optimizer 分片、scheduler/RNG 与训练数据状态，并额外导出 `global_step_N/actor/selector/`。

从 selector 初始化新实验：

```bash
SELECTOR_PATH=/absolute/path/to/global_step_50/actor/selector \
bash scripts/slsopd.sh \
    trainer.resume_mode=disable \
    trainer.default_local_dir=/mnt/checkpoints/slsopd-new-run \
    trainer.rollout_data_dir=/mnt/rollouts/slsopd-new-run
```

这不恢复 optimizer 或训练进度。旧独立 LIS wrapper 的完整 optimizer 分片可能不兼容，恢复校验发现不匹配会报错；不要绕过校验。

只输出 selector 文件：

```bash
VISION_SELECTOR_CHECKPOINT_ONLY=true \
bash scripts/slsopd.sh trainer.save_freq=10
```

输出 `global_step_N/selector/` 下的 `model.safetensors`、`config.json`、`selector_scorer.py`。此模式不可完整续训，也不保证只消耗 LIS 大小的瞬时内存。

需要将完整 FSDP checkpoint 合并为 Hugging Face actor：

```bash
bash tools/merge_checkpoint.sh /absolute/path/to/global_step_50
```

传 step 目录，不传 `actor/` 或 selector-only 目录。该工具会删除 step 目录中原有顶层普通文件再合并（包括可能的训练状态文件）；应在 checkpoint 副本上操作，保留原始可恢复目录。selector artifact 可用于 AngelSlim 评估，具体命令见 [README 的评估流程](README.md#使用-angelslim-评估)。

## 5. 原版 VOPD 入口

`scripts/run_vision_opd.sh` 和 `verl/trainer/config/vopd.yaml` 保留 `../vopd/` 原版内容，与 LIS 入口分离。原版使用 Qwen3.5-4B、legacy/EMA 教师、`bbox_images` 及原版 chat template。

```bash
python scripts/prepare_data.py --data-dir ./data
WORLD_SIZE=1 bash scripts/run_vision_opd.sh
```

原版数据脚本从 `yuanqianhao/Vision-OPD-6K` 下载学生/教师图片，生成 `data/train.parquet`；已有下载和解压数据可用 `--skip-download` 仅转换。下载步骤调用 `huggingface-cli`，需环境提供该命令。原版启动脚本要求设置 `WORLD_SIZE`，不沿用 SLSOPD 的模型与数据环境变量接口。

## 6. 验证

```bash
bash -n scripts/slsopd.sh
bash -n scripts/run_vision_opd.sh
python -m pytest tests -q
```

训练改动需在支持 BF16 的 GPU 上验证真实 VOPD 单独更新 LIS、开启 BCE 后的梯度、actor/vLLM 同步及两步训练后恢复。CPU/Gloo 和 mock vLLM 测试只能验证对应局部行为；报告中单独列出未执行的 CUDA 验证和依赖版本差异。

## 7. 训练后 AngelSlim 评测

`VISION_SELECTOR_CHECKPOINT_ONLY=true` 保存时已完成分片物化和格式导出，**不需要再转换，也不要运行 checkpoint 合并工具**。输出结构为：

```text
global_step_50/
└── selector/
    ├── model.safetensors
    ├── config.json
    └── selector_scorer.py
```

默认冻结 LLM 的训练可直接执行以下完整评测命令：

```bash
python scripts/run_angelslim_selector_eval.py \
    --selector-path /absolute/path/to/global_step_50 \
    --retention-budget 0.2 \
    --tasks mmmu_val docvqa_val chartqa \
    --output-dir ./eval_results/step50
```

入口会自动定位 `global_step_50/selector/`；直接传该 selector 子目录也可以。完整 checkpoint 则会定位到 `global_step_50/actor/selector/`。默认基础模型为 `Qwen/Qwen2.5-VL-3B-Instruct`，本地模型可用 `--model-path /absolute/path/to/model` 指定。真正的数据集评测需要 GPU、模型和 lmms-eval 运行依赖。


先无 GPU 检查三文件 artifact 并生成评测 YAML：

```bash
python scripts/run_angelslim_selector_eval.py \
    --selector-path /absolute/path/to/global_step_50 \
    --output-dir ./eval_results/step50-budget02 \
    --validate-only
```

移除 `--validate-only` 即调用 vendored `run_token_pruning_evaluation.py` 执行评测。默认保留率 0.2、batch size 1、任务 `mmmu_val docvqa_val chartqa`。`--selector-path` 可传 step、actor、selector 目录或官方 HF selector repo。普通 VisionSelector 使用 raw scores 排序，与原版 VisionSelector 的 hard Top-K 选择方式一致；YAML 留在输出目录，可用于直接调用原评测脚本。

若训练了 LLM 全参数，另用 `--model-path` 指向训练后的完整模型。冻结 LLM 时可使用原始基础模型。文件格式、验证范围及训练/评测差异见 [兼容性调查](docs/selector-checkpoint-compatibility.md)。

### 每隔 N 步只保存 LIS，供评测使用

例如每 10 个 trainer step 导出一次 LIS：

```bash
SAVE_FREQ=10 VISION_SELECTOR_CHECKPOINT_ONLY=true bash scripts/slsopd.sh
```

等价于 `VISION_SELECTOR_CHECKPOINT_ONLY=true bash scripts/slsopd.sh trainer.save_freq=10`。
`SAVE_FREQ` 默认 `-1`，不开启周期保存；最后一个训练 step 始终保存，即使不是 N 的倍数。
step 指 trainer 的全局训练迭代，不是 micro-batch 或单个回答。

输出为 `global_step_10/selector/`、`global_step_20/selector/` 等目录，各含三文件 artifact。
只保存 LIS，不保存完整模型、optimizer 或数据进度，不更新完整续训 checkpoint 指针。
该模式不应用 `max_actor_ckpt_to_keep` 的完整 checkpoint 轮转，会保留这些 LIS 导出。
已在同一目录存在的旧完整 checkpoint 不会被此开关删除；新实验应使用独立输出目录。

评测任一步：

```bash
python scripts/run_angelslim_selector_eval.py \
    --selector-path /absolute/path/to/global_step_20/selector \
    --retention-budget 0.2 \
    --output-dir ./eval_results/step20
```
