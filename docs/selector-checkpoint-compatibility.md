# Selector checkpoint 与 AngelSlim 兼容性调查

## 核对对象

官方仓库：<https://huggingface.co/AngelSlim/Qwen2.5-VL-3B-Instruct-Selector/tree/main>

本次固定 revision：`616808340d107a966a18114b20c683135f434345`。实际下载了配置、Python scorer 和权重进行验证，不仅对照文件名。

| 文件 / 参数 | 官方格式与项目导出 |
| --- | --- |
| `config.json` | `in_features=2048`、`hidden_dim=1024`、`init_scale=0.0001`、`architecture=TransformerScorer` |
| `selector_scorer.py` | 可动态导入的 `TransformerScorer`，forward 参数 `x`，q/k 点积后沿最后一维取均值 |
| `model.safetensors` | 不带模型前缀的四参数映射 |
| `k_proj.weight`, `q_proj.weight` | `[1024, 2048]` |
| `k_proj.bias`, `q_proj.bias` | `[1024]` |

官方权重为 BF16。本项目保存原生参数 dtype，不为导出强制转 BF16；默认 actor 主参数导出可以是 FP32。safetensors 和 AngelSlim loader 均支持，不能为匹配官方文件 dtype 而改变 actor 参数管理。

## 保存与加载链路

完整保存：`fsdp_workers.save_checkpoint` → 原生 FSDP sharded model/optimizer/extra state → 从带 shard metadata 的 model state dict 集体物化四个 selector tensor → rank 0 导出 `global_step_N/actor/selector/`。

Selector-only：`save_selector_checkpoint` → 原生 FSDP sharded state dict → 同样物化四参数 → rank 0 写 `global_step_N/selector/`。所有 rank 参与，不能直接保存局部参数视图；瞬时内存由正常 FSDP wrapping 决定。

训练初始化：`SELECTOR_PATH` → `load_selector_state` → 规范化四个参数名 → scorer `load_state_dict(strict=True)`。这是参数初始化，不恢复 optimizer、scheduler 或训练进度。

完整恢复使用原生 checkpoint manager。旧独立 wrapper 的 optimizer shard 可能不兼容，已有分片形状检查应保留。完整 checkpoint 不能直接当作 HF selector 目录，评测应使用其中的 selector artifact。

评测：`scripts/run_angelslim_selector_eval.py` → 校验 artifact → 生成持久化 `selector_eval.yaml` → 调用 `third_party/AngelSlim/tools/run_token_pruning_evaluation.py` → pruning adapter → `vision_selector_pruning` → 动态导入 artifact scorer 并加载权重。

## 本次加固

- 导入/导出校验四参数形状与浮点类型；导出张量连续化。
- 评测入口校验三个文件、配置维度及参数完整性，可自动解析 step、actor 或 selector 目录。
- 下载 allowlist 包含 scorer 源文件，支持官方 HF artifact 的预检。
- 实际 AngelSlim loader 改为严格加载，避免缺失权重时继续使用随机参数。
- 普通 VisionSelector 评测显式 `score_mode: raw`，直接按 scorer 排序；避免 soft sigmoid 饱和/舍入产生新并列。IDPruner 保留原策略，不能当成纯 LIS 对照。
- 限定 batch size 1，提前解析本地基础模型路径，并在子进程 PYTHONPATH 优先指定 vendored AngelSlim。
- `--validate-only` 无需加载完整模型或 GPU，只校验 artifact 并生成 YAML，不代表完整评测通过。

## 推荐评测命令

默认冻结 LLM 的训练，可搭配原始 Qwen2.5-VL-3B 基础模型：

```bash
python scripts/run_angelslim_selector_eval.py \
    --model-path Qwen/Qwen2.5-VL-3B-Instruct \
    --selector-path /absolute/path/to/global_step_50 \
    --retention-budget 0.2 \
    --tasks mmmu_val docvqa_val chartqa \
    --batch-size 1 \
    --output-dir ./eval_results/step50-budget02
```

首先可附加 `--validate-only` 检查。生成的 `selector_eval.yaml` 可直接交给用户指定的入口：

```bash
PYTHONPATH="$PWD/third_party/AngelSlim:${PYTHONPATH:-}" \
python third_party/AngelSlim/tools/run_token_pruning_evaluation.py \
    --model_path Qwen/Qwen2.5-VL-3B-Instruct \
    --configs ./eval_results/step50-budget02/selector_eval.yaml \
    --tasks mmmu_val docvqa_val chartqa \
    --batch_size 1 \
    --output_dir ./eval_results/step50-budget02
```

`LLM_TRAIN_MODE=full` 时必须使用对应训练后的完整基础模型权重作为 `--model-path`，不能搭配原始模型声称评测了完整训练结果。合并工具应在 checkpoint 副本上操作，参见 guide.md。

## 行为边界与验证结果

- 已验证官方 HF → 本项目严格加载 → 导出 → 实际 AngelSlim 动态 loader；四参数和 CPU raw scores 精确一致。
- 回归覆盖经过两步 optimizer 更新的 scorer 导出、实际 pruning strategy 选中位置一致、文本 token 保留，以及缺少权重时报错。这是小模型兼容测试，不是 Qwen 任务质量测试。
- 原生 CPU 双进程 FSDP 保存/恢复测试继续单独覆盖分片机制。
- 训练按每张图片独立打分，AngelSlim 当前策略对一个样本全部视觉 token 联合打分。多图片/视频不具备与训练相同的选择语义；单图片是直接对齐的验证范围。
- AngelSlim 保留数为 `round(N * budget)`，训练为 `floor(N * budget)`；可能相差一个 token，需记录实际数量。
- AngelSlim 动态 scorer 默认 FP32 构造并将输入转至 scorer dtype；它不是 actor/vLLM BF16 前向的一致性验证。未为 actor/vLLM 引入任何特殊转换。
- 当前训练是 soft gating，评测是 hard deletion；质量差异应通过真实评测测量。
- 当前机器无 CUDA，未执行完整 Qwen 模型加载、lmms-eval 数据集评测或报告任务分数。实际训练 artifact 和 GPU 运行仍是最后的端到端验证。
