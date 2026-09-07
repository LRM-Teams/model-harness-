# 0.8B SFT：先跑通，再判断有没有收益

数据已随仓库提供在 `data/teacher-sft-v1/`。先通过 G0 的模型推理/工具采集检查，再运行本页。SFT 使用独立 `.venv-sft` 环境，单卡执行，不启动 GRPO。训练期间停止 SGLang 和 G0 服务，让训练独占 GPU。当前开发机器无 CUDA GPU，因此这里的 GPU 训练代码尚未实机验收，不能把已通过的 CPU 数据检查写成训练成功。

## 一、20 步冒烟测试

Linux、Python 3.12、A100 40GB/80GB 完整卡，建议 16 vCPU、64GB RAM、100GB 可用磁盘。优先 NVIDIA 575.57.08 或更新驱动，匹配 CUDA 12.9 wheel。完整 8K 全参训练显存尚未实测；冒烟默认 2K、batch size 1、梯度累积 4。

```bash
git clone https://github.com/LRM-Teams/model-harness-.git
cd model-harness-
bash scripts/setup_sft.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_sft_smoke.sh artifacts/sft-smoke-001
```

脚本锁定 PyTorch 2.9.1 / CUDA 12.9 和 Transformers 5.3.0，下载数据中固定 revision 的 `Qwen/Qwen3.5-0.8B`。训练文本部分参数，冻结 vision/visual 参数；使用 FP32 主权重和 AdamW 状态、BF16 autocast、eager attention、gradient checkpointing、固定随机种子。不是 LoRA，也不是推理量化。

冒烟从 2K 内样本中按任务轮流选最多 32 条训练样本、8 条验证样本，训练 20 个 optimizer step。按任务均衡随机采样，目标是验证整条程序链路；这种小样本重复训练结果不能用于泛化结论。

每个样本将完整 prompt 渲染为推理前缀，再与 prompt+completion 的 token 前缀逐 ID 比较。不一致立即失败，拒绝猜测 label 边界。所有 prompt label 都设为 -100，只监督下一次 assistant 输出；超长样本整体跳过，无截断、无 packing。

完成后查看 `training_report.json`：

- `pipeline_passed: true`；20 个 optimizer step 完成，loss/梯度均有限，梯度非零。
- `weights_changed: true`，确认实际更新了权重。
- `reload_nll_abs_delta <= 0.0001`，确认保存后重新加载的 checkpoint 在相同样本上概率一致。
- `checkpoint/` 包含可加载模型及 tokenizer；日志在 `train_log.jsonl`。

如果 OOM，降低 `--max-length` 并使用新输出目录；不截断数据。输出目录不可复用，避免覆盖旧实验。失败时 `training_report.json` 保持未通过；先根据异常排查依赖、数据或显存，不把失败解释为模型能力差。

只做无 GPU 数据检查：

```bash
uv run --with 'transformers==5.3.0' --with 'jinja2>=3.1,<4' \
  python -m model_harness_g0.sft --check-data-only --mode train \
  --output artifacts/sft-data-check
```

## 二、初步学习实验

冒烟成功后，从原始 base 开始新的实验，先训练 200 步、学习率 1e-5。默认最多 8K；40GB 卡遇到显存压力可先用 4K 做实验，baseline 和 after 使用相同筛选后的验证样本。

```bash
CUDA_VISIBLE_DEVICES=0 .venv-sft/bin/python -m model_harness_g0.sft \
  --mode train --steps 200 --max-length 4096 --probes 16 \
  --output artifacts/sft-08b-200steps-4k
```

固定步数 + 按任务均衡采样，不称为严格的“一个 epoch”。保存 `config.json`、`selected_samples.json`、数据 hash、模型 revision、训练日志、before/after 报告。不同配置用不同目录，不能根据验证集反复挑参后再把它称为最终测试集。

## 三、什么算有效

| 层次 | 检查 | 能说明什么 |
|---|---|---|
| 工程跑通 | 非零梯度、参数改变、保存重载一致、20 步完成 | 训练代码链路可用 |
| 初步学到东西 | 相同验证样本上 token 平均 NLL、task 平均 NLL 比 base 低 | 对这些 teacher 行为的拟合有所改善；不是实际任务成功率 |
| 离线生成观察 | 比较 `probes_before.json` / `probes_after.json` 的同前缀 greedy 输出 | 是否更会输出预期工具名；当前只统计工具名序列，不检查参数正确性或实际执行 |
| 实际能力提升 | 固定 harness、工具预算、采样参数，在未参与训练的任务族上分别运行 base/SFT 并由原始 grader 打分 | 才能判断真实任务成功率是否提高 |

最后一层建议先用至少 30–50 个未参与训练的独立任务，每题 3 次采样，记录配对成功率、工具参数错误率、超时率和平均调用次数，同时给出任务层面的不确定性；这只是初步实验规模建议，非统计功效保证。0.8B 若没有成功率提升，仍可能已完成工程冒烟，但不能声称协同进化有效。

现有 validation 只有 5 个任务族，不能支撑广泛泛化结论。其工具返回来自 teacher 轨迹，模型没有自己执行工具，因此离线 probe 明确不等同于闭环任务评估。当前训练脚本始终写 `task_effectiveness_proven: false`，不使用 loss 下降自动宣告成功；闭环 AgentEval 评估接入留作后续工作。

参考：[Transformers 官方训练接口](https://github.com/huggingface/transformers/blob/main/docs/source/en/training.md)。本实现直接使用 PyTorch loop 来明确 loss mask、参数更新和重载检查，不依赖 TRL 自动 mask 的推断。

## 本次 CPU 验证记录

发布版数据全部 1,125 个样本已使用真实 Qwen3.5 tokenizer 完成 8K 长度检查和 prompt/completion token 前缀一致性检查，训练目标 token 数为 289,355 / 25,165（train / validation）。脱敏后的 9,594 次工具调用引用通过 JSON schema 参数检查（包含不同样本重复出现的历史调用）。已检查常见凭据模式及本机路径未残留。完整测试含真实 Pi CPU 集成测试共 29 项通过；GPU 前向、反向、checkpoint 重载尚待 GPU 机器运行。
