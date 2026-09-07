# DeepSeek V4 Flash 轨迹整理

当前产物是数据准备工具，不是 SFT trainer，也不改变 G0 零 optimizer step 的范围。输入为已有 ClawEval 的 task result JSON 与其引用的原始 Pi session JSONL；不读取 auth、models 或 settings 文件，不把评分反馈、judge 调用记录或参考答案写入训练对话。数据只保存在本地忽略目录 `artifacts/`，不要推送到代码仓库。

## 0.8B 的第一轮怎么训练

默认采用 **完整历史输入 + 下一次 assistant 输出** 的 prompt/completion 形式。示意：

```text
样本 1：任务                         → assistant 调用工具 A
样本 2：任务 + 调用 A + A 的结果       → assistant 调用工具 B
样本 3：上述历史 + 调用 B + B 的结果   → assistant 最终回答
```

每个 completion 是完整的一次 assistant 消息，可能包含多个并行工具调用。训练只对 completion 计算损失；历史 assistant、用户、工具返回、system 都不计算损失。不是把每次工具调用拆成没有上下文的孤立问答，也不是让模型预测工具返回。样本中不加入未来步骤或最终答案作为提示。

先去除 Pi `thinking` 内容块和 provider 签名，保留可见文本、工具调用与结果，适配第一轮 `enable_thinking=False`。这不等于删除多步任务过程。后续如果训练 thinking，需要另行定义 Qwen thinking 模板和目标格式，不能直接拼上 teacher reasoning。

逐步样本会重复读取前缀，计算量大于完整 episode 加 assistant mask，并不天然更省总计算。选择它是为了控制 8K 上限、明确目标损失和诊断每一步；不是为了把 349 条任务变成几千个“独立任务”。后续可以做整条 episode + assistant-only loss 的对照实验，不应把两种形式同时全量混入导致重复加权。

## 筛选规则

1. 只选 grading_results 非空且全部 `passed=true`、teacher 为 `deepseek-v4-flash*` 的轨迹。评分通过是弱质量信号，不能保证每一步正确。
2. 要求唯一原始 session 文件；缺失、分支、compaction、自定义事件、非文本内容、错误/中断/截断 assistant、未闭合工具链都隔离。
3. 要求原始工具调用 ID、工具名与返回一一配对。保留成功轨迹中的工具错误及其恢复过程，但单独统计。
4. 从同 trial 的 `mcp-cache.json` 恢复工具 JSON schema，验证所有被调用工具及参数；缺少 schema 的 Bash 等调用不猜测、不补造。
5. 不使用残留的 `APPEND_SYSTEM.md`：实查发现其工具列表与任务不一致，且它只是追加提示，不是完整模型请求。使用脚本中版本化的通用 system，明确标记 `adapted_generic_v1_not_original_pi_prompt`。这是适配后的 SFT 数据，不是精确重放 teacher 请求。缓存 schema 也不等同于历史请求快照。
6. 去除重复 trial 与同任务族下忽略临时 tool ID 后完全相同的对话。按任务名去除编号/语言前缀后的 suffix 分组，固定 hash 分配约 90% train / 10% validation，再拆步骤。不能随机按步骤切分。
7. 用固定 revision 的目标 tokenizer 和工具模板计算每个完整前缀 + completion 长度。超过 8192 token 的步骤整体跳过，不进行左截断，不截半个工具结果。前面短步骤仍可保留。

任务 suffix 分组能覆盖已知中英文同题，但不是完整语义去重；正式对外评测前仍需审核任务族划分。没有在 teacher 训练集出现过的任务才可用于独立泛化结论。

## 复现

在仓库根目录运行，无需 GPU；只下载 tokenizer 文件：

```bash
uv run --with 'transformers==5.3.0' --with 'jsonschema>=4,<5' --with 'jinja2>=3.1,<4' \
  python scripts/prepare_teacher_sft.py \
  --source /home/zhanghq17/trace \
  --output artifacts/teacher-sft-v4 \
  --tokenizer Qwen/Qwen3.5-0.8B --max-length 8192
```

`--revision` 可指定 tokenizer commit；默认 main 会在下载前解析成精确 commit 并记录。输出目录不能已存在，防止覆盖历史结果。省略 `--tokenizer` 时只做结构审核，文件后缀为 `.unlengthchecked.jsonl`，不能当成完成长度验收的训练集。

输出：

- `episodes.jsonl`：筛选后的完整标准化对话、工具 schema、来源 hash，供复查与整链训练对照。
- `train.jsonl` / `validation.jsonl`：`prompt`、单条 assistant `completion`、`tools`、token 数及来源 ID。
- `rejected.jsonl`：按 trial 记录排除原因；超长步骤单独记录在 `overlength.jsonl` 中。
- `report.json`：筛选统计、tokenizer commit、模板 hash、策略说明与文件 SHA256。

训练建议从 `completion_only_loss=True`、`packing=False` 开始。训练代码必须实测目标 Qwen 模板的 prompt/completion 边界与 label mask：工具返回全部为 -100、只有目标 assistant 有有效 label，不能仅依赖配置名推断正确。`enable_thinking=False` 也必须传入训练模板，和长度检查保持一致。长任务步骤更多，第一轮应按任务/episode 均衡采样，避免少数长轨迹主导梯度；当前文件不自动实现这种 sampler。

实现依据：[TRL 官方 SFT 文档](https://github.com/huggingface/trl/blob/main/docs/source/sft_trainer.md)、[TRL 模板与 mask 要求](https://github.com/huggingface/trl/blob/main/docs/source/chat_templates.md)。本提交不承诺任何特定 TRL 版本已完成 Qwen3.5 训练适配。

## 本次本地导出结果（2026-09-07）

已扫描 1,794 条 trial，其中 1,272 条评分通过；918 条缺少唯一原始 session。其余 354 条中，118 条缺工具 schema、5 条未完成、1 条含不支持的内容块、2 条参数不符合 schema，最终保留 **228 条 episode**。

它们产生 1,204 个 assistant 目标，按 Qwen tokenizer 实测排除 79 个超过 8K 的步骤，得到 **1,034 条训练样本、91 条验证样本**。训练集覆盖 96 个 task ID / 62 个任务族，验证集覆盖 10 个 task ID / 5 个任务族。全部步骤长度中位数 2,272、P90 6,677、最大 17,057 token（最大值包含被排除的步骤）。验证集任务族较少，结果只能作为初步验证。

目标 tokenizer revision：`2fc06364715b967f1860aea9cf38778875588b17`。本地输出目录：`artifacts/teacher-sft-v4`；以其 `report.json` 为准，不使用开发中间目录。已校验导出文件 hash、长度上限、assistant 目标及任务族不跨集合。新增 8 项数据准备测试通过；连同 G0 的真实 Pi CPU 集成测试，共 23 项通过。GPU SFT 训练尚未执行。
