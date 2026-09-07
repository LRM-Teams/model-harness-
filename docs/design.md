# G0 实现边界与上游契约

本仓库对应 0.8B 第一阶段 G0：先验证 transport、工具、身份、精确 token 轨迹和前向 logprob。它不需要 AgentEval 的内部数据或 grader；全部任务为公开合成协议用例。

## 选定路径

- AReaL commit：`b5f0820c307e9a02056131a54c6f7f92fa03ec55`。
- Pi：`@earendil-works/pi-coding-agent@0.84.3`。
- SGLang/Transformers/Torch：使用该 AReaL commit 的 lockfile/extra；安装结果另存 freeze。
- 正式模型：HF `Qwen/Qwen3.5-0.8B`，下载前将 revision 解析成 immutable SHA。

AReaL 官方 Online 服务会在 buffer 就绪后训练，因此这里不启动 PPOTrainer。采集服务为每个 trial 创建一个独立 `ArealOpenAI` cache，共享只推理的 `RemoteSGLangEngine`；导出直接读取 `InteractionWithTokenLogpReward.model_response`，不把 response 文本重新 tokenize。

FSDP 验证使用 `FSDPEngine.from_pretrained(..., learning_rate=None)`，`FinetuneSpec(total_train_epochs=0, ...)`；检查 optimizer=None，仅调用 forward。生成 token 在完整序列中的预测 logprob 索引是 `[input_length-1 : input_length+output_length-1]`。

## 数据与状态

- SQLite 保存 trial/session 唯一映射、key hash、状态、最终 reward；明文 key 不落库。
- 每次完整生成后原子写 trace，再返回 HTTP/SSE。未收到响应时可使用相同 request ID 重取。
- receipt 来自 Pi `after_provider_response` 的 completion header，加上 `message_end`；runner 比较完整 assistant 文本、工具 ID/名称/参数后确认协议。
- 每条 trace 保存 input IDs、output IDs、generation logprobs、policy version、实际请求、response、consumed 标志；loss mask 只覆盖当次新生成 token。
- `ACTIVE → ENDED`；故障/失联/重启为 `QUARANTINED`。没有任何状态可进入训练。
- 已结束会话的 key 仅保留结束请求重放能力，chat 被拒绝；管理端 export 可重复。敏感 key hash 也不出现在管理列表或导出。

SQLite 与 trace 原子文件写入不构成跨文件事务。崩溃后 active 会话一律 quarantine，避免把不确定的请求当作完整数据；本阶段宁可隔离也不猜测恢复。下一阶段如果需要训练级 exactly-once admission，需独立 batch ledger/提交协议。

## 和总体方案的关系

G0 通过后再做：真实 ClawEval 任务审计和语义 split、GT/grader 隔离、官方 reward bridge、episode-level GRPO 分组、LoRA 更新与 reload。当前模块清单只是模型结构 inventory，不能当作训练端/serving 端共同支持的 LoRA target manifest。

每个 worker 内一个 session 的调用串行，session 之间并发；Pi 工作目录独立，并限制为合成工具。MCP 用官方 Python SDK 执行 initialize→list_tools→call_tool，经 Pi extension 转交，无需修改用户 MCP 配置。

## 一手接口参考

- [AReaL Online Proxy](https://github.com/areal-project/AReaL/blob/b5f0820c307e9a02056131a54c6f7f92fa03ec55/docs/en/tutorial/online_proxy.md)
- [ArealOpenAI](https://github.com/areal-project/AReaL/blob/b5f0820c307e9a02056131a54c6f7f92fa03ec55/areal/experimental/openai/client.py)
- [RemoteSGLangEngine](https://github.com/areal-project/AReaL/blob/b5f0820c307e9a02056131a54c6f7f92fa03ec55/areal/engine/sglang_remote.py)
- [FSDPEngine：learning_rate=None 不创建 optimizer](https://github.com/areal-project/AReaL/blob/b5f0820c307e9a02056131a54c6f7f92fa03ec55/areal/engine/fsdp_engine.py)
- [Qwen3.5-0.8B 模型卡](https://huggingface.co/Qwen/Qwen3.5-0.8B)

Pinned API 使用在源码层面核对过；GPU 数值正确性由用户机器上的报告决定。SSE 为完成后分片的协议层测试，不声称已经验证引擎增量解码时延。
