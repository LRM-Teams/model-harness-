# G0 验证记录

日期：2026-09-07。开发机器：Linux、Python 3.12.3、Node.js 22、Pi 0.84.3。以下是 CPU 验证，不构成 GPU 验收结论。

```bash
RUN_PI_E2E=1 uv run pytest -q
# 15 passed in 22.14s
uv run ruff check src tests
uv run ruff format --check src tests
```

测试覆盖 session key 隔离、并发与重启清理、TTL/生成超时、幂等重试、断开后重试、消费确认、reward 冲突、SSE 分片及 Unicode 工具 JSON、采样限制、token trace 检查、真实 stdio MCP 和最终 gate 拒绝 mock/HF 替代品。

集成测试使用真实 Pi RPC 与显式 mock 模型，运行 20 个顺序会话和 4 个并发会话，覆盖全部六类任务；协议与任务检查均通过，结束后活跃 session 为 0。测试确认用户 Pi 配置不变、临时 agent 目录被清理。GitHub Actions 配置会执行相同 CPU 检查；此记录不代表远端 CI 已运行。

尚未验证：GPU 环境完整安装、真实 Qwen3.5-0.8B 工具成功率、SGLang kernel 运行、AReaL FSDP 实际前向、serving/FSDP logprob 误差与峰值显存。这些必须按 README 在 GPU 机器上执行，以 `gate_report.json` 的 `g0_passed: true` 为本阶段验收条件。若失败应保留报告定位原因，不降低门槛冒充通过。

G0 不执行 optimizer step；teacher SFT、GRPO、真实 benchmark 奖励及 harness 搜索未在本提交实现。
