# Qwen3.5-0.8B × AReaL × Pi：G0 采集验证

这是协同进化方案的**第一阶段代码**：不训练，先验证小模型通过 Pi 调用工具、多轮交互、会话隔离、精确 token/logprob 记录，以及 AReaL FSDP 前向概率重算。

```
Pi RPC → 本项目只采集 session 服务 → AReaL ArealOpenAI → SGLang /generate
                         ↓
                  SQLite + JSON token trace
                         ↓
           AReaL FSDP forward（optimizer=None）→ parity report
```

**本机已完成 CPU 测试与真实 Pi + mock 模型测试；Qwen GPU 路径留待 GPU 机器验收。** mock 报告永远不能通过最终 GPU 门禁。这里没有 SFT/GRPO 更新、REEF、真实 benchmark 数据或 AgentEval 私有源码；这些属于后续阶段。

## GPU 机器运行

需要 Linux x86_64、Python 3.12、Node.js 22+、Git、uv、可用 CUDA 驱动。**只跑本次 0.8B G0，建议申请 1 张 A100、16 vCPU、64GB 内存和至少 100GB 可用磁盘**，按下文单卡流程先采集、再停止 serving 做 FSDP 前向。两张卡可同时保留 serving 和前向进程，省去停服务的步骤；不是本阶段的硬性要求。实际显存以 GPU 验收日志为准。

| 申请项 | 建议 |
|---|---|
| GPU | 优先 1 × A100 40GB 或 80GB 完整卡；A800 可作为备选，先确认单卡显存和驱动 |
| CPU / 主机内存 | 16 vCPU / 64GB RAM，属于运行余量建议，尚未实测最低配置 |
| 磁盘 | 至少 100GB 可用空间，容纳模型、CUDA wheels、环境和日志 |
| 环境 | Ubuntu 22.04/24.04 的 PyTorch/CUDA 镜像；脚本另建 `.venv-gpu`，无需沿用镜像预装 PyTorch |
| 驱动 | 优先 NVIDIA 575.57.08 或更新，适配上游 CUDA 12.9 wheel，避免依赖旧驱动兼容模式 |
| 可选双卡 | 2 张 A100、32 vCPU、128GB RAM；可直接运行下面的双卡脚本，后续训练需求需另行评估 |

资源池名称 `A100-8` 不等于必须申请 8 张卡，也不能据此判断单卡显存；以平台实际配额为准。如果只能整机分配，需要比较总价再选。当前路径固定 BF16，不建议用 P100/V100；3090 的适配和余量未验证。H20 Premium 是否划算取决于报价，本项目没有实测依据。

安装依据上游锁文件：SGLang extra 对应 PyTorch `2.9.1+cu129`、Transformers `5.3.0`、SGLang `0.5.10.post1`，另固定 `flash-linear-attention==0.4.2`。最终安装版本会保存在 `artifacts/gpu-pip-freeze.txt`。硬件参考：[NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/)、[CUDA 12.9 Update 1 驱动说明](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html)。

### 1. 安装并锁定模型

```bash
git clone https://github.com/LRM-Teams/model-harness-.git
cd model-harness-
bash scripts/setup_gpu.sh
.venv-gpu/bin/mh-g0 prepare
```

安装脚本使用 AReaL 精确 commit 和该仓库 `uv.lock` 的 SGLang extra，然后安装本项目与固定 GDN kernel 版本；Pi 固定 `0.84.3`，安装在 `vendor/pi`，不修改日常 Pi 配置。需要能够访问 GitHub、Hugging Face、PyPI 和上游 CUDA wheel 源。

`prepare` 将 HF revision 解析为 commit，下载 `Qwen/Qwen3.5-0.8B`，计算全部 snapshot 文件 SHA256，生成 `artifacts/model-lock.json`。已有 lock 不覆盖。若指定历史模型版本：

```bash
.venv-gpu/bin/mh-g0 prepare --revision <HF_COMMIT> --output artifacts/model-lock.json
```

### 2. 启动 SGLang（终端 A）

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/start_sglang.sh
```

固定 BF16、8K context、单卡、`qwen3_coder` parser。等服务 ready 后再继续。

### 3. 启动只采集服务（终端 B）

生成一个本次运行专用管理 key；终端 C 使用同一个值：

```bash
export G0_ADMIN_KEY="$(.venv-gpu/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))')"
.venv-gpu/bin/mh-g0 serve --backend areal
```

管理 key 至少 24 字符。不要提交 key 或 artifacts。服务只监听 `127.0.0.1:8090`，SGLang 只监听 `127.0.0.1:30000`，所有步骤在同一机器执行。可通过 SSH 操作，不需要暴露端口。

启动时检查锁定模型文件、AReaL commit、SGLang 实际 model path，并执行固定合法 Qwen 工具解析 fixtures。模型或 parser 不匹配直接失败。

### 4. 跑完整 G0（终端 C）

将终端 B 的管理 key 设置到本终端后执行：

```bash
export G0_ADMIN_KEY='<与终端 B 相同的值>'
CUDA_VISIBLE_DEVICES=1 bash scripts/run_g0.sh artifacts/run-001
```

依次运行：

1. 20 个顺序 + 4 个并发的真实 Pi session。
2. 从六类成功用例各取一个 episode，对其中每次模型调用做 FSDP logprob 重算。
3. 输出最终 `gate_report.json`；任何阶段失败，脚本返回非零状态。

只有一张 GPU 时，先运行 smoke，停止采集服务和 SGLang，释放显存后单独运行 parity、verify：

```bash
.venv-gpu/bin/mh-g0 smoke --output artifacts/run-001 --pi "$PWD/vendor/pi/node_modules/.bin/pi"
# 在另两个终端停止采集服务和 SGLang 后：
CUDA_VISIBLE_DEVICES=0 .venv-gpu/bin/mh-g0 parity --run artifacts/run-001
.venv-gpu/bin/mh-g0 verify --run artifacts/run-001
```

## 看哪些结果

| 文件 | 用途 |
|---|---|
| `artifacts/environment.json`、`gpu-pip-freeze.txt` | GPU/驱动、依赖记录 |
| `model-lock.json` | 模型/tokenizer revision 和文件校验 |
| `run-001/manifest.json` | 代码 hash、Pi、模型和采样参数 |
| `run-001/compatibility_report.json` | 会话隔离、工具结果、成功覆盖和退出后的活跃会话数 |
| `run-001/<trial>/episode.json` | 实际输入 token、生成 token、旧策略 logprob、loss mask、完成/消费映射 |
| `run-001/<trial>/events.jsonl`、`pi.stderr` | Pi 请求、工具事件及诊断 |
| `run-001/module_inventory.json` | 线性层清单，为后续 LoRA targets 选择准备；不是 LoRA 兼容性结论 |
| `run-001/parity_report.json` | FSDP 与 rollout 的 mean/p99 logprob 误差 |
| `run-001/gate_report.json` | `g0_passed` 最终结果，且 `optimizer_steps=0` |

六类用例：纯文本、单工具、先 lookup 再 advance、同 session 多轮、真实 stdio MCP echo、工具临时失败后恢复。仅启用四个合成工具，禁用 Pi 内置 shell/文件工具及全局扩展/skills/context discovery。

要求所有 session 的协议检查通过，并且每类至少有一个成功任务示例。**小模型做错任务和接口损坏分开报告**，不要求所有模型采样均成功。

默认 parity 容差：每次调用 mean absolute delta ≤0.02 nats、p99 ≤0.1。失败先查版本/模板/数值精度，不自动放宽门槛。HF 直接前向仅用于定位：

```bash
CUDA_VISIBLE_DEVICES=1 .venv-gpu/bin/mh-g0 parity --run artifacts/run-001 --reference hf
```

HF 结果写 `hf_parity_report.json`，**不能替代 FSDP gate**。没有 GPU 时不要运行 `prepare` 下载模型来“试训练”。

## CPU 开发验证

```bash
uv sync --locked --extra dev
uv run pytest -q
uv run ruff check src tests
```

真实 Pi 的 CPU 集成测试（模型为明确标记的 mock，包含真实 MCP 子进程）：

```bash
npm install --prefix vendor/pi --save-exact @earendil-works/pi-coding-agent@0.84.3
RUN_PI_E2E=1 PI_BIN="$PWD/vendor/pi/node_modules/.bin/pi" uv run pytest -q
```

也可手动启动 mock 服务，在另一终端运行 smoke：

```bash
export G0_ADMIN_KEY='local-test-key-at-least-24-characters'
uv run mh-g0 serve --backend mock --state artifacts/mock-server
# 另一终端设置同一个 G0_ADMIN_KEY：
uv run mh-g0 smoke --output artifacts/mock-run --pi "$PWD/vendor/pi/node_modules/.bin/pi"
```

完整 CPU 测试可通过 `collection_checks_passed`，但 `verify` 应返回失败，防止 mock 冒充真实模型。

## 接口和范围说明

- 本项目提供的是 **G0 collection facade**，内部调用真实 AReaL `ArealOpenAI`/`RemoteSGLangEngine`，不是直接启动上游 Online RL trainer。`/rl/ack`、`/rl/abort_session`、`/rl/export` 是本项目接口，不能原样调用上游 gateway。
- SSE 当前在完整生成、持久化 token trace 后切块输出，用于检验 Pi 的文本/工具流式协议；不是低延迟逐 token serving，不用于首 token 延迟评测。
- 重试使用 `X-Request-ID`：相同 ID/请求返回同一 completion，不重新采样；同 ID 不同请求返回 409。Pi 自动重试关闭，控制端异常不盲目重试。
- Pi 在完成 assistant 消费后保存 receipt；服务保留未消费输出为 orphan，不伪造消费确认。runner 再比较 Pi 文本/工具 JSON 与 serving 结果是否一致。
- 会话可重复结束/导出，冲突 reward 被拒绝；原始 reward 仅作 API 契约测试，不进入训练。此版本不实现多轮 reward 传播、GRPO admission 或 batch 学习。
- 进程重启将未结束 session 标 quarantine；运行中超时由 reaper 清理。恢复时新建 trial ID，不假装内存中的 AReaL cache 可以恢复。已持久化 trace 可导出审计。
- 后续接 AgentEval 时复用现有 Pi connector 的独立目录能力，替换合成任务 runner，先通过 G1 的官方 reward 与 GT 隔离门禁，再做 SFT/GRPO。

设计与源码参考见 [docs/design.md](docs/design.md)，验证记录见 [docs/validation.md](docs/validation.md)。
