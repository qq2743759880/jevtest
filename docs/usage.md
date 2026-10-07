# 使用指南

## 离线与真实 API

默认 HTTP profile 为 offline：Rule、Jev mock、LLM stub。CLI 单臂 `--arm rule` 与 `--arm jev-mock` 不需要 key；`--arm jev` 和 `--arm llm` 会调用真实供应商，不要用它们验证离线安装。

真实 HTTP 模式启动前，通过进程环境设置 `TYPESAFE_API_KEY` 和 `ARK_API_KEY`，不要把凭据写进提交到仓库的文件。仅在你决定承担外部调用与费用时执行：

```powershell
python -m jevtest.http_api --host 127.0.0.1 --port 8765 --store runs/live.sqlite3 --profile live-advisory --allow-live-api
```

每条提交还需要页面确认。该配置总是进入人工审核；供应商适配器对每条请求分别限制 Jev／Ark 的尝试次数。本次交付未运行这个模式，不承诺真实服务可用性或效果。不要开启大批量真实调用来验证安装。

## MCP

安装 mcp extra 后，入口为 `jevtest-mcp` 或 `python -m jevtest.mcp_server`。通过进程环境变量 `JEVTEST_DB_PATH` 指定 SQLite 路径。其三个工具为 `triage_failure`、`get_triage_result`、`get_triage_evidence`。只提供离线分诊与读取，不执行修复、不提供网页审核的 MCP 工具。

把程序入口配置到支持 STDIO 的宿主时，按该宿主官方配置方式设置绝对路径和同一 SQLite 文件；本包没有替你更改任何宿主配置。HTTP 与 MCP 使用同一 store 才能共享数据；同时提交不意味着跨进程 exactly-once。

## 常见问题

- 无待办：先导入完整示例；某些规则命中可以直接输出，不进入人工队列。选 Quick Start 中的 R4 样例体验待办。
- 输入失败：保留示例全部必填字段，查看页面错误或 CLI stderr；输出 UNKNOWN 时继续人工分诊。
- 版本冲突：刷新当前记录再审核，避免用旧版本覆盖他人修改。
- 停服务后找不到记录：保持相同工作目录或改用 SQLite 的绝对路径。
- pip 找不到依赖：核对 Python 版本、网络和软件源；本包未提供全部第三方依赖的离线镜像。
