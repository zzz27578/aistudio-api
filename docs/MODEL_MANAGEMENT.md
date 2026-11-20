# 模型管理定制部署

维护仓库：<https://github.com/zzz27578/aistudio-api>。
上游基线：`8a919084cac1bbb08307870cd5cbef07e5f09866`（2026-11-20）。
`master` 保存定制功能与上游合并结果；`custom/model-management` 保存首次迁入的定制功能，
便于查看独立差异。更新检查仍比较 DeconstructedCube 主仓库，避免把 fork 自身的定制提交误报为上游更新。

控制台 `/models` 管理从上游发现的模型。配置持久化到数据目录中的
`model_catalog.json`，与 `config.yaml` 分离，重新拉取保留各模型的设置。

- `enabled` 控制基础模型及所有思考后缀是否可以调用。
- `visible` 控制基础模型是否出现在 Gemini `/v1beta/models` 和 OpenAI `/v1/models` 中，对所有 Key 生效。
  隐藏不是权限隔离，已知完整名称仍然可以调用。
- 每个思考等级后缀可分别启用及设定可见性。只允许上游明确声明的思考等级。
  如 `gemini-4-hydrogen-thinklow` 映射为 `models/gemini-4-hydrogen`，并注入
  `generationConfig.thinkingConfig.thinkingLevel=LOW`。覆盖请求的等级和预算，
  其他生成参数保留。账号选择、限额统计和实际请求均使用基础模型名。
- 基础模型名继续沿用原有默认规则和请求参数，不改变默认思考等级。
- 新模型默认启用且可见，思考后缀默认关闭。保存后即时生效，无需重启。
- 每个基础模型折叠卡片的“全部启用 / 全部停用”修改基础模型及其支持的思考、抗截断变体的启用状态，保留各项可见性；点击“保存”后生效。卡片显示启用数量，部分启用时可直接一键统一。

在 New API 的 Gemini 或 OpenAI 兼容渠道重新拉取模型后，选择这些后缀名称即可使用固定等级。
不要再将思考后缀映射为基础名称，否则会失去等级信息。

管理接口 `/api/model-catalog`（GET，`refresh=true` 强制拉取）及
`/api/model-catalog/{model_id}`（PUT）使用现有控制台密码鉴权，普通 API Key
不能修改配置。两种协议读取相同的模型目录与策略。

## OpenAI / Gemini 双协议

两种协议默认同时提供，同一个端口、相同 API Key、账号池和配额；不需要在服务端选择单一提供商。

- OpenAI 兼容 Base URL：`http://localhost:8080/v1`；支持 `/v1/models`、`/v1/models/{id}`、`/v1/chat/completions`，普通和 SSE 流式响应。
- Gemini Base URL：`http://localhost:8080`；保留 `/v1beta/models` 和原生生成、图片等接口。
- OpenAI 聊天支持文本、`image_url` 识图输入、`reasoning_effort`、函数工具历史、JSON 输出配置；固定思考后缀仍优先于请求等级与预算。
- 当前明确拒绝严格工具/schema 校验、`parallel_tool_calls=false`、`n>1`、logprobs 和未实现的关键参数。图片接受 base64 data URL 或经校验的公网页 URL，单图最大 10 MiB，每请求最多 8 张；不读取服务端本地或私网图片地址。
- 思考输出使用 `reasoning_content` 兼容扩展。工具调用需保留 `extra_content.google.thought_signature`；经过 NewAPI 或其他中间层时应确认其不丢弃扩展字段。
- 不提供 OpenAI Responses、Images、Audio 或 Embeddings API。图片生成仍用原生 Gemini；给模型看图与调用 Images 生成图片是不同能力。
- HTTP 错误与 SSE 错误保持对应协议格式；usage、终止原因和工具调用状态不因格式转换被伪造。

例如 OpenAI 客户端请求：

```json
{
  "model": "gemini-4-hydrogen-thinkhigh",
  "messages": [{"role": "user", "content": "解释一下协程"}],
  "stream": true
}
```

## 抗截断后缀（实验文本包装）

在模型卡片内分别启用普通包装别名、各思考等级的包装别名，并选择是否允许 Key 拉取。配置字段 `text_wrapper`、`wrapper_thinking_aliases` 独立于原有 `thinking_aliases`，默认关闭。旧 `model_catalog.json` 自动采用新增字段默认值，无需迁移账号数据。

- `gemini-4-hydrogen-抗截断`：原模型默认思考策略 + 文本包装。
- `gemini-4-hydrogen-thinkhigh-抗截断`：固定 HIGH + 文本包装；不要求普通 HIGH 别名也启用。

网关将文本对话转换为 `emit_complete_response` 工具承载的历史，再在签名前规范化最终请求；响应从 `display_stream` 还原为正文。所有变体在账号选择、配额与统计中仍归属基础模型，不产生额外额度。

这是参考 bridge 思路实现的可选功能，不是新模型或断线续传。正文会缓冲到生成完成后返回；流式连接保活与正常终态/usage 保留。只允许文本聊天，客户 tools、工具历史、JSON schema、图片或其他多模态组合明确报错，不会静默丢弃。普通模型继续支持原有功能。

其实际改善率需要后续真实调用对照；本次仓库检查不包含 Google 生成验证。包装不能保证不截断、突破 token 上限或自动续写。网关新增完整性检查：残缺 JSON 不能仅因修复出部分文字就伪报 STOP；完整响应包、明确终态、安全拦截与完整工具调用保留各自语义。

左下角“检查更新”调用 `/api/system/check-update`，查询固定 GitHub 主仓库
默认分支并比较完整 commit SHA；结果缓存 60 秒。显示当前主仓库基线、定制构建、
最新提交和差异链接。网络失败不会被误报为“已是最新”。该功能仅检查，不执行升级。

## 构建与升级

`Dockerfile.deployment` 从干净源码构建前端、Linux 签名器和运行镜像，不依赖服务器上的旧镜像。
Go 阶段运行签名器离线回归测试；前端使用 Bun 锁文件安装并执行 lint、类型检查和生产构建。
JS/CSS 使用内容哈希地址。Git 同步保存本 fork 重新构建的静态文件，Docker 也从本次源码重新构建，不混用上游旧 chunk。

签名器使用上游本地 stdio IPC 路线，`Dockerfile.deployment` 使用统一的 `build.sh --test` 与 `build.sh` 构建静态 Linux 二进制，不复用旧 HTTP signer。镜像构建标识为 `fork-8a91908-openai-20261120`；提交仓库不等于已部署该镜像。

```sh
git clone https://github.com/zzz27578/aistudio-api.git
cd aistudio-api
# 在 .env 中设置 AISTUDIO_WEB_PASSWORD，可用 AISTUDIO_HOST_PORT 修改外部端口（默认 7878）。
docker compose -f compose.fork.yml up -d --build
```

源码直接启动前，先在 `web/` 运行 `bun install --frozen-lockfile && bun run build`。
已有部署继续保留其原 Compose、环境变量和数据挂载，升级时只更新构建来源和镜像。
初次使用 `compose.fork.yml` 会把配置、账号和模型策略统一持久化到 `./data`。

后续从 fork 拉取已合并版本可运行 `git pull --ff-only`，检查后重新构建部署。
维护者同步上游时，使用 `git fetch upstream && git merge upstream/master`，解决冲突后运行
AGENT.md 中的检查。同步更新 `Dockerfile.deployment` 的 `UPSTREAM_REVISION`，并用构建参数
`CUSTOM_REVISION` 写入实际 fork commit SHA，方便核对线上来源。
直接使用原仓库镜像会失去新增功能。不要把运行中的账号、Cookie、`.env` 或模型策略提交进 Git。

升级前备份部署目录中的 `data`、`docker-compose.yml`、`.env`；保留原环境变量、账号、
端口和数据挂载。
构建验证通过后再通过该部署目录的 Compose 替换镜像。回滚时恢复备份的 Compose
并执行 `docker compose up -d --no-build`，不需要还原账号数据；旧版本会忽略新增的
`model_catalog.json`。保留旧镜像以便回滚。
