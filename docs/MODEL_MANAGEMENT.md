# 模型管理定制部署

维护仓库：<https://github.com/zzz27578/aistudio-api>。
上游基线：`85e22c3cf76683d470103675324c9c1f28ac6eca`。
`master` 保存定制功能与上游合并结果；`custom/model-management` 保存首次迁入的定制功能，
便于查看独立差异。更新检查仍比较 DeconstructedCube 主仓库，避免把 fork 自身的定制提交误报为上游更新。

控制台 `/models` 管理从上游发现的模型。配置持久化到数据目录中的
`model_catalog.json`，与 `config.yaml` 分离，重新拉取保留各模型的设置。

- `enabled` 控制基础模型及所有思考后缀是否可以调用。
- `visible` 控制基础模型是否出现在 Gemini `/v1beta/models` 中，对所有 Key 生效。
  隐藏不是权限隔离，已知完整名称仍然可以调用。
- 每个思考等级后缀可分别启用及设定可见性。只允许上游明确声明的思考等级。
  如 `gemini-4-hydrogen-thinklow` 映射为 `models/gemini-4-hydrogen`，并注入
  `generationConfig.thinkingConfig.thinkingLevel=LOW`。覆盖请求的等级和预算，
  其他生成参数保留。账号选择、限额统计和实际请求均使用基础模型名。
- 基础模型名继续沿用原有默认规则和请求参数，不改变默认思考等级。
- 新模型默认启用且可见，思考后缀默认关闭。保存后即时生效，无需重启。

在 New API 的 Gemini 渠道重新拉取模型后，选择这些后缀名称即可使用固定等级。
不要再将思考后缀映射为基础名称，否则会失去等级信息。

管理接口 `/api/model-catalog`（GET，`refresh=true` 强制拉取）及
`/api/model-catalog/{model_id}`（PUT）使用现有控制台密码鉴权，普通 API Key
不能修改配置。对外仍采用原生 Gemini 协议。

左下角“检查更新”调用 `/api/system/check-update`，查询固定 GitHub 主仓库
默认分支并比较完整 commit SHA；结果缓存 60 秒。显示当前主仓库基线、定制构建、
最新提交和差异链接。网络失败不会被误报为“已是最新”。该功能仅检查，不执行升级。

## 构建与升级

`Dockerfile.deployment` 从干净源码构建前端、Linux 签名器和运行镜像，不依赖服务器上的旧镜像。
Go 阶段运行签名器离线回归测试；前端使用 Bun 锁文件安装并执行 lint、类型检查和生产构建。
JS/CSS 使用内容哈希地址。Git 中沿用上游的静态文件，Docker 构建不会使用这些历史产物。

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
