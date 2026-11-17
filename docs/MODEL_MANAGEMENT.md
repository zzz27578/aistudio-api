# 模型管理定制部署

基于主仓库 `d4c3b7957ce491e820f4c774e741c2194b5c45b5`。

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

`Dockerfile.model-management` 复用已验证的 `Dockerfile.deployment` 基础镜像，
叠加新 Python 代码和由 `cd web && bun run build` 生成的静态页面。

```sh
docker build -f Dockerfile.model-management -t aistudio-api:upstream-d4c3b-20261116 .
```

升级主仓库时先将这些改动合并到新源码，重新运行 Python/前端检查并构建静态文件；
若 Python 依赖、系统依赖或 Linux 签名器源码/构建选项变化，使用 `Dockerfile.deployment`
构建新的基础镜像；否则可继续复用已验证的运行环境及签名器二进制。
指定 `BASE_IMAGE`、`UPSTREAM_REVISION`、`CUSTOM_BUILD` 构建定制镜像。
前端 JS/CSS 使用内容哈希文件名，升级时整体重建，避免新旧资源混用。
直接拉取官方镜像会失去这些
新增功能。不要把主仓库原始代码拷进数据目录。

升级前备份部署目录中的 `data`、`docker-compose.yml`、`.env`；保留原环境变量、账号、
端口和数据挂载。
构建验证通过后再通过该部署目录的 Compose 替换镜像。回滚时恢复备份的 Compose
并执行 `docker compose up -d --no-build`，不需要还原账号数据；旧版本会忽略新增的
`model_catalog.json`。保留旧镜像以便回滚。
