# GSM8K：五模式各10题快速比较

## 本轮做什么，以及结果决定什么

当前处于接受率定位期间的**横向基线比较**。
用同一批GSM8K题目比较DSpark eager/graph、
target-only eager/graph和已有NEXTN graph，观察问题是否随题目/算法/执行模式变化。
这不替代下一轮Draft最终hidden→head/Markov→Verify对齐诊断。

- 官方GSM8K test固定版本前10题，原始顺序、原始问题、zero-shot；原答案只保留在本地，不发送给模型。
- 源码内附 `gsm8k10.json` 和原MIT许可证；无需内网下载数据/安装EvalScope。
- 每组10题各一次；temperature=0、top_p=1、输出最多1024 token、尊重EOS。
- `request-rate=inf` 配合 `max-concurrency=1`，逐题串行；没有benchmark warmup请求。
- 每题独立cache_salt避免跨题/跨组前缀复用；不清全服务cache。期间不要让其他客户端向此端口发请求。
- 五组共50条正式请求；启动器自己的服务warmup不属于benchmark10题，指标取运行前后差。
- 当前graph指decode/verify图配置；prefill是否图化以解析配置为准。不能称“全部计算上图”。
- NEXTN沿用同事配置：steps=4、topk=1、draft_tokens=5；服务解析后algorithm可能显示EAGLE。
- DSpark使用QuaRot original候选、自有embedding/head和加载时Q-fold。

## 本地配置与执行

仓库不保存机器IP、个人目录、镜像地址和权重路径。先在仓库外准备自己的
`single.local.sh`，以下值均为占位符，替换为当前容器内可见的真实信息：

```bash
export SGLANG_REPO=/SET_ABSOLUTE_SGLANG_CHECKOUT
export MS1_STATE=/SET_ABSOLUTE_RUN_STATE
export MS1_HOST=SET_SERVER_ADDRESS
export MS1_PORT=8810
export TARGET_MODEL=/SET_ABSOLUTE_TARGET_CHECKPOINT
export DRAFT_MODEL=/SET_ABSOLUTE_DSPARK_CHECKPOINT
export SERVED_MODEL_NAME=model
```

两个终端都进入同一已有测试容器，并加载同一份本地配置：

```bash
source /SET_LOCAL_SETTINGS_FILE
cd "$SGLANG_REPO"
GSM_ARGS=(--host "$MS1_HOST" --port "$MS1_PORT" --state "$MS1_STATE"
  --target "$TARGET_MODEL" --draft "$DRAFT_MODEL" --served-model-name "$SERVED_MODEL_NAME")
```

使用原服务Python；若在EvalScope虚拟环境中，先执行`deactivate`。同步源码时应先
停止旧服务，再执行`git pull`，不要在运行过程中更换源码。

终端A启动服务：

```bash
python3 devtools/glm52_ms1/bench_gsm8k_modes.py launch dspark-eager "${GSM_ARGS[@]}"
```

等服务就绪，终端B运行同一组10题：

```bash
python3 devtools/glm52_ms1/bench_gsm8k_modes.py run dspark-eager "${GSM_ARGS[@]}"
```

本地模型路径同时传给社区benchmark的`--model`与`--tokenizer`，
`--served-model-name`单独决定HTTP模型名，避免把服务别名当成在线模型仓库查找。
固定GSM8K十题已随工具提供，无需在内网下载数据集或安装EvalScope。

`launch`只是选择已有 `single_dspark_static.sh` 的MODE/GRAPH，显式开启metrics；
无snapshot hook，不更改SGLang或kernel源码。`run`才是客户端，调用当前checkout的
`sglang.benchmark.serving`；只在这个客户端进程里捕获同次HTTP响应，不重发题目。

按顺序把上述两条命令中的模式替换为下表值：

| 顺序 | 模式值 | 服务算法 | 目标图配置 |
|---|---|---|---|
| 1 | `dspark-eager` | DSpark static，block8/verify9 | 禁用 |
| 2 | `dspark-graph` | DSpark static，block8/verify9 | 启用 |
| 3 | `target-eager` | target-only | 禁用 |
| 4 | `target-graph` | target-only | 启用 |
| 5 | `nextn-graph` | NEXTN，4/1/5 | 启用 |

每组客户端完成后，回终端A按Ctrl+C停止服务，再启动下一组。
不能只改客户端mode而不重启服务；工具会拒绝与 `/server_info` 不符的配置。
不使用旧 `with_proposal_snapshot.py` / `with_context_snapshot.py` 启动方式。
若graph组启动/请求失败，保留首个异常及前后日志；该组记录为未完成，不自动降级eager。

`--host`、`--target`、`--state`、`--served-model-name`必填；DSpark模式还必须提供`--draft`。
两端配置必须一致，尤其权重路径应使用当前容器实际可见的路径。
可加`--print-command`查看命令，不启动服务或发请求；输出含本地配置，留在内网。

直接调用`single_dspark_static.sh`时，同样要求`MS1_HOST`、`TARGET_MODEL`、`MS1_STATE`；
仅DSpark要求`DRAFT_MODEL`，`SERVED_MODEL_NAME`默认使用通用别名`model`。
如需创建容器，`start_container.sh`另要求在仓库外配置`IMAGE`与`CONTAINER_NAME`；
cache目录由`MS1_STATE`派生，保留逐个设备和宿主机核心目录挂载。

## 看哪些结果

每次客户端创建独立 `$MS1_STATE/evidence/gsm8k-时间-标识-模式/`：

- `summary.json`：每题和汇总A/P/N、接受率、结束原因、实际图计数及证据边界。
- `responses.jsonl`：同次原始响应；`responses.md`：问题、参考答案与模型输出，待人工审视。
- `requests.jsonl` / `protocol.json`：实际请求、题目顺序/源版本/工具SHA/固定参数。
- `benchmark.jsonl`：社区benchmark原报告；其Accept length和非流式TTFT不作为本轮结论。
- `server_info.before/after.json`、`metrics.before/after.txt`：前后配置/公共计数。

严格接受率是 `sum(accepted_drafts) / sum(proposed_drafts)`；不是平均各题百分比，
也不是首token接受概率。另列 `A/N` 与 `1+A/N` 供解释，不混同API的可见输出length/N。
Target-only没有投机接受率，记N/A。缺失/重复/冲突计数或请求失败记INCOMPLETE，
不把剩余成功题冒充10题成绩。数据集源、输出预算或采样参数变化后不可直接混比。

图证据来自 `sglang:cuda_graph_passes_total` 前后差，区分 `decode_cuda_graph`
与 `decode_none`。`target_replay_observed=true`只证明期间Target decode/verify走图；
公共指标是服务级batch计数，不能代替请求A/P/N，也不能证明Draft/Markov全链路回放。
Draft capture耗时会保存，Draft replay标记NOT_EXPOSED_BY_CURRENT_HTTP_API。

本轮非流式请求只便于完整保留原始计数。**原生benchmark把TTFT记成完整响应时延，
TPOT也不是真实流式逐token间隔**，不可用于TTFT/TPOT准出或宣称加速。
1024预算内未答完会标length；保存答案但不自动给精度PASS。
10题不等于正式GSM8K精度、发布方accepted length=6.41复现或压力接受率>50%验收。

下一步分流：

- DSpark在GSM8K仍低：保留该基线，继续最终hidden/head/Markov及Verify边界定位。
- eager与graph显著不同/报错：优先查图buffer、metadata及实际replay；不先改接受公式。
- 输出出现疑点：用对应target-only结果核对，不能只凭接受率判断正确性。
- 接受率达到50%以上：仅代表本批串行样本；后续继续更完整精度与正式压力/长输入测试。

每组可先回传终端摘要与 `summary.json`；无需等五组全部成功。完整响应和服务日志留内网。
工具为sync专用临时诊断，永久排除正式社区PR分支。
