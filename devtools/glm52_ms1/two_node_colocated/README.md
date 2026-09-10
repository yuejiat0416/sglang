# GLM-5.2：双机混部 static 自测脚本归档

状态：脚本准备与CPU自测；**双机/NPU未执行，不是转测通过记录**。本目录属于个人sync上的临时自测资产，永久排除正式社区PR。SGLang与kernel框架、算子实现均不在本轮改动范围内。

## 本轮目的、范围和结果分流

已有单机static eager请求与GSM8K接受率证据，仍缺双机、DP Attention、多请求、完整图执行和精度对照证据。这里把运行入口归到一个目录，复用社区服务与benchmark，补齐用户已明确的三条工作线：

1. 五模式：target-only eager、target-only graph、DSpark static eager、DSpark static graph、NEXTN graph。NEXTN沿用4 steps/topk1/draft5；DSpark block8/verify9。
2. GSM8K与GPQA-Diamond**分别默认10题**，保存答案、逐题评分、接受计数和target-only配对比较。全套脚本不等于默认跑完整数据集。
3. 131072输入/1024输出，prefix cache 0/50/90三档；先每档一条，再每档64条/并发8的候选压力点，可另设请求数、并发、持续时间。

环境检查失败先补配置/挂载/版本；启动失败保留两端日志；短请求失败先查功能；精度对照有丢分或未解析时核对逐题响应；graph没有replay证据时不能标图通过；长输入容量或实际命中不符时本档失败并停止后续档。只有这些证据齐全后，再按部门冻结的负载协议评估严格A/P>0.5。

NPU上所有命令由负责人执行。本目录不会SSH、自动切换五种服务、改主机sysctl、全局清缓存、安装/替换DeepEP或修改权重。默认打印启动命令；带`--execute`才运行本机服务。

## 目录与运行结果

| 入口 | 作用 |
|---|---|
| `config.example.json` | 两端共用的脱敏模板，路径、服务名、IP/NIC均为必填占位；TP32/DP8是待实测候选 |
| `container.sh` | 新容器可选入口：镜像/容器名/工作目录本地填写，逐个device，整个`/home:/home`及核心挂载 |
| `check_node.py` | 两端本地只读检查，随后离线比较两份环境证据 |
| `launch.py` | 五模式启动，保存两端各自的配置、源码/制品小文件指纹、完整服务日志 |
| `offline_dataset.py` | 离线整理GSM8K/GPQA固定题目，答案留在客户端 |
| `bench_accuracy.py` | 原社区chat benchmark、逐请求A/P/N、显式最终答案评分 |
| `bench_prefix.py` | 原社区native请求/流式计时、DP路由、GSP定长输入、三档缓存与负载 |
| `run_suite.py` | 打印五模式执行计划，或在当前服务上运行一个客户端阶段 |
| `report.py` | target-only逐题配对和结果文件盘点，不自动挑最高分 |
| `build_archive.py` | 连同明确依赖打包；不包含本地配置、模型、GPQA或运行结果 |

客户端结果位于配置的`state/evidence/two-node-colocated/`。每次独立目录；保留`summary.json`、请求/响应、benchmark结果、metrics与server_info。服务器启动证据在state下`evidence/launch-…-rank…-mode/`，包含`server.log`与退出状态。这些结果保留真实部署路径、地址、请求与模型元数据，属于本地调试证据，不随工具归档或加入Git。

## 1. 两端准备与配置

以下命令中的`/path/to/...`是说明占位，在服务器替换为自己的路径。先进入两端同版本SGLang仓库根目录；源码、模型与工作目录由本地配置指定，模板不携带真实路径、IP、容器名或镜像地址。

```bash
cd /path/to/sglang
```

已有正确容器可以继续使用。若新建容器，在**宿主机**填写本机使用的镜像、容器名和私有工作目录后运行；第二台独立填写。脚本保留完整`/home:/home`，因此`HOST_WORK_DIR`应是宿主机`/home`下的绝对路径，其cache目录在容器内保持同路径可见。

```bash
IMAGE='__FILL_APPROVED_IMAGE_REFERENCE__' \
CONTAINER_NAME='__FILL_LOCAL_CONTAINER_NAME__' \
HOST_WORK_DIR='__FILL_PRIVATE_WORK_DIR_BELOW_HOME__' \
bash devtools/glm52_ms1/two_node_colocated/container.sh
```

未填占位时脚本直接退出。请使用项目已经核准的A3/CANN版本镜像；模板不提供镜像地址，不替换已有容器，也不安装包。

在**容器内**复制私有配置，文件放在Git和工具归档目录之外：

```bash
cp devtools/glm52_ms1/two_node_colocated/config.example.json /path/to/local-config.json
```

编辑本地JSON，替换全部`__FILL_*__`：`repo`、`kernel_repo`、`state`、`target_model`、`draft_model`、`tokenizer`和`served_model_name`，以及两台的`host`、`hccl_socket_ifname`、`gloo_socket_ifname`。路径必须在容器内真实存在，两端使用相同配置。服务名由实际部署定义；IP/NIC根据预留机器填写，不能用单机`lo`。启动工具会拒绝未填写的模板，不猜网卡、IP、SoC、模型路径或防火墙配置。

候选为每节点16逻辑NPU、TP32、DP8、PP1、无CP、混部，context_length=133120。`--enable-dp-lm-head`来自当前DSpark在DP>1下的显式要求；五模式保留共同布局。`max_running_requests=8`在当前DP8规则下每DP组请求上限为1，实际值还要读取服务。context参数只放开长度，**不会凭空增加KV容量**；0/50/90都需容纳完整131072上下文及1024输出。

两台分别检查，第二台把rank改1：

```bash
python3 devtools/glm52_ms1/two_node_colocated/check_node.py local --config /path/to/local-config.json --rank 0
```

工具只读设备文件、挂载、NIC/IP、npu-smi、版本、Git及模型小文件/分片是否存在，不读取完整权重哈希。将两份`node.json`放到同一处后：

```bash
python3 devtools/glm52_ms1/two_node_colocated/check_node.py compare node0.json node1.json --output nodes-comparison.json
```

路径以各次输出为准。环境一致不等于跨机通信、完整权重字节或模型功能已通过。镜像digest由宿主机`docker inspect`另行保留；脚本不依赖docker.sock挂载。

## 2. 启动一个模式

先从`target-eager`建立双机基线。在node0终端：

```bash
python3 devtools/glm52_ms1/two_node_colocated/launch.py target-eager --config /path/to/local-config.json --rank 0 --execute
```

node1同一条命令把rank改1。等待node0服务就绪，再开第三个终端跑客户端。省略`--execute`仅打印。两台日志均自动落盘；Ctrl+C只停止本机本次启动进程组，另一台需分别停止。下一模式必须两台都切换，不把新旧模式混跑。

模式顺序建议：`target-eager` → `target-graph` → `dspark-eager` → `dspark-graph` → `nextn-graph`。DSpark显式static/original QuaRot并复用既有kernel源码overlay；不改已安装包。图启动参数只是请求开启，真实Target replay由metrics另记；**当前HTTP指标不证明Draft所有阶段的图回放**，还需服务日志/后续图阶段验证。

## 3. 数据与精度对照

GSM8K已有固定10题，可离线直接整理：

```bash
python3 devtools/glm52_ms1/two_node_colocated/offline_dataset.py prepare --dataset gsm8k --output /path/to/datasets/gsm8k.json
```

GPQA-Diamond数据先在内网PC准备，再传到服务器。传入本地CSV/JSONL，不联网下载；CSV列为`Question`、`Correct Answer`、`Incorrect Answer 1/2/3`。脚本记录文件哈希，但不能认证任意文件就是Diamond split，来源须随测试记录保留。

```bash
python3 devtools/glm52_ms1/two_node_colocated/offline_dataset.py prepare --dataset gpqa --input /path/to/source/gpqa_diamond.csv --output /path/to/datasets/gpqa.json
```

客户端通过`--data-dir`使用上述私有数据目录；省略时使用`state/datasets/two-node-colocated/`。默认各取10题，GPQA固定seed打乱选项，gold不会进入服务请求。缺数据明确BLOCKED。若日后要全数据，离线prepare的`--limit`填写样本总数，再运行客户端`--limit 0`；本次不要默认这么做。

当前启动的模式先跑1题64token连通/计数检查：

```bash
python3 devtools/glm52_ms1/two_node_colocated/run_suite.py run --config /path/to/local-config.json --mode target-eager --stage smoke --data-dir /path/to/datasets
```

然后该模式分别执行两个数据集，各10题、串行、最多4096输出、EOS有效：

```bash
python3 devtools/glm52_ms1/two_node_colocated/run_suite.py run --config /path/to/local-config.json --mode target-eager --stage accuracy --data-dir /path/to/datasets
```

其他模式替换`--mode`，同一批fixture与输出上限必须保持。4096是本归档可调整的诊断预算，不是部门冻结精度协议；提示增加明确最终答案格式，不能与不同提示或输出预算的历史结果直接当同协议复测。GSM按最后明确数字答案，GPQA按明确Answer:A-D评分；截断、未解析、失败保留在总样本分母，另列未定项；不执行生成代码，不拿推理中间文本冒充最终答案。

用各自的结果文件做同数据集对照：

```bash
python3 devtools/glm52_ms1/two_node_colocated/report.py compare --baseline TARGET_SUMMARY.json --candidate DSPARK_SUMMARY.json --output paired-comparison.json
```

输出逐题丢分/恢复、相同prompt IDs下首个不同输出token，以及样本精度差。10题不能证明1%精度波动或整体不下降。此本地显式答案评分不是EvalScope正式GPQA协议的替代品；正式转测协议冻结后需再对齐。

## 4. 128k/1k与接受率负载

先只读预检该模式的实际每DP组容量、并发槽位和缓存设置：

```bash
python3 devtools/glm52_ms1/two_node_colocated/bench_prefix.py check --config /path/to/local-config.json --mode dspark-eager
```

通过后先各档一条：

```bash
python3 devtools/glm52_ms1/two_node_colocated/run_suite.py run --config /path/to/local-config.json --mode dspark-eager --stage prefix
```

三档独立负载候选，各64条/并发8：

```bash
python3 devtools/glm52_ms1/two_node_colocated/run_suite.py run --config /path/to/local-config.json --mode dspark-eager --stage load --num-prompts 64 --concurrency 8
```

`--cache-hit 0|50|90`可单独跑一档，默认all依次跑三档，失败停止后续档。持续负载可加`--duration-seconds`，数值由你们最终压测标准确定；脚本不把未冻结时长自动标成验收。并发扫描显式改变`--concurrency`并保留所有结果，不自动挑最高接受率一组。

| 档位 | 输入token | 输出token | page128预期缓存token | 实际比例 |
|---|---:|---:|---:|---:|
| 0 | 131072 | 1024 | 0 | 0% |
| 50 | 131072 | 1024 | 65536 | 50% |
| 90 | 131072 | 1024 | 117888 | 89.94140625% |

从社区GSP随机文本生成方法构造prefix/suffix，重新tokenize定长，然后直接传input_ids，无chat模板增量。热档先预热每个将使用的DP组，正式请求通过`routed_dp_rank`落回该组；每请求suffix首token唯一，避免后续整题命中变100%。冷档每请求独立cache_salt。输出强制1024/ignore_eos，不把停止早的回答悄悄排除。

预热与测量分别归档。每条必须核对实际DP组、prompt/completion/cached token、retraction和A/P/N；任一不符不输出整档有效接受率。严格口径`sum(accepted_drafts)/sum(proposed_drafts)>0.5`，不含bonus、不平均百分比、不混合三档，也不把TP复制日志重复相加。target-only接受率N/A。

原生流式benchmark给TTFT、TPOT、吞吐与延迟分布；归档显示客户端准备时间及完整测量窗，不因`request-rate=inf`就假定硬件已打满。GSP是随机合成负载，接受率不能替代GSM8K/GPQA精度结论。本候选启动明确使用普通Radix；当前实验性CPP Radix不支持这里的cache_salt。如果自行另启服务，HTTP不能完整证明其环境设置，需核对启动日志，不能把未知当成兼容。

## 5. 尚待协议/实现补齐的范围

- **95% DDR/SSD池化缓存、相对不命中10倍**：不是这三档HBM prefix复用。需要实际分层后端、存储命中证据、比较指标及DSpark缓存恢复兼容性；当前脚本明确阻止把HiCache结果混入此协议。未实现/未执行，不标N/A或PASS。
- **小于64k/1k的平均TTFT<10秒、TPOT<50ms**：不能套到128k档；边界与具体输入点待冻结，本目录先保留128k开发必测项。
- **所有static内部机制都通过**：五模式HTTP测试不能替代全DP/全图阶段、abort/retract回收、长稳、KV恢复或算子正确性验证。此次未新增生产hook；当前目录覆盖部署、样本精度、压力/缓存可执行入口，不能将HTTP成功写成这些内部机制已通过。
- CANN9.0、A5、dynamic/compact、PP、CP、P/D分离是其他验证范围；当前脚本选择的TP32/DP8/PP1混部profile不是对框架已有能力新增限制。
- 双机IP/NIC、最终并发/时长/数据选择、正式精度协议以及graph实际覆盖仍待实机或确认；全部保留NOT_RUN/BLOCKED记录。

## 6. 归档、计划与源码对应

```bash
python3 devtools/glm52_ms1/two_node_colocated/run_suite.py plan --config /path/to/local-config.json --data-dir /path/to/datasets
python3 devtools/glm52_ms1/two_node_colocated/report.py inventory --evidence /path/to/state/evidence/two-node-colocated --output coverage.json
python3 devtools/glm52_ms1/two_node_colocated/build_archive.py --output /path/to/colocated-test-tools.tar.gz
```

archive按明确清单包含本目录和上一级单机客户端、启动入口、overlay/helper、指南、测试及GSM fixture依赖，附每文件SHA。仍需要匹配SGLang与kernel checkout；不打包任何模型、运行结果、本地IP配置或GPQA数据。归档不会收集目录里的额外配置/Python文件，不带本机用户、组和文件时间戳。压缩包是本地交付物，生成并不上传。

具体交付版本见压缩包的`colocated-tools-manifest.json`。以下是**已存在源码**，新增入口为本目录上表；本轮不修改这些源码：

- [speculative_hook.py](../../../python/sglang/srt/arg_groups/speculative_hook.py#L511)：DP>1 DSpark要求DP lm_head，解释双机相对单机增加的部署flag。
- [kv_cache_configurator.py](../../../python/sglang/srt/mem_cache/kv_cache_configurator.py#L2202)：全局请求上限按DP组分配，解释并发预检。
- [data_parallel_controller.py](../../../python/sglang/srt/managers/data_parallel_controller.py#L753)：显式DP路由，保证warm与measure对应。
- [tokenizer_manager.py](../../../python/sglang/srt/managers/tokenizer_manager.py#L2377)：响应DP rank，核验实际落点。
- [serving.py](../../../python/sglang/benchmark/serving.py#L658)：原生流式请求与时序；[gsm8k_mode_stats.py](../gsm8k_mode_stats.py)复用既有逐请求计数和图指标解析。

学习对应：项目完整学习手册第15.5节“Prefix cache与DSpark接口”用于理解缓存恢复涉及target特征与draft KV；第28.1节“计数分母”、28.7节“精度协议”、28.11—28.14节“长度/缓存/压力”解释各项采集与判读。教材未随工具归档，具体本地入口保留在内部审阅记录中。教材是概念/历史快照，当前运行接口以上述仓库源码为准。
