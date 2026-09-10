# GSP 128k/1k 三档缓存接受率用例（临时调试）

本轮在请求级比较长随机输入下的接受率及缓存条件。
GSP是随机词表生成的合成输入，与GSM8K内容不同，不能据两者接受率差异直接判定模型bug。
本工具属于个人sync上的临时客户端，不进入社区PR；服务、框架和算子保持原状。

## 本轮具体做什么

复用社区`sglang.benchmark.serving`、`generated-shared-prefix`生成器和原生`/generate`请求函数。
原生GSP会随机token→文本→重新分词，名义长度不保证准确；工具将生成结果定长为
**131072个token ID**直接发送，不再套chat模板。输出设置1024、ignore_eos=true，
并用响应实际token数复核，不将提前结束或服务限长算成完整128k/1k。

默认三档各测1条，串行；50/90档各有1条独立预热，只生成1token。
因此正常完整执行为**3条测量＋2条预热**，不额外重复、不混合三档接受率。
三档使用同一份完整输入、每档新cache_salt；GSP数据在本地模型tokenizer上生成，无需下载数据集。

| 档位 | 预热共同前缀（page128） | 实际目标比例 | 测量输入/输出 |
|---|---:|---:|---:|
| 0% | 无 | 0% | 131072/1024 |
| 50% | 65536 | 50% | 131072/1024 |
| 90% | 117888 | 89.94140625% | 131072/1024 |

运行时读取真实page_size，按`page_size * floor(131072 * ratio / page_size)`取整。
单条131072输入的90%本来就不是整数token，不能写成精确90%；结果保留档位和实际值。
预热发送共同前缀再加一个不同的尾token，等待请求完成后测量，避免无意热到整个输入。
不清空服务全局缓存；预热响应单独保存，接受率只用本条测量响应A/P。
这是普通prefix cache用例，缓存device/host/storage来源按返回记录，不宣称DDR/SSD池化验收。

负责人提供的64条/并发8命令是后续压测规模参考。当前先检查单条128k是否能装下，
并验证三档真实命中。原生命令1组64条可能把首批未热请求混入90%统计，
又未按实际长度纠正分词差异，因此不能直接代替本轮三档条件检查。
本轮不自动发起64条×3组请求，也不把单请求结果当“压测接受率>50%”准出。

## NPU上怎么执行

在**另一个终端、同一容器**中执行，现有服务终端保持运行。
使用原服务Python环境，不需要EvalScope。服务地址、权重和证据目录没有内置默认值，
从仓库外本地配置读取；与GSM8K指南使用同一组`MS1_HOST`、`MS1_PORT`、`TARGET_MODEL`、
`MS1_STATE`和`SGLANG_REPO`变量即可。

```bash
source /SET_LOCAL_SETTINGS_FILE
cd "$SGLANG_REPO"
GSP_ARGS=(--host "$MS1_HOST" --port "$MS1_PORT" --target "$TARGET_MODEL" --state "$MS1_STATE")
python3 devtools/glm52_ms1/bench_gsp_prefix.py check "${GSP_ARGS[@]}"
```

`check`只读取`/server_info`并保存preflight，不加载模型、不发送生成请求。
`--host`、`--target`、`--state`必填；tokenizer使用同一个本地target制品，
权重路径必须是当前容器内实际可见的路径。

若显示`PREFLIGHT_READY`，执行三档：

```bash
python3 devtools/glm52_ms1/bench_gsp_prefix.py run "${GSP_ARGS[@]}"
```

也可以只测一档，例如：

```bash
python3 devtools/glm52_ms1/bench_gsp_prefix.py run --cache-hit 0 "${GSP_ARGS[@]}"
```

`run`同样先做预检，失败会在任何生成请求前停止。成功后看到每档接受率、
实际缓存token和实际输出长度。Evidence在`$MS1_STATE/evidence/gsp-prefix-*`。
运行产物会记录实际地址和路径，留在内网；源码仓库只保留通用工具与占位配置。

## 容量不足怎么判读

实际KV容量若不足131072输入及输出所需空间，90%缓存命中也不减少完整上下文的容量要求。
工具读取顶层`max_req_input_len/max_total_num_tokens`以及各scheduler的`memory_usage.token_capacity`，
按当前源码的输出cap公式判断能否完整生成1024token。page128、无上下文分片时，
必要容量至少132225token，按128页对齐至少132352；这仍不是显存分配成功或NPU兼容保证。
`max_prefill_tokens=69632`是批调度预算，工具不会错误地把它当作单条输入硬上限。

若显示`PREFLIGHT_BLOCKED`，**只回传Evidence中的preflight.json**，无需反复尝试发送长请求。
我们依据真实容量讨论部署调整；工具不会自动加mem_fraction、开HiCache、改并行参数或截断输入。
本工具的TP16/DP1/单机范围是当前诊断范围，不代表DSpark框架能力上限。

## 看什么结果、下一步怎么走

- 回传根目录`summary.json`，三个场景分别记录`acceptance.accept_rate=A/P`，不平均或混合三档。
- `GSP_CASE_CONDITIONS_MET`只表示请求身份、131072/1024、目标cached_tokens、无retraction、计数齐全；不是质量/性能PASS。
- 若缓存/长度不符，保留计数并明确场景不成立，停止后续档；先查实际缓存、容量或prefix恢复路径。
- 若三档条件成立，比较各档接受率；热请求明显下降时再排查target/draft上下文恢复和首次proposal，不能立刻改FIA。
- 原生benchmark打印的`Accept length`可能来自服务历史累计，不是这条A/P；以本工具summary为准。
- 当前采用非流式取完整响应，原生TTFT实际为整请求延迟，TPOT=0不代表零耗时，不用于延迟准出。
- 框架图配置与完整draft/target图回放不同，本工具不授予graph PASS。

保留`input_ids.json`、`protocol.json`、`server_info.*.json`及各档`request.json`、
`warm-response.json`（若有）、`responses.json`、`benchmark.jsonl`，无需重跑来补同轮证据。

## 源码与学习位置

当前SGLang `sync/glm52-dspark-ms1`，开发基线`cdd5e4bb2f6d133c566c089002e3bc58421f9725`。

- [GSP生成与重新分词](../../python/sglang/benchmark/datasets/generated_shared_prefix.py)：保留其随机数据生成，仅临时客户端把最终输入变为精确ID数。
- [原生generate发送](../../python/sglang/benchmark/serving.py)：复用实际请求函数，ID列表进入input_ids；客户端捕获同次meta_info，不改变返回对象。
- [服务输出长度上限](../../python/sglang/srt/managers/scheduler.py)：预检使用该公式，避免服务将1024悄悄缩短。
- 教材对应28.11“长上下文与命中率定义”、28.12“缓存用例怎样准备和测量”：理解完整上下文长度、预热与测量边界。教材位于独立项目学习资料中；此归档不记录个人工作区绝对路径。教材实现快照与当前sync不同，当前代码以上述仓内位置为准。
