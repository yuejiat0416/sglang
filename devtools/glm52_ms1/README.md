# Temporary MS1 validation tools

This directory supports the temporary `sync/glm52-dspark-ms1` development branch.
It is separate from the SGLang feature commits intended for upstream review.

## 当前排查第1步：接受率的逐轮顺序

已有真实请求53轮、10个草稿token接受，总接受率约2.36%；48轮在第一个草稿
token就被拒绝。汇总histogram没有时间顺序，本轮先区分从头低接受和后期下降。
这里“第一个草稿token”不是服务输出的第一个token。

只使用社区已有 `core,reqs` 记录，新增独立HTTP采集器；启动脚本、模型、
Attention、kernel及公共推理流程不改。保持原来的问题、temperature=0、
max_tokens=64、static/eager、TP16/DP1；本轮不同时更换gamma或开图。

先在原启动终端按Ctrl+C停止自己的服务，退出后更新SGLang同步分支并启动：

```bash
cd /home/tyj/glm52/sglang
git pull --ff-only
SGLANG_DSPARK_DEBUG_DUMP=core,reqs bash devtools/glm52_ms1/single_dspark_static.sh
```

这次kernel仓没有更新，不需重装包或重建容器。新增环境变量只对本次启动
生效，原启动脚本会继承它。诊断会增加记录/拷贝开销，不用于性能结论。

服务ready后，在**同一容器的另一个终端**执行：

```bash
cd /home/tyj/glm52/sglang
python3 devtools/glm52_ms1/collect_acceptance_trace.py
```

采集器先读取`/server_info`确认已有逐轮记录，再发一次原chat请求，最后导出
记录。请求使用唯一rid，仅增加返回元数据和token IDs的选项；无其他推理请求、
预热或清理cache操作。默认直连`http://61.47.19.71:8810`，忽略HTTP代理，
与此前curl的`--noproxy '*'`一致。需要指定其他已确认服务时用`--url`。

请回传终端摘要。完整的request、response、server_info前后快照、逐轮trace和
summary保存在打印出的`/home/tyj/glm52-ms1/evidence/acceptance-trace-*/`中，
不写Git仓库。采集器的Git SHA不是运行服务源码证明，仍保留本次启动日志及
已有kernel overlay的来源记录。

| 结果 | 含义与下一步 |
|---|---|
| `TRACE_COLLECTED` | 逐轮计数及输出token与本请求核对一致，已取得定位材料；不是模型质量PASS。看接受序列，从头低优先查配对/加载/hidden/proposal，后期下降优先查首次下降前后的KV/位置/commit；两类都保留target verify调查 |
| `TRACE_REVIEW_REQUIRED` | 计数、token、前缀连续性或请求生命周期有待核查。保留全部数据，先确认记录归属与是否发生retraction，不直接断言模型或KV错误 |
| `COLLECTION_FAILED` | 请求或取证出口失败，回传摘要；已有JSON仍保留。若服务启用记录后启动失败，回传首个traceback，属于取证入口验证，尚不能归因原低接受率 |

当前overlap可能多记录已经结束请求的后续worker轮。工具以API计数的前N轮为
候选，核对A/P/N、histogram、输出token及长度连续性，其余行单独保留；有
retraction时不自动判定候选归属。EOS/长度上限还可能裁掉末轮输出后缀，不能
用`completion_tokens == 1 + sum(acc_len)`当硬判据，也不能为凑输出长度扣减A。
原始计数用于接受率，临时buffer无效尾部不当成用户输出。

源码关联：`dspark_observability.py`的`ReqDetail`、`DsparkInfoDumper.dump`与
`scheduler.get_internal_state`提供记录；新增工具只处理HTTP JSON，不导入
SGLang/Torch/NPU。对应学习手册27.4的anchor、`gamma+1`与下一轮位置关系；
具体本地源码和学习链接在项目交付记录中。本轮暂不采集hidden/logits/cache内容，
有序记录仍不足时，再按首个差异对齐下一轮诊断范围。

CPU采集器检查（不启动模型）：

```bash
python3 devtools/glm52_ms1/test_collect_acceptance_trace.py -v
```

## 本轮：单机 DSpark static 首请求

本轮从192单算子测试进入整模型联调：检查真实草稿加载、上下文KV注入、
proposal和target验证能否连起来。性能后补，不作为启动本轮的前提。
本机CPU配置/加载回归不能代替下面的NPU执行；旧71项kernel结果也不能
替代最终简化入口的47项复测。

在已有0904容器里使用两仓的 `sync/glm52-dspark-ms1` 分支。停止自己正在
占用整机的服务后，逐行更新并启动（不重建容器、不下载模型）：

```bash
cd /home/tyj/glm52/sgl-kernel-npu
git pull --ff-only
git log -1 --oneline
cd /home/tyj/glm52/sglang
git pull --ff-only
git log -1 --oneline
bash devtools/glm52_ms1/single_dspark_static.sh
```

kernel本轮使用 `ae0fd2cf4498e5c23d644a2854dc377fd87a3ff7`。
`--ff-only`表示只接受可以直接前进的更新，遇到本地分叉就停止。
脚本根据自身位置选择SGLang源码，kernel默认是旁边的`sgl-kernel-npu`目录。
默认host为`61.47.19.71`。负责人确认当前容器仍使用旧挂载：宿主
`/home/weights`映射为容器`/workspace/weight`，因此target默认路径为
`/workspace/weight/GLM-5.2-w8a8`，draft为
`/workspace/weight/GLM-5.2-DSpark-NPU-0805`。本次沿用该容器，不需要重建。
这也解释了首轮启动的`Repo id must be...`：当时脚本使用了该容器看不到的
`/home/weights/GLM-5.2-w8a8`。修正路径后仍需继续验证模型加载和真实请求。

下方`start_container.sh`采用整个`/home:/home`的新挂载方案；仅通过Git更新
脚本不会改变已建容器的挂载。如果以后切换到新建容器，可以使用现有参数覆盖：

```bash
TARGET_MODEL=/home/weights/GLM-5.2-w8a8 \
DRAFT_MODEL=/home/weights/GLM-5.2-DSpark-NPU-0805 \
bash devtools/glm52_ms1/single_dspark_static.sh
```

配方沿用同事的TP16、DP1、DeepEP auto、prefill、显存和请求上限；移除NEXTN
配置，加入DSpark static、block8/window9、draft unquant和ascend Attention。
第一步使用`--disable-cuda-graph`便于定位普通执行问题。它只是本次诊断配方，
没有新增框架层的graph、DP、PP或PD限制，也没有修改宿主CPU调频/sysctl。
如需先看实际命令，可运行：

```bash
bash devtools/glm52_ms1/single_dspark_static.sh --print-command
```

`with_kernel_checkout.py`为本次服务新建
`/home/tyj/glm52-ms1/kernel-overlay-*/`。其中冻结一份候选Python入口，
其余包文件与`.so`链接到镜像安装位置；仅本次命令的`PYTHONPATH`优先使用它，
spawn子进程继承相同路径。不会覆盖site-packages、重编译kernel或替换DeepEP。
`manifest.json`记录kernel Git状态、源码哈希、命令和路径，`import.json`
记录新解释器的实际导入位置。包版本仍属于镜像，候选代码身份看Git和源码哈希。
其他文件是链接，运行期间保持镜像依赖不变。这是临时联调方式，正式wheel安装
验证仍待后续完成。

等待服务ready后，在另一个已进入容器的终端发一次greedy请求：

```bash
curl -sS --noproxy '*' http://61.47.19.71:8810/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.2-w8a8","messages":[{"role":"user","content":"请用三句话解释为什么天空看起来是蓝色的。"}],"temperature":0,"max_tokens":64,"stream":false,"return_spec_tokens_details":true}'
```

请回传启动日志、两仓`git log`结果和完整JSON响应。关注响应
`sglext.spec_tokens_details.spec_verify_ct`：大于0才有本请求执行投机验证的
证据，尽量覆盖至少两轮。它与接受率一起用于排查；单次接受率不作性能或质量
验收。如果只ready还没有真实请求，或响应没有投机统计，不能直接记DSpark通过。
导入预检失败则回传对应目录的`manifest.json`和`import.stderr.txt`；模型失败
则保留首个完整traceback，先按真实报错定位，再决定改动位置。

eager首请求通过后再进行整图验证；下列开关已备好，本轮不用同时跑多个服务：

```bash
GRAPH=1 bash devtools/glm52_ms1/single_dspark_static.sh
```

后补target-only对照使用同脚本的`MODE=target-only`，并保持两边`GRAPH`值一致。
`GRAPH=1`恢复同事的`--cuda-graph-bs 16`；不同图模式的结果不能直接归因于DSpark。
完整推理、图回放、请求边界、量化模型质量、DPA多组、双机与PD仍需分别验证。

临时工具的CPU检查：

```bash
python3 devtools/glm52_ms1/test_with_kernel_checkout.py -v
python3 devtools/glm52_ms1/test_single_dspark_static.py -v
```

## Native 192 experiment (single NPU)

`probe_native192.py` tests the installed `sgl_kernel_npu` fused kernel with real
192-element blocks. `native192_probe_ops.py` contains three small compiler probes.
Neither file is imported by SGLang. The tool does not modify the installed host
function, install packages, change compiler defaults, start distributed workers,
or load a model. Its baseline is the agreed Triton-Ascend **3.2.2** image
(`triton.__version__` may correctly report **3.2.0**).

In the existing container, use the same initialized CANN/ATB environment as the
working target-only service. Select an available logical NPU; if your target
service occupies the whole machine, stop your service first. From the checkout:

```bash
cd /home/tyj/glm52/sglang
python3 devtools/glm52_ms1/probe_native192.py
```

The default is logical NPU 0. Use `--device 1` (or another available index) to
change it. These are indices in the process's visible device list. The first
run includes JIT compilation and can take several minutes. No second machine is
needed. Results are saved automatically under
`/home/tyj/glm52-ms1/evidence/native192-<UTC-time>-<pid>/`, outside Git.

The first command runs these checks in order and stops at the first failure:

1. An existing **128-dimensional public-host call** as an environment control.
2. Actual-length 192 load/store, reshape/reduction, and 96+96 slice/rotation.
3. The installed full fusion, with `Q_BLOCK_SIZE=KV_BLOCK_SIZE=192` and the
   original grid formula. It uses BF16 Q/K/V and norm weights, FP32 sin/cos,
   full NeoX RoPE, no bias, and epsilon `1e-5`. Nine token/head combinations,
   a guaranteed second row-loop iteration, zero and near-zero input are checked.

The script calls the underlying kernel directly for the experiment. The public
host's current power-of-two assertion stays intact. Q/K are compared with an
independent CPU FP32 formula without intermediate BF16 rounding; V must be
bitwise identical. The diagnostic absolute tolerance `5e-2` comes from the
existing kernel's test, **not** a new GLM model-quality acceptance threshold.
Per-output error metrics are saved for review.

After the first command passes, optionally run graph validation and diagnostic
timing. This reruns the eager checks before capturing one `[8, 4, 192]` case:

```bash
python3 devtools/glm52_ms1/probe_native192.py --graph --benchmark
```

Graph checks use one side stream, stable tensor addresses, and two changed-input
replays, poisoning output buffers before each replay. Capture uses
`auto_dispatch_capture=True`, as in the current SGLang NPU graph backend. Event
timings exclude JIT/capture, but include gaps on the measured stream; they are
diagnostic measurements, not an end-to-end speedup or performance acceptance.

Result interpretation:

- `EAGER_DIAGNOSTIC_PASS_GRAPH_NOT_RUN`: eager checks passed; graph was not run.
- `EAGER_AND_GRAPH_DIAGNOSTIC_PASS`: eager plus this single-operator graph passed.
- `FAILED`: inspect `active_stage`, the metrics, and `traceback.txt`.
- `RUNNING` left behind after a crash/interruption: incomplete evidence, never a pass.

Return `report.json`; on failure also return `traceback.txt` and the terminal
error. `run.log`, the installed kernel source, and the actual `arange` source are
saved alongside them. Versions, selected-device properties, compiler metadata
and source hashes identify what actually ran. A source different from the local
audit is recorded, not mislabeled as the audited commit. No automatic fallback
to 2x128 is performed: first identify whether a failure is environmental,
compiler-related, numerical, or specific to graph execution.

New reports record `kernel_function_ast_schema` with the AST fingerprint. This
schema ignores source locations and empty `type_params` fields added in Python
3.12; nonempty type parameters and calculation changes remain significant.
`kernel_function_source_sha256` separately hashes the saved UTF-8 source after
dedenting, normalizing line endings to LF and keeping one final newline, so
comments remain available for audit. Older reports without the schema used
Python-version-dependent AST dumps; their hashes must not be compared directly
with new ones. Existing reports are not rewritten.

Local tool tests (CPU PyTorch; no Triton/NPU installation required):

```bash
python3 devtools/glm52_ms1/test_probe_native192.py -v
```

These tests do not count as NPU evidence. This experiment does not establish
SGLang DSpark, actual TP/DP execution, A5 support, or milestone completion. The
temporary tools stay on the sync branch and are not promoted as feature code.

## Existing container and target-only tools

The following `/home:/home` recipe is for a future container migration. The
current DSpark run uses the existing container's `/workspace/weight` mount as
described above; do not recreate it for this path fix.

`start_container.sh` contains the team's A3 Docker command with each device
listed explicitly. Run it on the host with Bash; it only creates an interactive
container, without model loading or preflight. For the second node, set
`CONTAINER_NAME=tyj-glm52-ms1-target-node1` when invoking it. The image and host
mounts are visible in the script; after entering the container, change directory
to `/home/tyj/glm52/sglang` before using the commands below.

The script mounts the host's entire `/home` at `/home`, following the team's
container command. Host and container paths are identical: the target is at
`/home/weights/GLM-5.2-w8a8`, and personal configuration, cache, and logs remain
under `/home/tyj/glm52-ms1`. The separate `/workspace/weight` alias is no longer
created. Driver, firmware, and the other system mounts are unchanged.

An existing container retains the mounts selected when it was created; pulling
this script or restarting that container does not update them. Only when you
choose to migrate to `/home:/home`, create a new container on the host:

```bash
cd /home/tyj/glm52/sglang
CONTAINER_NAME=tyj-glm52-ms1-home-node0 bash devtools/glm52_ms1/start_container.sh
```

Use this new name with `docker exec` and `docker inspect`. In an existing local
configuration, change `model_path` to `/home/weights/GLM-5.2-w8a8` and update
`source_commit` to the reviewed commit being run. Copying the example with
`cp -n` will not update an existing configuration. Files already stored under
the host-mounted `/home/tyj` remain available in the new container; files saved
only inside the old container remain there.

The helper provides environment preflight, a launch command preview, foreground
launch, and three short raw generation requests. It does not install dependencies
or download model weights. Use an existing Ascend environment and local weights.

## Configure a run

Copy `target-only.example.json` to a directory outside the checkout. Set the
absolute `source_dir`, the exact full Git `source_commit` selected for testing,
the container-visible model path, image reference, rendezvous address, and
network interfaces. Run both nodes at the same reviewed commit. Git mode rejects
source changes instead of silently testing a different production tree.

The initial recipe is two nodes with TP32/DP8, ModelSlim target weights, BF16
computation, and prefill/decode graphs disabled. Each node needs 16 visible NPU
logical devices. This is a diagnostic configuration, not a performance result.
Coordinate device and port use before starting a model on a shared server.
The container itself can be used on one node, but this helper's launch recipe
still requires two nodes. The mount update does not add a single-node model
launch recipe.

```bash
python devtools/glm52_ms1/ms1_target_only.py preflight \
  --config /absolute/path/target-only.local.json --node-rank 0 \
  --out /absolute/path/evidence/node0-preflight-01

python devtools/glm52_ms1/ms1_target_only.py command \
  --config /absolute/path/target-only.local.json --node-rank 0 \
  --out /absolute/path/evidence/node0-command-01

python devtools/glm52_ms1/ms1_target_only.py launch \
  --config /absolute/path/target-only.local.json --node-rank 0 \
  --out /absolute/path/evidence/node0-launch-01
```

Use rank 1 and separate evidence directories on the second node. `launch` sets
the source import path and offline environment, repeats preflight, then runs the
server in the foreground. Use Ctrl-C to stop this launch. Command preview alone
does not set the runtime environment in another shell.

When node 0 is ready, run in another terminal in the same environment:

```bash
python devtools/glm52_ms1/ms1_target_only.py smoke \
  --config /absolute/path/target-only.local.json \
  --launch-evidence /absolute/path/evidence/node0-launch-01 \
  --base-url http://127.0.0.1:30000 \
  --out /absolute/path/evidence/node0-smoke-01
```

`SMOKE_ONLY_PASS` means that the three basic requests returned nonempty text with
normal completion metadata. It does not establish task accuracy, DSpark support,
performance, or milestone completion. Keep full configuration and logs in the
execution environment, and correlate the checked endpoint with the launched
process.

Stop the running model before pulling another commit into the same checkout.
After `git pull --ff-only`, verify the delivered SHA, update `source_commit` in
the local configuration, and use fresh evidence directories. Do not commit local
configuration, proxy credentials, caches, or run logs.

## Test the helper

```bash
python devtools/glm52_ms1/test_ms1_target_only.py -v
```

These tests exercise the helper and real temporary Git repositories. Model/NPU
probe and HTTP test doubles do not count as actual model or device validation.
