# Temporary MS1 validation tools

This directory supports the temporary `sync/glm52-dspark-ms1` development branch.
It is separate from the SGLang feature commits intended for upstream review.

## 当前轮：固定 token 前缀，核对 Target prefill 与历史 verify

**2026-09-09实机更新：首次请求触发输入logprob融合算子编译崩溃。**
`row_logsumexp_topk → bishengir-compile`在Ascend910_9362上SIGSEGV，
服务随后退出。本次没有取得prefill/verify对照结果。之前“现有服务无需重启”
只适用于未发生故障的准备状态；本次需恢复服务。

使用社区已有开关关闭输入logprob的快速实现，保留log_softmax/top-k备用计算；
在**服务端终端**运行以下启动命令，原已成功配置的其他参数保持：

```bash
cd /home/tyj/glm52/sglang
SGLANG_ENABLE_FAST_INPUT_LOGPROBS=0 \
SGLANG_NPU_GLM_DSPARK_QUAROT=original \
SGLANG_DSPARK_DEBUG_DUMP=core,reqs \
bash devtools/glm52_ms1/single_dspark_static.sh
```

等待服务ready后，再在另一个终端运行下方客户端命令。该开关在服务构造
InputLogprobProcessor时读取，只在客户端设置无效；本轮不改运行源码、权重、
kernel或默认启动脚本。输入分数计算有额外开销，不拿恢复后的诊断延迟作性能
结论。现有原请求/失败Evidence保留，新执行会写新的Evidence目录。

已有12请求总接受率为27.72%。这一轮继续定位原因：把同一轮的完整前缀、
anchor和draft候选固定下来，看Target重新prefill得到的预测与历史verify决定
是否一致。它不是提接受率的新补丁，也不是独立target-only精度或压测。

从已保存的天空题中选择首轮、首次零接受和首次全接受，重合的轮只选一次；
每轮复算两次，最多6条串行HTTP请求，每条只生成1个token。不重分词、不套新
聊天模板，不新增题目。给每次请求独立cache_salt并检查cached_tokens=0，避免
复用旧KV；不清全局缓存。当前DSPARK服务的prefill仍会注入draft hidden，
overlap可能执行额外未结算工作，不能理解成完全不执行draft。

在**当前服务所在容器的另一个终端**执行，原服务保持运行：

```bash
cd /home/tyj/glm52/sglang
git pull --ff-only
python3 devtools/glm52_ms1/probe_verify_prefill.py --source /home/tyj/glm52-ms1/evidence/acceptance-suite-20260909T063207Z-6e53c35a/01-sky_baseline-r1 --run
```

不带`--run`时只读取旧证据、准备`plan.json`，不联系服务。不要将`--source`
指向套件总目录，需要指向上述单请求子目录。服务按上方恢复后，运行客户端
无需安装依赖或再次重启模型；恢复开关本身不需要更新代码。
脚本在model采样默认模式下只读当前target的`generation_config.json`，确认
没有repetition penalty等干扰；文件和历史加载版本未变仍是比较前提。

本轮新增工具入口是[probe_verify_prefill.py](probe_verify_prefill.py)，CPU测试
在[test_probe_verify_prefill.py](test_probe_verify_prefill.py)。仅这两个临时文件
及本README发生变化；SGLang运行代码和kernel仓无改动。

旧`core,reqs`记录没有完整verify logits，只能知道**已接受前缀及bonus**的预测。
脚本只比较这些已知位置，其余标记unknown；同时记录top-5和前两名分差，
分数并列单独展示。没有自定精度容差，也不输出模型正确性PASS。

| 本轮结果 | 下一步 |
|---|---|
| 同前缀的两次prefill自身不同 | 先定位Target/执行稳定性及分数并列，暂不归罪draft |
| prefill重复稳定，但已知位置与历史verify不同 | 查Target prefill/verify计算、位置、KV和数值差异；不同本身尚不能断言错误 |
| 已知位置一致 | 缩小这些位置的verify疑点，下一轮采集真实hidden并核对draft输入/上下文KV；不能宣告整个verify正确 |
| 采集失败或中断 | 保留当前Evidence，先看异常；不自动重试或继续剩余请求 |

请回传新Evidence目录里的`report.json`。`COMPARISON_COLLECTED`只表示采集完成；
超时或中断也不保证服务端已停止处理。原始请求、响应、逐项比较、配置及文件
指纹都留在同一目录，必要时再取对应个案，无需先上传全部历史日志。

源码定位、实际diff与本地验证见[本轮交付](/Users/yuejiat/workspace/model-inference/glm52-dspark-npu-project/reviews/2026-09-09-verify-prefill/README.md)。
学习对应[第18章：规划、打包、验证和状态提交](/Users/yuejiat/workspace/model-inference/glm52-dspark-npu-project/learning/glm52-dspark-complete-guide.md:3552)，重点理解anchor、候选与bonus为何错开一位。

## Previous：固定多输入、较长输出的接受率诊断

QuaRot加载日志及原64-token请求已核对：16rank完成FC转换，草稿使用自有
embedding/head，接受率由2.36%改善至28.95%。本轮保持当前服务，检查改善
是否跨内容稳定，以及回答是否完整；不是正式业务数据集或性能压测。

固定用例在 `acceptance-cases.json`：天空问题（原提示词）、中文内存/硬盘说明、
英文DNS说明、Python去重函数、规则库存算术、记录摘要。先按此顺序执行
一轮，再以相同顺序执行第二轮，共12条串行请求。每条 `temperature=0`、
`max_tokens=512`，允许自然结束；没有增加system消息或thinking开关。
512是本轮诊断预算，不保证模型一定答完；不按结果重跑或挑选样本。

**当前服务继续运行。** 本轮更新仅在devtools，另一个容器终端执行：

```bash
cd /home/tyj/glm52/sglang
git pull --ff-only
python3 devtools/glm52_ms1/collect_acceptance_suite.py
```

该命令发HTTP请求，不启动模型、不改服务配置、不重建容器。沿用当前
static/eager、TP16/DP1、gamma8/verify9、original与core,reqs。采集前后核对
固定用例文件里的服务配置，避免请求误发到不同配置却被当作本轮结果。
模型/kernel代码不变，服务无需重启；客户端的git SHA不冒充运行服务的SHA。

每条会输出A/P/N、接受率和结束原因，全部可信完成后才输出
`SUITE_COLLECTED` 和 `Aggregate`。整体使用 `sum(A)/sum(P)`，不平均各请求
百分比；其中严格>0.5只是对用户参考门槛的观察，不能据开发小样本宣布转测通过。

所有证据保存在打印的 `Evidence` 目录：

- `suite-plan.json`、`collector.json`：冻结用例、参数、客户端来源和文件指纹。
- 每请求子目录：原始请求/响应、前后server_info、逐轮trace、summary和套件记录。
- `suite-summary.json`：整体与逐条结果、未执行请求、重复输出token是否一致、缓存数。
- `responses.md`：按原顺序排列完整回答及人工审视要点；不会执行模型生成的代码。

两轮之间不清缓存或统计；重复请求可命中缓存，缓存数单独记录。逐轮数据按
独立rid筛选，旧请求不累入本次A/P/N。server_info导出包含保留历史并可能等待
数据拷贝完成，有采集开销，本轮延时不作为性能基线。

若超时、Ctrl+C、服务配置变化或记录核对失败，工具保留已获得的证据，停止
后续请求，不自动重试。`SUITE_INCOMPLETE` 不提供整套接受率，不能只拿已成功
的部分宣布通过；无投机轮时标记 `NO_SPECULATIVE_ROUNDS`，不记成0%或模型错误。
HTTP超时/中断不保证服务端请求已结束，先保留现场，不立刻重新执行整套。

`length`表示用完输出预算，保留回答待人工判断完整性；`stop`也不自动代表内容
正确。`Output token match`仍仅是同一DSpark请求内部记录核对，不是target-only对拍。

请回传终端汇总、`suite-summary.json` 和 `responses.md`。若有异常，保留对应
请求子目录，先判断是记录/服务问题，还是模型输出问题；不改变block或统计分母。
若多个输入均稳定改善，继续多输入正确性和后续部署验证；若普遍偏低，下一步
审视真实hidden→FC/hidden_norm及草稿/verify对照方案；若仅某类输入偏低，先查
任务配对与具体输出；不能仅凭本轮结果盲改192、Q尺度或epsilon。

实际diff和验证范围见[本轮交付](/Users/yuejiat/workspace/model-inference/glm52-dspark-npu-project/reviews/2026-09-09-acceptance-suite/README.md)。

## Previous：GLM NPU QuaRot 加载候选与真实请求

本轮把前两次诊断支持的候选接入运行代码：草稿加载自身的 embedding/head，
两个 embedding 入口都使用它们；原始 FC 在加载时以 CPU FP32 分块右乘
target 的 Q，再回到原参数 dtype（当前 BF16）。推理仍走原 FC→hidden_norm、
192 融合和 proposal/verify/commit，不新增每轮旋转。

新模式默认关闭。`original` 是你对草稿制品的明确声明：这是一份未经本次
转换的原始草稿，不是自动识别 checkpoint 坐标。只为 GLM DSA 的 ModelSlim
QuaRot target 与对应 NPU dense DSpark 路径启用。当前已核对的0805草稿可用于
这轮候选；不要把已转换过的制品再次声明为 `original`。

本地自测和实际 diff 审视后，先停止当前测试服务，在原容器中执行：

```bash
cd /home/tyj/glm52/sglang
git pull --ff-only
SGLANG_NPU_GLM_DSPARK_QUAROT=original SGLANG_DSPARK_DEBUG_DUMP=core,reqs bash devtools/glm52_ms1/single_dspark_static.sh
```

这是服务启动，不是重复上一轮 CPU probe。模型路径仍沿用当前容器的
`/workspace/weight`，镜像、CANN、kernel checkout 和单机并行参数沿用原脚本。
临时前缀只作用于这次命令。无需修改共享权重目录、重建容器或重装算子包。

启动日志应同时出现 `GLM DSpark original mode: folding FC`（含 Q/target 路径）、
`GLM DSpark FC load-time Q folding finished`（耗时）及
`DSpark draft uses its checkpoint-local embedding and LM head.`。
初次加载多了 CPU 转换工作，各rank可能重复执行；目前没有加载耗时或性能
达标结论。转换完成后释放临时矩阵，新增常驻词表由随后的 KV 预算计入。

服务就绪后运行 `python3 devtools/glm52_ms1/collect_acceptance_trace.py`；它会
发送与上一轮相同的请求，无需先手工重复发送。保留
这次启动日志、实际 git HEAD、server_info 和 trace 输出。需要观察首个草稿
候选的拒绝是否缓解，并按同一 A/P/N 口径与原 `10/424/53` 比较；不改请求来
选择性展示高接受率。上面的启动命令已保留既有 `core,reqs` 记录设置。

本轮的 `F_i @ Q` 是有已知尺度残差的实机候选，不是 Q 的精确逆。单个请求
改善不能代替实际 NPU 数值、质量/并发/部署回归、性能及压测接受率>0.5的
准出验证。不要因本地单测通过就记为 MS1 已完成。

关闭新模式并重启原脚本即回到之前的共享词表行为；该回退也会回到原来的
低接受率候选，不代表问题已解决。源权重未被写回。
checkpoint 重载传入的仍应是原始参数；绕过模型 loader 的 direct tensor 更新
属于运行参数接口，不能期待它自动完成此转换。

对应源码阶段为 DSpark 模型加载及 target 特征进入草稿 KV；当前实现和逐文件
审视见[本轮交付](/Users/yuejiat/workspace/model-inference/glm52-dspark-npu-project/reviews/2026-09-09-quarot-runtime/README.md)，
学习入口为[32.12～32.16](/Users/yuejiat/workspace/model-inference/glm52-dspark-npu-project/learning/glm52-dspark-complete-guide.md:5832)。

## Previous：加载适配设计与固定输入 FC 对照（仅 CPU）

这轮诊断及完整报告已经收到；下面保留原方法，不要求重复执行。

上一轮已经收到词表取样结果：embedding 乘 Q 后接近 target，head 还需要
对应的 norm 处理；Q 往返探针则出现约 0.296% 的幅值收缩。这轮检查草稿
接收五层 target hidden 的另一条接口，不能仅凭词表相似就开始修改 FC。

三项工作的关系：自有 embedding/head 与两条 embedding 调用入口是一组；
五层 hidden→FC 的坐标适配是另一组，可并行设计，最终一起进入真实请求验证。
本轮工具不改变共享模块、FC、proposal、target 或 kernel 的运行行为。

在当前容器的 SGLang 同步分支更新本轮仅含工具/测试/本说明的提交后，服务可
保持运行，不需要重装包或重启容器。在仓库目录执行：

```bash
python3 devtools/glm52_ms1/probe_quarot_fc.py
```

默认沿用 `/workspace/weight/GLM-5.2-w8a8` 和
`/workspace/weight/GLM-5.2-DSpark-NPU-0805`，使用 NumPy 和一个 CPU 线程。
读取完整 Q（144MiB）、draft FC（BF16 360MiB，解码后 F32 720MiB）及
hidden_norm（12KiB）。不读完整模型，内存预计约 1～2GiB 加解释器/BLAS
开销；线程限制不意味着不消耗共享 CPU 和内存带宽。

工具生成固定 seed 的三行合成输入，幅值分别为 1、0.01、0.0001，用于区分
普通幅值和 epsilon 影响明显的情形。它们不是捕获的 target hidden。

| 报告内容 | 实际验证范围 |
|---|---|
| `unchanged_fc_pre/post_norm` | 将合成 H 变换为 H×Q 后直接使用原 FC，分别在归一化前后与原 H 的输出比较 |
| `q_fold_equivalent_pre/post_norm` | 用 `(H×Q)×Q.T` 经原 FC 计算完整输出，检查加载时 FC 右乘 Q 的数学候选和真实 hidden_norm 权重；这通过改变乘法顺序计算，**没有转换全部 FC 权重** |
| `sampled_weights` | 默认实际转换 16 个 FC 输出行的全部输入列，分别按 FP32 和模拟 BF16 参数存储比较投影；矩阵乘法仍为 FP32，不是 NPU 精度验证 |
| `q_structure` / `q_roundtrip` | 全部 Q 行的范数/元素绝对值范围及固定输入的往返误差；不构造完整 Q×Q.T，不拟合补偿系数或求逆 |

只转换少量输出行，可避免五次 6144 维完整矩阵乘法。所选行的结果也不冒充
完整转换后的 FC→hidden_norm 验证；真实权重全量转换、NPU 算术和实际 target
特征需要在后续候选实现里验证。报告没有自定义精度通过阈值。

输出目录为 `/home/tyj/glm52-ms1/evidence/quarot-fc-<UTC>-<uuid>/`。
请保留并回传终端结果和 `report.json`。`FIXED_INPUT_DIAGNOSTIC_COLLECTED`
仅表示计算完成；`NUMERICAL_REVIEW_REQUIRED` 表示非有限值或零范数等需核查；
`FAILED` 表示输入/工具错误。任何状态都不代表接受率或性能通过。

若 Q 候选明显改善且归一化后保留误差可解释，再评审加载适配；若方向、尺度或
采样 BF16 转换不符，先解决该项数值差异，不用重启服务盲试补偿。
可选参数：`--target`、`--draft`、`--out`、`--threads`、`--weight-rows`
（默认16，工具上限64行）。固定输入不受模型请求和随机采样参数影响。

源码阶段：[已有 FC→hidden_norm](/Users/yuejiat/workspace/model-inference/worktrees/sglang-glm52-dspark-ms1-sync/python/sglang/srt/models/dflash.py:662)
随后用于草稿上下文 KV；[外部 embedding 选择](/Users/yuejiat/workspace/model-inference/worktrees/sglang-glm52-dspark-ms1-sync/python/sglang/srt/speculative/dspark_components/dspark_draft.py:253)
属于另一项调用覆盖。学习对应[32.15 加载与固定输入验证](/Users/yuejiat/workspace/model-inference/glm52-dspark-npu-project/learning/glm52-dspark-complete-guide.md:5888)
与[32.16 方案及代价](/Users/yuejiat/workspace/model-inference/glm52-dspark-npu-project/learning/glm52-dspark-complete-guide.md:5917)。

```bash
python3 devtools/glm52_ms1/test_probe_quarot_fc.py -v
```

## Previous diagnosis: sampled QuaRot vocabulary weights (CPU only)

The ordered trace shows low acceptance from the start. The target export records
QuaRot and provides `global_rotation` as F32 `[6144, 6144]`; its separate
`rot.weight` is BF16 with the same shape. These headers identify the inputs, but
do not prove that the target and draft weights use compatible coordinates.
This step compares a few embedding/head rows on disk. Unlike the earlier trace
collector, it does not send requests or inspect the running service.

This delivery changes only `probe_quarot_vocab.py`, its CPU test, and this README.
After fetching the delivered commit, inspect its file list with
`git show --stat <delivered-commit>` before updating the checkout. If the update
contains only these three files, keep the service running: the stop/restart
instructions in the older sections below do not apply to this tool-only update.
Updates containing runtime or launch changes need their own restart instructions.

In the current container, from `/home/tyj/glm52/sglang`, run:

```bash
python3 devtools/glm52_ms1/probe_quarot_vocab.py
```

Defaults are target `/workspace/weight/GLM-5.2-w8a8`, draft
`/workspace/weight/GLM-5.2-DSpark-NPU-0805`, 32 evenly spaced token IDs plus valid
mask/special IDs from the configs (deduplicated), and one CPU thread.
The tool uses the Python standard library and NumPy, without importing Torch,
using an NPU, installing dependencies, or accessing the network. It reads only
selected vocabulary rows, the full Q (144 MiB), and the full R (72 MiB, converted
to F32 for calculation). Expect several hundred MiB plus Python/BLAS overhead;
the CPU thread limit does not eliminate shared CPU or memory-bandwidth usage.
It does not modify model files, runtime/kernel code, or launch parameters.

Optional arguments are `--target`, `--draft`, `--out` (evidence parent directory),
`--samples 32`, `--token-ids '0,1,154856'`, and `--threads 1`. Each run creates a
new `quarot-vocab-<UTC-time>-<uuid>/` directory below
`/home/tyj/glm52-ms1/evidence/` by default. Return the terminal summary and
`report.json`; retain the complete evidence directory on the server.
Explicit token IDs replace the automatic selection. This diagnostic limits the
total selection to 256 unique IDs to keep vocabulary reads bounded.

For matching token rows, the comparisons are:

| Boundary | Candidates compared with the target row |
|---|---|
| Embedding | Original draft row; `E_draft @ Q` |
| LM head | Original draft row; `W_draft @ Q`; `(W_draft @ Q) @ R` |

The last candidate tests the export contract `R = Q.T @ D_gamma @ Q`, where
`D_gamma` contains the original target final-norm scales. It is not an inverse
rotation. This round requires both Q and R, whose presence was already confirmed
in the current target directory. Since R is stored in BF16, this candidate has additional rounding
error compared with an export computed from higher-precision inputs. No inverse,
full gamma reconstruction, fitted correction, or cosine pass threshold is used.

| Result | Meaning and next decision |
|---|---|
| `COORDINATE_COMPARISON_COLLECTED` | Calculations completed, not a correctness PASS. Review per-row errors across the predefined candidates. A consistent coordinate relation supports investigating that contract; if none explains the rows, check export provenance and weight pairing before changing inference. |
| `NUMERICAL_REVIEW_REQUIRED` | A numerical anomaly, such as a non-finite value, needs review. Keep the evidence and resolve it before interpreting coordinate comparisons. |
| `FAILED` | An input or tool operation failed. Review the reported error first; this is not evidence that the model itself is incorrect. |

Sampled agreement does not verify every weight, the draft FC, actual target
hidden states, model quality, acceptance, or performance. In particular, the
five captured target residuals are a different interface from NEXTN's final
hidden state; neither this test nor a matching head authorizes applying R to
those five features. The existing 192-dimensional fusion stays unchanged.

Source context: dense DSpark shares target vocabulary modules in
[`attach_shared_modules`](/Users/yuejiat/workspace/model-inference/worktrees/sglang-glm52-dspark-ms1-sync/python/sglang/srt/models/dspark.py:514),
while its target features enter
[`project_target_hidden`](/Users/yuejiat/workspace/model-inference/worktrees/sglang-glm52-dspark-ms1-sync/python/sglang/srt/models/dflash.py:662).
Learning sections
[26.4: checkpoint/loader contract](/Users/yuejiat/workspace/model-inference/glm52-dspark-npu-project/learning/glm52-dspark-complete-guide.md:4443)
and [27.3: hidden-state contract](/Users/yuejiat/workspace/model-inference/glm52-dspark-npu-project/learning/glm52-dspark-complete-guide.md:4583)
explain the two boundaries; the guide is a conceptual reference from an older
code snapshot, not proof of this export's transformations.

CPU tool checks (no model startup):

```bash
python3 devtools/glm52_ms1/test_probe_quarot_vocab.py -v
```

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
