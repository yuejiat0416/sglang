# 协议附录与工具索引

**完整操作统一看[端到端手册](../README.md#step9)。本轮测试运行本目录run_tests.py，顶部直接改MODE/地址；无需JSON配置。**
本文件不再维护另一套配置/启动步骤。本目录保留已有双机TP32测试客户端；launch.py与config.example.json属于旧启动方式，不再要求负责人用它们启动服务。

## 各阶段收集什么

| 阶段 | 输入与计数 | 结论边界 |
|---|---|---|
| smoke | 1道GSM、64输出预算 | 连通与计数，不是精度 |
| accuracy | GSM8K与GPQA分别10题、串行、最多4096输出、EOS有效 | 同题同提示与target-only配对，不是正式全量精度 |
| prefix / quick | 三档各1条131072/1024，独立预热 | 真实长度、缓存、A/P成立才算场景有效 |
| load | 三档各64条、并发8；可显式设最少持续时间 | 协议须冻结，inf不证明硬件满载 |

最终答案格式：GSM要求明确数字，GPQA要求Answer: A—D。gold仅保留在客户端；未解析、截断或失败保留在分母，不把推理中间数字当最终答案。10题不能证明1%精度波动或整模型无回退。此评分不是EvalScope正式协议的等价替代。

GSP复用社区生成器，将重新分词后的输入定长为token ID；热档预热每个使用的DP组，正式请求回到该组，新suffix避免意外全命中。0档使用独立cache_salt。按缓存页向下取整，page128时90档为117888/131072=89.94140625%。每条核验真实长度、cached_tokens、DP落点、retraction、A/P/N。本目录客户端按双机DP配置严格核验实际rank；单机客户端在上层目录。

接受率是测量请求sum(A)/sum(P)，不含bonus、预热，不平均单题比例，不混三档，不重复相加TP复制日志。Target-only为N/A。流式请求保存原生TTFT/TPOT及完整测量窗；持续负载还可能受客户端数据准备影响。

## 容量、部署与图证据

- 本目录既有测试配置面向133120总长；请求长度必须以运行时实际cap为准。0/50/90命中都不能抵扣完整上下文容量。
- quick检查C1；load独立检查C8。DP8/C8按各组负载检查，缓存比例不能绕过容量检查。
- 环境检查保留源码SHA、配置、小文件指纹及分片大小，不校验全部权重字节；双节点一致不证明HCCL通信已通过。
- Target图metrics区分decode_cuda_graph与decode_none；公共接口未暴露Draft全链路replay，不能仅凭graph启动参数标全部上图。
- HiCache的DDR/SSD命中、95%/10倍、其他输入长度SLO及P/D、PP、CP等不属于本轮工具覆盖范围。该边界不限制框架已有能力。

## 已存在源码

以下是当前调用依据，不是本轮修改的框架代码：

- [DSpark配置检查](../../../python/sglang/srt/arg_groups/speculative_hook.py#L511)：DP>1要求DP lm_head。
- [KV容量与请求槽位](../../../python/sglang/srt/mem_cache/kv_cache_configurator.py#L2202)：全局请求上限按DP分配。
- [DP控制器](../../../python/sglang/srt/managers/data_parallel_controller.py#L753)：显式路由。
- [请求入口](../../../python/sglang/srt/managers/tokenizer_manager.py#L796)：DP1忽略显式rank0路由；[响应位置](../../../python/sglang/srt/managers/tokenizer_manager.py#L2376)保留实际DP信息。
- [社区benchmark](../../../python/sglang/benchmark/serving.py#L658)：原生发送、流式计时。
- [GSP生成](../../../python/sglang/benchmark/datasets/generated_shared_prefix.py)：合成输入来源。

服务脚本及主要文件用途在启动手册第8节。临时工具及本附录不进入正式PR；生产模型、FIA、192算子未在本轮改动。学习对应完整手册15.5“缓存接口”、28.1“接受计数”、28.7“精度协议”、28.11—28.14“长度/缓存/压力”；教材是概念/历史快照，不代替当前运行源码。

## 打包范围

build_archive.py仅按明确清单打包工具及所需helper、公开GSM10题和许可证，不打包完整SGLang/kernel、填好的JSON、权重、GPQA或实测证据。归档保留每文件SHA，去除本机UID/GID/时间元数据。

本次负责人要求启动手册和脚本保留实际个人路径、仓库和镜像；**文档已不是脱敏版本**。构建器不会自动清除文档正文的环境信息，不能把去除tar元数据称作整个包已脱敏。
