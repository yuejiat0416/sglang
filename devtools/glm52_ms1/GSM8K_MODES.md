# 历史协议附录：GSM8K五模式（1024预算）

**新机器准备和两个服务启动脚本统一见[启动手册](README.md)。本文件只保留旧工具协议，复现历史成绩时参考；不再维护独立操作步骤。**

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
