# GLM-5.2：从新机器到单双机启动、双机跑测与DeepEP预案

**2026-09-13当前执行：A3 68/70双机、TP32/DP4、DSpark static graph，gamma5、verify6，使用EvalScope 1.11.1跑GPQA-Diamond全部198题并采集接受计数及图证据。完整顺序见[9.6](#gpqa-full-current)。** 复用已准备的全量脚本，不运行GSM8K或GSP。以下68/69、十题/20题与GPQA暂停描述均为历史阶段；精度91.2%±1个百分点与接受率严格>0.5沿用此前自测参考，并非本轮实测结论。

此前精度安排：GSM8K取20题、GPQA-Diamond取20题，分别运行和评分，操作保留在[9.5](#accuracy-twenty)。各模式使用同样的20题和输出预算。该流程不是本轮全量EvalScope入口。

此前性能安排：quick/performance已改为原生bench_serving完整CLI流程，见[原生GSP quick](#capacity-first)和[原生GSP压测](#gsp-load-current)。原生GSP的输入长度和共享比例以实测为准，Accept length不能冒充A/P接受率；旧自定义GSP流程已停用于quick/performance。这里的性能安排不作为本轮GPQA执行前置。

**启动服务只用两个脚本，参数直接写在脚本开头。无需single.json、two.json、local.env或single.local.sh。**

- [单机完整启动脚本](single_dspark_static.sh)：在61.47.19.71运行。
- [双机混部完整启动脚本](two_node_dspark_static.sh)：在61.47.19.71、61.47.19.70各运行一次。

已有容器和模型时，直接跳到[单机启动](#step5)或[双机启动](#step6)。新机器按前四节准备。

已有容器、权重和两个仓库时，本轮直接从[9.6](#gpqa-full-current)开始。**服务复用服务器上现有双机.sh；GPQA全量测试用run_gpqa_diamond.py，参数在文件顶部。不需要手写JSON配置。** 旧two_node_colocated/run_tests.py继续用于此前的小样本和GSP场景。

2026-09-10本轮交付：四组精度抽样、五组长输入性能、结果对比与DeepEP离线构建预案。保留现有服务配方，不修改框架或算子。单机static eager已有实测；双机、整模型graph和128k实测结果仍由负责人在NPU执行后确认，以下不是已通过报告。

1. [执行位置与目录](#step1)
2. [拉代码](#step2)
3. [准备权重](#step3)
4. [拉镜像、创建容器](#step4)
5. [单机启动：改一行graph开关](#step5)
6. [双机启动：两端各运行同一脚本](#step6)
7. [检查服务、停止与更新](#step7)
8. [现有工具用途](#step8)
9. [本轮双机：准备数据和测试入口](#step9)
10. [精度对照：十题流程与全量GSM8K准备](#step10)
11. [128k/1k容量、三档quick与并发4压测](#step11)
12. [生成对比表、判断下一步](#step12)
13. [DeepEP预案：离线准备、编译、启用与回退](#step13)
14. [本轮范围、源码与学习位置](#step14)

<a id="step1"></a>
## 1. 看懂执行位置与目录

- **宿主机终端**：Xshell/SSH刚登录服务器后的终端，执行Git、复制权重、Docker命令。
- **容器终端**：创建容器或docker exec后进入的bash，执行SGLang和测试。
- **服务终端A**：启动模型，看到server is fired up后继续占用终端是正常的。
- **客户端终端B**：另开SSH、进入同一容器，运行测试。不要把客户端命令粘到正在跑服务的终端。
- **节点0/1（rank 0/1）**：两台服务器的编号，不是NPU卡号。单机只有节点0。
- **混部**：一个服务同时处理prefill和decode；双机是两台共同承载这个服务，不是P/D分离，也不是两台各起一份完整服务。

本项目固定目录：

~~~text
/home/tyj/glm52/
├── sglang/                      SGLang代码，含本手册及测试工具
└── sgl-kernel-npu/               kernel候选代码

/home/tyj/glm52-ms1/              日志、结果、缓存，不是代码或模型
├── datasets/                    GSM8K、GPQA各10题
├── cache/                       下载缓存
├── evidence/                    环境报告及比较结果
└── kernel-overlay-*/             启动时自动创建的192维算子加载目录

/home/weights/
├── GLM-5.2-w8a8/                完整量化target模型
└── GLM-5.2-DSpark-NPU-0805/      独立DSpark草稿
~~~

原先MS1_STATE指的就是/home/tyj/glm52-ms1这类“运行产物目录”。**本手册直接写路径，不要求你设置这个Shell变量。**下载cache不等于模型KV cache的命中证据。

新容器挂载整个/home:/home，所以宿主和容器中的路径相同，不再改成/workspace/weight。双机两端同名目录不表示文件自动共享，每台都要准备。

<a id="step2"></a>
## 2. 新宿主机拉代码

本章在**宿主机**执行。新服务器需已安装匹配的NPU驱动和Docker：

~~~bash
npu-smi info
docker version
ip -br -4 address
df -h /home
~~~

前两条检查设备和Docker；第三条显示本机IP/网卡，第一列是网卡名，IP填写时去掉/24等后缀；最后检查磁盘空间。命令不存在或无权限先解决宿主环境，容器不会代装宿主驱动。

创建目录并拉两个仓库：

~~~bash
mkdir -p /home/tyj/glm52
mkdir -p /home/tyj/glm52-ms1/datasets /home/tyj/glm52-ms1/cache /home/tyj/glm52-ms1/evidence
cd /home/tyj/glm52
git clone --branch sync/glm52-dspark-ms1 https://github.com/yuejiat0416/sglang.git
git clone --branch sync/glm52-dspark-ms1 https://github.com/yuejiat0416/sgl-kernel-npu.git
~~~

--branch表示克隆后直接使用本次个人临时调试分支；两个仓库分支同名但内容不同。私仓认证用你已有的Git凭据，不把令牌写到命令或文档。已经克隆的目录不用再clone，更新看第7节。

若Git必须通过代理，只有代理地址和端口需要向网络提供人确认。下面是**替代**上面第一条clone的写法，不要两条都执行：

~~~bash
git -c http.proxy='http://代理实际地址:代理实际端口' clone --branch sync/glm52-dspark-ms1 https://github.com/yuejiat0416/sglang.git
~~~

替换单引号里的完整地址，保留单引号；这不是已知可执行代理。kernel同理。-c只作用这次命令，不修改同事全局Git配置；不关闭SSL校验。完全离线可由负责人提供完整Git bundle后clone，不能用缺.git的网页ZIP替代版本检查。

查看两个实际版本：

~~~bash
cd /home/tyj/glm52/sglang
git branch --show-current
git rev-parse HEAD
cd /home/tyj/glm52/sgl-kernel-npu
git branch --show-current
git rev-parse HEAD
~~~

双机要保证SGLang两端SHA一致、kernel两端SHA一致。成功clone个人分支不代表已经与最新社区main对齐；这项在正式转测前另外核验。

<a id="step3"></a>
## 3. 新宿主机准备权重

Git只下载代码，不下载模型，镜像也不替代这两套权重。先在**宿主机**检查：

~~~bash
ls -lh /home/weights/GLM-5.2-w8a8/config.json
ls -lh /home/weights/GLM-5.2-w8a8/quant_model_description.json
ls -lh /home/weights/GLM-5.2-w8a8/optional/quarot.safetensors
ls -lh /home/weights/GLM-5.2-DSpark-NPU-0805/config.json
ls -lh /home/weights/GLM-5.2-DSpark-NPU-0805/model.safetensors
~~~

如果部门已在本机放了完整的同批制品，不重复复制。若是另一台新机器，可从此前已经跑通过的61.47.19.71拉到当前机器。**以下命令在新机器宿主机执行；如果当前就是.71且权重已在，不要对自己复制。**

~~~bash
mkdir -p /home/weights/GLM-5.2-w8a8
mkdir -p /home/weights/GLM-5.2-DSpark-NPU-0805
rsync -a --partial --info=progress2 root@61.47.19.71:/home/weights/GLM-5.2-w8a8/ /home/weights/GLM-5.2-w8a8/
rsync -a --partial --info=progress2 root@61.47.19.71:/home/weights/GLM-5.2-DSpark-NPU-0805/ /home/weights/GLM-5.2-DSpark-NPU-0805/
~~~

root@61.47.19.71是这里的“复制来源”，需要你有该机登录权限。若部门另有制品机，改这一来源和源路径，不改模型内容。rsync会提示认证；没有rsync可用Xftp上传完整文件夹。

末尾/保留，表示复制文件夹内容，避免多套一层目录。不带--delete；已有其他版本时另建目的目录，不混写。完整复制tokenizer、量化描述、索引、全部分片、optional及target的MTP/旋转文件。双机每台都需要完整target和draft，不是各一半；已共享挂载完整模型时只核对。

有制品方SHA256清单时在模型目录用sha256sum -c 清单文件名核验。后面的检查只查分片存在和小文件指纹，不能证明完整权重字节一致。没有收到这批内部量化制品的网上下载URL，所以使用已确认的内网制品，不随意换成网上同名模型。

<a id="step4"></a>
## 4. 拉镜像、创建容器：完整Docker命令

### 4.1 拉镜像

在每台**宿主机**执行：

~~~bash
docker pull swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.1.0-a3-20260904
docker image inspect swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.1.0-a3-20260904 --format '{{json .RepoDigests}}'
~~~

第一条下载你指定的0904镜像，第二条查看digest，双机应一致。仓库要求登录时按部门提供的docker login方法操作。Git代理不会替Docker配置代理。

### 4.2 创建并进入新容器

下面**整段复制到宿主机终端**。每行末尾的反斜杠表示命令还没结束，要保留，不要在它后面加空格。最后一行没有反斜杠。

~~~bash
docker run -it \
  --name glm52-test \
  --privileged \
  --ipc=host \
  --net=host \
  --shm-size=16g \
  --device=/dev/davinci0 \
  --device=/dev/davinci1 \
  --device=/dev/davinci2 \
  --device=/dev/davinci3 \
  --device=/dev/davinci4 \
  --device=/dev/davinci5 \
  --device=/dev/davinci6 \
  --device=/dev/davinci7 \
  --device=/dev/davinci8 \
  --device=/dev/davinci9 \
  --device=/dev/davinci10 \
  --device=/dev/davinci11 \
  --device=/dev/davinci12 \
  --device=/dev/davinci13 \
  --device=/dev/davinci14 \
  --device=/dev/davinci15 \
  --device=/dev/davinci_manager \
  --device=/dev/hisi_hdc \
  -v /usr/local/sbin:/usr/local/sbin:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /var/queue_schedule:/var/queue_schedule \
  -v /home:/home \
  -e CONTAINER_NAME=glm52-test \
  -e HF_HOME=/home/tyj/glm52-ms1/cache/huggingface \
  -e TRANSFORMERS_CACHE=/home/tyj/glm52-ms1/cache/huggingface \
  -e HUGGINGFACE_HUB_CACHE=/home/tyj/glm52-ms1/cache/huggingface/hub \
  -e TORCH_HOME=/home/tyj/glm52-ms1/cache/torch \
  -e XDG_CACHE_HOME=/home/tyj/glm52-ms1/cache/xdg \
  -e PIP_CACHE_DIR=/home/tyj/glm52-ms1/cache/pip \
  --entrypoint=bash \
  swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.1.0-a3-20260904
~~~

容器名已填glm52-test，镜像已填完整，不需要设置IMAGE、CONTAINER_NAME或MS1_STATE变量。每个设备单独挂载，核心目录和完整/home都已包含。不需要调用旧建容器.sh脚本。

若提示容器重名，宿主机执行docker ps -a查看；确认是自己的同一容器再进入，或把上面glm52-test换一个名字。不要删除同事容器。双机不同宿主可使用同一个容器名。

**为什么第一次exit后docker ps看不到容器？** 上面的docker run -it以bash作为容器主进程，退出这个初始bash会停止容器；命令没有--rm，不会因为这次exit自动删除。docker ps只显示运行中的容器，docker ps -a才同时显示已停止的容器。

如果已经exit，在**创建它的同一台宿主机**执行以下命令恢复原容器，不需要重新docker run：

~~~bash
docker ps -a
docker start glm52-test
docker exec -it glm52-test bash
~~~

glm52-test是本文的容器名；若创建时改过名字，后两条使用docker ps -a最后一列NAMES中的实际名字。以后通过docker exec进入，exit只退出新开的终端，不会结束原来的主bash。[Docker官方exec说明](https://docs.docker.com/reference/cli/docker/container/exec/)及[ps说明](https://docs.docker.com/reference/cli/docker/container/ls/)。停止过容器意味着里面原有的SGLang进程也已停止；docker start只恢复容器主进程，不会替你重新运行双机服务脚本。

创建成功后进入**容器bash**，执行：

~~~bash
cd /home/tyj/glm52/sglang
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
npu-smi info
ls /home/weights/GLM-5.2-w8a8/config.json /home/weights/GLM-5.2-DSpark-NPU-0805/config.json
python3 -m pip show torch torch-npu triton-ascend sgl-kernel-npu
~~~

使用镜像自带Python，不启用EvalScope虚拟环境，不重装Torch/CANN。设备或模型不可见先解决挂载/文件问题。

### 4.3 以后怎样进入、怎样开客户端终端

新开SSH连接，在**宿主机**执行：

~~~bash
docker exec -it glm52-test bash
~~~

进入**容器**后，每个新终端都执行：

~~~bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
cd /home/tyj/glm52/sglang
~~~

后面的两个启动命令都从这个仓库根目录执行。

容器已停止时先在宿主机执行docker start glm52-test，再docker exec；不要重复docker run。

<a id="step5"></a>
## 5. 单机启动

在61.47.19.71的**容器**里，打开`/home/tyj/glm52/sglang/devtools/glm52_ms1/single_dspark_static.sh`。
脚本开头已经填好IP和模型路径。正常切换只改这两行：

~~~bash
MODE='dspark'
GRAPH=0
~~~

`MODE='dspark'`用DSpark；`MODE='target-only'`只用target；`MODE='nextn'`用同事的NEXTN配方。
`GRAPH=0`是eager；把这一整行改成`GRAPH=1`就是graph。**不改后面的if或启动命令，不需要删冒号、问号或补花括号。**

保存后运行：

~~~bash
cd /home/tyj/glm52/sglang
bash devtools/glm52_ms1/single_dspark_static.sh
~~~

这就是完整启动步骤。脚本自行加载CANN/ATB，设置源码路径并启动服务。看到`server is fired up`后，这个终端继续被占用是正常的。

| 想运行的模式 | MODE这一行 | GRAPH这一行 |
|---|---|---|
| DSpark static eager | MODE='dspark' | GRAPH=0 |
| DSpark static graph | MODE='dspark' | GRAPH=1 |
| target-only eager | MODE='target-only' | GRAPH=0 |
| target-only graph | MODE='target-only' | GRAPH=1 |
| NEXTN graph | MODE='nextn' | GRAPH=1 |

eager/graph是SGLang原有执行方式，不是DSpark引入的。原始命令加`--disable-cuda-graph`选择eager；去掉它，并按同事配方设置`--cuda-graph-bs 16`选择启用图的配置。`GRAPH=0/1`只是脚本替你增删这些参数；`--cuda-graph-bs`设置捕获批量，不是独立开关。NPU沿用带cuda字样的参数名。

**当前TP32/DP4的图批次说明（2026-09-11，sync 7a93c2be53源码核对）**：保持当前DP Attention/DeepEP、CP1且未额外开启two-batch-overlap时，对齐单位为Attention TP=32/4=8。公共捕图选择要求`batch×每请求token行数`能被8整除；DSpark static的Target Verify每请求9行，所以目标验证图批次为8、16等，最小8；草稿每请求8行，仅此对齐规则允许batch1。batch指每DP的请求槽位，实际少量请求可补齐到图批次，不要求先凑够8个真实请求。若服务仍是`--max-running-requests 8`，每DP实际请求池大小为2，图上限向上对齐为8；脚本写`--cuda-graph-bs 16`也会被公共选择逻辑裁成8。只写1会让目标验证图没有合法批次。这里核验了源码批次选择函数，未在NPU捕图；不能据此承诺显存足够或整模型graph已通过。当前继续GRAPH=0完成11.0容量与长输入检查，之后首次graph候选可显式用`--cuda-graph-bs 8`，两台保持一致。源码见[get_batch_sizes_to_capture](/Users/yuejiat/workspace/model-inference/worktrees/sglang-glm52-dspark-ms1-sync/python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py:64)、[对齐规则](/Users/yuejiat/workspace/model-inference/worktrees/sglang-glm52-dspark-ms1-sync/python/sglang/srt/utils/common.py:3908)和[每DP请求池上限](/Users/yuejiat/workspace/model-inference/worktrees/sglang-glm52-dspark-ms1-sync/python/sglang/srt/mem_cache/kv_cache_configurator.py:2202)。分组概念见学习手册[24.4 Dense draft在DP Attention下如何分组](/Users/yuejiat/workspace/model-inference/glm52-dspark-npu-project/learning/glm52-dspark-complete-guide.md:4268)，该章integration快照只用于解释分组，当前捕图取值以上述sync源码为准。

单机沿用TP16/DP1。DSpark是block8/verify9；NEXTN是steps4/topk1/draft5。
`CONTEXT_LENGTH=16384`用于先跑短题；准备128k输入/1k输出时改为`CONTEXT_LENGTH=133120`。
这是允许的序列长度，实际KV容量仍要另行检查，不能据这一行认定能装下128k。

脚本沿用`with_kernel_checkout.py`加载本次192维Python算子候选，其他二进制仍用镜像安装包；不要求重编译、不替换已安装包。此步骤不包含a3_stuck分支的DeepEP二进制修复。
`MS1_STATE='/home/tyj/glm52-ms1'`只是工具缓存和运行记录的位置，已填好，不需要先生成任何JSON。

**旧镜像怎样运行分支里的DSpark适配？** 镜像提供Python、Torch/torch-npu、CANN、Triton及已安装二进制依赖；服务实际用哪份Python代码由两个现有启动脚本决定。容器已挂载/home，两台分别准备同一提交的两个仓库，顶部SGLANG_REPO和KERNEL_REPO指向它们即可：

| 部分 | 当前脚本的加载方式 |
|---|---|
| SGLang框架/模型的已提交适配 | 将/home/tyj/glm52/sglang/python放在PYTHONPATH前面，再运行python3 -m sglang.launch_server；加载该目录当前检出的源码，而非仅依赖镜像预装SGLang |
| 192维融合算子 | with_kernel_checkout.py读取/home/tyj/glm52/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/norm/split_qkv_rmsnorm_rope.py，复制到本次私有加载目录；其余sgl_kernel_npu文件和.so链接到镜像安装包，导入核验成功后启动服务 |
| 其他依赖和DeepEP二进制 | 继续使用当前容器已安装版本；拉取kernel源码不会自动更新.so或DeepEP备用包 |

192适配当前是Triton Python源码，首次调用按需编译/使用编译缓存，不要求先build整个kernel包。若后续改的是C++/AscendC等预编译二进制，必须另行构建并安装相应wheel或重打镜像；目前的单文件组合不会自动带入kernel仓其他改动。两台仓库应各自检出相同的审定提交；同名分支不能代替commit一致性。

仍只执行原single_dspark_static.sh或two_node_dspark_static.sh，不需要手工export PYTHONPATH或复制安装包。启动日志会打印Kernel overlay evidence、Frozen candidate的路径/commit以及Kernel import verified；对应目录的manifest.json和import.json保存候选哈希、导入来源与使用的镜像二进制路径。该核验确认算子导入来源，不等于模型或性能通过。普通docker exec新开的终端不继承服务脚本设置的PYTHONPATH，因此直接import看到镜像旧包是可能的，不能据此否定服务的覆盖加载。kernel候选在启动时复制，更新两个仓库后都需重启服务才能稳定测试新版本。

本轮被测版本应描述为“基础镜像 + SGLang提交 + kernel候选提交（及如有的另装DeepEP包）”，不能只给镜像标签就声称它已包含个人分支代码。无需为已有Python适配重建当前容器；镜像/依赖兼容性和整模型graph/128k仍以实测为准。

<a id="step6"></a>
## 6. 双机混部启动

两台都完成前四节，使用相同的镜像、SGLang/kernel提交和完整模型。双机是两台机器一起承载同一个服务，客户端只访问61.47.19.71:8810。

在**两台容器**里打开同一个文件：
`/home/tyj/glm52/sglang/devtools/glm52_ms1/two_node_dspark_static.sh`。

61.47.19.71上，开头写：

~~~bash
MODE='dspark'
GRAPH=0
NODE_RANK=0
~~~

61.47.19.70上，开头写：

~~~bash
MODE='dspark'
GRAPH=0
NODE_RANK=1
~~~

两台的MODE、GRAPH和模型参数保持一致，**只有NODE_RANK不同**。切graph就在两台都改`GRAPH=1`。

脚本已填两台IP、HTTP端口8810和分布式初始化端口50000。网卡默认按本机通往对端的路由自动读取，并打印实际选择；如果部门要求HCCL专用网卡，在开头`HCCL_SOCKET_IFNAME=''`的单引号内填真实名字，Gloo同理。不要填lo。

如果旧脚本报`Configured NIC ... does not exist`，但`ls /sys/class/net`能看到该网卡，这是旧检查可能将`ip`命令失败误报为网卡不存在。已填写两个网卡名时，新检查直接读取`/sys/class/net/网卡名`，不依赖`ip`命令。保留现场配置，在原文件找到下面这一行：

~~~bash
  if [ "$PRINT_COMMAND" -eq 0 ] && ! ip link show dev "$nic" >/dev/null 2>&1; then
~~~

只将这一行改为：

~~~bash
  if [ "$PRINT_COMMAND" -eq 0 ] && [ ! -d "/sys/class/net/$nic" ]; then
~~~

随后重新执行同一个启动脚本，保留现场填写的IP、网卡名和NODE_RANK。此修复只处理启动前检查，仍由`with_kernel_checkout.py`组合分支192算子与镜像依赖，不需要手改安装包。网卡目录存在不等于双机通信已通；如果继续报不存在，先核对是否在此前列出网卡的同一个容器。若进入kernel加载或SGLang初始化，按下一条实际异常定位。

在61.47.19.71的容器运行：

~~~bash
cd /home/tyj/glm52/sglang
bash devtools/glm52_ms1/two_node_dspark_static.sh
~~~

随后在61.47.19.70的容器运行**同一条命令**：

~~~bash
cd /home/tyj/glm52/sglang
bash devtools/glm52_ms1/two_node_dspark_static.sh
~~~

不要等第一台完全ready才启动第二台：第一台要等第二台加入。两台都启动后再发客户端请求。

双机沿用待实测的TP32/DP8配置，启用DP Attention和DP lm_head；不是本轮给框架新增并行限制。
`CONTEXT_LENGTH=133120`用于计划中的长输入；先测短题时两台都可改16384。context上限不等于实际KV容量。

<a id="step7"></a>
## 7. 检查、停止与更新

另开SSH终端，进入节点0容器。下面同样适用于单机或双机：

~~~bash
docker exec -it glm52-test bash
curl -sS --noproxy '*' http://61.47.19.71:8810/health
curl -sS --noproxy '*' http://61.47.19.71:8810/v1/models
curl -sS --noproxy '*' http://61.47.19.71:8810/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.2-w8a8","messages":[{"role":"user","content":"你好"}],"temperature":0,"max_tokens":64,"return_spec_tokens_details":true}'
~~~

HTTP正常返回只证明当前请求能完成。DSpark接受计数查看`sglext.spec_tokens_details`中的accepted/proposed及accept_rate，不能据单条请求宣布精度、graph或压力准出。

停止服务：回到启动服务的终端按**Ctrl+C**，等返回命令提示符再改GRAPH/MODE重新运行。双机两端都停止、都改同样模式、都重新运行。

需要保存服务日志时，第一次启动就使用下面的替代命令（与不带tee的命令二选一）：

~~~bash
mkdir -p /home/tyj/glm52-ms1
bash devtools/glm52_ms1/single_dspark_static.sh 2>&1 | tee /home/tyj/glm52-ms1/single-server.log
~~~

双机把文件名改成`two_node_dspark_static.sh`、日志名改成`two-server.log`，每台的日志留在各自宿主机。再次使用相同日志名会覆盖上次，需保留旧记录时先改日志名。

更新个人分支代码，先停止服务，在每台宿主机依次执行：

~~~bash
cd /home/tyj/glm52/sglang
git status --short
git pull
cd /home/tyj/glm52/sgl-kernel-npu
git status --short
git pull
~~~

你编辑过启动脚本，`git status`会显示它被修改。若pull提示冲突，保留现场，不执行hard reset；把提示发来再处理。不要为了更新代码删模型、运行目录或容器。

<a id="step8"></a>
## 8. 现有文件用途

| 文件/目录 | 用途 |
|---|---|
| single_dspark_static.sh | 当前单机完整启动脚本 |
| two_node_dspark_static.sh | 当前双机混部完整启动脚本 |
| two_node_colocated/run_tests.py | 本轮双机测试唯一日常入口：准备数据、精度、缓存检查、压力与对比表 |
| run_gsm8k_single.sh | 单机服务启动后的GSM8K十题入口；只需让顶部MODE与服务一致，然后直接运行 |
| with_kernel_checkout.py | 两个启动脚本内部使用的192维算子加载工具，无需单独操作 |
| bench_gsm8k_modes.py / GSM8K_MODES.md | 已有单机GSM8K十题客户端及协议；旧launch配置方法已由本页两个脚本替代 |
| bench_gsp_prefix.py / GSP_PREFIX.md | 已有单机131072/1024三档缓存诊断客户端及协议 |
| two_node_colocated/ | 先前的双机测试客户端、结果比较和归档工具；里面的launch.py/配置模板不再作为服务启动入口 |
| probe_*、collect_*、其他test_* | 各轮临时诊断和本地测试，启动服务不需要逐一执行 |

数据集和压力测试的客户端不负责启动、停止或切换服务；旧测试配置不应成为服务启动的前置步骤。后续章节把本轮跑测命令放在本页。

<a id="step9"></a>
## 9. 本轮双机：准备数据和测试入口

**目的**：先锁定相同的题目、服务版本与测试口径，再比较。当前单机GSM8K接受率不能替代双机精度、graph或长输入压力结果。本节只准备客户端数据，不占用模型服务。

本页先沿用节点0=61.47.19.71、节点1=61.47.19.70，每台16逻辑NPU，TP32/DP8。若本次节点0换成61.47.19.69，在两台的two_node_dspark_static.sh里把NODE0_HOST改为61.47.19.69，同时把run_tests.py顶部HOST改为61.47.19.69。两个仓库在/home/tyj/glm52下是**并列**关系；不要在sglang目录里面再clone kernel。

先按第7节更新两端代码。模型和镜像按第2至4节准备；源码提交、镜像digest、模型制品须一致。客户端在节点0同一个容器另开终端运行，使用镜像自带Python，不进入evalscope-venv。

### 9.1 GPQA先在能联网的PC下载，再传服务器（只跑GSM8K时跳过）

GSM8K十题已在仓库gsm8k10.json中，服务器不用访问Hugging Face。GPQA用作者公开的[官方仓库dataset.zip](https://github.com/idavidrein/gpqa)，作者在README公开了解压密码。**在能联网的PC执行**下面整段，只提取Diamond CSV，不下载模型：

~~~bash
mkdir -p glm52-test-data
cd glm52-test-data
curl -fL https://raw.githubusercontent.com/idavidrein/gpqa/main/dataset.zip -o gpqa-dataset.zip
python3 - <<'PY'
import hashlib
import zipfile
from pathlib import Path
with zipfile.ZipFile("gpqa-dataset.zip") as archive:
    names = [n for n in archive.namelist() if Path(n).name == "gpqa_diamond.csv"]
    if len(names) != 1:
        raise SystemExit("没有找到唯一的gpqa_diamond.csv，保留压缩包并检查来源")
    content = archive.read(names[0], pwd=b"deserted-untie-orchid")
Path("gpqa_diamond.csv").write_bytes(content)
print("gpqa_diamond.csv SHA256:", hashlib.sha256(content).hexdigest())
PY
~~~

把gpqa_diamond.csv通过内网允许的传输方式送到节点0的/home/tyj/glm52-ms1/datasets/。可用SFTP上传；PC能SSH访问节点0时，从上述PC目录执行：

~~~bash
scp gpqa_diamond.csv root@61.47.19.68:/home/tyj/glm52-ms1/datasets/
~~~

如果网络隔离，先转到内网PC，再从内网PC上传同一个文件。**不需要把GPQA传给节点1；数据只由节点0的客户端读取。**题目、答案、生成响应和权重留在内网，Git只同步工具。

### 9.2 默认十题，当前只跑GSM8K

**当前先跑通GSM8K十题，GPQA暂停。** 服务保持当前双机DSpark eager，在节点0同一容器另开客户端终端；先把`two_node_colocated/run_tests.py`顶部的`HOST`改为当前节点0地址，`MODE`设为`"dspark-eager"`，模型目录及TP/DP与运行服务一致。只运行：

~~~bash
cd /home/tyj/glm52/sglang
python3 devtools/glm52_ms1/two_node_colocated/run_tests.py accuracy --dataset gsm8k
~~~

这会自动准备和读取仓库自带的GSM8K十题，不检查GPQA文件，不需要联网下载或EvalScope，也不用单独执行`prepare`。每题串行、temperature=0、最多4096输出token，尊重EOS。客户端继续调用已有bench_serving及A/P/N采集、评分，不改变服务计算。十题之外或全量test见9.4。

先看`DATASET_COLLECTED`、十题完成情况、`correct`、截断/未解析数量及`Acceptance`。这是小样本功能/精度与接受率观察，不能作为压力>50%的准出。首组完整后再切target-only eager，以同样十题和预算补对照；再验证graph。任何请求失败先保留对应`summary.json`并定位，不追加GPQA或128k压测。后续恢复GPQA时用`--dataset gpqa`单独准备/运行，或不填该参数按下面的双数据集流程处理。

在节点0**容器客户端终端**执行：

~~~bash
cd /home/tyj/glm52/sglang
python3 devtools/glm52_ms1/two_node_colocated/run_tests.py prepare
~~~

生成datasets/gsm8k-10.json和datasets/gpqa-10.json。它们是**自动保存的样本文件，不是需要你填写的配置文件**。当前选源文件前10行；GPQA四个选项使用seed42固定洗牌。同一轮所有模式复用这两份文件，不重复抽题、不混两数据集。来源和SHA保存在样本中。若GPQA缺失，工具明确提示，不会偷偷换成其他题集。

### 9.3 测试时只改一个Python文件

打开/home/tyj/glm52/sglang/devtools/glm52_ms1/two_node_colocated/run_tests.py。顶部已填地址和路径，日常只改MODE这一行：

~~~python
MODE = "dspark-eager"
~~~

保留双引号，替换里面的文字，不改后面的Python函数。其他顶部参数含义：

| 参数 | 已填的值和含义 |
|---|---|
| HOST / PORT | 节点0的HTTP地址和端口，61.47.19.71 / 8810 |
| TARGET_MODEL / DRAFT_MODEL | 容器能读到的两个完整权重目录；若旧容器仍挂/workspace/weight，应与服务脚本一起改成真实路径 |
| DATASETS | 两份十题样本与GPQA原CSV所在目录 |
| RESULTS | 本轮所有模式的结果总目录；五组保持一致，另开一轮比较时换新目录名 |
| ACCURACY_MAX_TOKENS | 每题最多4096输出token；允许EOS提前结束，所有模式一致 |
| GSM8K_LIMIT | 默认10；可改100或1319，按源文件顺序选前N题；1319对应官方main/test全量 |
| GSM8K_SOURCE | 默认空字符串，使用自带十题；超过十题时填已上传的官方test.jsonl绝对路径，不填URL |
| PERFORMANCE_REQUESTS | 每档缓存64条正式请求，三档共192条；不含预热 |
| PERFORMANCE_CONCURRENCY | 仓库默认8；本轮DP4改为4，表示客户端总并发4，每DP一路；quick仍固定并发1 |

RESULTS默认/home/tyj/glm52-ms1/dual-node-validation-20260910。日志留在第6节指定的位置；测试数据自动写在RESULTS下，不用你创建更多配置。

<a id="gsm8k-full"></a>
### 9.4 跑更多GSM8K或全量test：仍用同一个脚本

**注意：`accuracy --dataset gsm8k`只选择数据集，不是“全量”开关。当前仓库默认仍为`GSM8K_LIMIT = 10`、`GSM8K_SOURCE = ""`，直接使用默认值会跑自带十题；必须按下文改成1319及完整源文件路径。** 运行开头应打印`gsm8k: 固定1319题 -> /home/tyj/glm52-ms1/datasets/gsm8k-1319.json`；若打印固定10题/`gsm8k-10.json`，本次就是十题。已有运行直接查看当次开头和结果中的`accuracy.samples`，无需重跑确认。源文件不足1319题时会报错，不会自动退回十题。

**当前节点0为61.47.19.68，节点1为61.47.19.69。** 全量精度通常指GSM8K **main/test的1319题**，不是把7473条训练数据混入测试。[官方test文件](https://github.com/openai/grade-school-math/blob/master/grade_school_math/data/test.jsonl)。当前仓库只自带十题，因此超过十题必须提供完整本地数据。评分、请求内容和接受计数继续复用原工具；只改客户端，不重启服务、不安装EvalScope。

先在**能联网的PC终端**下载（Windows PowerShell将`curl`写成`curl.exe`）：

~~~bash
curl -fL "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl" -o gsm8k-test.jsonl
~~~

得到`gsm8k-test.jsonl`。若外网PC不能连接服务器，按现有文件传输方式先送到内网PC；再从**能连接68的PC终端**上传：

~~~bash
scp gsm8k-test.jsonl root@61.47.19.68:/home/tyj/glm52-ms1/datasets/
~~~

也可以用SFTP把同一个文件放到该目录。此前十题测试已经使用这个目录；`/home:/home`挂载下容器路径相同。数据只需在68，不需要传给69。

**68已回读为d5066f95e旧版，虽手工添加了GSM8K_LIMIT/SOURCE，旧函数仍写死十题，不能据变量存在判断已更新。** 本次只补全量精度需要的两个文件，固定取7a93c2be53；后续GSP入口变更不混入本次更新。若当前有测试客户端在运行，等它结束后，在**68容器的客户端终端**执行，服务不用停止：

~~~bash
cd /home/tyj/glm52/sglang
cp devtools/glm52_ms1/two_node_colocated/run_tests.py /tmp/glm52-run-tests-before-full-20260911.py
cp devtools/glm52_ms1/two_node_colocated/offline_dataset.py /tmp/glm52-offline-dataset-before-full-20260911.py
git fetch origin sync/glm52-dspark-ms1
git restore --source=7a93c2be53a1898b54127a4601cb9a329fcba2aa -- devtools/glm52_ms1/two_node_colocated/run_tests.py devtools/glm52_ms1/two_node_colocated/offline_dataset.py
~~~

若fetch报错，先处理该错误，不继续后面的restore。此更新会把`run_tests.py`顶部恢复为该提交的默认值，**必须按下面填回68的已知配置**；备份保留当前所有设置，.sh、服务端代码和kernel不更新。这里只更新指定文件，不移动整个Git分支；因此随后git log仍可能显示旧提交，实际功能以准备命令的题数为准。修改文件：

~~~bash
vi devtools/glm52_ms1/two_node_colocated/run_tests.py
~~~

按`i`编辑，设置为：

~~~python
MODE = "dspark-eager"
HOST = "61.47.19.68"
TP_SIZE = 32
DP_SIZE = 4
PERFORMANCE_REQUESTS = 64
PERFORMANCE_CONCURRENCY = 4
GSM8K_LIMIT = 1319
GSM8K_SOURCE = "/home/tyj/glm52-ms1/datasets/gsm8k-test.jsonl"
ACCURACY_MAX_TOKENS = 4096
~~~

`MODE`仍必须与当前服务一致：target-only eager填`"target-eager"`，DSpark eager填`"dspark-eager"`；graph则对应`"target-graph"`或`"dspark-graph"`。**改客户端MODE不会切换服务器。** 模型路径、TP/DP保持与服务一致。只想先扩大到100题，把`GSM8K_LIMIT`改为100，完整文件路径不变。只替换引号里面的路径/地址，数字1319不加引号；按`Esc`、输入`:wq`、回车保存。

更新后先执行一次准备检查；只读取本地数据并生成题目文件，不访问服务、不执行NPU：

~~~bash
python3 devtools/glm52_ms1/two_node_colocated/run_tests.py prepare --dataset gsm8k
~~~

应打印`gsm8k: 固定1319题 -> /home/tyj/glm52-ms1/datasets/gsm8k-1319.json`；若报完整源文件不存在或不足1319题，先补正确数据，不继续精度运行。已在CPU复现d5066旧依赖只换上述两文件，完整官方源得到1319个唯一请求、bench的num-prompts=1319，短源被拒绝。数据可以趁quick排错时先上传；**实际运行排在11.1的GSP并发4压测之后**。本轮先测当前`dspark-eager`，然后仍然只执行：

~~~bash
python3 devtools/glm52_ms1/two_node_colocated/run_tests.py accuracy --dataset gsm8k
~~~

脚本自动生成`datasets/gsm8k-1319.json`（100题则`gsm8k-100.json`），保留旧`gsm8k-10.json`；缺文件或题数不足时停止，不会把十题重复发送凑成全量。各模式使用同一文件、同一题数和输出预算；结果在启动时打印的Evidence目录，包含`summary.json`与逐请求`responses.jsonl`。先看完成题数、正确数、未解析/截断及失败项，再比较target-only与DSpark的逐题得分变化和sum(A)/sum(P)。不要把十题target-only与1319题DSpark直接比较。

仍为**并发1、非流式精度采集**，不受`PERFORMANCE_CONCURRENCY=4`影响，TPOT/TTFT不可用于性能验收；投机模式记录接受率，target-only为不适用。全量可能运行较长时间，旧十题耗时不能预测新DP4/分块配方的准确时长。评分分母保留全部1319题；4096 token截断、未解析或请求失败项单独列出，不能剔除后提高正确率。覆盖完整test不代表与模型卡不同提示/预算下的成绩可直接对比。随后须用相同1319题、提示和4096预算补target-only对照，才能判断DSpark有没有精度下降；已有十题对照不够。本轮已在CPU核对1319个唯一题目、答案解析和请求中不含标准答案，尚未在NPU实测全量，不自动启动或续跑。

<a id="accuracy-twenty"></a>
### 9.5 当前改为GSM8K、GPQA各20题，分别测试

以9.4已能运行全量的客户端为前提，不再拉代码。本轮复用原选题流程，固定取各自源文件前20行，不称为随机抽样；GPQA只对四个选项按seed42固定洗牌。分别生成gsm8k-20.json和gpqa-20.json，旧十题/全量文件保留，不混合计分。后续target-only与DSpark复用相同文件、提示和4096预算。

若全量测试还在运行，在**运行全量命令的客户端终端**按Ctrl+C停止；服务终端不停止。在68容器的客户端终端编辑：

~~~bash
cd /home/tyj/glm52/sglang
vi devtools/glm52_ms1/two_node_colocated/run_tests.py
~~~

按i，修改顶部已有的同名行，保留当前MODE/HOST/TP/DP和其他服务对应配置：

~~~python
GSM8K_LIMIT = 20
GSM8K_SOURCE = "/home/tyj/glm52-ms1/datasets/gsm8k-test.jsonl"
ACCURACY_MAX_TOKENS = 4096
~~~

GPQA当前还在dataset_settings函数中固定为10，不需要添加一个不会被读取的新变量。按Esc，输入`/gpqa_diamond.csv`并回车找到下面这行，只把10改20，保持原缩进：

~~~python
        count, source = 20, directory / "gpqa_diamond.csv"
~~~

按Esc，输入:wq，回车保存。数据文件须在`/home/tyj/glm52-ms1/datasets/`：GSM8K沿用全量gsm8k-test.jsonl，GPQA用gpqa_diamond.csv（至少20道题）；仅有旧gpqa-10.json不能代替完整CSV。若尚未准备GPQA，按9.1从官方zip提取并只上传68。

先用原命令同时准备两份小样本，不调用模型服务：

~~~bash
python3 devtools/glm52_ms1/two_node_colocated/run_tests.py prepare
~~~

应分别打印`gsm8k: 固定20题`、`gpqa: 固定20题`，文件名分别为gsm8k-20.json和gpqa-20.json。文件缺失/不足20题时先补数据。然后分别执行，第一套结束后再启动第二套：

~~~bash
python3 devtools/glm52_ms1/two_node_colocated/run_tests.py accuracy --dataset gsm8k
~~~

~~~bash
python3 devtools/glm52_ms1/two_node_colocated/run_tests.py accuracy --dataset gpqa
~~~

每套仍为并发1、最多4096 token、允许EOS结束；PERFORMANCE_CONCURRENCY不控制精度并发。输出各自Evidence、correct=N/20、Acceptance以及summary.json，失败/未解析/截断不从分母删除。20题用于快速对照，一题对应5个百分点，不当作全量精度通过。若之后切换graph或target-only，保持这两份20题与预算一致，不直接拿旧10题/全量成绩对比。

本地已按以上改法核验prepare、两个独立accuracy入口、各20个请求和原生num-prompts20，以及GPQA不足20题时拒绝执行。使用真实GSM8K源和合成GPQA解析样例，未跑NPU或取得GPQA真实准确率；没有新脚本/依赖/推送。已有源位置为[dataset_settings](/Users/yuejiat/workspace/model-inference/worktrees/sglang-glm52-dspark-ms1-sync/devtools/glm52_ms1/two_node_colocated/run_tests.py:69)，学习[28.7评测协议](/Users/yuejiat/workspace/model-inference/glm52-dspark-npu-project/learning/glm52-dspark-complete-guide.md:4747)用于理解固定样本、提示与预算；integration教材不替代当前sync代码。

<a id="gpqa-full-current"></a>
### 9.6 当前：68/70双机、DSpark graph、EvalScope全量GPQA-D

**2026-09-13当前顺序**：按负责人要求直接跑graph全量，跳过十题GSM8K。使用68为node0、70为node1，TP32/DP4、static、沿用已准备的gamma5/verify6配方。GPQA-Diamond全量为198题（不是GPQA其它子集），同一次EvalScope运行同时评分和采集A/P/N。NPU操作由负责人执行；尚未取得本轮graph实机结果。

#### 为什么原来是8，本次为什么同时改5和6

0805草稿制品的`block_size=8`被解析为默认gamma=8。它不是从accept len推导出的最优值，也不是A3或双机强制要求。当前源码允许CLI覆盖：`--speculative-dspark-block-size`优先确定gamma，`--speculative-num-draft-tokens`必须等于gamma+1；只改其中一个会发生冲突。**本次同时设置5和6，不改权重config.json。** 启动时可能看到checkpoint gamma8与运行gamma5不一致的warning，这是显式覆盖的预期提示；仍需看到服务ready且客户端核准5/6。

这处改动处于**配置→draft proposal→target verify→accept/commit**：gamma5每轮产生5个草稿候选，Target验证输入为anchor+5候选，共6行。全部接受时最多输出5个草稿token和1个bonus；提前拒绝时，额外的target token是纠正token。验证输入的anchor与新输出的bonus不是同一个token。

依据当前sync基线`35edd9ec4b5a7329fcccf85dc71c248a2fa71353`：[配置优先级及gamma+1校验](/Users/yuejiat/workspace/model-inference/worktrees/sglang-glm52-dspark-ms1-sync/python/sglang/srt/arg_groups/speculative_hook.py:627)、[运行时显式覆盖](/Users/yuejiat/workspace/model-inference/worktrees/sglang-glm52-dspark-ms1-sync/python/sglang/srt/speculative/dspark_components/dspark_config.py:108)。学习手册[17.2 gamma的名称合同](/Users/yuejiat/workspace/model-inference/glm52-dspark-npu-project/learning/glm52-dspark-complete-guide.md:3449)帮助区分两种计数，[18.2/18.3 草稿与静态验证](/Users/yuejiat/workspace/model-inference/glm52-dspark-npu-project/learning/glm52-dspark-complete-guide.md:3575)解释每轮的输入和输出。教材是integration概念快照，本轮参数合同以此sync源码为准。

若N为请求×验证轮数，A为被接受草稿token总数，static gamma固定时P=gamma×N。服务日志的平均接受长度`L=1+A/N`，接受率`R=A/P=(L-1)/gamma`。原gamma8、L约3.5对应31.25%；改gamma5后**若L仍为3.5，恰好50%，还没通过严格>0.5**。新配置需要L>3.5；L必须重新实测，不能保证缩短窗口后保持原值。缩短窗口可能减少无效候选计算，是否加速另行实测。本轮不把接受率提高等同于模型能力或性能提高。

#### 第一步：两端修改已有启动脚本并启动graph

**不要用仓库模板覆盖服务器已跑通的.sh。** 保留已经跑通的DP4显存/分块参数、context/KV133120、max-running-requests4、DeepEP和192 overlay；70也使用相同代码、模型制品和有效网卡。只更新节点地址、graph开关，并核对gamma5/verify6。

等旧客户端结束，在两台**服务终端**停止本次旧服务。然后在68和70的服务容器分别打开同一个文件：

~~~bash
cd /home/tyj/glm52/sglang
vi devtools/glm52_ms1/two_node_dspark_static.sh
~~~

按`i`进入编辑。两台顶部都设置以下值，另外68的`NODE_RANK=0`，70的`NODE_RANK=1`：

~~~bash
MODE='dspark'
GRAPH=1
NODE0_HOST='61.47.19.68'
NODE1_HOST='61.47.19.70'
~~~

DSpark分支保留`--speculative-dspark-block-size 5 --speculative-num-draft-tokens 6`。若现场仍是8/9，把这两个值同时改为5/6，与本轮全量工具一致；不修改NEXTN分支。在后半段含`--cuda-graph-bs`的行把16改8：

~~~bash
if [ "$GRAPH" = 1 ]; then SERVER_ARGS+=(--cuda-graph-bs 8); else SERVER_ARGS+=(--disable-cuda-graph); fi
~~~

保留`ENABLE_METRICS=1`。按`Esc`、输入`:wq`、回车保存。TP32/DP4的Attention TP为8；本轮Target宽6、Draft宽5，bs8可满足两种宽度的8行对齐，实际捕图列表还会受runtime约束。不再套用上一轮gamma8、Target宽9时的最小档位推导。static由`SGLANG_RAGGED_VERIFY_MODE=static`设置，P=5N本身不能证明static模式。

在**68服务容器**启动并保存本次服务日志：

~~~bash
cd /home/tyj/glm52/sglang
mkdir -p /home/tyj/glm52-ms1/evidence
set -o pipefail
bash devtools/glm52_ms1/two_node_dspark_static.sh 2>&1 | tee /home/tyj/glm52-ms1/evidence/server-gpqa-gamma5-node0.log
~~~

在**70服务容器**运行：

~~~bash
cd /home/tyj/glm52/sglang
mkdir -p /home/tyj/glm52-ms1/evidence
set -o pipefail
bash devtools/glm52_ms1/two_node_dspark_static.sh 2>&1 | tee /home/tyj/glm52-ms1/evidence/server-gpqa-gamma5-node1.log
~~~

服务占用终端是正常的。68启动后就启动70，不要等68ready才启动70。查看`Capture target verify NPU graph ... end`和`Capture draft verify NPU graph ... end`，等68显示ready再进入第二步。若捕图OOM/通信异常，先处理第一处错误，不直接调大mem fraction。同名服务日志会由tee重写；若重跑，先保留上次两份日志。

#### 第二步：在68另开客户端终端，跑一次全量

客户端使用`/home/tyj/glm52-ms1/evalscope-venv`中的EvalScope 1.11.1；服务进程继续使用镜像Python。本轮工具已经推到个人`sync/glm52-dspark-ms1`。因为服务器上的双机启动脚本有手工调整，不直接pull覆盖现场；在**68容器的客户端终端**只更新本次需要的GPQA文件：

~~~bash
cd /home/tyj/glm52/sglang
git fetch origin sync/glm52-dspark-ms1
git restore --source origin/sync/glm52-dspark-ms1 -- devtools/glm52_ms1/run_gpqa_diamond.py
~~~

它顶部已设`GRAPH=True`、地址68，复用同目录原有`gsm8k_mode_stats.py`分析图计数。不改旧run_tests.py的十题设置，也不改两台现场启动脚本。

若执行评测命令时报`/home/tyj/glm52-ms1/evalscope-venv/bin/python: No such file or directory`，说明当前68容器还没有这套客户端环境。在**68容器的客户端终端**直接执行下面四条命令；使用阿里PyPI镜像，只安装到独立目录，不修改服务Python，也不需要停止双机服务：

~~~bash
mkdir -p /home/tyj/glm52-ms1
python3 -m venv /home/tyj/glm52-ms1/evalscope-venv
/home/tyj/glm52-ms1/evalscope-venv/bin/python -m pip install -i https://mirrors.aliyun.com/pypi/simple evalscope==1.11.1
/home/tyj/glm52-ms1/evalscope-venv/bin/python -c "from evalscope import TaskConfig, run_task; print('EvalScope 1.11.1 import OK')"
~~~

最后一行打印`EvalScope 1.11.1 import OK`即表示客户端依赖已经准备好。只在68准备；70仅运行服务，不安装EvalScope。

数据沿用[9.1](#step9)从GPQA作者仓库取得的`/home/tyj/glm52-ms1/datasets/gpqa_diamond.csv`。必须是完整198题的Diamond原始CSV，不能使用gpqa-20.json或仅20行CSV。只需在68准备，题目不传70。测试脚本先核对198个唯一题目和必需字段；在自己的输出目录内部复制为独立CSV目录，供EvalScope原生loader读取。实际检查确认直接CSV文件路径会失败，父目录又混有GSM文件，因此内部复制是必要的最小处理，不增加用户步骤或手写配置。

另开SSH，在**68宿主机**进入已有容器：

~~~bash
docker exec -it glm52-test bash
~~~

在这个**客户端容器终端**执行全量命令：

~~~bash
cd /home/tyj/glm52/sglang
/home/tyj/glm52-ms1/evalscope-venv/bin/python devtools/glm52_ms1/run_gpqa_diamond.py
~~~

若现场容器用了不同名称，docker exec只替换glm52-test为原容器名。不需要source另一个配置文件；可选`--check-only`只检查已有依赖和完整CSV，不发模型请求，但不代替实际EvalScope加载与服务验证。

本轮参数固定为：`gpqa_diamond`、train、0-shot、全198题、每题1次、并发4、temperature1.0、max_tokens65536、seed42、官方规则评分；不设limit、不复用旧回答、不自动重试/重跑取最好值、不配置外部judge。提示、选项排列、答案提取和accuracy均由EvalScope 1.11.1实现。采样温度与输出预算取自[仓库GLM-5.2 GPQA配方](/Users/yuejiat/workspace/model-inference/worktrees/sglang-glm52-dspark-ms1-sync/test/registered/npu/accuracy/glm5_2/test_npu_glm_5_2_w4a8_16p_gpqa.py:99)，该社区例子为W4A8/NEXTN，不能称已复现当前W8A8/DSpark的91.2原始协议。负责人确认没有此前91.2命令，91.2继续作为本次指定参考值。

为保存服务响应中的`sglext`原始计数，本轮使用非流式，超时7200秒，且将社区并发32改为当前服务相适应的4；不改变提示/评分或服务采样算法。EvalScope流式拼接会丢失sglext，普通CLI无法保留精确计数，因此脚本只在官方`on_response`回调保存A/P/N；无框架改动、无新的部署依赖。非流式等待期间客户端可能长时间没有逐token输出，服务日志仍显示推理进展；这里的客户端延迟不用于TTFT/TPOT验收。

#### 第三步：读精度和接受率，决定下一步

脚本开始和结束均打印本次目录：`/home/tyj/glm52-ms1/evidence/gpqa-d-graph-gamma5-…/`，每次自动新建，不会把旧回答当本轮结果。目录中：

- `summary.json`：198题完成情况、正确数、精度、A/P/N、接受率、接受长度、length结束数，以及两个独立判定。
- `inference.jsonl`：同次198个唯一响应的服务端原始接受计数，含response_id，可关联EvalScope原始回答。按请求完整计数覆盖尾轮和各DP，不重复累计TP副本。
- `reports/`、`reviews/`、`predictions/`：EvalScope官方报告、评分、题目及原始回答；`server-info.json`、`run-settings.json`记录本次运行设置。
- `metrics.before.txt`、`metrics.after.txt`及`summary.json`中的`graph`：记录测试前后Target decode/verify图计数增量；若抓取失败则保存error.json，不能当零计数。测试期间其它客户端保持空闲。HTTP未提供独立draft replay计数，Target图增长不能证明每轮draft也回放。

精度按此前自测参考**90.2%≤accuracy≤92.2%**，198题单次对应**179～182题正确**；高于92.2%也按约定标记超出区间，应复核协议，不解释成质量退化。接受率按**sum(A)/sum(P)>0.5**，严格用整数`2×sum(A)>sum(P)`判断，0.5不通过。length结束仍留在198分母并单独列出，不删题；失败、缺计数、重复响应或未评分全198题均为INCOMPLETE。客户端发题前核对graph backend和enable-metrics，结束后再次核对模式；未观察到Target图计数增长则标INCOMPLETE，已完成精度及接受计数仍保留，另存evaluation_status。两个数值条件满足且取得图回放证据才输出PASS；仍不是全链路graph、压力性能或相对target-only不降精度的结论。

同时查看**两端服务终端**的`accept len: …, accept rate: …`日志。它们是各DP周期观察，只保留两位小数、可能漏掉未输出的最后统计窗口；不要直接平均各行或各DP，也不要将`0.50`当严格通过。最终全程结论以本轮`inference.jsonl`里的服务端计数汇总为准，不用completion_tokens反推。源码见[日志公式](/Users/yuejiat/workspace/model-inference/worktrees/sglang-glm52-dspark-ms1-sync/python/sglang/srt/managers/scheduler_components/metrics_reporter.py:896)、[响应计数入口](/Users/yuejiat/workspace/model-inference/worktrees/sglang-glm52-dspark-ms1-sync/python/sglang/srt/entrypoints/openai/utils.py:158)；学习[28.9/28.10 接受率与采集陷阱](/Users/yuejiat/workspace/model-inference/glm52-dspark-npu-project/learning/glm52-dspark-complete-guide.md:4780)解释为何不能只看长度或平均百分比。

如果精度和接受率都通过，保存本轮资料后再进入性能对比；精度偏低先查看截断/原始回答与协议，必要时同题同参数做target-only对照；精度通过但接受率≤0.5时看gamma5的真实长度与请求分布，不改分母或重跑择优；启动/请求/依赖失败则先修对应故障，未完成的运行不计为精度失败或成功。参考[EvalScope官方API与本地数据说明](https://evalscope.readthedocs.io/en/latest/get_started/basic_usage.html)。

本地验证：既有CSV加载与SDK回调检查保留为上轮证据；本轮增加graph模式、计数缺失/回退/重置检查，详见[实际diff和逐项审视](/Users/yuejiat/workspace/model-inference/glm52-dspark-npu-project/reviews/2026-09-10-two-node-colocated-tests/README.md:3)。NPU全量、实际精度、接受率及graph回放待负责人实测；工具提交`53adfe7c35`已推到个人sync分支，随后仅补运行文档。

<a id="step10"></a>
## 10. 精度对照：四组十题流程，全量前固定同一协议

<a id="graph-smoke-current"></a>
### 10.1 当前68/69、TP32/DP4：先做DSpark graph十题验证

**阶段与目的**：当前eager已能启动并执行精度请求；本轮检查相同部署开启graph后能否捕图、启动和完成连续生成，并在同一次10题请求中记录正确数和接受率。尚未验证整模型graph，不把旧192单算子graph通过当作本轮结论。128k/1k的容量要求继续保留；短GSM8K通过后，仍需另测长输入与并发。

等当前精度测试结束并记下Evidence目录，才在68、69两台服务终端分别按`Ctrl+C`停止旧服务。**不用git pull覆盖服务脚本，也不更换容器、kernel overlay或DeepEP包。** 在两台的当前服务容器内打开已经跑通的文件：

~~~bash
cd /home/tyj/glm52/sglang
vi devtools/glm52_ms1/two_node_dspark_static.sh
~~~

按`i`进入编辑。顶部设置`MODE='dspark'`（若当前是target-only，也要切回dspark），把`GRAPH=0`改成`GRAPH=1`。在文件后半段找到含`--cuda-graph-bs`的这一行，仅把捕图档位从16改为8，完整行如下；`else`后面的禁用参数留着，它只在GRAPH=0时使用：

~~~bash
if [ "$GRAPH" = 1 ]; then SERVER_ARGS+=(--cuda-graph-bs 8); else SERVER_ARGS+=(--disable-cuda-graph); fi
~~~

按`Esc`、输入`:wq`、回车保存。两台只调整上述模式和捕图选项，其余沿用当前成功的DSpark eager配置：

| 项目 | 本轮设置 |
|---|---|
| 节点 | 68的NODE_RANK=0；69的NODE_RANK=1；两台NODE0_HOST为68、NODE1_HOST为69 |
| 并行与并发上限 | TP32、DP4、EP32、max-running-requests=4；DP Attention与DP LM head保留 |
| 容量 | CONTEXT_LENGTH=133120、MAX_TOTAL_TOKENS=133120；不能为了短题启动改回16k |
| 显存与分块 | mem-fraction-static、chunked-prefill-size、max-prefill-tokens保留本次eager成功值；两台一致 |
| DSpark | static、block8、verify9、QuaRot original、draft unquant及当前192算子overlay保留 |
| 图 | GRAPH=1，--cuda-graph-bs 8；此次先验证decode/target verify及draft模型图，prefill图以实际backend为准 |

**为什么是8**：本轮TP32/DP4且CP1，每组Attention TP为8；当前DeepEP路径要求图的token行数满足8的对齐。Target Verify每请求9行，因此`batch_size × 9`要整除8，最小正档位是8（未启用two-batch-overlap）。这不是最多只能并发8，也不是客户端必须并发8；十题仍并发1，runtime按图档位补齐计算。draft每请求8行，捕图列表还会受自身请求池和并行上下文约束，以其启动日志为准。显式只给8可避免不必要的大档位配置；不承诺捕图显存按比例下降或一定能启动。

两台分别运行同一个命令，**不要等68启动完成才启动69**：

~~~bash
bash devtools/glm52_ms1/two_node_dspark_static.sh
~~~

查看两端启动日志中的`Capture target verify NPU graph begin/end`和`Capture draft verify NPU graph begin/end`；begin行会打印实际backend、每请求行数、捕图bs和余量。等68显示`server is fired up`，再在68的另一个客户端终端运行测试。若捕图报OOM，先保留第一处异常及前面的捕图bs/显存日志；不要直接上调mem-fraction-static，它不会减少现有权重或固定KV的实际占用，也不保证增加捕图可用空间。此次尚未实机捕图，不能提前归因于DeepEP问题。

在68客户端打开测试入口：

~~~bash
cd /home/tyj/glm52/sglang
vi devtools/glm52_ms1/two_node_colocated/run_tests.py
~~~

按`i`，将顶部这几项设为以下值；`GSM8K_SOURCE = ""`表示本次使用仓库自带十题，不删除已经上传的全量文件。模型路径及`ACCURACY_MAX_TOKENS`沿用刚才eager精度测试，避免预算不同：

~~~python
MODE = "dspark-graph"
HOST = "61.47.19.68"
TP_SIZE = 32
DP_SIZE = 4
GSM8K_LIMIT = 10
GSM8K_SOURCE = ""
~~~

按`Esc`、输入`:wq`、回车保存，然后只运行：

~~~bash
python3 devtools/glm52_ms1/two_node_colocated/run_tests.py accuracy --dataset gsm8k
~~~

开头应打印`gsm8k: 固定10题`。本次只测GSM8K、不跑GPQA；温度0、并发1、真实EOS，同一批请求同时评分和采集A/P/N，不需要再发另一轮接受率请求。结束后把终端摘要及Evidence目录下`summary.json`贴回：检查10题是否全部完成、正确数、截断/未解析项、`acceptance.accept_rate=sum(A)/sum(P)`，以及`graph.target_replay_observed`是否为true。服务没有其他测试流量时，图计数增长支持Target Verify实际回放；HTTP没有独立draft replay计数，draft捕图完成也不单独证明每轮draft都回放。该十题结果不等于全量精度或压力接受率>50%准出，非流式TTFT/TPOT不用于性能比较。

本轮只对齐参数和补充手册，没有修改SGLang框架、算子或测试入口。核对基线为sync `35edd9ec4b`：图档位过滤在`python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py:get_batch_sizes_to_capture`，对齐因子在`python/sglang/srt/utils/common.py:get_cuda_graph_batch_size_alignment`，Target/Draft NPU捕图日志在`python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py:capture_decode_graph`，测试图计数在`devtools/glm52_ms1/two_node_colocated/client_common.py:graph_evidence`。项目学习手册23.7解释eager与graph证据为何分开，24.4解释dense draft的TP/DP分组；教材integration快照用于概念说明，本轮判断依据上述sync源码。

### 10.2 后续四组精度对照

**目的**：比较同题同提示下DSpark与target-only的最终答案，并检查graph是否引入变化。这里测真实题目，允许提前结束；不是128k/1k性能负载。

按下面表逐组执行。每换一组，先在**两台服务终端**停止旧服务，把两台.sh顶部MODE、GRAPH改成表中值，NODE_RANK保持各自0/1，再分别运行原双机脚本。客户端Python的MODE也改成对应值。

| 顺序 | 两台服务.sh的MODE | 两台GRAPH | 客户端run_tests.py的MODE |
|---|---|---:|---|
| 1 | 'target-only' | 0 | "target-eager" |
| 2 | 'dspark' | 0 | "dspark-eager" |
| 3 | 'target-only' | 1 | "target-graph" |
| 4 | 'dspark' | 1 | "dspark-graph" |

两端各启动一次，命令相同，rank不同：

~~~bash
cd /home/tyj/glm52/sglang
bash devtools/glm52_ms1/two_node_dspark_static.sh
~~~

等节点0显示server is fired up。在节点0**客户端终端**运行：

~~~bash
python3 devtools/glm52_ms1/two_node_colocated/run_tests.py accuracy
~~~

每运行一次，先GSM8K十题、再GPQA十题，分别保存正确率、逐题结果及同次请求的A/P/N。工具先检查服务实际模式，不会以你写的MODE名称代替服务配置。GSM要求最终明确数字，GPQA要求Answer: A/B/C/D；标准答案只供客户端评分，不发给模型。

判读和后续：

- 有HTTP/通信错误：当前组不完整，先修错误，暂不做性能结论。
- 有length截断或最终答案未解析：保留在10题分母中，不能算正确。先人工检查响应；需要提高输出预算时，统一修改ACCURACY_MAX_TOKENS，并重跑所有比较组，不能只给某一组更多预算。
- DSpark把target答对的题答错：对比表列出lost_correct和首次token差异，回到同题的proposal/verify/accept定位；不能用其他题提高总分来掩盖。
- eager正常、graph异常：优先查graph输入刷新、metadata、回放和提交状态；不直接归因于草稿训练或192算子。
- 四组完成后可说“这两份十题样本是否观察到精度下降”。**10题分辨率为10个百分点，不能据此证明整个模型精度不下降。**

当前DSpark沿用static、block8/verify9、QuaRot original、TP32/DP4/EP32；不打开模拟接受率、不用临时替换Attention来美化结果。这里“开启static能力”与“已全部验证通过”分开：

| 能力 | 本轮证据 | 仍需留意 |
|---|---|---|
| proposal → verify → accept/commit | 请求返回、A/P/N、最终答案及连续生成 | 正常文本不单独证明每处KV正确 |
| eager / graph | 分别启动和同题比较；前后/metrics记录target decode/verify图计数 | 配置graph不等于实际回放 |
| DSpark草稿图 | 保留两端完整启动日志，确认草稿捕图完成；必要时另取实际回放trace | HTTP目前没有独立draft replay计数，不能仅凭target图计数宣布全链路图通过 |
| 多DP并发与prefix复用 | 下一节DP4/C4压力及三档实际cache计数 | 短题C1不覆盖这些能力 |
| 其他static边界（abort、retract、chunked prefill等） | 当前请求记录retraction并拒绝混入常规性能结果；分块prefill待本轮quick验证 | chunk参数启用不等于功能通过；异常回收还需专门用例 |

本轮不通过修改SGLang或kernel默认行为来绕开失败。新的框架补丁仍单独给你审视。

<a id="step11"></a>
## 11. 原生bench_serving：128k/1k GSP quick与并发4压测

**2026-09-11按负责人要求改为原生CLI完整流程。** 当前 `run_tests.py quick` 和 `performance` 都直接启动 `python3 -m sglang.benchmark.serving`，使用社区 `generated-shared-prefix` 数据集。输入生成、请求发送、服务路由、流式计时与指标计算均由原生代码执行。取消自定义输入裁剪、cache_salt、固定DP路由、前缀预热和响应解析hook。不新增运行脚本，不修改SGLang框架、算子、模型或服务启动配置。

**当前仍为随机合成GSP，不是连贯自然文本。** 原生 `gen_prompt()` 随机抽词表token再decode；每组是公共前缀加独立后缀。默认单轮，不将前一条回答加入下一条历史。此前的自然文本候选未接入本入口；本轮按最新要求先恢复原生跑法。

<a id="capacity-first"></a>
### 11.0 更新客户端后，跑原生quick

现有68/69服务保持已跑通的TP32/DP4、DSpark eager、context与KV上限133120，以及当前显存和chunk参数。当前容量check已通过；此前未分块长prefill发生过OOM，负责人随后已能运行0%和50%档，但低接受率原因仍未闭环。更新本客户端无需重启服务。启动参数历史解释留在本轮审视记录，不再作为新启动步骤。

**仅更新客户端两个文件，不覆盖两端已手改的启动脚本。** 在68服务容器的客户端终端执行。第一行复制现有客户端留底，之后只取本轮客户端和手册：

~~~bash
cd /home/tyj/glm52/sglang
cp devtools/glm52_ms1/two_node_colocated/run_tests.py /home/tyj/glm52-ms1/run_tests.before-native.py
git fetch origin sync/glm52-dspark-ms1
git restore --source=origin/sync/glm52-dspark-ms1 -- devtools/glm52_ms1/two_node_colocated/run_tests.py devtools/glm52_ms1/README.md
~~~

本轮默认已填好HOST=61.47.19.68、MODE=dspark-eager、TP_SIZE=32、DP_SIZE=4和/home/weights的两份模型路径。若现场模型路径、MODE或RESULTS不同，打开下面文件，只把顶部对应等号右边的值改为现场值，其他代码不改：

~~~bash
vi devtools/glm52_ms1/two_node_colocated/run_tests.py
~~~

按i编辑，文字保留双引号，数字不加引号；Esc后输入:wq并回车保存。原来的完整文件备份在/home/tyj/glm52-ms1/run_tests.before-native.py，可对照顶部值，不要用旧文件覆盖新逻辑。check仍只是已有只读容量检查，不负责发请求；服务配置没变且已READY时不用重复check。

**先让其他测试客户端停止发请求。每档原生命令会通过--flush-cache清理这套测试服务的缓存。** 然后执行：

~~~bash
python3 devtools/glm52_ms1/two_node_colocated/run_tests.py quick
~~~

当前DP4下，quick每档8条、并发1，按0%、50%、90%顺序运行，共24条正式请求。数量取2×DP_SIZE，使普通轮询路由有机会在第二轮复用第一轮留下的前缀；客户端不固定DP，也不保证调度一定均匀。关闭额外预热，保留冷启动请求，因此整档实际命中率会低于后续热请求的命中率。它不再是旧版每档1条的quick，24条长请求会比旧版耗时更长。

原生命令使用的长度参数如下：

| 共享前缀标签 | gsp-system-prompt-len | gsp-question-len | gsp-output-len |
|---|---:|---:|---:|
| 0% | 0 | 131072 | 1024 |
| 50% | 65536 | 65536 | 1024 |
| 90% | 117964 | 13108 | 1024 |

共同参数为 `--gsp-num-groups 1 --gsp-range-ratio 1 --request-rate inf --seed 42 --temperature 0 --warmup-requests 0 --flush-cache --cache-report --output-details`。保持原生流式输出和默认ignore_eos，不使用chat模板或token裁剪。相比负责人命令，仅补足名义长度合计131072、改为当前GLM-5.2路径、设置本轮请求数/并发和缓存/输出参数。

**131072是生成参数的合计，不再承诺实际输入精确等于131072。** 原生GSP经过decode、换行拼接和encode，实际长度可能不同；看原生JSONL的input_lens。0/50/90也只是共享前缀比例，实际缓存命中看cache_report和逐请求cached_tokens。BOS、页边界、DP路由、缓存淘汰均会影响实测；不能把标签当成命中达标。

### 11.1 quick结果怎么看

每轮结果保存在终端打印的Evidence目录，路径仍在既有RESULTS下：

~~~text
/home/tyj/glm52-ms1/dual-node-validation-20260910/evidence/two-node-colocated/gsp-native-dspark-eager-时间-编号/
~~~

目录内prefix0.jsonl、prefix50.jsonl、prefix90.jsonl是原生benchmark结果；同名.log保留完整终端输出，.command.txt保存实际执行命令。没有旧版cache*/summary.json，也不手工构造一份等价结果。原生CLI异常退出或完成数不足、输出不足1024时，入口停止后续档，保留已有日志。

重点看completed、input_lens/output_lens、errors、cache_report、mean_ttft_ms、mean_tpot_ms、output_throughput和accept_length。由于是流式，TTFT/TPOT可以用于观察；输出长度仍要求1024。实际输入长度或命中比例不达验收要求时，只能记为本次原生负载结果，不能给精确128k/指定缓存命中的通过结论。

**原生结果当前只保存Accept length，没有逐请求accepted/proposed草稿计数。** 该值来自服务内部统计，可能受DP和统计窗口影响；不直接换算为A/P，也不标记接受率>50%通过。之前工具的逐请求A/P能力没有冒充原生功能保留下来。精确压力接受率采集仍有缺口，后续单独处理；GSM8K精度入口原有的逐请求接受计数不受影响。

<a id="gsp-load-current"></a>
### 11.2 quick完成后跑原生并发4压测

现有入口顶部已设PERFORMANCE_REQUESTS=64、PERFORMANCE_CONCURRENCY=4。在同一个客户端终端执行：

~~~bash
python3 devtools/glm52_ms1/two_node_colocated/run_tests.py performance
~~~

每档64条、最多4条同时在途，共192条。等价于负责人提供的原生命令将并发8改为当前4，再分别设置三档前后缀长度；不是自定义worker固定每DP一路。服务按原有路由和容量排队。C4只代表客户端最多4条在途，不保证设备持续满载；目前服务可启动、C1可执行并不证明C4峰值可承载。

每档重新执行原生CLI并清缓存。清缓存后直接测量，没有额外前缀预热，因此冷请求包含在测量中。三档分别保留结果，不能混合比例后算通过。某档失败先看.log与服务首处异常，不自动降低并发、缩短输入、重启或重复请求。

压测后按[9.4](#gsm8k-full)将GSM8K_LIMIT设1319、GSM8K_SOURCE设完整本地test.jsonl路径，再执行：

~~~bash
python3 devtools/glm52_ms1/two_node_colocated/run_tests.py accuracy --dataset gsm8k
~~~

这是全量题面精度与A/P接受率采集，和原生随机GSP保持分开；两种测试不要同时运行。本轮GPQA暂停，graph和target-only/NEXTN后续分别起对应服务、改客户端MODE后复用命令。

### 11.3 本轮源码与验证边界

阶段为客户端测试协议替换，不是proposal/verify/kernel修复。原生入口见[run_tests.py](two_node_colocated/run_tests.py)的native_gsp；旧[bench_prefix.py](two_node_colocated/bench_prefix.py)只供check及历史诊断读取，quick/performance不调用其run_scenario。基线为sync 7a93c2be53；本轮临时devtools永久排除正式PR。学习[28.14 压测工具能提供什么](/Users/yuejiat/workspace/model-inference/glm52-dspark-npu-project/learning/glm52-dspark-complete-guide.md:4886)用于区分共享比例与实际命中，是integration概念快照；执行以当前sync原生源码为准。

本地验证包括真实子进程边界、命令选项、日志/JSONL保留、非零退出以及原生零退出但请求失败/输出不足时停止。Mac上的原生模块完整导入因没有Triton受限，未运行NPU或宣称CLI实机通过。负责人下一步执行quick并回传三个原生JSONL；完整后再performance及全量GSM8K。

<a id="step12"></a>
## 12. 生成一份对比表

**本节旧汇总支持精度与历史自定义GSP，不读取本轮gsp-native结果。** 本轮原生性能直接看第11节的prefix0/50/90.jsonl；不要将旧comparison.md中的GSP数据当成本轮原生结果。GSM8K逐题配对仍可使用下面命令。

在节点0客户端运行，不发新请求：

~~~bash
python3 devtools/glm52_ms1/two_node_colocated/run_tests.py report
~~~

直接打开/home/tyj/glm52-ms1/dual-node-validation-20260910/comparison.md。
同目录comparison.json保留逐题配对、丢失/恢复正确题、首次token差异、各档A/P/N、延迟倍率、图观测和所有原始结果路径。

报告采用每模式/阶段**最新一次尝试，包括失败**，历史都保留；不是选最高分。未跑、失败、真实输入或协议不同均会标出。检查不完整记录之后再重跑；换镜像/权重/DeepEP包时建议新建RESULTS轮次，避免把前后环境混在同一张对比表。

提交本轮结果时保留：comparison.md/json、各阶段summary.json、两端完整启动日志、代码commit与镜像digest、实际DeepEP包版本/路径。GPQA题面、响应全文、权重和.npy留内网，常规讨论只给聚合指标与错误摘要。

下一步由结果决定：精度或graph不一致先定位；缓存/长度不匹配先修场景；上述均成立但接受率<50%才分析该负载下proposal与verify的失配；成立且>50%后再看吞吐/TPOT是否真的优于target-only和NEXTN。

<a id="step13"></a>
## 13. DeepEP预案：提前编译备用，确认问题后再启用

**目的**：把编译、依赖和回退提前准备好，避免遇到已确认的A3混部通信问题时才临时找包。可以提前下载源码、检查环境、编译wheel；本节不要求现在替换服务包，更不把所有挂死都认定为同一个bug。

### 13.1 修复是什么、替换哪个包

本轮2026-09-10重新核对：

- 个人修复分支固定在[fb0d5017c1465500fac8ab60ece7108242aad863](https://github.com/666syh/sgl-kernel-npu/tree/fb0d5017c1465500fac8ab60ece7108242aad863)，不是此前5209d62d。后续若分支前进，仍按这个固定提交构建本轮候选。
- 主社区已合入[PR #785：A3混部DeepEP挂死修复](https://github.com/sgl-project/sgl-kernel-npu/pull/785)，合入提交87153b4；本次核验主干f4b858ce与上述候选的csrc/deepep/ops目录无差异。**这不表示0904镜像中的二进制已经带修复**；不能单靠包版本日期判断。
- 该修复为normal与low-latency通信重新划分HCCL window中的通知、状态和selector区域，避免混部模式切换时互相覆盖。新布局由DEEPEP_HYBRID_DEPLOYMENT启用。数据区仍有共享约束，不代表可以任意并发两种通信。
- 安装对象是完整**deep_ep wheel**，包括Python接口、deep_ep_cpp扩展及vendors/hwcomputing自定义算子。不是只换一个.py或.so，也不是重装sgl_kernel_npu、Torch、torch-npu、CANN或驱动。
- 现有192维Python算子overlay继续由启动.sh加载；它不会自动重编译或替代DeepEP。

固定候选和当前主干的build.sh内容相同。本地实际执行过help：**正确写法是bash build.sh -h；--help不被这个脚本接受**。只编DeepEP使用-a deepep；直接bash build.sh会构建其他包，本预案不用。

### 13.2 现在就查两端环境

在**两台现有测试容器**分别执行，可以在服务空闲时查询：

~~~bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
npu-smi info -t board -i 0 -c 0
python3 -c "import acl; print('Actual SoC:', acl.get_soc_name())"
python3 -m pip show torch torch-npu deep-ep sgl-kernel-npu triton-ascend wheel setuptools pybind11
command -v python3 pip3 cmake g++ make msopgen ccec
cmake --version
uname -m
python3 -c "import sys; print(sys.executable); print(sys.version)"
~~~

核对两台的Python版本/架构、Torch、torch-npu、CANN开发工具一致，再共享同一个wheel。之前是Python3.12.13、Torch2.10.0、torch-npu2.10.0.post4、triton-ascend3.2.2，**以本次实际容器为准**。pip3也必须指向这个Python，不能在EvalScope环境编译而给服务环境安装。

**SoC必须单独核实**：之前设备属性是Ascend910_9362。固定build.sh的A3构建档只有Ascend910_9382；自动检测看到通用Ascend910也会选择9382，甚至无npu-smi时默认9382。这是仓库的A3编译配方，不能把打印出来的9382当成实际硬件型号。

以下命令只展示当前CANN中的相关平台定义，帮助核验编译目标；不会修改配置：

~~~bash
python3 - <<'PY'
import os
from pathlib import Path
root = Path(os.environ["ASCEND_HOME_PATH"])
for name in ("Ascend910_9362.ini", "Ascend910_9382.ini"):
    found = list(root.rglob(name))
    print(name, [str(p) for p in found])
    for path in found:
        for line in path.read_text().splitlines():
            if any(key in line for key in ("SoC_version", "AIC_version", "AIV_version", "CCEC", "ai_core_cnt", "vector_core_cnt")):
                print(line)
PY
~~~

能找到两份配置、Short_SoC_version一致也不能单独保证二进制兼容。后面给的是仓库支持的**A3候选构建命令**，可以提前编出备用包；对9362设备的支持仍需同事/工具链兼容依据及实机小请求确认。若msopgen、编译器或运行时明确报不支持该SoC，保留首个错误；不能自行把允许列表改成9362来宣称已支持。当前本地没有CANN/NPU，尚未替你完成这个实机检查。

顶层CMake要求版本至少3.20；config_envs.cmake还会直接import pybind11，不能只检查Torch。缺cmake、g++、make、msopgen、ccec或CANN头文件，优先使用部门配套的开发镜像/工具链准备；不要因此在推理环境升级Torch/CANN。尤其内网环境要在测试前发现这些缺口。

### 13.3 联网PC准备一份可离线构建的源码

**在能联网的PC执行**，不在现有sglang或192 kernel checkout里面操作。固定源码放到单独的临时构建目录：

~~~bash
mkdir -p glm52-deepep-package
cd glm52-deepep-package
git clone --branch a3_stuck --single-branch https://github.com/666syh/sgl-kernel-npu.git sgl-kernel-npu-a3-stuck
cd sgl-kernel-npu-a3-stuck
git checkout fb0d5017c1465500fac8ab60ece7108242aad863
mkdir -p csrc/deepep/ops/third_party
git clone --branch catlass-v1-stable --single-branch https://gitcode.com/cann/catlass.git csrc/deepep/ops/third_party/catlass
git rev-parse HEAD
git -C csrc/deepep/ops/third_party/catlass rev-parse HEAD
git -C csrc/deepep/ops/third_party/catlass rev-parse 'catlass-v1-stable^{commit}'
cd ..
mkdir -p build-wheels
python3 -m pip download --index-url https://mirrors.aliyun.com/pypi/simple/ --only-binary=:all: --no-deps --dest build-wheels wheel==0.45.1 setuptools pybind11
tar -czf glm52-deepep-a3-stuck-fb0d5017.tar.gz sgl-kernel-npu-a3-stuck build-wheels
~~~

这里保留两个.git目录；不能用不含.git的GitHub ZIP代替。DeepEP构建内部会检查csrc/deepep/ops/third_party/catlass，HEAD与本地catlass-v1-stable必须相同，才跳过联网fetch。普通顶层third_party/catlass不能代替这个路径。归档固定了此次实际Catlass提交，后续无需再跟随网络分支更新。

如果GitCode访问失败，先在能访问该站的PC完成这一步，不要把缺Catlass的压缩包送进去等到编译时失败。wheel是build.sh会自动补装的小依赖；setuptools/pybind11也随包准备，遵循仓库CI没有固定它们版本的方式，实际版本由这份压缩包固定。服务器已安装时保留现有版本，只补缺失包；其余工具链通过13.2确认。

把glm52-deepep-a3-stuck-fb0d5017.tar.gz转到节点0的/home/tyj/glm52/，传法与GPQA相同。例如在能访问节点0的PC执行：

~~~bash
scp glm52-deepep-a3-stuck-fb0d5017.tar.gz root@61.47.19.71:/home/tyj/glm52/
~~~

### 13.4 节点0容器编译，只生成备用wheel

编译会大量使用CPU和内存，**不要与性能测量同时运行**。以下都在节点0测试容器执行。新建临时目录，若已存在则先核对内容，不覆盖另一轮构建。

~~~bash
cd /home/tyj/glm52
tar -xzf glm52-deepep-a3-stuck-fb0d5017.tar.gz
cd /home/tyj/glm52/sgl-kernel-npu-a3-stuck
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
git rev-parse HEAD
git -C csrc/deepep/ops/third_party/catlass rev-parse HEAD
git -C csrc/deepep/ops/third_party/catlass rev-parse 'catlass-v1-stable^{commit}'
bash build.sh -h
python3 - <<'PY'
import importlib.metadata as metadata
import subprocess
import sys
for name in ("wheel", "setuptools", "pybind11"):
    try:
        print(name, metadata.version(name), "已安装，保留")
    except metadata.PackageNotFoundError:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                        "--find-links", "/home/tyj/glm52/build-wheels", name], check=True)
PY
~~~

源码HEAD必须是fb0d5017开头；两个Catlass值必须相同。接着运行仓库提供的A3候选构建档：

~~~bash
set -o pipefail
export PIP_NO_INDEX=1
export PIP_FIND_LINKS=/home/tyj/glm52/build-wheels
VERSION=1.0.0 bash build.sh -a deepep Ascend910_9382 2>&1 | tee build-deepep.log
~~~

**只在上述命令成功退出后**继续。出现错误先看build-deepep.log里的第一个错误，后面的打包失败通常只是连带结果。这里没有修改build.sh或算子源码，也没有pip安装新DeepEP；输出备用包不影响当前服务加载的已安装包。

检查产物并保存指纹：

~~~bash
python3 - <<'PY'
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
wheels = list(Path("output").glob("deep_ep*.whl"))
if len(wheels) != 1:
    raise SystemExit(f"需要唯一wheel，发现{len(wheels)}个；先核对本次构建产物，不能通配安装多份")
wheel = wheels[0]
with zipfile.ZipFile(wheel) as archive:
    names = archive.namelist()
    extension = [n for n in names if "deep_ep_cpp" in n and n.endswith(".so")]
    vendors = [n for n in names if "/vendors/hwcomputing/" in n]
    if not extension or not vendors:
        raise SystemExit("wheel缺少扩展或vendors自定义算子，不能安装")
manifest = {
    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "catlass_commit": subprocess.check_output(["git", "-C", "csrc/deepep/ops/third_party/catlass", "rev-parse", "HEAD"], text=True).strip(),
    "wheel": wheel.name,
    "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
    "extension": extension,
    "vendor_files": len(vendors),
    "device_validation": "NOT_RUN",
}
Path("output/build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY
python3 -m pip freeze > output/build-python-packages.txt
~~~

wheel名称应包含fb0d5017和当前Python/架构信息；CANN版本后缀是否出现取决于安装元数据，不能只看wheel文件名验证ABI。两端环境相同再复用同一wheel；不同则在各自匹配的容器独立构建，不能硬装。

把产物传到节点1，**在节点0宿主机**执行：

~~~bash
ssh root@61.47.19.70 mkdir -p /home/tyj/glm52/deepep-wheel
scp /home/tyj/glm52/sgl-kernel-npu-a3-stuck/output/deep_ep*.whl /home/tyj/glm52/sgl-kernel-npu-a3-stuck/output/build-manifest.json root@61.47.19.70:/home/tyj/glm52/deepep-wheel/
mkdir -p /home/tyj/glm52/deepep-wheel
cp /home/tyj/glm52/sgl-kernel-npu-a3-stuck/output/deep_ep*.whl /home/tyj/glm52/sgl-kernel-npu-a3-stuck/output/build-manifest.json /home/tyj/glm52/deepep-wheel/
~~~

此时两台都有同一候选包，**可以先保持原包跑测试**。暂时不用就停在这里。

### 13.5 什么情况下启用

先保留两台日志和首个异常，区分：

| 现象 | 优先核查 |
|---|---|
| 卡在Init torch distributed，还未进MoE | 节点0 IP、rank、端口、两端启动、网卡与HCCL拓扑 |
| 明确OOM / allocation failed | 每个rank的权重/KV/graph空间，不把普通capture建议当OOM证据 |
| normal/low-latency切换或MoE dispatch/combine中卡死，伴HCCL窗口/状态等待问题 | 对照PR #785修复条件、已加载DeepEP及HYBRID设置；此时使用备用包做对照 |

该分支A3通信说明依赖机内/机间HCCS拓扑；HTTP能互通或ping正常并不证明两台属于支持的HCCS通信域。连接拓扑不成立也不能靠装修复包解决。

不要只因堆栈出现aclnnMoeLowLatencyDispatchV2就宣布命中。DEEP_USE_MODE=ops时low-latency可走torch_npu接口，而新自定义包修复的是另一条实现；下面对照固定DEEP_USE_MODE=default。**是否原问题就是此bug，仍需相同输入、配置和前后日志支持。**

### 13.6 两端备份，再安装同一份deep_ep

确认要测试候选后，先停两台模型服务，确认没有该服务残留占用NPU。在**两台宿主机分别执行**，保存当前测试容器文件系统为本地回退镜像：

~~~bash
docker commit glm52-test glm52-before-deepep:20260910
~~~

这个名字已填完整，不需你造IMAGE变量。它保存容器内的原包，但**不会备份/home挂载的代码、模型、结果**。这里也不改这些文件；若已有同名回退镜像，应换个日期/编号保留旧版本。记录本次SGLang/kernel的git rev-parse HEAD。

然后在**两台容器分别执行相同命令**，仅选唯一的deep_ep wheel：

~~~bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
cd /home/tyj/glm52
python3 - <<'PY'
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path
root = Path("/home/tyj/glm52/deepep-wheel")
manifest = json.loads((root / "build-manifest.json").read_text())
wheel = root / manifest["wheel"]
if hashlib.sha256(wheel.read_bytes()).hexdigest() != manifest["sha256"]:
    raise SystemExit("wheel SHA与构建记录不同，停止安装")
with zipfile.ZipFile(wheel) as archive:
    names = archive.namelist()
    if not any("/vendors/hwcomputing/" in name for name in names):
        raise SystemExit("没有vendors，停止安装")
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--force-reinstall", str(wheel)], check=True)
PY
~~~

--no-deps保证不借机重装Torch等依赖。固定仓库README另要求顶层deep_ep_cpp链接；安装后在**两台容器**运行以下完整段，它只处理这个包的扩展入口，遇到独立旧二进制会保留为备份：

~~~bash
python3 - <<'PY'
import importlib.metadata as metadata
import shutil
from pathlib import Path
dist = metadata.distribution("deep-ep")
site = Path(dist.locate_file(""))
package = site / "deep_ep"
extensions = list(package.glob("deep_ep_cpp*.so"))
if len(extensions) != 1:
    raise SystemExit(f"deep_ep内扩展数量不是1：{extensions}")
source = extensions[0]
for old in site.glob("deep_ep_cpp*.so"):
    if old.is_symlink():
        old.unlink()
    else:
        backup = old.with_name(old.name + ".before-a3-stuck")
        if backup.exists():
            raise SystemExit(f"已有旧扩展备份，请先核对：{backup}")
        shutil.move(str(old), str(backup))
(site / source.name).symlink_to(Path("deep_ep") / source.name)
print("deep-ep版本：", dist.version)
print("扩展链接：", (site / source.name).resolve())
PY
python3 - <<'PY'
import hashlib
import importlib.metadata as metadata
import os
from pathlib import Path
import torch
import torch_npu
import deep_ep
import deep_ep_cpp
package = Path(deep_ep.__file__).resolve().parent
extension = Path(deep_ep_cpp.__file__).resolve()
assert extension.parent == package, (extension, package)
assert (package / "vendors/hwcomputing").is_dir()
print("deep-ep version:", metadata.version("deep-ep"))
print("deep_ep:", deep_ep.__file__)
print("deep_ep_cpp:", extension)
print("extension sha256:", hashlib.sha256(extension.read_bytes()).hexdigest())
print("ASCEND_CUSTOM_OPP_PATH:", os.environ.get("ASCEND_CUSTOM_OPP_PATH"))
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
PY
~~~

两端版本、wheel SHA、扩展SHA应一致。导入成功只证明包入口能加载，不证明通信/SoC/graph已经可用。源码中的deep_ep初始化会把配套vendors目录加入自定义算子路径；保留这些打印，排除误载其他旧目录。

### 13.7 启用新布局、重启、先小请求再压测

先把两台.sh设为MODE='dspark'、GRAPH=0。在**两台各自服务终端**，每次候选测试启动前执行：

~~~bash
export DEEPEP_HYBRID_DEPLOYMENT=1
export DEEP_USE_MODE=default
cd /home/tyj/glm52/sglang
bash devtools/glm52_ms1/two_node_dspark_static.sh
~~~

当前启动.sh会继承这两个变量，作用于全部32个rank。**不能只在客户端设置，不能只设置节点0。**该分支用“环境变量是否存在”判断HYBRID，写0或空字符串不等于关闭；关闭需unset。

先等服务就绪，客户端MODE="dspark-eager"，运行accuracy完成两套短题并记录新包结果；再切graph跑accuracy，核验原来的卡死触发步骤；两组均可用再进入check、quick、performance。若只在原问题的特定输入/模式切换上失败，还要用同一触发条件复现对照，不能以一条短题成功就宣布修复。

若启用候选后继续本轮完整性能对比，**所有被比较模式都使用这同一DeepEP包和环境**，并给run_tests.py的RESULTS换一个新轮次目录，不混用原包target-only结果。

### 13.8 回退到替换前的容器包

两台都停止服务。**在两台宿主机分别执行**，保留候选容器便于查日志，不删除：

~~~bash
docker stop glm52-test
docker rename glm52-test glm52-deepep-candidate
~~~

然后使用[第4.2节的完整docker run命令](#step4)创建新glm52-test，**只把最后一行镜像名整行换成下面这行**，其余设备和/home挂载保持原样：

~~~text
  glm52-before-deepep:20260910
~~~

这是恢复13.6保存的原容器包，不是重新安装一个名字相似的PyPI包。新容器服务终端执行：

~~~bash
unset DEEPEP_HYBRID_DEPLOYMENT
unset DEEP_USE_MODE
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
python3 -m pip show deep-ep
python3 -c "import deep_ep; print(deep_ep.__file__)"
cd /home/tyj/glm52/sglang
bash devtools/glm52_ms1/two_node_dspark_static.sh
~~~

两端一起回退，不能一端新包一端旧包。/home里的临时构建目录和证据继续保留；服务恢复后重新核对模式和小请求。

<a id="step14"></a>
## 14. 本轮范围、源码与学习位置

本轮修改范围是devtools客户端和这份手册：不改python/sglang中的模型、调度、attention、graph，也不改kernel算子；不自动连接NPU、切服务或安装DeepEP。临时诊断入口仍只在sync/glm52-dspark-ms1分支；未来正式PR的测试入口需另外收敛，不能把本工具交付称为主社区准出完成。

| 当前环节 | 已有源码与本轮改动 | 阅读目的 |
|---|---|---|
| 双机配置/加载 | [two_node_dspark_static.sh](two_node_dspark_static.sh)，本轮保持服务参数 | 看MODE/GRAPH怎样变成已有启动参数 |
| 精度请求与Accept统计 | [bench_accuracy.py](two_node_colocated/bench_accuracy.py)、[gsm8k_mode_stats.py](gsm8k_mode_stats.py)；新增[run_tests.py](two_node_colocated/run_tests.py)直接传顶部设置，复用评分与计数 | 同次HTTP响应既评分也取A/P/N |
| cache与压力 | [bench_prefix.py](two_node_colocated/bench_prefix.py)复用[serving.py](../../python/sglang/benchmark/serving.py)发送/计时，新增直接参数入口和记录实际服务配置 | 看每DP预热、实际长度/缓存校验和聚合分母 |
| 对照结果 | [report.py](two_node_colocated/report.py)新增一张精度/性能矩阵，比较实际输入指纹，保留最新失败 | 防止比较不同请求或择优挑结果 |
| DeepEP包构建 | [固定build.sh](https://github.com/666syh/sgl-kernel-npu/blob/fb0d5017c1465500fac8ab60ece7108242aad863/build.sh#L35)、[Catlass准备](https://github.com/666syh/sgl-kernel-npu/blob/fb0d5017c1465500fac8ab60ece7108242aad863/csrc/deepep/ops/build.sh#L24)、[打包](https://github.com/666syh/sgl-kernel-npu/blob/fb0d5017c1465500fac8ab60ece7108242aad863/build.sh#L453)、[加载vendors](https://github.com/666syh/sgl-kernel-npu/blob/fb0d5017c1465500fac8ab60ece7108242aad863/python/deep_ep/deep_ep/__init__.py#L5) | 区分源码下载、编译、安装、实际运行四步 |

本轮与学习手册对应：23.7“Eager和Graph必须分开建立能力”、28.1“四种token数与接受率分母”、28.7“评测协议如何与参考值对齐”、28.11—28.12“长上下文与命中率/准备测量”、28.14—28.15“压测工具/公平比较”。概念及历史快照见负责人工作区[完整学习手册](/Users/yuejiat/workspace/model-inference/glm52-dspark-npu-project/learning/glm52-dspark-complete-guide.md:4197)；远程SGLang仓不包含该本地项目手册。DeepEP本轮构建没有专属DSpark教材章节，本页第13节和固定源码是直接依据。
