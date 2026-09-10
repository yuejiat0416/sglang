# GLM-5.2：新机器准备与单双机服务启动

**启动服务只用两个脚本，参数直接写在脚本开头。无需single.json、two.json、local.env或single.local.sh。**

- [单机完整启动脚本](single_dspark_static.sh)：在61.47.19.71运行。
- [双机混部完整启动脚本](two_node_dspark_static.sh)：在61.47.19.71、61.47.19.70各运行一次。

已有容器和模型时，直接跳到[单机启动](#step5)或[双机启动](#step6)。新机器按前四节准备。

本轮只收敛服务启动方式，保留现有DSpark、target-only、NEXTN参数；不修改框架或算子实现。单机static eager已有实测，整模型graph与双机仍待实机验证。

1. [执行位置与目录](#step1)
2. [拉代码](#step2)
3. [准备权重](#step3)
4. [拉镜像、创建容器](#step4)
5. [单机启动：改一行graph开关](#step5)
6. [双机启动：两端各运行同一脚本](#step6)
7. [检查服务、停止与更新](#step7)
8. [现有工具用途](#step8)

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

--branch表示克隆后直接使用本次个人临时调试分支；两个仓库分支同名但内容不同。私仓认证用你已有的Git凭据，不把令牌写到命令或文档。已经克隆的目录不用再clone，更新看第9章。

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

单机沿用TP16/DP1。DSpark是block8/verify9；NEXTN是steps4/topk1/draft5。
`CONTEXT_LENGTH=16384`用于先跑短题；准备128k输入/1k输出时改为`CONTEXT_LENGTH=133120`。
这是允许的序列长度，实际KV容量仍要另行检查，不能据这一行认定能装下128k。

脚本沿用`with_kernel_checkout.py`加载本次192维Python算子候选，其他二进制仍用镜像安装包；不要求重编译、不替换已安装包。此步骤不包含a3_stuck分支的DeepEP二进制修复。
`MS1_STATE='/home/tyj/glm52-ms1'`只是工具缓存和运行记录的位置，已填好，不需要先生成任何JSON。

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
| run_gsm8k_single.sh | 单机服务启动后的GSM8K十题入口；只需让顶部MODE与服务一致，然后直接运行 |
| with_kernel_checkout.py | 两个启动脚本内部使用的192维算子加载工具，无需单独操作 |
| bench_gsm8k_modes.py / GSM8K_MODES.md | 已有单机GSM8K十题客户端及协议；旧launch配置方法已由本页两个脚本替代 |
| bench_gsp_prefix.py / GSP_PREFIX.md | 已有单机131072/1024三档缓存诊断客户端及协议 |
| two_node_colocated/ | 先前的双机测试客户端、结果比较和归档工具；里面的launch.py/配置模板不再作为服务启动入口 |
| probe_*、collect_*、其他test_* | 各轮临时诊断和本地测试，启动服务不需要逐一执行 |

本页当前交付重点是两个服务脚本。数据集和压力测试的客户端不负责启动、停止或切换服务；旧测试配置不应成为上述服务启动的前置步骤。
