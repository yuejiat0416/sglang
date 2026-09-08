#!/usr/bin/env bash
# A3 container options follow the SGLang quickstart and the team's example.
# Set CONTAINER_NAME before running this file to select the second node name.

IMAGE=swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.1.0-a3-20260904
CONTAINER_NAME=${CONTAINER_NAME:-tyj-glm52-ms1-target-node0}

docker run -it \
  --name "${CONTAINER_NAME}" \
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
  -v /home/weights:/workspace/weight:ro \
  -v /home/tyj:/home/tyj \
  -e CONTAINER_NAME="${CONTAINER_NAME}" \
  -e HF_HOME=/home/tyj/glm52-ms1/cache/huggingface \
  -e TRANSFORMERS_CACHE=/home/tyj/glm52-ms1/cache/huggingface \
  -e HUGGINGFACE_HUB_CACHE=/home/tyj/glm52-ms1/cache/huggingface/hub \
  -e TORCH_HOME=/home/tyj/glm52-ms1/cache/torch \
  -e XDG_CACHE_HOME=/home/tyj/glm52-ms1/cache/xdg \
  -e PIP_CACHE_DIR=/home/tyj/glm52-ms1/cache/pip \
  --entrypoint=bash \
  "${IMAGE}"
