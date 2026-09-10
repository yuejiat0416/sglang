#!/usr/bin/env bash
# A3 container options follow the SGLang quickstart and the team's example.
# Set IMAGE, CONTAINER_NAME and MS1_STATE in a local file outside the checkout.

: "${IMAGE:?Set IMAGE to the available A3/CANN 9.1 image reference}"
: "${CONTAINER_NAME:?Set CONTAINER_NAME to your test container name}"
: "${MS1_STATE:?Set MS1_STATE to an absolute directory available through the /home mount}"

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
  -v /home:/home \
  -e CONTAINER_NAME="${CONTAINER_NAME}" \
  -e HF_HOME="${MS1_STATE}/cache/huggingface" \
  -e TRANSFORMERS_CACHE="${MS1_STATE}/cache/huggingface" \
  -e HUGGINGFACE_HUB_CACHE="${MS1_STATE}/cache/huggingface/hub" \
  -e TORCH_HOME="${MS1_STATE}/cache/torch" \
  -e XDG_CACHE_HOME="${MS1_STATE}/cache/xdg" \
  -e PIP_CACHE_DIR="${MS1_STATE}/cache/pip" \
  --entrypoint=bash \
  "${IMAGE}"
