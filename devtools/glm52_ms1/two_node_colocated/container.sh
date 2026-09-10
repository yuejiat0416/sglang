#!/usr/bin/env bash
# A3 container template. Fill these values on each test host before execution.
# Private image names, local work directories and container names stay local.
set -euo pipefail

: "${IMAGE:?Set IMAGE to the locally approved image reference}"
: "${CONTAINER_NAME:?Set CONTAINER_NAME for this test host}"
: "${HOST_WORK_DIR:?Set HOST_WORK_DIR to a private directory below /home}"
if [[ "${IMAGE}${CONTAINER_NAME}${HOST_WORK_DIR}" == *"__FILL_"* ]]; then
  echo "Replace every __FILL_*__ placeholder on the test host." >&2
  exit 2
fi
case "${HOST_WORK_DIR}" in
  /home/?*) ;;
  *) echo "HOST_WORK_DIR must be an absolute directory below the /home mount." >&2; exit 2 ;;
esac
case "${HOST_WORK_DIR}/" in
  */../*|*/./*) echo "HOST_WORK_DIR must not contain . or .. path segments." >&2; exit 2 ;;
esac
CACHE_ROOT="${HOST_WORK_DIR%/}/cache"

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
  -e "HF_HOME=${CACHE_ROOT}/huggingface" \
  -e "TRANSFORMERS_CACHE=${CACHE_ROOT}/huggingface" \
  -e "HUGGINGFACE_HUB_CACHE=${CACHE_ROOT}/huggingface/hub" \
  -e "TORCH_HOME=${CACHE_ROOT}/torch" \
  -e "XDG_CACHE_HOME=${CACHE_ROOT}/xdg" \
  -e "PIP_CACHE_DIR=${CACHE_ROOT}/pip" \
  --entrypoint=bash \
  "${IMAGE}"
