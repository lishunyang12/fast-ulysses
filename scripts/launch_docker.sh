#!/bin/bash

docker run -it \
  --name REDACTED-CONTAINER-inference \
  --init \
  --privileged \
  --runtime=nvidia \
  --gpus all \
  --network=host \
  --ipc=private \
  --shm-size=32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --mount "type=bind,src=/data/sonlin,dst=/workspace" \
  --mount "type=bind,src=/data/cache/huggingface,dst=/root/.cache/huggingface" \
  -w /workspace \
  nvcr.io/nvidia/pytorch:26.07-py3 \
  /bin/bash