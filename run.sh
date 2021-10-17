#!/usr/bin/env bash

set -e

THIS_DIR=$(realpath $(dirname $0))
LOCAL_DATASET_DIR=$(realpath "${THIS_DIR}/dataset")
DATASET_SOURCE_DIR=${DATASET_SOURCE_DIR:-${LOCAL_DATASET_DIR}}

if [[ ! -d "$DATASET_SOURCE_DIR" ]]; then
    echo "Path to an existing source dataset is required as an argument. Dataset dir should have structure root/classes/images."
    exit
fi

IMAGE_NAME=arch-recognizer/trainings
docker build \
    --tag ${IMAGE_NAME} \
    --target=train-stage \
    --build-arg="USER=$(whoami)" \
    -- ${THIS_DIR}/

RUN_COMMAND="python /workspace/arch_recognizer $@"
docker run \
    --runtime=nvidia \
    --publish=6006:6006 \
    --volume="${THIS_DIR}/:/workspace/" \
    --volume="${DATASET_SOURCE_DIR}/:/workspace/dataset/:ro" \
    --env "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-1,2}" \
    --env "NVIDIA_DRIVER_CAPABILITIES=compute,utility" \
    --env "CUDA_DEVICE_ORDER=PCI_BUS_ID" \
    --env "TF_FORCE_GPU_ALLOW_GROWTH=true" \
    --env "TF_GPU_THREAD_MODE=gpu_private" \
    --env "TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-2}" \
    --rm -- ${IMAGE_NAME} \
    /bin/bash -c "${RUN_COMMAND}"
