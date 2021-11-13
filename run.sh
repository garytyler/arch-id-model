#!/usr/bin/env bash

set -e

THIS_DIR=$(realpath $(dirname $0))
IMAGE_NAME=arch-id

if [[ "${@}" =~ .*([ ]-d[ =]|[ ]--dataset-dir[ =]).* ]]; then
    echo 'Use DATASET_DIR environment variable instead of -d/--dataset-dir arguments'
    exit 22
fi

if [[ "${@}" =~ .*([ ]-o[ =]|[ ]--output-dir[ =]).* ]]; then
    echo 'Use OUTPUT_DIR environment variable instead of -o/--output-dir arguments'
    exit 22
fi

docker build \
    --tag ${IMAGE_NAME} \
    --target=train-stage \
    --build-arg="USERNAME=$(whoami)" \
    -- ${THIS_DIR}/

echo $NVIDIA_VISIBLE_DEVICES
RUN_CMD=(
    docker run
    --publish=6006:6006
    # mount ./ to /srv
    -v="${THIS_DIR?}:/srv:rw"
    # if /srv/dataset exists, void it with an empty volume
    $(if [ -d "${THIS_DIR}/dataset" ]; then echo "-v=:/srv/dataset:ro"; fi)
    # if /srv/output exists, void it with an empty volume
    $(if [ -d "${THIS_DIR}/output" ]; then echo "-v=:/srv/output:ro"; fi)
    # if DATASET_DIR is given, mount it to /dataset, else mount ./dataset to /dataset
    -v="${DATASET_DIR:-${THIS_DIR}/dataset}:/dataset:ro"
    # if OUTPUT_DIR is given, mount it to /output, else bind ./output to /output
    -v="${OUTPUT_DIR:-${THIS_DIR}/output}:/output"
    --gpus="${GPUS:-all}"
    --env="NVIDIA_DRIVER_CAPABILITIES=compute,utility"
    --env="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES?}"
    --env="CUDA_DEVICE_ORDER=PCI_BUS_ID"
    --env="TF_FORCE_GPU_ALLOW_GROWTH=true"
    --env="TF_GPU_THREAD_MODE=gpu_private"
    --env="TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-2}"
    --env="TF_ENABLE_AUTO_MIXED_PRECISION=${TF_ENABLE_AUTO_MIXED_PRECISION:-1}"
    --rm
    --name training
    -- ${IMAGE_NAME}
    python -m arch_id
    "--dataset-dir=/dataset"
    "--output-dir=/output"
    ${@}
)
"${RUN_CMD[@]}"
