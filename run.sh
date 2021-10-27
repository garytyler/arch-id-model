#!/usr/bin/env bash

set -e

echo $@
THIS_DIR=$(realpath $(dirname $0))
IMAGE_NAME=arch-recognizer

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

RUN_CMD=(
    docker run
    --publish=6006:6006
    -v="${THIS_DIR?}:/workspace:rw"                                               # mount ./ to /workspace
    $(if [ -d "${THIS_DIR}/dataset" ]; then echo "-v=:/workspace/dataset:ro"; fi) # if /workspace/dataset exists, void it with an empty volume
    $(if [ -d "${THIS_DIR}/output" ]; then echo "-v=:/workspace/output:ro"; fi)   # if /workspace/output exists, void it with an empty volume
    -v="${DATASET_DIR:-${THIS_DIR}/dataset}:/dataset:ro"                          # if DATASET_DIR is given, mount it to /dataset, else mount ./dataset to /dataset
    -v="${OUTPUT_DIR:-${THIS_DIR}/output}:/output"                                # if OUTPUT_DIR is given, mount it to /output, else bind ./output to /output
    --gpus="${GPUS:-all}"
    --env="NVIDIA_DRIVER_CAPABILITIES=compute,utility"
    --env="NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-1,2}"
    --env="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2}"
    --env="CUDA_DEVICE_ORDER=PCI_BUS_ID"
    --env="TF_FORCE_GPU_ALLOW_GROWTH=true"
    --env="TF_GPU_THREAD_MODE=gpu_private"
    --env="TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-2}"
    --env="TF_ENABLE_AUTO_MIXED_PRECISION=${TF_ENABLE_AUTO_MIXED_PRECISION:-1}"
    --rm
    --name training
    -- ${IMAGE_NAME}
    python -m arch_recognizer
    "--dataset-dir=/dataset"
    "--output-dir=/output"
    ${@}
)
"${RUN_CMD[@]}"
