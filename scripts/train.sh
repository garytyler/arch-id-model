#!/usr/bin/env bash

msg() {
    echo >&2 $(basename $(readlink -f "$0")): "$*"
}

if [[ ! "$1" ]]; then
    msg "notebook name required."
    exit
fi

REPO_DIR="$(dirname $(dirname $(realpath -s $0)))"
cd "$REPO_DIR"
nvidia-docker run -it --rm \
    -v "$REPO_DIR:/tmp:rw" -w /tmp \
    tensorflow/tensorflow:latest-gpu-jupyter \
    bash -c "cd notebooks/$1 && python $1.py"
