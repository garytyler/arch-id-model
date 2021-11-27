<div align="center">
<h1>arch-id-model</h1>
<h3><a href="https://architectureid.ai">ArchitectureID.ai</a>’s data model development.</h3>
<i>For the <a href="https://architectureid.ai" target="_blank">ArchitectureID.ai</a> web application repo, see <a href="https://github.com/garytyler/arch-id-web" target="_blank">arch-id-web</a>.</i>
<br/>
<br/>
</div>

# Quick Start

The recommended method for running the program is in a container built from the included `./Dockerfile` using [non-root (optional)](https://docs.docker.com/engine/security/rootless/) docker with [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) for GPU support. The included run script `./run.sh` will call the necessary docker commands to build the container and run the program in it.

The run script `./run.sh` accepts the following environment variables:

- `DATASET_DIR` (required):\
  dataset root directory with structure: root_dir/category_dir/image_files. replaces `--dataset-dir` CLI arg.
- `OUTPUT_DIR` (required):\
  directory for all program output. can be reused for multiple sessions. a new sub-directory will be created for each session. replaces `--output-dir` cli arg.
- `CUDA_VISIBLE_DEVICES` (optional):\
  a comma-separated list of integers reflecting the Bus ID of the GPUs to expose to the container with [NVIDIA Container Toolkit](<(https://github.com/NVIDIA/nvidia-docker)>). defaults to 0.
- `TF_CPP_MIN_LOG_LEVEL` (optional):\
  set C++ tensorflow log level. accepts one of 0,1,2,3. defaults to 2.
- `TF_ENABLE_AUTO_MIXED_PRECISION` (optional):\
  boolean to enable mixed precision. defaults to 1.

\*Notice the run script accepts environment variables `DATASET_DIR` and `OUTPUT_DIR` in place of CLI args `--dataset-dir` and `--output-dir`. All other CLI options can be passed to the script as they would if calling the program directly from your shell.

For example, to train on a GPU with Bus ID #2 with a minimum accuracy of 0.7 for a maximum of 200 epochs, with dataset directory `~/dataset` and output directory `~/output`:

```sh
CUDA_VISIBLE_DEVICES=2 DATASET_DIR=~/dataset OUTPUT_DIR=~/output ./run.sh train --min-accuracy=.7 --max-epochs=1000
```

For CLI help:

```sh
DATASET_DIR=~/dataset OUTPUT_DIR=~/output ./run.sh train -h
```
