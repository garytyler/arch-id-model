FROM tensorflow/tensorflow:latest-gpu-jupyter AS train-stage

WORKDIR /workspace/

# Use non-root user
ARG USERNAME=${USERNAME:-developer}

# Create non-root user
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN !(id -u "$USERNAME") 1>/dev/null 2>&1 \
    && groupadd -f --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# Add sudo
RUN apt-get update \
    && apt-get install --no-install-recommends -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Upgrade pip and install project dependencies from pip
RUN pip install --upgrade pip
RUN pip install -U \
    sklearn \
    tensorflow-datasets \
    tensorboard-plugin-profile

# Set as non-root user
USER $USERNAME

FROM train-stage AS dev-stage

# Set as non-root user
USER $USERNAME

# Install docker
RUN sudo apt-get update \
    && sudo apt-get install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - \
    && sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
    && sudo apt-get update \
    && sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Install NVIDIA container toolkit
RUN DISTRIBUTION=$(. /etc/os-release && echo $ID$VERSION_ID) \
    && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
    && curl -s -L https://nvidia.github.io/nvidia-docker/$DISTRIBUTION/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list \
    && sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# Install development dependencies from apt
USER root
RUN apt-get update \
    && apt-get install direnv

# Install development dependencies from pip
RUN pip install -U \
    black \
    isort \
    flake8 \
    mypy \
    docker

# Set as non-root user
USER $USERNAME
