# Build training environment
FROM tensorflow/tensorflow:latest-gpu-jupyter AS train-stage

# Set working dir
WORKDIR /workspace/

# Create non-root user
ARG USER=${USER}
ARG USER_UID=1000
ARG USER_GID=${USER_UID}
RUN !(id -u "${USER}") 1>/dev/null 2>&1 \
    && groupadd -f --gid ${USER_GID} ${USER} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USER}

# Add sudo
RUN apt-get update \
    && apt-get install --no-install-recommends -y sudo \
    && echo ${USER} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USER} \
    && chmod 0440 /etc/sudoers.d/${USER}

# Upgrade pip and install project dependencies from pip
RUN pip install --upgrade pip
RUN pip install -U \
    sklearn \
    tensorflow-datasets \
    tensorboard-plugin-profile

# Set as non-root user
USER ${USER}

# Build development environment
FROM train-stage AS dev-stage

# Set as root user
USER root

# Install development dependencies from apt
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
USER ${USER}
