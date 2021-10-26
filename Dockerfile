# Build training environment
FROM tensorflow/tensorflow:2.6.0-gpu-jupyter AS train-stage

# Set working dir
WORKDIR /workspace/

# Create non-root user
# ARG USER=${USER}
ARG USERNAME=${USERNAME}
ARG USER_UID=1000
ARG USER_GID=${USER_UID}
RUN !(id -u "${USERNAME}") 1>/dev/null 2>&1 \
    && groupadd -f --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME}

# Add sudo
RUN apt-get update \
    && apt-get install --no-install-recommends -y sudo \
    && echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

# Upgrade pip and install project dependencies from pip
RUN pip install --upgrade pip
RUN pip install -U \
    sklearn==0.0 \
    tensorflow-datasets==4.4.0 \
    tensorboard-plugin-profile==2.5.0

# Set as non-root user
USER ${USERNAME}

# Build development environment
FROM train-stage AS dev-stage

# Set as root user
USER root

# Install development dependencies from apt
RUN apt-get update \
    && apt-get install direnv

# Install development dependencies from pip
RUN pip install -U \
    black==21.9b0 \
    isort==5.9.3 \
    flake8==4.0.1 \
    mypy==0.910

# Set as non-root user
USER ${USERNAME}
