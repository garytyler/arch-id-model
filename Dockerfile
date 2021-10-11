FROM tensorflow/tensorflow:latest-gpu-jupyter AS train-stage

# Use non-root user
ARG USERNAME=${USERNAME:-developer}

# Create non-root user
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN !(id -u "$USERNAME") 1>/dev/null 2>&1 \
    && groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# Add sudo
RUN apt-get update \
    && apt-get install --no-install-recommends -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Upgrade pip and install project dependencies from pip
RUN pip install --upgrade pip sklearn

# Set as non-root user
USER $USERNAME

FROM train-stage AS dev-stage

# Set as non-root user
USER $USERNAME

WORKDIR /workspace/

# Install docker
COPY --chown=$USERNAME:$USERNAME /install-docker.sh ./
RUN chmod u+x ./install-docker.sh
RUN /bin/bash -c './install-docker.sh'

# Install development dependencies from apt
USER root
RUN apt-get update \
    && apt-get install direnv

# Install development dependencies from pip
RUN pip install black isort flake8 mypy docker

# Set as non-root user
USER $USERNAME
