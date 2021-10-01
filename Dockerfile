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

# Set as non-root user
USER $USERNAME

FROM train-stage AS dev-stage

# Add dev dependencies
USER root
RUN apt-get update \
    && apt-get install direnv

# Upgrade pip and install python project dependencies
RUN pip install --upgrade pip sklearn black isort flake8 mypy

# Set as non-root user
USER $USERNAME
