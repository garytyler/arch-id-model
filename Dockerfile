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

RUN ulimit -n 4092

RUN chown $USERNAME:$USERNAME /.config/matplotlip

# Set as non-root user
USER $USERNAME
