# Build training environment
FROM tensorflow/tensorflow:2.6.0-gpu-jupyter AS training

# Set working dir
WORKDIR /srv

# Upgrade pip and install project dependencies from pip
RUN pip install --upgrade pip
RUN pip install -U \
    sklearn==0.0 \
    tensorflow-datasets==4.4.0 \
    tensorboard-plugin-profile==2.5.0

# Build development environment
FROM training AS development

# Update apt packages
RUN apt-get update

# Install development dependencies from apt
RUN apt-get install direnv

# Install development dependencies from pip
RUN pip install -U \
    black==21.7b0 \
    isort==5.9.3 \
    flake8==4.0.1 \
    mypy==0.910
