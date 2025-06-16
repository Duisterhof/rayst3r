FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive

# Declare a build argument for CUDA_ARCH
ARG CUDA_ARCH
# Install system dependencies, including Python 3.11, its development headers, and GCC 9
RUN apt-get update -y && \
    apt-get install -y python3.11 python3.11-venv python3-pip git ffmpeg && \
    # Install Python 3.11 development headers - ESSENTIAL for Python.h
    apt-get install -y python3.11-dev && \
    # Install and set GCC 9 as default compiler
    apt-get install -y gcc-9 g++-9 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90 && \
    # Set Python 3.11 and pip as default
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
    # Clean up apt caches to reduce image size
    rm -rf /var/lib/apt/lists/*

WORKDIR /src


COPY . /src/
# Upgrade pip and install core Python dependencies
RUN python3.11 -m pip install --upgrade pip
RUN python3.11 -m pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
RUN python3.11 -m pip install -r requirements_eval.txt

# Build the custom PyTorch extension (curope)
WORKDIR /src/extensions/curope
RUN TORCH_CUDA_ARCH_LIST="${CUDA_ARCH}" python3.11 -m pip install . --no-build-isolation

WORKDIR /src

CMD ["uvicorn", "eval_server:app", "--host", "0.0.0.0", "--port", "6000"]