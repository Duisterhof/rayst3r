FROM pytorch/pytorch:2.9.0-cuda13.0-cudnn9-devel

# Silence pip errors. We don't care about system packages.
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Note that autodetection of GPU compute capability is not available during
# docker build. We select Ampere+ architectures. Blackwell is not yet
# officially supported in PyTorch IIRC.
ENV CMAKE_CUDA_ARCHITECTURES="80;86;89;90"
ENV TORCH_CUDA_ARCH_LIST="8.6+PTX;8.9;9.0"
ENV MAX_JOBS=16

WORKDIR /app
COPY . /app

# System deps for CUDA extensions & rendering utils. Note that xformers build
# take a long time and needs quite a bit of RAM
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  git \
  git-lfs \
  curl \
  ca-certificates \
  pkg-config \
  nano \
  ffmpeg \
  python3-dev \ 
  python3-pip \
  git \
  cmake \
  ninja-build \
  build-essential \
  libomp-dev \
  && rm -rf /var/lib/apt/lists/*

RUN git apply torch2.9.0.patch \
  && pip3 install -r requirements.txt \
  && pip3 install xformers --index-url https://download.pytorch.org/whl/cu130 \
  && cd extensions/curope/ \
  && python setup.py build_ext --inplace

# Pre-download Hugging Face models used in eval_wrapper/eval.py and app.py
RUN python3 -c "import torch; from huggingface_hub import hf_hub_download; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg'); hf_hub_download('bartduis/rayst3r', 'rayst3r.pth'); hf_hub_download('Ruicheng/moge-vitl', 'model.pt')" 

CMD ["/bin/bash"]



