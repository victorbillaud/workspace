# Use NVIDIA's official CUDA 12.8 base image with cuDNN on Ubuntu 22.04
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Set non-interactive mode for apt and install basic dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
software-properties-common build-essential wget curl git ca-certificates && \
add-apt-repository ppa:deadsnakes/ppa && apt-get update && \
apt-get install -y --no-install-recommends python3.11 python3.11-dev python3.11-distutils && \
apt-get clean && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.11
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.11 get-pip.py && rm get-pip.py

COPY builder /builder

# Install PyTorch, Torchvision, Torchaudio with CUDA 12.8 support, and Transformers
ENV TORCH_VERSION=2.7.0
ENV CUDA_VERSION=12.8
RUN python3.11 -m pip install torch torchvision torchaudio \
-f https://download.pytorch.org/whl/torch_stable.html && \
python3.11 -m pip install transformers==4.34.0 && \
python3.11 -m pip install --no-cache-dir -r ./builder/requirements.txt

# (Optional) Install additional LLM support libraries
# RUN python3.11 -m pip install accelerate bitsandbytes datasets

# Set Python3.11 as the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Download Faster Whisper Models
RUN chmod +x /builder/download_models.sh
RUN --mount=type=secret,id=hf_token /builder/download_models.sh

# Create a workspace directory (for persistent volume mount) and set it as working dir
RUN mkdir /workspace
WORKDIR /workspace

# Default command to keep container alive (so we can exec into it or run commands)
CMD ["sleep", "infinity"]