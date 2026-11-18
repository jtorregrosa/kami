# kami-server image: PyTorch + TRELLIS.2 (and its CUDA extensions) + diffusers.
#
# Only the dependencies live here: `kami up` sends kami/server.py and kami/guard.py to the
# pod on every boot, so a change to them needs no rebuild. Weights are not baked in
# either; they download on boot (or come from the network volume).
#
# Build (no GPU needed; ~30-60 min, the CUDA extensions compile):
#   docker build -t ghcr.io/<you>/kami-server:0.1.0 docker
#   docker push ghcr.io/<you>/kami-server:0.1.0
# or let .github/workflows/server-image.yml build and push it.

# TRELLIS.2's tested stack: torch 2.6.0 + CUDA 12.4 (setup.sh --new-env installs the same).
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ARG TRELLIS_REF=main
# GPUs the extensions are compiled for: Ampere (A100 8.0; A5000, A6000, 3090 8.6),
# Ada (4090, L40, L40S, 6000 Ada 8.9), Hopper (H100, H200 9.0). Blackwell needs CUDA 12.8.
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    MAX_JOBS=4 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends git sudo ca-certificates ninja-build \
        libjpeg-dev libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN git clone --recursive --depth 1 -b "${TRELLIS_REF}" https://github.com/microsoft/TRELLIS.2.git /opt/TRELLIS.2

WORKDIR /opt/TRELLIS.2

# setup.sh picks CUDA or ROCm by looking for nvidia-smi, which a GPU-less build does not
# have: a stand-in answers during the build only. It must go afterwards, or it would
# shadow the real nvidia-smi the NVIDIA runtime mounts (the guard reads GPU memory with it).
# Without --new-env, setup.sh installs into this image's Python, next to its torch.
RUN printf '#!/bin/sh\nexit 0\n' > /usr/local/bin/nvidia-smi && chmod +x /usr/local/bin/nvidia-smi \
 && bash -c ". ./setup.sh --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm" \
 && rm -f /usr/local/bin/nvidia-smi \
 && rm -rf /tmp/extensions /root/.cache

# Text -> image: diffusers with Z-Image support (0.36+), and what its pipelines load.
RUN pip install "diffusers>=0.36" accelerate sentencepiece protobuf

ENV PYTHONPATH=/opt/TRELLIS.2 \
    OPENCV_IO_ENABLE_OPENEXR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fail the build here, not on a billed pod, if an import is broken (TRELLIS.2's pipeline
# module pulls in its extensions).
RUN python -c "import torch, diffusers, o_voxel; \
from trellis2.pipelines import Trellis2ImageTo3DPipeline; \
print('torch', torch.__version__, 'cuda', torch.version.cuda)"

WORKDIR /root
