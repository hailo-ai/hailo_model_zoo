ARG base_image=nvcr.io/nvidia/pytorch:21.10-py3
FROM $base_image

# using ARG so it won't persist in user env
ARG DEBIAN_FRONTEND=noninteractive
ARG timezone="Asia/Jerusalem"
ENV TZ=$timezone

RUN apt-get update && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    apt-get -y --no-install-recommends install vim git build-essential python3-opencv sudo tmux && \
    # solve mpi conflicts
    { which mpirun && apt-get remove -y libopenmpi3 || true ; }

ARG repo=https://github.com/hailo-ai/YOLOX
ARG yolox_branch=yolox-hailo-model

RUN mkdir /workspace/data && \
    git clone $repo --branch $yolox_branch && \
    cd YOLOX && pip install -r requirements.txt && pip install -e . && \
    pip install cython==0.29.24
ENV YOLOX_DATADIR=/workspace/data

WORKDIR /workspace/YOLOX

RUN wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth -q
RUN wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/OD/yolox_hailo/2023-09-23/yolox_hailo_before_prune.pth -q; \
    wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/OD/yolox_hailo/2023-09-23/yolox_hailo_p50.pth -q; \
    wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/OD/yolox_hailo/2023-09-23/README.md -q

RUN pip install "sparseml==1.5.4"

ENV NM_BYPASS_TORCH_VERSION=1
