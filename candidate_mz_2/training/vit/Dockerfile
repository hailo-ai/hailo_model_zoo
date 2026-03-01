ARG base_image=pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
ARG timezone
FROM $base_image

ENV DEBIAN_FRONTEND=noninteractive, TZ=$timezone

RUN apt-get update && \
    apt-get -y --no-install-recommends install git build-essential python3-opencv wget vim

RUN git clone https://github.com/hailo-ai/pytorch-image-models && \
    cd pytorch-image-models && \
    git checkout Hailo-0.x && \
    pip install -U pip && \
    pip install -r requirements.txt && \
    pip install tensorboard==2.9.0 && \
    pip install protobuf==3.20.1 && \
    pip install onnx==1.12.0 && \
    pip install onnxruntime==1.13.1 && \
    wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_tiny/pretrained/2023-08-29/vit_tiny_un_patch16_224.pth.tar


ENV PYTHONPATH=/workspace/pytorch-image-models
WORKDIR /workspace/pytorch-image-models

ARG user=hailo
ARG group=hailo
ARG uid=1000
ARG gid=1000

