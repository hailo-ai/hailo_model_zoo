ARG base_image=pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
ARG timezone
FROM $base_image

ENV DEBIAN_FRONTEND=noninteractive, TZ=$timezone

RUN apt-get update && \
    apt-get -y --no-install-recommends install git build-essential python3-opencv wget vim

RUN git clone --depth 1 https://github.com/hailo-ai/deep-person-reid.git && \
    cd deep-person-reid && \
    python3 -m pip install --upgrade Pillow && \
    pip install -r requirements.txt && \
    python setup.py develop

ENV PYTHONPATH=/workspace/deep-person-reid
WORKDIR /workspace/deep-person-reid

RUN mkdir models
ADD https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/MCPReID/reid/repvgg_a0_person_reid_512/2022-04-18/repvgg_a0_person_reid_512.pth ./models
ADD https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/MCPReID/reid/repvgg_a0_person_reid_2048/2022-04-18/repvgg_a0_person_reid_2048.pth ./models
