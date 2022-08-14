ARG base_image=pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
ARG timezone
FROM $base_image
ENV DEBIAN_FRONTEND=noninteractive, TZ=$timezone

RUN apt-get update && \
    apt-get -y --no-install-recommends install git build-essential python3-opencv wget vim sudo

RUN pip install mmcv-full==1.6.0 && \
    git clone https://github.com/hailo-ai/mmpose.git && \
    cd mmpose && pip install -e . && \
    mkdir -p data/coco/annotations && mkdir pretrained && \
    pip install tensorboard onnx onnxruntime onnx-simplifier netron ipdb && \
    wget -P ./data/coco/ https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SinglePersonPoseEstimation/mspn_regnetx_800mf/pretrained/2022-07-12/coco_gt_as_results.json && \
    wget -P ./pretrained https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SinglePersonPoseEstimation/mspn_regnetx_800mf/pretrained/2022-07-12/best_AP_epoch_310.pth

WORKDIR /workspace/mmpose

ARG user=hailo
ARG group=hailo
ARG uid=1000
ARG gid=1000

RUN groupadd --gid $gid $group && \
    adduser --uid $uid --gid $gid --shell /bin/bash --disabled-password --gecos "" $user && \
    chmod u+w /etc/sudoers && echo "$user ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && chmod -w /etc/sudoers && \
    chown -R $user:$group /workspace/mmpose
