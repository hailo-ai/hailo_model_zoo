ARG base_image=pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
ARG timezone
FROM $base_image

ENV DEBIAN_FRONTEND=noninteractive, TZ=$timezone

RUN apt-get update && \
    apt-get -y --no-install-recommends install git build-essential python3-opencv wget vim sudo

RUN git clone https://github.com/hailo-ai/yolov5.git --branch v2.0.1 && \
    cd yolov5 && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -U 'coremltools==6.1' 'onnx==1.13.0' 'scikit-learn==0.19.2'

ENV PYTHONPATH=/workspace/yolov5
WORKDIR /workspace/yolov5

RUN cd /workspace/yolov5 && wget https://github.com/ultralytics/yolov5/releases/download/v2.0/yolov5s.pt -q
RUN cd /workspace/yolov5 && wget https://github.com/ultralytics/yolov5/releases/download/v2.0/yolov5m.pt -q
RUN cd /workspace/yolov5 && wget https://github.com/ultralytics/yolov5/releases/download/v2.0/yolov5l.pt -q
RUN cd /workspace/yolov5 && wget https://github.com/ultralytics/yolov5/releases/download/v2.0/yolov5x.pt -q
RUN cd /workspace/yolov5 && wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5m/pretrained/2022-04-19/yolov5m_wo_spp.pt -q

ARG user=hailo
ARG group=hailo
ARG uid=1000
ARG gid=1000

RUN groupadd --gid $gid $group && \
    adduser --uid $uid --gid $gid --shell /bin/bash --disabled-password --gecos "" $user && \
    chmod u+w /etc/sudoers && echo "$user ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && chmod -w /etc/sudoers && \
    chown -R $user:$group /workspace