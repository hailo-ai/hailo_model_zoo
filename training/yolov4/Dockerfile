ARG base_image=nvcr.io/nvidia/pytorch:21.06-py3
ARG timezone
FROM $base_image

ENV DEBIAN_FRONTEND=noninteractive, TZ=$timezone

RUN apt-get update && \
    apt-get -y --no-install-recommends install git build-essential python3-opencv wget vim
RUN pip install gdown==4.6.0

RUN git clone -b hailo_ai https://github.com/hailo-ai/pytorch-YOLOv4.git && \
    git clone https://github.com/hailo-ai/darknet.git

ENV PYTHONPATH=/workspace/darknet
WORKDIR /workspace/darknet
RUN sed -i "1s/GPU=0/GPU=1/" Makefile && \
    sed -i "2s/CUDNN=0/CUDNN=1/" Makefile && \
    sed -i "3s/CUDNN_HALF=0/CUDNN_HALF=1/" Makefile && \
    sed -i "4s/OPENCV=0/OPENCV=1/" Makefile

RUN make

RUN cd /workspace/darknet && wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29 -q && \
    wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights -q
