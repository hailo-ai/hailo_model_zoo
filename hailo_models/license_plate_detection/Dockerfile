ARG base_image=nvcr.io/nvidia/pytorch:21.06-py3
ARG timezone
FROM $base_image

ENV DEBIAN_FRONTEND=noninteractive, TZ=$timezone

RUN apt-get update && \
    apt-get -y --no-install-recommends install git build-essential python3-opencv wget

RUN git clone https://github.com/Tianxiaomo/pytorch-YOLOv4.git
RUN git clone https://github.com/AlexeyAB/darknet.git

ENV PYTHONPATH=/workspace/darknet
WORKDIR /workspace/darknet
RUN sed -i "1s/GPU=0/GPU=1/" Makefile && \
    sed -i "2s/CUDNN=0/CUDNN=1/" Makefile

RUN make

RUN mkdir ./data/obj

COPY ./src/obj.names ./data
COPY ./src/obj.data ./data
COPY ./src/tiny_yolov4_license_plates.cfg ./cfg
ADD https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/LPR/lp_detector/tiny_yolov4_license_plate/2021-11-30/tiny_yolov4_license_plates.weights .

COPY ./src/export.patch /workspace/pytorch-YOLOv4/
RUN cd /workspace/pytorch-YOLOv4/ && \
    git checkout 4ccef0e && \
    git apply export.patch && rm export.patch
