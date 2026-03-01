ARG base_image=pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
ARG timezone
FROM $base_image

ENV DEBIAN_FRONTEND=noninteractive, TZ=$timezone

RUN apt-get update && \
    apt-get -y --no-install-recommends install git build-essential python3-opencv wget vim

RUN git clone https://github.com/hailo-ai/DAMO-YOLO.git && \
    cd DAMO-YOLO && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

RUN pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI

ENV PYTHONPATH=/workspace/DAMO-YOLO
WORKDIR /workspace/DAMO-YOLO

RUN cd /workspace/DAMO-YOLO && \
    wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/damoyolo_tinynasL20_T/pretrained/2022-12-19/damoyolo_tinynasL20_T_418.pth -q && \
    wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/damoyolo_tinynasL25_S/pretrained/2022-12-19/damoyolo_tinynasL25_S_456.pth -q && \
    wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/damoyolo_tinynasL35_M/pretrained/2022-12-19/damoyolo_tinynasL35_M_487.pth -q
