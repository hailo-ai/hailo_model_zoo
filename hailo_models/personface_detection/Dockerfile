ARG base_image=pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
ARG timezone
FROM $base_image

ENV DEBIAN_FRONTEND=noninteractive, TZ=$timezone

RUN apt-get update && \
    apt-get -y --no-install-recommends install git build-essential python3-opencv wget vim

RUN git clone https://github.com/hailo-ai/yolov5.git --branch v2.0 && \
    cd yolov5 && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -U 'coremltools>=4.1' 'onnx>=1.9.0' 'scikit-learn==0.19.2'

ENV PYTHONPATH=/workspace/yolov5
WORKDIR /workspace/yolov5

RUN cd /workspace/yolov5
COPY ./src/yolov5s_personface.yaml ./models
COPY ./src/personface_data.yaml ./data
ADD https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/MCPReID/personface_detector/yolov5s_personface/2022-04-01/yolov5s_personface.pt ./weights
