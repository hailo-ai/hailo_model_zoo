ARG base_image=pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
ARG timezone
FROM $base_image

ENV DEBIAN_FRONTEND=noninteractive, TZ=$timezone

RUN apt-get update && \
    apt-get -y --no-install-recommends install git build-essential python3-opencv wget

RUN git clone https://github.com/hailo-ai/yolov5.git --branch v2.0.1 && \
    cd yolov5 && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -U 'coremltools==6.1' 'onnx==1.13.0' 'scikit-learn==0.19.2'

ENV PYTHONPATH=/workspace/yolov5
WORKDIR /workspace/yolov5

RUN cd /workspace/yolov5
COPY ./src/vehicle_detector.yaml ./models
COPY ./src/vehicles.yaml ./data
ADD https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/LPR/vehicle_detector/yolov5m_vehicles/2022-02-23/yolov5m_vehicles.pt ./weights
