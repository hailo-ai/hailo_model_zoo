ARG base_image=pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
ARG timezone="Asia/Jerusalem"
FROM $base_image

ENV DEBIAN_FRONTEND=noninteractive, TZ=$timezone

RUN apt-get update && \
    apt-get -y --no-install-recommends install git build-essential python3-opencv wget

RUN git clone https://github.com/ultralytics/yolov5.git --branch v2.0 && \
    cd yolov5 && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -U 'coremltools>=4.1' 'onnx>=1.9.0' 'scikit-learn==0.19.2'

ENV PYTHONPATH=/workspace/yolov5
WORKDIR /workspace/yolov5

RUN cd /workspace/yolov5 && wget https://github.com/ultralytics/yolov5/releases/download/v2.0/yolov5s.pt -q
RUN cd /workspace/yolov5 && wget https://github.com/ultralytics/yolov5/releases/download/v2.0/yolov5m.pt -q
RUN cd /workspace/yolov5 && wget https://github.com/ultralytics/yolov5/releases/download/v2.0/yolov5l.pt -q
RUN cd /workspace/yolov5 && wget https://github.com/ultralytics/yolov5/releases/download/v2.0/yolov5x.pt -q