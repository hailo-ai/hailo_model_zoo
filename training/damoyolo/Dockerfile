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
    wget https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/clean_models/before_distill/damoyolo_tinynasL20_T_418.pth -q && \
    wget https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/clean_models/before_distill/damoyolo_tinynasL25_S_456.pth -q && \
    wget https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/clean_models/before_distill/damoyolo_tinynasL35_M_487.pth -q
