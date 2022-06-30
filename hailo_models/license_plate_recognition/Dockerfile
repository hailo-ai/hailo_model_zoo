ARG base_image=pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
ARG timezone
FROM $base_image

ENV DEBIAN_FRONTEND=noninteractive, TZ=$timezone

RUN apt-get update && \
    apt-get -y --no-install-recommends install git build-essential python3-opencv wget

RUN git clone https://github.com/hailo-ai/LPRNet_Pytorch.git && \
    cd LPRNet_Pytorch && \
    pip install --upgrade pip && \
    pip install -U 'imutils==0.5.4' 'opencv-python>=4.5.5' 'imgaug==0.4.0' 'tensorboard==2.7.0' 'torchsummary' 'pandas==1.3.5' 'strsimpy==0.2.1' 'numpy==1.19.2' 'jupyter' 'protobuf==3.20.1'

WORKDIR /workspace/LPRNet_Pytorch/
RUN cd /workspace/LPRNet_Pytorch/ && \
    mkdir pre_trained && \
    mkdir dataset
COPY ./src/lp_autogenerate.ipynb ./dataset
ADD https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/LPR/ocr/lprnet/2022-03-09/lprnet.pth ./pre_trained
