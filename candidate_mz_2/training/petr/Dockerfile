ARG timezone

FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install wget -yq
RUN apt-get install build-essential g++ gcc python3.8-venv -y
RUN apt-get install libgl1-mesa-glx libglib2.0-0 -y
RUN apt-get install openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev git -y
RUN apt-get install sudo -y

WORKDIR /workspace

ARG user=hailo
ARG group=hailo
ARG uid=1000
ARG gid=1000
RUN groupadd --gid $gid $group
RUN adduser --uid $uid --gid $gid --shell /bin/bash --disabled-password --gecos "" $user
RUN chmod u+w /etc/sudoers && echo "$user ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && chmod -w /etc/sudoers

ENV VIRTUAL_ENV=/workspace/my_venv PATH="/workspace/my_venv/bin:$PATH"
RUN python3 -m venv /workspace/my_venv && . /workspace/my_venv/bin/activate

RUN apt-get install python3-pip -y
RUN python3 -m pip install --upgrade pip

RUN pip install --default-timeout=100000 future torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

RUN apt-get install python3-dev

RUN pip install nuscenes-devkit==1.1.10 scikit-image==0.19.3 pandas==1.1.4 torchvision==0.10.0 numpy==1.19.5 einops==0.7.0 yapf==0.40.1 plyfile==1.0.2 onnx==1.15.0 onnxruntime==1.8.0 onnxsim==0.4.31

RUN pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html 
RUN pip install mmsegmentation==0.20.2
RUN git clone https://github.com/open-mmlab/mmdetection.git  && \
    cd mmdetection && git checkout v2.24.1 && \
    pip install -r requirements/build.txt && \
    python3 setup.py develop && cd .. 

RUN git clone  https://github.com/open-mmlab/mmdetection3d.git
RUN cd mmdetection3d && git checkout v0.17.1 && \
    pip install -r requirements/build.txt

RUN git clone https://github.com/hailo-ai/PETR.git && \
    cd PETR && git checkout petrv2-repvgg && \
    mkdir ckpts && mkdir data && \
    ln -s ../mmdetection3d ./mmdetection3d

RUN chown -R $user:$group /workspace

RUN cd /workspace/PETR/ckpts && \
    wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection3d/Detection3d-Nuscenes/petrv2/pretrained/2024-08-13/fcos3d_repvgg_h2_epoch_12_remapped.pth -q
    
WORKDIR /workspace/PETR

# On runtime, install mmdetd3d
CMD ["/bin/bash", "-c", "cd mmdetection3d;python3 setup.py develop;cd ..;tail -f /dev/null"]
