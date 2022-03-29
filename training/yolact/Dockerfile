ARG base_image=pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
ARG timezone
FROM $base_image

ENV DEBIAN_FRONTEND=noninteractive, TZ=$timezone
# Install requirements
RUN apt-get update && \
    apt-get -y --no-install-recommends install git sudo build-essential python3-opencv wget

# Clone HAILO's fork of the YOLACT repo and install pip packages
RUN git clone https://github.com/hailo-ai/yolact.git --branch Model-Zoo-1.5 && \
    cd yolact && \
    pip install --upgrade pip && \
    pip install cython && \
    pip install opencv-python pillow pycocotools matplotlib && \
    pip install pycls

ENV PYTHONPATH=/workspace/yolact
WORKDIR /workspace/yolact

# Download backbone YAMLs
RUN mkdir yamls && \
    cd yamls && \
    wget https://raw.githubusercontent.com/facebookresearch/pycls/main/configs/dds_baselines/regnetx/RegNetX-600MF_dds_8gpu.yaml && \
    wget https://raw.githubusercontent.com/facebookresearch/pycls/main/configs/dds_baselines/regnetx/RegNetX-800MF_dds_8gpu.yaml && \
    wget https://raw.githubusercontent.com/facebookresearch/pycls/main/configs/dds_baselines/regnetx/RegNetX-1.6GF_dds_8gpu.yaml

# Download RegNetX pretrained weights
RUN mkdir weights && \
    cd weights && \
    wget https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906442/RegNetX-600MF_dds_8gpu.pyth && \
    wget https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906036/RegNetX-800MF_dds_8gpu.pyth && \
    wget https://dl.fbaipublicfiles.com/pycls/dds_baselines/160990626/RegNetX-1.6GF_dds_8gpu.pyth

ARG user=hailo
ARG group=hailo
ARG uid=1000
ARG gid=1000

RUN apt-get update && apt-get install sudo
RUN groupadd --gid $gid $group && \
    adduser --uid $uid --gid $gid --shell /vin/vash --disabled-password --gecos "" $user && \
    chmod u+w /etc/sudoers && echo "$user ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && chmod -w /etc/sudoers && \
    chown -R $user:$group /workspace

