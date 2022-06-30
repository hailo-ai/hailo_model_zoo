FROM tensorflow/tensorflow:1.15.2-gpu-py3

# Fix nvidia changing keys (2022-04)
RUN rm /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-key del 7fa2af80 && \
    curl -O -J https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    rm cuda-keyring_1.0-1_all.deb

# using ARG so it won't persist in user env
ARG DEBIAN_FRONTEND=noninteractive
ARG timezone="Asia/Jerusalem"
ENV TZ=$timezone

# Install apt dependencies
RUN apt-get update && apt-get -y --no-install-recommends install \
    git \
    python3-opencv \
    python3-cairocffi \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    sudo \ 
    tmux \
    vim \
    wget

RUN git clone https://github.com/hailo-ai/models.git /home/tensorflow/models \
    && cd /home/tensorflow/models \
    && git checkout 10ee28d

# Compile protobuf configs
RUN (cd /home/tensorflow/models/research/ && protoc object_detection/protos/*.proto --python_out=.)

# Run TensorFlow Object Detection API setup
WORKDIR /home/tensorflow/models/research/
RUN cp object_detection/packages/tf1/setup.py ./

ENV PATH="/home/tensorflow/.local/bin:${PATH}"
RUN python -m pip install -U pip && \
    python -m pip install .

# Get pre-trained ssd-mobilenet-v1 model
RUN curl -o ./ssd_mobilenet_v1.tar.gz  http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz \
    && tar -xzf  ssd_mobilenet_v1.tar.gz && cp ssd_mobilenet_v1_coco_2018_01_28/pipeline.config ./ \
    && rm ssd_mobilenet_v1.tar.gz

# Update pipeline.config
RUN sed -i pipeline.config -e "s/PATH_TO_BE_CONFIGURED\/model.ckpt/\/home\/tensorflow\/models\/research\/ssd_mobilenet_v1_coco_2018_01_28\/model.ckpt/" \
    -e "s/mscoco_train.record/<your-own-dataset>_train.record-?????-of-00100/" \
    -e "s/mscoco_val.record/<your-own-dataset>_val.record-?????-of-00010/"

ARG user=hailo
ARG group=hailo
ARG uid=1000
ARG gid=1000

RUN groupadd --gid $gid $group && \
    adduser --uid $uid --gid $gid --shell /bin/bash --disabled-password --gecos "" $user && \
    chmod u+w /etc/sudoers && echo "$user ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && chmod -w /etc/sudoers && \
    chown -R $user:$group /home/tensorflow
