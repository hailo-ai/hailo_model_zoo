ARG base_image=nvcr.io/nvidia/pytorch:21.10-py3
FROM $base_image

# using ARG so it won't persist in user env
ARG DEBIAN_FRONTEND=noninteractive
ARG timezone="Asia/Jerusalem"
ENV TZ=$timezone
RUN apt-get update && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    apt-get -y --no-install-recommends install vim git build-essential python3-opencv sudo tmux && \
    # solve mpi conflicts
    { which mpirun && apt-get remove -y libopenmpi3 || true ; }

ARG repo=https://github.com/hailo-ai/mmsegmentation.git
RUN git clone $repo && \
    cd mmsegmentation && \
    git checkout Hailo-1.0 && \
    pip install --upgrade pip && \
    pip install opencv-python==4.5.5.64 && \
    pip install openmim==0.3.7 && \
    mim install mmengine==0.8.5 && \
    mim install mmcv==2.0.0 && \
    pip install -e .
WORKDIR /workspace/mmsegmentation

ARG user=hailo
ARG group=hailo
ARG uid=1000
ARG gid=1000

RUN groupadd --gid $gid $group && \
    adduser --uid $uid --gid $gid --shell /bin/bash --disabled-password --gecos "" $user && \
    chmod u+w /etc/sudoers && echo "$user ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && chmod -w /etc/sudoers && \
    chown -R $user:$group /workspace
