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

# workaround broken opencv-python 4.6
RUN pip install opencv-python==4.5.5.64
#newer pytorch-lightning do not include ProgressBar
RUN pip install pytorch-lightning==1.6.5

ARG repo=https://github.com/hailo-ai/nanodet.git
RUN git clone $repo && \
    cd nanodet && pip install -r requirements.txt

WORKDIR /workspace/nanodet
RUN pip install -e .
RUN mkdir -p  pretrained && python google_drive.py 1nWZZ1qXb1HuIXwPSYpEyFHHqX05GaFer ./pretrained/nanodet-RepVGG-A0_416.ckpt

ARG user=hailo
ARG group=hailo
ARG uid=1000
ARG gid=1000

RUN groupadd --gid $gid $group && \
    adduser --uid $uid --gid $gid --shell /bin/bash --disabled-password --gecos "" $user && \
    chmod u+w /etc/sudoers && echo "$user ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && chmod -w /etc/sudoers && \
    chown -R $user:$group /workspace

