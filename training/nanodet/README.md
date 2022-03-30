# Nanodet Retraining
* To learn more about NanoDet look [**here**](https://github.com/hailo-ai/nanodet)
---

## Prerequisites
  * docker ([installation instructions](https://docs.docker.com/engine/install/ubuntu/))
  * nvidia-docker2 ([installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
  > **_NOTE:_**  In case you use Hailo Software Suite docker, make sure you are doing all the following instructions outside this docker.
## Environment Preparations
1. Build the docker image:
    
    <code stage="docker_build">
    cd <span val="dockerfile_path">hailo_model_zoo/training/nanodet</span>

    docker build -t nanodet:v0 --build-arg timezone=`cat /etc/timezone` .
    </code>

    the following optional arguments can be passed via --build-arg:
    
    - `timezone` - a string for setting up timezone. E.g. "Asia/Jerusalem"
    - `user` - username for a local non-root user. Defaults to 'hailo'.
    - `group` - default group for a local non-root user. Defaults to 'hailo'.
    - `uid` - user id for a local non-root user.
    - `gid` - group id for a local non-root user.

2. Start your docker:

    <code stage="docker_run">
    docker run <span val="replace_none">--name "your_docker_name"</span> -it --gpus all <span val="replace_none">-u "username"</span> --ipc=host -v <span val="local_vol_path">/path/to/local/data/dir</span>:<span val="docker_vol_path">/path/to/docker/data/dir</span>  nanodet:v0
    </code>
    
      - `docker run` create a new docker container.
      - `--name <your_docker_name>` name for your container.
      - `-u <username>` same username as used for building the image.
      - `-it` runs the command interactively.
      - `--gpus all` allows access to all GPUs.
      - `--ipc=host` sets the IPC mode for the container.
      - `-v /path/to/local/data/dir:/path/to/docker/data/dir` maps `/path/to/local/data/dir` from the host to the container. You can use this command multiple times to mount multiple directories.
      - `nanodet:v0` the name of the docker image.

## Training and exporting to ONNX
1. Prepare your data: <br>
    Data is expected to be in coco format. More information can be found [here](https://cocodataset.org/#format-data)

2. Training: <br>
    Configure your model in a .yaml file. We'll use /workspace/nanodet/config/legacy_v0.x_configs/RepVGG/nanodet-RepVGG-A0_416.yml in this guide.
    Modify the path for the dataset in the .yaml configuration file:
    ```
    data:
      train:
        name: CocoDataset
        img_path: <path-to-train-dir>
        ann_path: <path-to-annotations-file>
        ...
      val:
        name: CocoDataset
        img_path: <path-to-validation-dir>
        ann_path: <path-to-annotations-file>
        ...
    ```

    Start training with the following command:
    
    <code stage="retrain">
    <span val="replace_none">cd /workspace/nanodet</span>
    
    python tools/train.py ./config/legacy_v0.x_configs/RepVGG/nanodet-RepVGG-A0_416.yml
    </code>

    In case you want to use the pretrained nanodet-RepVGG-A0_416.ckpt, which was predownloaded into your docker modify your configurationf file:
    ```
    schedule:
      load_model: ./pretrained/nanodet-RepVGG-A0_416.ckpt
    ```

    Modifying the batch size and the number of GPUs used for training can be done also in the configuration file:
    ```
    device:
      gpu_ids: [0]
      workers_per_gpu: 1
      batchsize_per_gpu: 128
    ```

3. Exporting to onnx
    After training, run the following command:
    
    <code stage="export">
    python tools/export_onnx.py --cfg_path ./config/legacy_v0.x_configs/RepVGG/nanodet-RepVGG-A0_416.yml --model_path /workspace/nanodet/workspace/RepVGG-A0-416/model_last.ckpt
    </code>

  > **_NOTE:_**  Your trained model will be found under the following path: /workspace/nanodet/workspace/<backbone-name>/model_last.ckpt, and exported onnx will be written to /workspace/nanodet/nanodet.onnx

<br>

---

## Compile the Model using Hailo Model Zoo
You can generate an HEF file for inference on Hailo-8 from your trained ONNX model.
In order to do so you need a working model-zoo environment.
Choose the corresponding YAML from our networks configuration directory, i.e. <code>hailo_model_zoo/cfg/networks/nanodet_repvgg.yaml</code>, and run compilation using the model zoo:  
  
  <code stage="compile">
  python <span val="mz_main_path">hailo_model_zoo/main.py</span> compile --ckpt <span val="local_path_to_onnx">nanodet.onnx</span> --calib-path <span val="calib_set_path">/path/to/calibration/imgs/dir/</span> --yaml <span val="yaml_file_path">nanodet_repvgg.yaml</span>
  </code>

  * <code>--ckpt</code> - path to your ONNX file.
  * <code>--calib-path</code> - path to a directory with your calibration images in JPEG format
  * <code>--yaml</code> - path to your configuration YAML file. In case you have made some changes in the model, you might need to update its start/end nodes names / number of classes and so on.  <br>
  The model zoo will take care of adding the input normalization to be part of the model.

The compiled model can be used in the [**Hailo TAPPAS**](https://hailo.ai/developer-zone/tappas-apps-toolkit/) to generate a full application.