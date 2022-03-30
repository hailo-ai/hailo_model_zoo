# FCN Retraining
  * To learn more about FCN look [**here**](https://github.com/hailo-ai/mmsegmentation)
---

### Prerequisites
  * docker ([installation instructions](https://docs.docker.com/engine/install/ubuntu/))
  * nvidia-docker2 ([installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
  > **_NOTE:_**  In case you use Hailo Software Suite docker, make sure you are doing all the following instructions outside this docker.
### Environment Preparations
1. Build the docker image:
    
    <code stage="docker_build">
    cd <span val="dockerfile_path">hailo_model_zoo/training/fcn</span>
    
    docker build -t fcn:v0 --build-arg timezone=`cat /etc/timezone` .
    </code>

    the following optional arguments can be passed via --build-arg:

    - `timezone` - a string for setting up timezone. E.g. "Asia/Jerusalem"
    - `user` - username for a local non-root user. Defaults to 'hailo'.
    - `group` - default group for a local non-root user. Defaults to 'hailo'.
    - `uid` - user id for a local non-root user.
    - `gid` - group id for a local non-root user.

2. Start your docker:
    
    <code stage="docker_run">
    docker run <span val="replace_none">--name "your_docker_name"</span> -it --gpus all  <span val="replace_none">-u "username"</span> --ipc=host -v <span val="local_vol_path">/path/to/local/data/dir</span>:<span val="docker_vol_path">/path/to/docker/data/dir</span>  fcn:v0
    </code>
    
      - `docker run` create a new docker container.
      - `--name <docker-name>` name for your container.
      - `-u <username>` same username as used for building the image.
      - `-it` runs the command interactively.
      - `--gpus all` allows access to all GPUs.
      - `--ipc=host` sets the IPC mode for the container.
      - `-v /path/to/local/data/dir:/path/to/docker/data/dir` maps `/path/to/local/data/dir` from the host to the container. You can use this command multiple times to mount multiple directories.
      - `fcn:v0` the name of the docker image.

## Training and exporting to ONNX
1. Prepare your data: <br>
    Data is expected to be in coco format, and by default should be in /workspace/data/<dataset_name>.

    The expected structure is as follows:
    ```
    /workspace
    |-- mmsegmentation
    `-- |-- data
            `-- cityscapes
                |-- gtFine
                |   | -- train
                |   |    | -- aachem
                |   |    | -- | -- *.png
                |   |    ` -- ...
                |   ` -- test
                |        | -- berlin
                |        | -- | -- *.png
                |        ` -- ...
                `-- leftImg8bit
                    | -- train
                    | -- | -- aachem
                    | -- | -- | -- *.png
                    | -- ` -- ...
                    ` -- test
                         | -- berlin
                         | -- | -- *.png
                         ` -- ...
            
    ```

    more information can be found [here](https://github.com/hailo-ai/mmsegmentation/blob/master/docs/en/dataset_prepare.md#cityscapes)

2. Training: <br>
    Configure your model in a .py file. We'll use /workspace/mmsegmentation/configs/fcn/fcn_r18_hailo.py in this guide.

    start training with the following command:
    
    <code stage="retrain">
    cd /workspace/mmsegmentation

    ./tools/dist_train.sh configs/fcn/fcn8_r18_hailo.py <span val="gpu_num">2</span>
    </code>

    Where 2 is the number of GPUs used for training.

3. Exporting to onnx  
    After training, run the following command:
    
    <code stage="export">
    cd /workspace/mmsegmentation
    
    python ./tools/pytorch2onnx.py configs/fcn/fcn_r18_hailo.py --opset-version 11 --checkpoint ./work_dirs/fcn8_r18_hailo/latest.pth --shape 1024 1920 --output-file fcn.onnx
    </code>

<br>

---

## Compile the Model using Hailo Model Zoo
You can generate an HEF file for inference on Hailo-8 from your trained ONNX model.  
In order to do so you need a working model-zoo environment.
Choose the corresponding YAML from our networks configuration directory, i.e. <code>hailo_model_zoo/cfg/networks/fcn16_resnet_v1_18.yaml</code>, and run compilation using the model zoo:  
  
  <code stage="compile">
  python <span val="mz_main_path">hailo_model_zoo/main.py</span> compile --ckpt <span val="local_path_to_onnx">fcn.onnx</span> --calib-path <span val="calib_set_path">/path/to/calibration/imgs/dir/</span> --yaml <span val="yaml_file_path">fcn16_resnet_v1_18.yaml</span>
  </code>

  * <code>--ckpt</code> - path to your ONNX file.
  * <code>--calib-path</code> - path to a directory with your calibration images in JPEG format
  * <code>--yaml</code> - path to your configuration YAML file. In case you have made some changes in the model, you might need to update its start/end nodes names / number of classes and so on.  <br>
  The model zoo will take care of adding the input normalization to be part of the model.

The compiled model can be used in the [**Hailo TAPPAS**](https://hailo.ai/developer-zone/tappas-apps-toolkit/) to generate a full application.