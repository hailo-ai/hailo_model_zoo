# Centerpose Retraining
  * To learn more about CenterPose look [**here**](https://github.com/hailo-ai/centerpose)
---

## Prerequisites
  * docker ([installation instructions](https://docs.docker.com/engine/install/ubuntu/))
  * nvidia-docker2 ([installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
  > **_NOTE:_**  In case you use Hailo Software Suite docker, make sure you are doing all the following instructions outside of this docker.
## Environment Preparations
1. Build the docker image:
    
    <code stage="docker_build">
    cd <span val="dockerfile_path">hailo_model_zoo/training/centerpose</span>

    docker build -t centerpose:v0 --build-arg timezone=\`cat /etc/timezone\` .
    </code>

	the following optional arguments can be passed via --build-arg:
  
	- `timezone` - a string for setting up timezone. E.g. "Asia/Jerusalem"
	- `user` - username for a local non-root user. Defaults to 'hailo'.
	- `group` - default group for a local non-root user. Defaults to 'hailo'.
	- `uid` - user id for a local non-root user.
	- `gid` - group id for a local non-root user.

2. Start your docker:
    
    <code stage="docker_run">
    docker run <span val="replace_none">--name "your_docker_name"</span> -it --gpus all <span val="replace_none">-u "username"</span> --ipc=host -v <span val="local_vol_path">/path/to/local/data/dir</span>:<span val="docker_vol_path">/path/to/docker/data/dir</span>  centerpose:v0
    </code>

      - `docker run` create a new docker container.
      - `--name <your_docker_name>` name for your container.
      - `-it` runs the command interactively.
      - `--gpus all` allows access to all GPUs.
      - `--ipc=host` sets the IPC mode for the container.
      - `-v /path/to/local/data/dir:/path/to/docker/data/dir` maps `/path/to/local/data/dir` from the host to the container. You can use this command multiple times to mount multiple directories.
      - `centerpose:v0` the name of the docker image.

## Training and exporting to ONNX
1. Prepare your data: <br>
    Data is expected to be in coco format, and by default should be in /workspace/data/<dataset_name>.

    The expected structure is as follows:
    ```
    /workspace
    |-- data
    `-- |-- coco
        `-- |-- annotations
            |   |-- instances_train2017.json
            |   |-- instances_val2017.json
            |   |-- person_keypoints_train2017.json
            |   |-- person_keypoints_val2017.json
            |   |-- image_info_test-dev2017.json
        `-- |-- images
            |---|-- train2017
            |---|---|-- *.jpg
            |---|-- val2017
            |---|---|-- *.jpg
            |---|-- test2017
            `---|---|-- *.jpg
    ```
    The path for the dataset can be configured in the .yaml file, e.g. centerpose/experiments/regnet_fpn.yaml

2. Training: <br>
    Configure your model in a .yaml file. We'll use /workspace/centerpose/experiments/regnet_fpn.yaml in this guide.

    start training with the following command:
    
    <code stage="retrain">
    cd /workspace/centerpose/tools

    python -m torch.distributed.launch --nproc_per_node <span val="gpu_num">4</span> train.py --cfg ../experiments/regnet_fpn.yaml
    </code>

    Where 4 is the number of GPUs used for training.
    If using a different number, update both this and the used gpus in the .yaml configuration.

3. Exporting to onnx
    After training, run the following command:
    
    <code stage="export">
    cd /workspace/centerpose/tools

    python export.py --cfg ../experiments/regnet_fpn.yaml --TESTMODEL /workspace/out/regnet1_6/<span val="model_best_to_last">model_best.pth</span>
    </code>


<br>

---

## Compile the Model using Hailo Model Zoo
You can generate an HEF file for inference on Hailo-8 from your trained ONNX model.
In order to do so you need a working model-zoo environment.
Choose the corresponding YAML from our networks configuration directory, i.e. <code>hailo_model_zoo/cfg/networks/centerpose_regnetx_1.6gf_fpn.yaml</code>, and run compilation using the model zoo:  

  <code stage="compile">
  python <span val="mz_main_path">hailo_model_zoo/main.py</span> compile --ckpt <span val="local_path_to_onnx">coco_pose_regnet1.6_fpn.onnx</span> --calib-path <span val="calib_set_path">/path/to/calibration/imgs/dir/</span> --yaml <span val="yaml_file_path">centerpose_regnetx_1.6gf_fpn.yaml</span>
  </code>

  * <code>--ckpt</code> - path to your ONNX file.
  * <code>--calib-path</code> - path to a directory with your calibration images in JPEG format
  * <code>--yaml</code> - path to your configuration YAML file. In case you have made some changes in the model, you might need to update its start/end nodes names / number of classes and so on.  <br>
  The model zoo will take care of adding the input normalization to be part of the model.

The compiled model can be used in the [**Hailo TAPPAS**](https://hailo.ai/developer-zone/tappas-apps-toolkit/) to generate a full application.