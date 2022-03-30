# YOLACT Retraining
 * To learn more about Yolact look [**here**](https://github.com/hailo-ai/yolact/tree/Model-Zoo-1.5)
---
## Prerequisites
  * docker ([installation instructions](https://docs.docker.com/engine/install/ubuntu/))
  * nvidia-docker2 ([installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
  > **_NOTE:_**  In case you use Hailo Software Suite docker, make sure you are doing all the following instructions outside this docker.
## Environment Preparations
1. Build the docker image:
	
	<code stage="docker_build">
	cd <span val="dockerfile_path">hailo_model_zoo/training/yolact</span>

	docker build --build-arg timezone=`cat /etc/timezone` -t yolact:v0 .
	</code>

	the following optional arguments can be passed via --build-arg:

	- `timezone` - a string for setting up timezone. E.g. "Asia/Jerusalem"
	- `user` - username for a local non-root user. Defaults to 'hailo'.
	- `group` - default group for a local non-root user. Defaults to 'hailo'.
	- `uid` - user id for a local non-root user.
	- `gid` - group id for a local non-root user.
	- This command will build the docker image with the necessary requirements using the Dockerfile exists in the yolact directory.
  
2. Start your docker:

	<code stage="docker_run">
	docker run <span val="replace_none">--name "your_docker_name"</span> -it --gpus all --ipc=host -v <span val="local_vol_path">/path/to/local/data/dir</span>:<span val="docker_vol_path">/path/to/docker/data/dir</span>  yolact:v0
	</code>

      - `docker run` create a new docker container.
      - `--name <your_docker_name>` name for your container.
      - `-it` runs the command interactively.
      - `--gpus all` allows access to all GPUs.
      - `--ipc=host` sets the IPC mode for the container.
      - `-v /path/to/local/data/dir:/path/to/docker/data/dir` maps `/path/to/local/data/dir` from the host to the container. You can use this command multiple times to mount multiple directories.
      - `yolact:v0` the name of the docker image.

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
            |---|-- train2017
            |---|---|-- *.jpg
            |---|-- val2017
            |---|---|-- *.jpg
            |---|-- test2017
            `---|---|-- *.jpg
    ```
	For training on custom datasets see [**here**](https://github.com/hailo-ai/yolact/tree/Model-Zoo-1.5#custom-datasets)
  
2. Train your model:
	Once your dataset is prepared, you can start training your model:
	
	<code stage="retrain">
	python train.py --config=yolact_regnetx_800MF_config
	</code>

	* <code>yolact_regnetx_800MF_config</code> - configuration using the regnetx_800MF backbone. Two other available options are <code>yolact_regnetx_600MF_config</code> and <code>yolact_regnetx_1600MF_config</code>.
  
3. Export to ONNX:
	In orded to export your trained YOLACT model to ONNX run the following script:
	
	<code stage="export">
	python export.py --config=yolact_regnetx_800MF_config --trained_model=<span val="docker_path_to_trained_model">path/to/trained/model</span> --export_path=<span val="docker_path_to_onnx">path/to/export/model.onnx</span>
	</code>

	* <code>--config</code> - same configuration used for training.
	* <code>--trained_model</code> - path to the weights produced by the training process.
	* <code>--export_path</code> - path to export the ONNX file to. Include the <code>.onnx</code> extension.

<br>

---

## Compile the Model using Hailo Model Zoo
You can generate an HEF file for inference on Hailo-8 from your trained ONNX model.
In order to do so you need a working model-zoo environment.
Choose the corresponding YAML from our networks configuration directory, i.e. <code>hailo_model_zoo/cfg/networks/yolact.yaml</code>, and run compilation using the model zoo:  
  
  <code stage="compile">
  python <span val="mz_main_path">hailo_model_zoo/main.py</span> compile <span val="replace_none">yolact</span> --ckpt <span val="local_path_to_onnx">yolact.onnx</span> --calib-path <span val="calib_set_path">/path/to/calibration/imgs/dir/</span> --yaml <span val="yaml_file_path">yolact_regnetx_800mf_20classes.yaml</span>
  </code>


> **_NOTE:_** The yolact_regnetx_800mf_20classes is an example yaml where some of the classes were removed. In case you want to remove classes you can remove them in tha yaml file, under *channel_remove*.
  * <code>--ckpt</code> - path to your ONNX file.
  * <code>--calib-path</code> - path to a directory with your calibration images in JPEG format
  * <code>--yaml</code> - path to your configuration YAML file. In case you have made some changes in the model, you might need to update its start/end nodes names / number of classes and so on.  <br>
  The model zoo will take care of adding the input normalization to be part of the model.

The compiled model can be used in the [**Hailo TAPPAS**](https://hailo.ai/developer-zone/tappas-apps-toolkit/) to generate a full application.