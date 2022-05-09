 # Train Person-Face Detection on a Custom Dataset
 Here we describe how to finetune Hailo's person-face detection network with your own custom dataset.
## Prerequisites
* docker ([installation instructions](https://docs.docker.com/engine/install/ubuntu/))
* nvidia-docker2 ([installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
> **_NOTE:_**  In case you use Hailo Software Suite docker, make sure you are doing all the following instructions outside of this docker.

## Environment Preparations

1. **Build the docker image:**
    ```
    cd hailo_model_zoo/hailo_models/personface_detection/
    docker build  --build-arg timezone=`cat /etc/timezone` -t personface_detection:v0 .
    ```
    - This command will build the docker image with the necessary requirements using the Dockerfile that exists in this directory.

2. **Start your docker:**
    ```
    docker run --name <your_docker_name> -it --gpus all --ipc=host -v /path/to/local/drive:/path/to/docker/dir personface_detection:v0
    ```
      - `docker run` create a new docker container.
      - `--name <your_docker_name>` name for your container.
      - `-it` runs the command interactively.
      - `--gpus all` allows access to all GPUs.
      - `--ipc=host` sets the IPC mode for the container.
      - `-v /path/to/local/data/dir:/path/to/docker/data/dir` maps `/path/to/local/data/dir` from the host to the container. You can use this command multiple times to mount multiple directories.
      - `personface_detection:v0` the name of the docker image.

## Finetuning and exporting to ONNX
1. **Train the network on your dataset**<br>
Once the docker is started, you can train the person-face detector on your custom dataset. We recommend following the instructions for YOLOV5 training that can be found in [**here**](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#11-create-datasetyaml). The important steps are specified below:

    - Update the dataset config file <code>data/personface_data.yaml</code> with the paths to your training and validation images files.
        ```
        #update your data paths
        train: /path/to/personface/images/train/
        val: /path/to/personface/images/val

        # number of classes
        nc: 2

        # class names
        names: ['person', 'face']
        ```
    - Start training on your dataset starting from our pre-trained weights in ```weights/yolov5s_personface.pt``` (you can also download it from [here](
    https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/MCPReID/personface_detector/yolov5s_personface/2022-04-01/yolov5s_personface.pt)

        ```
        python train.py --data ./data/personface_data.yaml --cfg ./models/yolov5s_personface.yaml --weights ./weights/yolov5s_personface.pt --epochs 300 --batch 128 --device 1,2,3,4
        ```

2. **Export to ONNX**<br>
Export the model to ONNX using the following command:
    ```
    python models/export.py --weights ./runs/exp<#>/weights/best.pt --img-size 640 --batch-size 1
    ```
    * The best model's weights will be saved under the following path: ```./runs/exp<#>/weights/best.pt```
    , where <#> is the experiment number.
    * Export at 640x640 with batch size 1
<br>

---
## Compile the Model using Hailo Model Zoo<br>
You can generate an HEF file for inference on Hailo-8 from your trained ONNX model. In order to do so you need a working model-zoo environment.
Choose the model YAML from our networks configuration directory, i.e. `hailo_model_zoo/cfg/networks/yolov5s_personface.yaml`, and run compilation using the model zoo:
```
hailomz compile --ckpt yolov5s_personface.onnx --calib-path /path/to/calibration/imgs/dir/ --yaml yolov5s_personface.yaml
```

* <code>--ckpt</code> - path to your ONNX file.
* <code>--calib-path</code> - path to a directory with your calibration images in JPEG format
* <code>--yaml</code> - path to your configuration YAML file.
<br>

The model zoo will take care of adding the input normalization to be part of the model. The compiled model can be used in the [**Hailo TAPPAS**](https://hailo.ai/developer-zone/tappas-apps-toolkit/) to generate a full application.