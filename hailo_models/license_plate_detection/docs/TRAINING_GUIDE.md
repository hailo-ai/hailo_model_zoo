# Train License Plate Detection on a Custom Dataset
 Here we describe how to finetune Hailo's license plate detection network on your own custom dataset.
## Prerequisites
* docker ([installation instructions](https://docs.docker.com/engine/install/ubuntu/))
* nvidia-docker2 ([installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
> **_NOTE:_**  In case you use Hailo Software Suite docker, make sure you are doing all the following instructions outside this docker.

## Environement Preparations

1. **Build the docker image**  
    ```
    cd model_zoo/hailo_models/license_plate_detection/
    docker build --build-arg timezone=`cat /etc/timezone` -t license_plate_detection:v0 .
    ```
    - This command will build the docker image with the necessary requirements using the Dockerfile exists in this directory.  

2. **Start your docker:**
    ```
    docker run -it --gpus all --ipc=host -v /path/to/local/drive:/path/to/docker/dir license_plate_detection:v0
    ```
## Finetuning and exporting to ONNX
1. **Train the network on your dataset**<br>
Once the docker is started, you can train the license plate detector on your custom dataset. We recommend following the instructions for YOLOv4 training that can be found in [**here**](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects). The important steps are specified below:

    - Update `data/obj.data` with paths to your training and validation `.txt` files, which contain the list of the image paths<sup>*</sup>.
        ```
        classes = 1
        train  = data/train.txt
        valid  = data/val.txt
        names = data/obj.names
        backup = backup/
        ```
        \* Tip: specify the paths to the training and validation images in the training and validation `.txt` files relative to `/workspace/darknet/`

    - Place your training and validation images and labels in your data folder.

    - Start training on your dataset starting from our pre-trained weights in ```tiny_yolov4_license_plates.weights``` (or download it from [here](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/LPR/lp_detector/tiny_yolov4_license_plates/2021-12-23/tiny_yolov4_license_plates.weights))
        ```
        ./darknet detector train data/obj.data ./cfg/tiny_yolov4_license_plates.cfg tiny_yolov4_license_plates.weights -map
        ```

2. **Export to ONNX**<br>
    Export the model to ONNX using the following command:
    ```
    python ../pytorch-YOLOv4/demo_darknet2onnx.py ./cfg/tiny_yolov4_license_plates.cfg /path/to/trained.weights /path/to/some/image.jpg 1
    ```