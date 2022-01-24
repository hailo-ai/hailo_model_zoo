 # Train Vehicle Detection on a Custom Dataset
 Here we describe how to finetune Hailo's vehicle detection network with your own custom dataset.
## Prerequisites
* docker ([installation instructions](https://docs.docker.com/engine/install/ubuntu/))
* nvidia-docker2 ([installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
> **_NOTE:_**  In case you use Hailo Software Suite docker, make sure you are doing all the following instructions outside this docker.

## Environement Preparations

1. **Build the docker image**  
    cd model_zoo/hailo_models/vehicle_detection/
    docker build  --build-arg timezone=`cat /etc/timezone` -t vehicle_detection:v0 .
    ```
    - This command will build the docker image with the necessary requirements using the Dockerfile exists in this directory.  

2. **Start your docker:**
    ```
    docker run -it --gpus all --ipc=host -v /path/to/local/drive:/path/to/docker/dir vehicle_detection:v0
    ```

## Finetuning and exporting to ONNX
1. **Train the network on your dataset**<br>
Once the docker is started, you can train the vehicle detector on your custom dataset. We recommend following the instructions for YOLOV5 training that can be found in [**here**](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#11-create-datasetyaml). The important steps are specified below:

    - Update the dataset config file <code>data/vehicles.yaml</code> with the paths to your training and validation images files.
        ```
        #update your data paths
        train: /path/to/vehicles/training/images/
        val: /path/to/vehicles/validation/images/
        
        # number of classes
        nc: 1
        
        # class names
        names: ['vehicle']
        ```
    - Start training on your dataset starting from our pre-trained weights in ```weights/yolov5m_vehicles.pt``` (you can also download it from [here](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/LPR/vehicle_detector/yolov5m_vehicles/2021-11-16/yolov5m_vehicles.pt))
        ```
        python train.py --data ./data/vehicles.yaml --cfg ./models/yolov5m.yaml --weights ./weights/yolov5m_vehicles.pt --epochs 300 --batch 16
        ```  

2. **Export to ONNX**<br>
Export the model to ONNX using the following command:
    ```
    python models/export.py --weights /path/to/trained/model.pt --img 640 --batch 1  # export at 640x640 with batch size 1
    ```