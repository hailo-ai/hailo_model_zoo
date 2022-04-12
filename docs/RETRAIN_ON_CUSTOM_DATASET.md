# Retrain on Custom Dataset

This page will help you ramping-up your training environment for various tasks and architectures.
Each architecture has its own README which describes how to:
   1. Setup the environment using a compatible Dockerfile.
   2. Start the training process.
   3. Export your model.

> **Important**:  
    Retraining is not available inside the docker version of Hailo Software Suite. In case you use it, clone the hailo_model_zoo outside of the docker, and perform the retraining there:  
    ```git clone https://github.com/hailo-ai/hailo_model_zoo.git``` 
<br>

## Object Detection
* [YOLOv3](../training/yolov3/README.md)
* [YOLOv4](../training/yolov4/README.md)
* [YOLOv5](../training/yolov5/README.md)
* [YOLOX](../training/yolox/README.md)
* [NanoDet](../training/nanodet/README.md)
* [SSD](../training/ssd/README.md)

<br>

## Pose Estimation
* [CenterPose](../training/centerpose/README.md)

<br>

## Semantic Segmentation
* [FCN](../training/fcn/README.md)
<br>

## Instance Segmentation
* [YOLACT](../training/yolact/README.md)
