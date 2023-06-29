=========================
Retrain on Custom Dataset
=========================

This page will help you ramping-up your training environment for various tasks and architectures.
Each architecture has its own README which describes how to:


#. Setup the environment using a compatible Dockerfile.
#. Start the training process.
#. Export your model.

**Important:**
Retraining is not available inside the docker version of Hailo Software Suite. In case you use it, clone the hailo_model_zoo outside of the docker, and perform the retraining there:
``git clone https://github.com/hailo-ai/hailo_model_zoo.git``


**Object Detection**


* `YOLOv3 <../training/yolov3/README.rst>`_
* `YOLOv4 <../training/yolov4/README.rst>`_
* `YOLOv5 <../training/yolov5/README.rst>`_
* `YOLOX <../training/yolox/README.rst>`_
* `DAMO-YOLO <../training/damoyolo/README.rst>`_
* `NanoDet <../training/nanodet/README.rst>`_
* `SSD <../training/ssd/README.rst>`_

**Pose Estimation**

* `CenterPose <../training/centerpose/README.rst>`_

**Semantic Segmentation**

* `FCN <../training/fcn/README.rst>`_

**Instance Segmentation**

* `YOLACT <../training/yolact/README.rst>`_

**Face Recognition**

* `ArcFace <../training/arcface/README.rst>`_
