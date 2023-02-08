Changelog
=========

**v2.6.1**

* Bug fixes

**v2.6**

* Update to use Dataflow Compiler v3.22.0 (`developer-zone <https://hailo.ai/developer-zone/>`_)
* Updated to use HailoRT 4.12.0 (`developer-zone <https://hailo.ai/developer-zone/>`_)
* ViT (`Vision Transformer <https://arxiv.org/pdf/2010.11929.pdf>`_) - new classification network with transformers-encoder based architecture 
* New instance segmentation variants:

  * yolov5n_seg
  * yolov5s_seg
  * yolov5m_seg
  
* New object detecion variants for high resolution images:

  * yolov7e6
  * yolov5n6_6.1
  * yolov5s6_6.1
  * yolov5m6_6.1
  
* New flag ``--performance`` to reproduce highest performance for a subset of networks
* Hailo model-zoo log is now written into ``sdk_virtualenv/etc/hailo/modelzoo/hailo_examples.log``
* Bug fixes 

**v2.5**

* Update to use Dataflow Compiler v3.20.1 (`developer-zone <https://hailo.ai/developer-zone/>`_)
* Model scripts use new bgr to rgb conversion
* New Yolact variants - with all COCO classes:

  * yolact_regnetx_800mf
  * yolact_regnetx_1.6gf
  
* Bug fixes 

**v2.4**

* Updated to use Dataflow Compiler v3.20 (`developer-zone <https://hailo.ai/developer-zone/>`_)
* Required FPS was moved from models YAML into the models scripts
* Model scripts use new change activation syntax
* New models:

  * Face Detection - scrfd_500m / scrfd_2.5g / scrfd_10g
  
* New tasks: 

  1. Super-Resolution

    * Added support for BSD100 dataset
    * The following models were added: espcn_x2 / espcn_x3 / espcn_x4
  2.  Face Recognition

    * Support for LFW dataset
    * The following models were added:

      #. arcface_r50
      #. arcface_mobilefacenet
    * Retraining docker for arcface architecture

* Added support for new hw-arch - hailo8l 

**v2.3**

* Updated to use Dataflow Compiler v3.19 (`developer-zone <https://hailo.ai/developer-zone/>`_)
* New models:

  * yolov6n
  * yolov7 / yolov7-tiny
  * nanodet_repvgg_a1_640
  * efficientdet_lite0 / efficientdet_lite1 / efficientdet_lite2
  
* New tasks:

  * mspn_regnetx_800mf - single person pose estimation
  * face_attr_resnet_v1_18 - face attribute recognition

* Single person pose estimation training docker (mspn_regnetx_800mf)
* Bug fixes

**v2.2**

* Updated to use Dataflow Compiler v3.18 (`developer-zone <https://hailo.ai/developer-zone/>`_)
* CLI change:

  * Hailo model zoo CLI is now working with an entry point - hailomz
  * quantize sub command was changed to optimize
  * Hailo model zoo data directory by default will be ``~/.hailomz``

* New models:
  
  * yolov5xs_wo_spp_nms - a model which contains bbox decoding and confidence thresholding on Hailo-8
  * osnet_x1_0 - person ReID network
  * yolov5m_6.1 - yolov5m network from the latest tag of the repo (6.1) including silu activation

* New tasks:
  
  * person_attr_resnet_v1_18 - person attribute recognition

* ReID training docker for the Hailo model repvgg_a0_person_reid_512/2048

**NOTE:**  Ubuntu 18.04 will be deprecated in Hailo Model Zoo future version

**NOTE:**  Python 3.6 will be deprecated in Hailo Model Zoo future version

**v2.1**

* Updated to use Dataflow Compiler v3.17 (`developer-zone <https://hailo.ai/developer-zone/>`_)
* Parser commands were moved into model scripts
* Support Market-1501 Dataset
* Support a new model zoo task - ReID
* New models:

  * | yolov5s_personface - person and face detector
  * | repvgg_a0_person_reid_512 / repvgg_a0_person_reid_2048 - ReID networks which outputs a person embedding
    | These models were trained in-house as part of our upcoming new application
  * | stdc1 - Segmentation architecture for Cityscapes
      
**v2.0**

* Updated to use Dataflow Compiler v3.16 (`developer-zone <https://hailo.ai/developer-zone/>`_) with TF version 2.5 which require CUDA11.2
* Updated to use HailoRT 4.6 (`developer-zone <https://hailo.ai/developer-zone/>`_)
* Retraining Dockers - each retraining docker has a corresponding README file near it. New retraining dockers:

  * SSD
  * YOLOX
  * FCN

* New models:

  * yolov5l

* Introducing Hailo Models, in-house pretrained networks with compatible Dockerfile for retraining

  * yolov5m_vehicles (vehicle detection)
  * tiny_yolov4_license_plates (license plate detection)
  * lprnet (license plate recognition)

* Added new documentation to the `YAML structure <YAML.rst>`_

   
**v1.5**

* Remove HailoRT installation dependency.
* Retraining Dockers

  * YOLOv3
  * NanoDet
  * CenterPose
  * Yolact

* New models:

  * unet_mobilenet_v2

* Support Oxford-IIIT Pet Dataset
* New mutli-network example: detection_pose_estimation which combines the following networks:

  * yolov5m_wo_spp_60p
  * centerpose_repvgg_a0

* Improvements:

  * nanodet_repvgg mAP increased by 2%

* | New Tasks:

  * | hand_landmark_lite from MediaPipe
  * | palm_detection_lite from MediaPipe
  
  | Both tasks are without evaluation module.


**v1.4**

* Update to use Dataflow Compiler v3.14.0 (`developer-zone <https://hailo.ai/developer-zone/>`_)
* Update to use HailoRT 4.3.0 (`developer-zone <https://hailo.ai/developer-zone/>`_)
* Introducing `Hailo Models <HAILO_MODELS.rst>`_ - in house pretrained networks with compatible Dockerfile for easy retraining:

  * yolov5m_vehicles - vehicle detector based on yolov5m architecture
  * tiny_yolov4_license_plates - license plate detector based on tiny_yolov4 architecture

* New Task: face landmarks detection

  * tddfa_mobilenet_v1
  * Support 300W-LP and AFLW2k3d datasets

* New features:

  * Support compilation of several networks together - a.k.a `multinets <GETTING_STARTED.rst#compile-multiple-networks-together>`_
  * CLI for printing `network information <GETTING_STARTED.rst#info>`_

* Retraining Guide:

  * New training guide for yolov4 with compatible Dockerfile
  * Modifications for yolov5 retraining

**v1.3**

* Update to use Dataflow Compiler v3.12.0 (`developer-zone <https://hailo.ai/developer-zone/>`_)
* New task: indoor depth estimation

  * fast_depth
  * Support NYU Depth V2 Dataset

* New models:

  * resmlp12 - new architecture support `paper <https://arxiv.org/abs/2105.03404>`_
  * yolox_l_leaky

* Improvements:

  * ssd_mobilenet_v1 - in-chip NMS optimization (de-fusing)

* Model Optimization API Changes

  * Model Optimization parameters can be updated using the networks' model script files (\*.alls)
  
  * Deprecated: quantization params in YAMLs

* Training Guide: new training guide for yolov5 with compatible Dockerfile

**v1.2**

* New features:

  * YUV to RGB on core can be added through YAML configuration.
  * Resize on core can be added through YAML configuration.

* Support D2S Dataset
* New task: instance segmentation

  * yolact_mobilenet_v1 (coco)
  * yolact_regnetx_800mf_20classes (coco)
  * yolact_regnetx_600mf_31classes (d2s)

* New models:

  * nanodet_repvgg
  * centernet_resnet_v1_50_postprocess
  * yolov3 - `darkent based <https://github.com/AlexeyAB/darknet>`_
  * yolox_s_wide_leaky
  * deeplab_v3_mobilenet_v2_dilation
  * centerpose_repvgg_a0
  * yolov5s, yolov5m - original models from `link <https://github.com/ultralytics/yolov5/tree/v2.0>`_
  * yolov5m_yuv - contains resize and color conversion on HW

* Improvements:

  * tiny_yolov4
  * yolov4

* IBC and Equalization API change
* Bug fixes

**v1.1**

* Support VisDrone Dataset 
* New task: pose estimation 

  * centerpose_regnetx_200mf_fpn 
  * centerpose_regnetx_800mf 
  * centerpose_regnetx_1.6gf_fpn 

* New task: face detection 

  * lightfaceslim 
  * retinaface_mobilenet_v1 

* New models: 

  * hardnet39ds 
  * hardnet68 
  * yolox_tiny_leaky 
  * yolox_s_leaky 
  * deeplab_v3_mobilenet_v2 

* Use your own network manual for YOLOv3, YOLOv4_leaky and YOLOv5.

**v1.0**

* Initial release
* Support for object detection, semantic segmentation and classification networks
