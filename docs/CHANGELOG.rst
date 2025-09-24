**v2.17**

* Update to use Dataflow Compiler v3.33.0 (`developer-zone <https://hailo.ai/developer-zone/>`_)
* Update to use HailoRT 4.23.0 (`developer-zone <https://hailo.ai/developer-zone/>`_)

* Performance Improvements for clip_vit_b_16_image_encoder and clip_vit_b_32_image_encoder models
* Add support for TinyCLIP family models:

  * tinyclip_vit_8m_16_text_3m_yfcc15m
  * tinyclip_vit_39m_16_text_19m_yfcc15m
  * tinyclip_vit_40m_32_text_19m_laion400m
  * tinyclip_vit_61m_32_text_29m_laion400m

* Bug fixes

**v2.16**

* Update to use Dataflow Compiler v3.32.0 (`developer-zone <https://hailo.ai/developer-zone/>`_)
* Update to use HailoRT 4.22.0 (`developer-zone <https://hailo.ai/developer-zone/>`_)

* Removed support for hailo15h, hailo15m, hailo15l and hailo10h
* Removed support for clip_vit_l_14 and clip_vit_l_14_laion2B models

* Bug fixes

**v2.15.1**

* Included missing files for CLIP models

**v2.15**

* Update to use Dataflow Compiler v3.31.0 (`developer-zone <https://hailo.ai/developer-zone/>`_)
* Update to use HailoRT 4.21.0 (`developer-zone <https://hailo.ai/developer-zone/>`_)

* New Models:

  * `CLIP <https://arxiv.org/pdf/2103.00020>`_ - ViT-Base-16, ViT-Base-32, ViT-Large-14 (336x336 resolution) - Contrastive Language-Image Pre-training model [Hailo-15H and Hailo-10H only]
  * `Real-ESRGAN <https://arxiv.org/pdf/2107.10833>`_ - x4 - Super Resolution model [Hailo-15H and Hailo-10H only]
  * `PoolFormer <https://arxiv.org/pdf/2111.11418>`_ - s12 - Vision Transformer classification model [Hailo-15M/H and Hailo-10H only]

* Bug fixes

**v2.14**

* Update to use Dataflow Compiler v3.30.0 (`developer-zone <https://hailo.ai/developer-zone/>`_)
* Update to use HailoRT 4.20.0 (`developer-zone <https://hailo.ai/developer-zone/>`_)

* New cascade API (experimental)

  * Currently supports PETRv2, bird-eye-view network for 3D object detection, see ``petrv2_repvggB0.yaml`` for configurations.

  * The user needs existing hars/hefs: both ``petrv2_repvggB0_backbone_pp_800x320`` & ``petrv2_repvggB0_transformer_pp_800x320``

  * full_precision evaluation: ``hailomz cascade eval petrv2``

  * hardware evaluation: ``hailomz cascade eval petrv2 --override target=hardware``

* New task:

  * Human Action Recognition

    * Added support for (partial) Kinetics-400 dataset

    * Added r3d_18 to support this task

* New Models:

  * `YOLOv11 <https://arxiv.org/pdf/2410.17725>`_ - nano, small, medium, large, x-large - Latest YOLO detectors
  * `CLIP <https://arxiv.org/pdf/2103.00020>`_ - ViT-Large-14-Laion2B - Contrastive Language-Image Pre-training model [H15H and H10H only]
  * `SWIN <https://arxiv.org/pdf/2103.14030>`_ - tiny, small - Shifted-Windows Transformer based classification model
  * `DaViT <https://arxiv.org/pdf/2204.03645>`_ - tiny - Dual Attention Vision Transformer classification model [H15H and H10H only]
  * `LeViT <https://arxiv.org/pdf/2104.01136>`_ - levit128, levit192, levit384 - Transformer based classification model
  * `EfficientFormer <https://arxiv.org/pdf/2212.08059>`_ - l1 - Transformer based classification model
  * `Real-ESRGAN <https://arxiv.org/pdf/2107.10833>`_ - x2 - Super Resolution model
  * `R3D_18 <https://pytorch.org/vision/stable/models.html#video-classification>`_ - r3d_18 - Video Classification network for Human Action Recognition [H8 only]

* Bug fixes

**v2.13**

* Update to use Dataflow Compiler v3.29.0 (`developer-zone <https://hailo.ai/developer-zone/>`_)
* Update to use HailoRT 4.19.0 (`developer-zone <https://hailo.ai/developer-zone/>`_)

* Using jit_compile which reduces dramatically the emulation inference time of the Hailo Model Zoo models.

* New tasks:

  * BEV: Multi-View 3D Object Detection

    * Added support for NuScenes dataset

    * Added PETRv2 with the following configuration:

      1. Backbone: RepVGG-B0 (800x320 input resolution)

      2. Transformer: 3 decoder layers, detection queries=304, replaced LN with UN

* New Models:

  * `CAS-ViT <https://arxiv.org/pdf/2408.03703>`_ - S, M, T - Convolutional-Attention based classification model
  * `YOLOv10 <https://arxiv.org/pdf/2405.14458>`_ - base, x-large - Latest YOLO detectors
  * `CLIP <https://arxiv.org/pdf/2103.00020>`_ Text Encoders - ResNet50x4, ViT-Large

* New retraining Docker containers for:

  * PETR - Multi-View 3D Object Detection

* Introduced new flags for hailomz CLI:

  * ``--ap-per-class`` for measuring average-precision per-class. Relevant for object detection and instance segmentation tasks.

* Bug fixes

**v2.12**

* Update to use Dataflow Compiler v3.28.0 (`developer-zone <https://hailo.ai/developer-zone/>`_)
* Update to use HailoRT 4.18.0 (`developer-zone <https://hailo.ai/developer-zone/>`_)

* Target ``hardware`` now supports Hailo-10H device

* New Models:

  * Original ViT models - tiny, small, base - Transformer based classification models
  * DeiT models - tiny, small, base - Transformer based classification models
  * DETR (resnet50) - Transformer based object detection model
  * fastvit_sa12 - Fast transformer based classification model
  * levit256 - Transformer based classification model
  * YOLOv10 - nano, small - Latest YOLO detectors
  * RepGhostNet1.0x, RepGhostNet2.0x - Hardware-Efficient classification models

* New postprocessing support on NN Core:

  * yolov6 tag 0.2.1

* Added support for person attribute visualization

* Bug fixes

**v2.11**

* Update to use Dataflow Compiler v3.27.0 (`developer-zone <https://hailo.ai/developer-zone/>`_)
* Update to use HailoRT 4.18.0 (`developer-zone <https://hailo.ai/developer-zone/>`_)

* New Models:

  * FastSAM-s - Zero-shot Instance Segmentation
  * Yolov9c - Latest Object Detection model of the YOLO family

* Using HailoRT-pp for postprocessing of the following variants:

  * nanodet

  Postprocessing JSON configurations are now part of the cfg directory.

* Introduced new flags for hailomz CLI:

  * ``--start-node-names`` and ``--end-node-names`` for customizing parsing behavior.
  * ``--classes`` for adjusting the number of classes in post-processing configuration.

  The ``--performance`` flag, previously utilized for compiling models with their enhanced model script if available, now offers an additional functionality.
  In instances where a model lacks an optimized model script, this flag triggers the compiler's Performance Mode to achieve the best performance

  These flags simplify the process of compiling models generated from our retrain dockers.

* Bug fixes

**v2.10**

* Update to use Dataflow Compiler v3.26.0 (`developer-zone <https://hailo.ai/developer-zone/>`_)
* Update to use HailoRT 4.16.0 (`developer-zone <https://hailo.ai/developer-zone/>`_)

* Using HailoRT-pp for postprocessing of the following variants:

  * yolov8

* Profiler change:

  * Removal of ``--mode`` flag from ``hailomz profile`` command, which generates a report according to provided HAR state.

* CLI change:

  * ``hailo8`` target is deprecated in favor of ``hardware``

* Support KITTI Stereo Dataset
* New Models:

  * vit_pose_small - encoder based transformer with layernorm for pose estimation
  * segformer_b0_bn - encoder based transformer with batchnorm for semantic segmentation

* Bug fixes

**v2.9**

* Update to use Dataflow Compiler v3.25.0 (`developer-zone <https://hailo.ai/developer-zone/>`_)
* Update to use HailoRT 4.15.0 (`developer-zone <https://hailo.ai/developer-zone/>`_)
* A new CLI-compatible API that allows users to incorporate format conversion and reshaping capabilities into the input:

.. code-block::

   hailomz compile yolov5s --resize 1080 1920 --input-conversion nv12_to_rgb

* New transformer models added:

  * vit_pose_small_bn - encoder based transformer with batchnorm for pose estimation
  * clip_resnet_50x4 - Contrastive Language-Image Pre-Training for zero-shot classification

* New retraining dockers for vit variants using unified normalization.
* New Models:

  * yolov8s_pose / yolov8m_pose - pose estimation
  * scdepthv3 - depth-estimation
  * dncnn3 / dncnn_color_blind - image denoising
  * zero_dce_pp - low-light enhancement
  * stereonet - stereo depth estimation

* Using HailoRT-pp for postprocessing of the following models:

  * efficientdet_lite0 / efficientdet_lite1 / efficientdet_lite2

**v2.8**

* Update to use Dataflow Compiler v3.24.0 (`developer-zone <https://hailo.ai/developer-zone/>`_)
* Update to use HailoRT 4.14.0 (`developer-zone <https://hailo.ai/developer-zone/>`_)
* The Hailo Model Zoo now supports the following vision transformers models:

  * vit_tiny / vit_small / vit_base - encoder based transformer with batchnorm for classification
  * detr_resnet_v1_18_bn - encoder/decoder transformer for object detection
  * clip_resnet_50 - Contrastive Language-Image Pre-Training for zero-shot classification
  * yolov5s_c3tr - object detection model with a MHSA block

* Using HailoRT-pp for postprocessing of the following variants:

  * yolov5
  * yolox
  * ssd
  * efficientdet
  * yolov7

* New Models:

  * repvgg_a1 / repvgg_a2 - classification
  * yolov8_seg: yolov8n_seg / yolov8s_seg / yolov8m_seg - instance segmentation
  * yolov6n_0.2.1 - object detection
  * zero_dce - low-light enhancement

* New retraining dockers for:

  * yolov8
  * yolov8_seg

* Enable compilation for hailo15h device
* Enable evaluation of models with RGBX / NV12 input format
* Bug fixes

**v2.7**

* Update to use Dataflow Compiler v3.23.0 (`developer-zone <https://hailo.ai/developer-zone/>`_)
* Updated to use HailoRT 4.13.0 (`developer-zone <https://hailo.ai/developer-zone/>`_)
* Inference flow was moved to new high-level APIs
* New object detection variants:

  * yolov8: yolov8n / yolov8s / yolov8m / yolov8l / yolov8x
  * damoyolo: damoyolo_tinynasL20_T / damoyolo_tinynasL25_S / damoyolo_tinynasL35_M

* New transformers based models:

  * vit_base - classification model
  * yolov5s_c3tr - object detection model with a self-attention block

* Examples for using HailoRT-pp - support for seamless integration of models and their corresponding postprocessing

  * yolov5m_hpp

* Configuration YAMLs and model-scripts for networks with YUY2 input format
* DAMO-YOLO retraining docker
* Bug fixes

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
  * yolov5l_seg

* New object detection variants for high resolution images:

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

**NOTE:**\  Ubuntu 18.04 will be deprecated in Hailo Model Zoo future version

**NOTE:**\  Python 3.6 will be deprecated in Hailo Model Zoo future version

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
* New multi-network example: detection_pose_estimation which combines the following networks:

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
