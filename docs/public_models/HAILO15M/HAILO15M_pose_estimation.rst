
Public Pre-Trained Models
=========================

.. |rocket| image:: ../../images/rocket.png
  :width: 18

.. |star| image:: ../../images/star.png
  :width: 18

Here, we give the full list of publicly pre-trained models supported by the Hailo Model Zoo.

* Benchmark Networks are marked with |rocket|
* Networks available in `TAPPAS <https://github.com/hailo-ai/tappas>`_ are marked with |star|
* Benchmark and TAPPAS  networks run in performance mode
* All models were compiled using Hailo Dataflow Compiler v3.30.0



.. _Pose Estimation:

---------------

COCO
^^^^

.. list-table::
   :widths: 31 9 7 11 9 8 8 8 7 7 7 7
   :header-rows: 1

   * - Network Name
     - mAP
     - HW Accuracy
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
     - Profile Html     
   * - centerpose_regnetx_1.6gf_fpn  |star| 
     - 53.54
     - 53.28
     - 47
     - 63
     - 640x640x3
     - 14.28
     - 64.58
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/PoseEstimation/centerpose_regnetx_1.6gf_fpn/pretrained/2022-03-23/centerpose_regnetx_1.6gf_fpn.zip>`_
     - `link <https://github.com/tensorboy/centerpose>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo15m/centerpose_regnetx_1.6gf_fpn.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo15m/centerpose_regnetx_1.6gf_fpn_profiler_results_compiled.html>`_    
   * - centerpose_regnetx_800mf   
     - 44.06
     - 42.95
     - 57
     - 74
     - 512x512x3
     - 12.31
     - 86.12
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/PoseEstimation/centerpose_regnetx_800mf/pretrained/2021-07-11/centerpose_regnetx_800mf.zip>`_
     - `link <https://github.com/tensorboy/centerpose>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo15m/centerpose_regnetx_800mf.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo15m/centerpose_regnetx_800mf_profiler_results_compiled.html>`_    
   * - centerpose_repvgg_a0   
     - 39.16
     - 39.14
     - 99
     - 149
     - 416x416x3
     - 11.71
     - 28.27
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/PoseEstimation/centerpose_repvgg_a0/pretrained/2021-09-26/centerpose_repvgg_a0.zip>`_
     - `link <https://github.com/tensorboy/centerpose>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo15m/centerpose_repvgg_a0.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo15m/centerpose_repvgg_a0_profiler_results_compiled.html>`_    
   * - yolov8m_pose   
     - 64.26
     - 61.97
     - 39
     - 53
     - 640x640x3
     - 26.4
     - 81.02
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/PoseEstimation/yolov8/yolov8m/pretrained/2023-06-11/yolov8m_pose.zip>`_
     - `link <https://github.com/ultralytics/ultralytics>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo15m/yolov8m_pose.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo15m/yolov8m_pose_profiler_results_compiled.html>`_    
   * - yolov8s_pose   
     - 59.2
     - 56.71
     - 84
     - 124
     - 640x640x3
     - 11.6
     - 30.2
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/PoseEstimation/yolov8/yolov8s/pretrained/2023-06-11/yolov8s_pose.zip>`_
     - `link <https://github.com/ultralytics/ultralytics>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo15m/yolov8s_pose.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo15m/yolov8s_pose_profiler_results_compiled.html>`_
