


Public Models
=============

All models were compiled using Hailo Dataflow Compiler v5.2.0.

|

Pose Estimation
===============

|

Link Legend
-----------

|

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - **Key / Icon**
     - **Description**
   * - ⭐
     - Networks used by `Hailo-apps <https://github.com/hailo-ai/hailo-apps-infra>`_.
   * - **S**
     - Source – Link to the model’s open-source repository.
   * - **PT**
     - Pretrained – Download the pretrained model file (ZIP format).
   * - **HEF, NV12, RGBX**
     - Compiled Models – Links to models in various formats:
       - **HEF:** RGB format
       - **NV12:** NV12 format
       - **RGBX:** RGBX format
   * - **PR**
     - Profiler Report – Download the model’s performance profiling report.

|

Coco
----

|

.. list-table::
   :header-rows: 1
   :widths: 31 9 7 11 9 8 8 8 9

   
   * - Network Name
     - float AP
     - Hardware AP
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
   
   
   
   
   
   
   

   * - centerpose_regnetx_800mf
     - 44.1
     - 42.9
     - 130
     - 194
     - | `S <https://github.com/tensorboy/centerpose>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/PoseEstimation/centerpose_regnetx_800mf/pretrained/2021-07-11/centerpose_regnetx_800mf.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/centerpose_regnetx_800mf_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/centerpose_regnetx_800mf.hef>`_
     - 512x512x3
     - 12.31
     - 86.12
   
   
   
   
   
   
   

   * - yolov8m_pose⭐
     - 64.3
     - 61.9
     - 79.8
     - 137
     - | `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/PoseEstimation/yolov8/yolov8m/pretrained/2023-06-11/yolov8m_pose.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/yolov8m_pose_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/yolov8m_pose.hef>`_
     - 640x640x3
     - 26.4
     - 81.02
   
   
   
   
   
   
   

   * - yolov8s_pose⭐
     - 59.2
     - 56.8
     - 163
     - 290
     - | `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/PoseEstimation/yolov8/yolov8s/pretrained/2023-06-11/yolov8s_pose.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/yolov8s_pose_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/yolov8s_pose.hef>`_
     - 640x640x3
     - 11.6
     - 30.2