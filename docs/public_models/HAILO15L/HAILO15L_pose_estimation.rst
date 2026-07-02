


Public Models
=============

* System host: Intel® Core™ i5-9400 CPU @ 2.90GHz
* Hailo Dataflow Compiler Version v5.4.0
* Measurement conditions: Measuring from the SoC, room temperature

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
     - Source – Link to the model's open-source repository.
   * - **PT**
     - Pretrained – Download the pretrained model file (ZIP format).
   * - **HEF, NV12, RGBX**
     - Compiled Models – Links to models in various formats:
       - **HEF:** RGB format
       - **NV12:** NV12 format
       - **RGBX:** RGBX format
   * - **PR**
     - Profiler Report – Download the model's performance profiling report.

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
     - 43.8
     - 42.7
     - 41.8
     - 48.7
     - | `S <https://github.com/tensorboy/centerpose>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/PoseEstimation/centerpose_regnetx_800mf/pretrained/2021-07-11/centerpose_regnetx_800mf.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15l/centerpose_regnetx_800mf_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15l/centerpose_regnetx_800mf.hef>`_
     - 512x512x3
     - 12.31
     - 86.12
   
   
   
   
   
   
   

   * - yolov8m_pose⭐
     - 63.8
     - 61.7
     - 38.3
     - 47.4
     - | `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/PoseEstimation/yolov8/yolov8m/pretrained/2023-06-11/yolov8m_pose.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15l/yolov8m_pose_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15l/yolov8m_pose.hef>`_
     - 640x640x3
     - 26.4
     - 81.02