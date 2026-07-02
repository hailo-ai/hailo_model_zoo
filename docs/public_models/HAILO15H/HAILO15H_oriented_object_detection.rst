


Public Models
=============

* System host: Intel® Core™ i5-9400 CPU @ 2.90GHz
* Hailo Dataflow Compiler Version v5.4.0
* Measurement conditions: Measuring from the SoC, room temperature

|

Oriented Object Detection
=========================

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

Dotav1
------

|








.. list-table::
   :header-rows: 1
   :widths: 31 9 7 11 9 8 8 8 9

   * - Network Name
     - float mAP
     - Hardware mAP
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
   
   
   
   
   
   
   

   * - yolov11m_obb
     - 55.1
     - 54.1
     - 29.6
     - 45.6
     - | `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-DOTA/yolo/yolov11m/2025-11-19/yolo11m-obb.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/yolov11m_obb_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/yolov11m_obb.hef>`_
     - 1024x1024x3
     - 33.99
     - 183.59
   
   
   
   
   
   
   

   * - yolov11n_obb
     - 49.7
     - 48.4
     - 169
     - 312
     - | `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-DOTA/yolo/yolov11n/2025-11-19/yolo11n-obb.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/yolov11n_obb_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/yolov11n_obb.hef>`_
     - 1024x1024x3
     - 3.80
     - 17.25
   
   
   
   
   
   
   

   * - yolov11s_obb
     - 53.2
     - 52.3
     - 76.8
     - 143
     - | `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-DOTA/yolo/yolov11s/2025-11-19/yolo11s-obb.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/yolov11s_obb_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/yolov11s_obb.hef>`_
     - 1024x1024x3
     - 12.6
     - 57.94
   
   
   
   
   
   
   

   * - yolov11x_obb
     - 56.9
     - 56.0
     - 8.54
     - 10.9
     - | `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-DOTA/yolo/yolov11x/2025-11-19/yolo11x-obb.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/yolov11x_obb_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/yolov11x_obb.hef>`_
     - 1024x1024x3
     - 60.08
     - 521.61