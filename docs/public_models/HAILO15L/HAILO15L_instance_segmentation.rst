


Public Models
=============

* System host: Intel® Core™ i5-9400 CPU @ 2.90GHz
* Hailo Dataflow Compiler Version v5.4.0
* Measurement conditions: Measuring from the SoC, room temperature

|

Instance Segmentation
=====================

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
     - float mAP-segmentation
     - Hardware mAP-segmentation
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
   
   
   
   
   
   
   

   * - yolact_regnetx_1.6gf
     - 27.4
     - 27.0
     - 21.8
     - 25.4
     - | `S <https://github.com/dbolya/yolact>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolact_regnetx_1.6gf/pretrained/2022-11-30/yolact_regnetx_1.6gf.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15l/yolact_regnetx_1.6gf_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15l/yolact_regnetx_1.6gf.hef>`_
     - 512x512x3
     - 30.09
     - 125.34
   
   
   
   
   
   
   

   * - yolact_regnetx_800mf
     - 25.6
     - 25.4
     - 30.9
     - 38.1
     - | `S <https://github.com/dbolya/yolact>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolact_regnetx_800mf/pretrained/2022-11-30/yolact_regnetx_800mf.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15l/yolact_regnetx_800mf_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15l/yolact_regnetx_800mf.hef>`_
     - 512x512x3
     - 28.3
     - 116.75
   
   
   
   
   
   
   

   * - yolov5l_seg
     - 39.5
     - 39.3
     - 13.2
     - 14.8
     - | `S <https://github.com/ultralytics/yolov5>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5l/pretrained/2022-10-30/yolov5l-seg.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15l/yolov5l_seg_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15l/yolov5l_seg.hef>`_
     - 640x640x3
     - 47.89
     - 147.88
   
   
   
   
   
   
   

   * - yolov5m_seg⭐
     - 36.9
     - 36.4
     - 38.1
     - 45.0
     - | `S <https://github.com/ultralytics/yolov5>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5m/pretrained/2022-10-30/yolov5m-seg.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15l/yolov5m_seg_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15l/yolov5m_seg.hef>`_
     - 640x640x3
     - 32.60
     - 70.94
   
   
   
   
   
   
   

   * - yolov5n_seg⭐
     - 23.2
     - 23.0
     - 174
     - 217
     - | `S <https://github.com/ultralytics/yolov5>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5n/pretrained/2022-10-30/yolov5n-seg.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15l/yolov5n_seg_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15l/yolov5n_seg.hef>`_
     - 640x640x3
     - 1.99
     - 7.1
   
   
   
   
   
   
   

   * - yolov5s_seg
     - 31.3
     - 30.6
     - 97.2
     - 121
     - | `S <https://github.com/ultralytics/yolov5>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5s/pretrained/2022-10-30/yolov5s-seg.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15l/yolov5s_seg_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15l/yolov5s_seg.hef>`_
     - 640x640x3
     - 7.61
     - 26.42
   
   
   
   
   
   
   

   * - yolov8n_seg
     - 30.0
     - 29.4
     - 164
     - 220
     - | `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov8/yolov8n/pretrained/2023-03-06/yolov8n-seg.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15l/yolov8n_seg_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15l/yolov8n_seg.hef>`_
     - 640x640x3
     - 3.4
     - 12.04
   
   
   
   
   
   
   

   * - yolov8s_seg
     - 36.5
     - 36.1
     - 73.1
     - 93.7
     - | `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov8/yolov8s/pretrained/2023-03-06/yolov8s-seg.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15l/yolov8s_seg_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15l/yolov8s_seg.hef>`_
     - 640x640x3
     - 11.8
     - 42.6