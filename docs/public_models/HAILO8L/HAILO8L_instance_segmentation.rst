


Public Models
=============

* System host: Intel® Core™ i5-9400 CPU @ 2.90GHz
* Hailo Dataflow Compiler Version v2.19.0
* Measurement conditions: PCIe Gen 3 x 4 lanes, room temperature

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
     - 27.6
     - 27.2
     - 40.9
     - 63.1
     - | `S <https://github.com/dbolya/yolact>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolact_regnetx_1.6gf/pretrained/2022-11-30/yolact_regnetx_1.6gf.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/yolact_regnetx_1.6gf_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/yolact_regnetx_1.6gf.hef>`_
     - 512x512x3
     - 30.09
     - 125.34
   
   
   
   
   
   
   

   * - yolact_regnetx_800mf
     - 25.6
     - 25.4
     - 41.3
     - 65.3
     - | `S <https://github.com/dbolya/yolact>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolact_regnetx_800mf/pretrained/2022-11-30/yolact_regnetx_800mf.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/yolact_regnetx_800mf_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/yolact_regnetx_800mf.hef>`_
     - 512x512x3
     - 28.3
     - 116.75
   
   
   
   
   
   
   

   * - yolo26m_seg
     - 43.5
     - 41.3
     - 16.0
     - 21.7
     - | `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolo26/yolo26m_seg/pretrained/2026-02-12/yolo26m-seg.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/yolo26m_seg_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/yolo26m_seg.hef>`_
     - 640x640x3
     - 23.6
     - 131.8
   
   
   
   
   
   
   

   * - yolo26x_seg
     - 46.4
     - 44.6
     - 9.24
     - 13.2
     - | `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolo26/yolo26x_seg/pretrained/2026-02-12/yolo26x-seg.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/yolo26x_seg_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/yolo26x_seg.hef>`_
     - 640x640x3
     - 62.8
     - 336.7
   
   
   
   
   
   
   

   * - yolov11l_seg
     - 42.9
     - 42.6
     - 18.2
     - 30.5
     - | `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov11/yolov11l_seg/2026-03-23/yolo11l-seg.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/yolov11l_seg_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/yolov11l_seg.hef>`_
     - 640x640x3
     - 27.7
     - 142.5
   
   
   
   
   
   
   

   * - yolov11m_seg
     - 41.7
     - 40.7
     - 24.3
     - 35.9
     - | `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov11/yolov11m_seg/2026-03-23/yolo11m-seg.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/yolov11m_seg_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/yolov11m_seg.hef>`_
     - 640x640x3
     - 22.4
     - 123.4
   
   
   
   
   
   
   

   * - yolov11n_seg
     - 32.1
     - 31.2
     - 149
     - 342
     - | `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov11/yolov11n_seg/2026-03-23/yolo11n-seg.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/yolov11n_seg_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/yolov11n_seg.hef>`_
     - 640x640x3
     - 2.9
     - 9.81
   
   
   
   
   
   
   

   * - yolov11s_seg
     - 37.7
     - 37.2
     - 71.0
     - 145
     - | `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov11/yolov11s_seg/2026-03-23/yolo11s-seg.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/yolov11s_seg_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/yolov11s_seg.hef>`_
     - 640x640x3
     - 10.1
     - 35.6
   
   
   
   
   
   
   

   * - yolov11x_seg
     - 43.8
     - 43.3
     - 11.3
     - 15.9
     - | `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov11/yolov11x_seg/2026-03-23/yolo11x-seg.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/yolov11x_seg_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/yolov11x_seg.hef>`_
     - 640x640x3
     - 62.1
     - 319.4
   
   
   
   
   
   
   

   * - yolov5l_seg⭐
     - 39.8
     - 39.3
     - 27.0
     - 41.3
     - | `S <https://github.com/ultralytics/yolov5>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5l/pretrained/2022-10-30/yolov5l-seg.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/yolov5l_seg_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/yolov5l_seg.hef>`_
     - 640x640x3
     - 47.89
     - 147.88
   
   
   
   
   
   
   

   * - yolov5m_seg⭐
     - 37.1
     - 36.6
     - 53.0
     - 83.6
     - | `S <https://github.com/ultralytics/yolov5>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5m/pretrained/2022-10-30/yolov5m-seg.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/yolov5m_seg_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/yolov5m_seg.hef>`_
     - 640x640x3
     - 32.60
     - 70.94
   
   
   
   
   
   
   

   * - yolov5n_seg⭐
     - 23.3
     - 22.9
     - 145
     - 220
     - | `S <https://github.com/ultralytics/yolov5>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5n/pretrained/2022-10-30/yolov5n-seg.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/yolov5n_seg_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/yolov5n_seg.hef>`_
     - 640x640x3
     - 1.99
     - 7.1
   
   
   
   
   
   
   

   * - yolov5s_seg⭐
     - 31.6
     - 30.7
     - 102
     - 160
     - | `S <https://github.com/ultralytics/yolov5>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5s/pretrained/2022-10-30/yolov5s-seg.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/yolov5s_seg_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/yolov5s_seg.hef>`_
     - 640x640x3
     - 7.61
     - 26.42
   
   
   
   
   
   
   

   * - yolov8m_seg⭐
     - 40.6
     - 40.1
     - 42.0
     - 69.8
     - | `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov8/yolov8m/pretrained/2023-03-06/yolov8m-seg.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/yolov8m_seg_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/yolov8m_seg.hef>`_
     - 640x640x3
     - 27.3
     - 110.2
   
   
   
   
   
   
   

   * - yolov8n_seg⭐
     - 30.3
     - 29.7
     - 168
     - 342
     - | `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov8/yolov8n/pretrained/2023-03-06/yolov8n-seg.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/yolov8n_seg_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/yolov8n_seg.hef>`_
     - 640x640x3
     - 3.4
     - 12.04
   
   
   
   
   
   
   

   * - yolov8s_seg⭐
     - 36.6
     - 36.4
     - 88.0
     - 159
     - | `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov8/yolov8s/pretrained/2023-03-06/yolov8s-seg.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/yolov8s_seg_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/yolov8s_seg.hef>`_
     - 640x640x3
     - 11.8
     - 42.6