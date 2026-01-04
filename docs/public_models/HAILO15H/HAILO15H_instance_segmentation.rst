


Public Models
=============

All models were compiled using Hailo Dataflow Compiler v5.2.0.

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
     - 27.3
     - 63.9
     - 93.6
     - | `S <https://github.com/dbolya/yolact>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolact_regnetx_1.6gf/pretrained/2022-11-30/yolact_regnetx_1.6gf.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/yolact_regnetx_1.6gf_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/yolact_regnetx_1.6gf.hef>`_
     - 512x512x3
     - 30.09
     - 125.34
   
   
   
   
   
   
   

   * - yolact_regnetx_800mf
     - 25.6
     - 25.4
     - 77.6
     - 125
     - | `S <https://github.com/dbolya/yolact>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolact_regnetx_800mf/pretrained/2022-11-30/yolact_regnetx_800mf.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/yolact_regnetx_800mf_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/yolact_regnetx_800mf.hef>`_
     - 512x512x3
     - 28.3
     - 116.75
   
   
   
   
   
   
   

   * - yolov5l_seg
     - 39.8
     - 39.3
     - 43.3
     - 67.5
     - | `S <https://github.com/ultralytics/yolov5>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5l/pretrained/2022-10-30/yolov5l-seg.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/yolov5l_seg_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/yolov5l_seg.hef>`_
     - 640x640x3
     - 47.89
     - 147.88
   
   
   
   
   
   
   

   * - yolov5m_seg⭐
     - 37.1
     - 36.6
     - 65.8
     - 96.5
     - | `S <https://github.com/ultralytics/yolov5>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5m/pretrained/2022-10-30/yolov5m-seg.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/yolov5m_seg_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/yolov5m_seg.hef>`_
     - 640x640x3
     - 32.60
     - 70.94
   
   
   
   
   
   
   

   * - yolov5n_seg⭐
     - 23.3
     - 23.1
     - 189
     - 170
     - | `S <https://github.com/ultralytics/yolov5>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5n/pretrained/2022-10-30/yolov5n-seg.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/yolov5n_seg_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/yolov5n_seg.hef>`_ `RGBX <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/yolov5n_seg_rgbx.hef>`_
     - 640x640x3
     - 1.99
     - 7.1
   
   
   
   
   
   
   

   * - yolov5s_seg
     - 31.6
     - 30.7
     - 178
     - 166
     - | `S <https://github.com/ultralytics/yolov5>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5s/pretrained/2022-10-30/yolov5s-seg.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/yolov5s_seg_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/yolov5s_seg.hef>`_
     - 640x640x3
     - 7.61
     - 26.42
   
   
   
   
   
   
   

   * - yolov8m_seg
     - 40.6
     - 40.2
     - 65.9
     - 111
     - | `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov8/yolov8m/pretrained/2023-03-06/yolov8m-seg.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/yolov8m_seg_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/yolov8m_seg.hef>`_
     - 640x640x3
     - 27.3
     - 110.2
   
   
   
   
   
   
   

   * - yolov8n_seg
     - 30.3
     - 29.8
     - 237
     - 335
     - | `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov8/yolov8n/pretrained/2023-03-06/yolov8n-seg.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/yolov8n_seg_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/yolov8n_seg.hef>`_
     - 640x640x3
     - 3.4
     - 12.04
   
   
   
   
   
   
   

   * - yolov8s_seg
     - 36.6
     - 36.4
     - 141
     - 242
     - | `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov8/yolov8s/pretrained/2023-03-06/yolov8s-seg.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/yolov8s_seg_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/yolov8s_seg.hef>`_
     - 640x640x3
     - 11.8
     - 42.6