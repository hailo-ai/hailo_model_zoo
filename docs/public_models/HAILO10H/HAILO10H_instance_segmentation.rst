
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
* All models were compiled using Hailo Dataflow Compiler v5.0.0

Link Legend

The following shortcuts are used in the table below to indicate available resources for each model:

* S – Source: Link to the model’s open-source code repository.
* PT – Pretrained: Download the pretrained model file (compressed in ZIP format).
* H, NV, X – Compiled Models: Links to the compiled model in various formats:
            * H: regular HEF with RGB format
            * NV: HEF with NV12 format
            * X: HEF with RGBX format

* PR – Profiler Report: Download the model’s performance profiling report.



.. _Instance Segmentation:

---------------------

COCO
^^^^

.. list-table::
   :widths: 31 9 7 11 9 8 8 8 9
   :header-rows: 1

   * - Network Name
     - float mAP
     - Hardware mAP
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
   * - yolact_regnetx_1.6gf
     - 27.3
     - 27.0
     - 66
     - 82
     - `S <https://github.com/dbolya/yolact>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolact_regnetx_1.6gf/pretrained/2022-11-30/yolact_regnetx_1.6gf.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo15h/yolact_regnetx_1.6gf.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo15h/yolact_regnetx_1.6gf_profiler_results_compiled.html>`_
     - 512x512x3
     - 30.09
     - 125.34
   * - yolact_regnetx_800mf
     - 25.4
     - 25.2
     - 77
     - 99
     - `S <https://github.com/dbolya/yolact>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolact_regnetx_800mf/pretrained/2022-11-30/yolact_regnetx_800mf.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo15h/yolact_regnetx_800mf.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo15h/yolact_regnetx_800mf_profiler_results_compiled.html>`_
     - 512x512x3
     - 28.3
     - 116.75
   * - yolov5m_seg
     - 36.6
     - 36.1
     - 64
     - 94
     - `S <https://github.com/ultralytics/yolov5>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5m/pretrained/2022-10-30/yolov5m-seg.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo15h/yolov5m_seg.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo15h/yolov5m_seg_profiler_results_compiled.html>`_
     - 640x640x3
     - 32.60
     - 70.94
   * - yolov5n_seg  |star|
     - 23.0
     - 22.7
     - 247
     - 171
     - `S <https://github.com/ultralytics/yolov5>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5n/pretrained/2022-10-30/yolov5n-seg.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo15h/yolov5n_seg.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo15h/yolov5n_seg_profiler_results_compiled.html>`_
     - 640x640x3
     - 1.99
     - 7.1
   * - yolov5s_seg
     - 30.8
     - 30.0
     - 191
     - 152
     - `S <https://github.com/ultralytics/yolov5>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5s/pretrained/2022-10-30/yolov5s-seg.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo15h/yolov5s_seg.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo15h/yolov5s_seg_profiler_results_compiled.html>`_
     - 640x640x3
     - 7.61
     - 26.42
   * - yolov8m_seg
     - 40.2
     - 39.8
     - 64
     - 96
     - `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov8/yolov8m/pretrained/2023-03-06/yolov8m-seg.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo15h/yolov8m_seg.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo15h/yolov8m_seg_profiler_results_compiled.html>`_
     - 640x640x3
     - 27.3
     - 110.2
   * - yolov8n_seg
     - 29.8
     - 29.3
     - 310
     - 212
     - `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov8/yolov8n/pretrained/2023-03-06/yolov8n-seg.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo15h/yolov8n_seg.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo15h/yolov8n_seg_profiler_results_compiled.html>`_
     - 640x640x3
     - 3.4
     - 12.04
   * - yolov8s_seg
     - 36.3
     - 36.0
     - 141
     - 164
     - `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov8/yolov8s/pretrained/2023-03-06/yolov8s-seg.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo15h/yolov8s_seg.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo15h/yolov8s_seg_profiler_results_compiled.html>`_
     - 640x640x3
     - 11.8
     - 42.6
.. list-table::
   :header-rows: 1

   * - Network Name
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
     - Profile Report
   * - yolov5l_seg
     - 42
     - 60
     - `S <https://github.com/ultralytics/yolov5>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5l/pretrained/2022-10-30/yolov5l-seg.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo15h/yolov5l_seg.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo15h/yolov5l_seg_profiler_results_compiled.html>`_
     - 640x640x3
     - 47.89
     - 147.88
