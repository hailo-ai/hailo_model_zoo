
Public Pre-Trained Models - Instance Segmentation Hailo15M
==========================================================

.. |rocket| image:: ../../images/rocket.png
  :width: 18

.. |star| image:: ../../images/star.png
  :width: 18

Here, we give the full list of publicly pre-trained models supported by the Hailo Model Zoo.

* Network available in `Hailo Benchmark <https://hailo.ai/developer-zone/benchmarks/>`_ are marked with |rocket|
* Networks available in `TAPPAS <https://hailo.ai/developer-zone/tappas-apps-toolkit/>`_ are marked with |star|
* Benchmark, TAPPAS and Recommended networks run in performance mode
* All models were compiled using Hailo Dataflow Compiler v3.27.0


.. _Instance Segmentation:

Instance Segmentation
---------------------

COCO
^^^^

.. list-table::
   :widths: 31 9 7 11 9 8 8 8 7 7 7 7
   :header-rows: 1

   * - Network Name
     - mAP
     - Quantized
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
     - NV12 Compiled
   * - yolact_regnetx_1.6gf
     - 27.57
     - 27.27
     - 34
     - 43
     - 512x512x3
     - 30.09
     - 125.34
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolact_regnetx_1.6gf/pretrained/2022-11-30/yolact_regnetx_1.6gf.zip>`_
     - `link <https://github.com/dbolya/yolact>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.11.0/hailo15m/yolact_regnetx_1.6gf.hef>`_
     - None
   * - yolact_regnetx_800mf
     - 25.61
     - 25.5
     - 40
     - 50
     - 512x512x3
     - 28.3
     - 116.75
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolact_regnetx_800mf/pretrained/2022-11-30/yolact_regnetx_800mf.zip>`_
     - `link <https://github.com/dbolya/yolact>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.11.0/hailo15m/yolact_regnetx_800mf.hef>`_
     - None
   * - yolov5l_seg
     - 39.78
     - 39.09
     - 20
     - 24
     - 640x640x3
     - 47.89
     - 147.88
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5l/pretrained/2022-10-30/yolov5l-seg.zip>`_
     - `link <https://github.com/ultralytics/yolov5>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.11.0/hailo15m/yolov5l_seg.hef>`_
     - None
   * - yolov5m_seg
     - 37.05
     - 36.32
     - 43
     - 56
     - 640x640x3
     - 32.60
     - 70.94
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5m/pretrained/2022-10-30/yolov5m-seg.zip>`_
     - `link <https://github.com/ultralytics/yolov5>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.11.0/hailo15m/yolov5m_seg.hef>`_
     - None
   * - yolov5n_seg  |star|
     - 23.35
     - 22.75
     - 148
     - 163
     - 640x640x3
     - 1.99
     - 7.1
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5n/pretrained/2022-10-30/yolov5n-seg.zip>`_
     - `link <https://github.com/ultralytics/yolov5>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.11.0/hailo15m/yolov5n_seg.hef>`_
     - None
   * - yolov5s_seg
     - 31.57
     - 30.8
     - 82
     - 114
     - 640x640x3
     - 7.61
     - 26.42
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5s/pretrained/2022-10-30/yolov5s-seg.zip>`_
     - `link <https://github.com/ultralytics/yolov5>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.11.0/hailo15m/yolov5s_seg.hef>`_
     - None
   * - yolov8m_seg
     - 40.6
     - 39.85
     - 30
     - 41
     - 640x640x3
     - 27.3
     - 110.2
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov8/yolov8m/pretrained/2023-03-06/yolov8m-seg.zip>`_
     - `link <https://github.com/ultralytics/ultralytics>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.11.0/hailo15m/yolov8m_seg.hef>`_
     - None
   * - yolov8n_seg
     - 30.32
     - 29.68
     - 143
     - 242
     - 640x640x3
     - 3.4
     - 12.04
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov8/yolov8n/pretrained/2023-03-06/yolov8n-seg.zip>`_
     - `link <https://github.com/ultralytics/ultralytics>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.11.0/hailo15m/yolov8n_seg.hef>`_
     - None
   * - yolov8s_seg
     - 36.63
     - 36.13
     - 66
     - 94
     - 640x640x3
     - 11.8
     - 42.6
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov8/yolov8s/pretrained/2023-03-06/yolov8s-seg.zip>`_
     - `link <https://github.com/ultralytics/ultralytics>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.11.0/hailo15m/yolov8s_seg.hef>`_
     - None
