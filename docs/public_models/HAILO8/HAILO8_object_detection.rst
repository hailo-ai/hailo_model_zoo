
Public Pre-Trained Models
=========================

.. |rocket| image:: ../../images/rocket.png
  :width: 18

.. |star| image:: ../../images/star.png
  :width: 18

Here, we give the full list of publicly pre-trained models supported by the Hailo Model Zoo.

* Network available in `Hailo Benchmark <https://hailo.ai/products/ai-accelerators/hailo-8-ai-accelerator/#hailo8-benchmarks/>`_ are marked with |rocket|
* Networks available in `TAPPAS <https://github.com/hailo-ai/tappas>`_ are marked with |star|
* Benchmark and TAPPAS  networks run in performance mode
* All models were compiled using Hailo Dataflow Compiler v5.1.0

Link Legend

The following shortcuts are used in the table below to indicate available resources for each model:

* S – Source: Link to the model’s open-source code repository.
* PT – Pretrained: Download the pretrained model file (compressed in ZIP format).
* H, NV, X – Compiled Models: Links to the compiled model in various formats:
            * H: regular HEF with RGBX format
            * NV: HEF with NV12 format
            * X: HEF with RGBX format

* PR – Profiler Report: Download the model’s performance profiling report.



.. _Object Detection:

----------------

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
   * - centernet_resnet_v1_18_postprocess
     - 26.37
     - 24.95
     - 405
     - 405
     - `S <https://cv.gluon.ai/model_zoo/detection.html>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/centernet/centernet_resnet_v1_18/pretrained/2023-07-18/centernet_resnet_v1_18.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/centernet_resnet_v1_18_postprocess.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/centernet_resnet_v1_18_postprocess_profiler_results_compiled.html>`_
     - 512x512x3
     - 14.22
     - 31.21
   * - centernet_resnet_v1_50_postprocess
     - 31.77
     - 29.32
     - 87
     - 178
     - `S <https://cv.gluon.ai/model_zoo/detection.html>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/centernet/centernet_resnet_v1_50_postprocess/pretrained/2023-07-18/centernet_resnet_v1_50_postprocess.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/centernet_resnet_v1_50_postprocess.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/centernet_resnet_v1_50_postprocess_profiler_results_compiled.html>`_
     - 512x512x3
     - 30.07
     - 56.92
   * - damoyolo_tinynasL20_T
     - 42.8
     - 42.27
     - 564
     - 564
     - `S <https://github.com/tinyvision/DAMO-YOLO>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/damoyolo_tinynasL20_T/pretrained/2022-12-19/damoyolo_tinynasL20_T.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/damoyolo_tinynasL20_T.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/damoyolo_tinynasL20_T_profiler_results_compiled.html>`_
     - 640x640x3
     - 11.35
     - 18.02
   * - damoyolo_tinynasL25_S
     - 46.53
     - 45.28
     - 228
     - 228
     - `S <https://github.com/tinyvision/DAMO-YOLO>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/damoyolo_tinynasL25_S/pretrained/2022-12-19/damoyolo_tinynasL25_S.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/damoyolo_tinynasL25_S.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/damoyolo_tinynasL25_S_profiler_results_compiled.html>`_
     - 640x640x3
     - 16.25
     - 37.64
   * - damoyolo_tinynasL35_M
     - 49.7
     - 47.66
     - 63
     - 130
     - `S <https://github.com/tinyvision/DAMO-YOLO>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/damoyolo_tinynasL35_M/pretrained/2022-12-19/damoyolo_tinynasL35_M.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/damoyolo_tinynasL35_M.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/damoyolo_tinynasL35_M_profiler_results_compiled.html>`_
     - 640x640x3
     - 33.98
     - 61.64
   * - detr_resnet_v1_18_bn
     - 33.91
     - 31.67
     - 30
     - 75
     - `S <https://github.com/facebookresearch/detr>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/detr/detr_resnet_v1_18/2022-09-18/detr_resnet_v1_18_bn.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/detr_resnet_v1_18_bn.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/detr_resnet_v1_18_bn_profiler_results_compiled.html>`_
     - 800x800x3
     - 32.42
     - 61.87
   * - detr_resnet_v1_50
     - 38.38
     - 35.09
     - 12
     - 28
     - `S <https://github.com/facebookresearch/detr>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/detr/detr_resnet_v1_50/2024-03-05/detr_resnet_v1_50.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/detr_resnet_v1_50.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/detr_resnet_v1_50_profiler_results_compiled.html>`_
     - 800x800x3
     - 41.1
     - 120.4
   * - efficientdet_lite0
     - 27.32
     - 26.56
     - 108
     - 299
     - `S <https://github.com/google/automl/tree/master/efficientdet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/efficientdet/efficientdet_lite0/pretrained/2023-04-25/efficientdet-lite0.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/efficientdet_lite0.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/efficientdet_lite0_profiler_results_compiled.html>`_
     - 320x320x3
     - 3.56
     - 1.94
   * - efficientdet_lite1
     - 32.27
     - 31.76
     - 76
     - 223
     - `S <https://github.com/google/automl/tree/master/efficientdet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/efficientdet/efficientdet_lite1/pretrained/2023-04-25/efficientdet-lite1.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/efficientdet_lite1.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/efficientdet_lite1_profiler_results_compiled.html>`_
     - 384x384x3
     - 4.73
     - 4
   * - efficientdet_lite2
     - 35.95
     - 34.72
     - 50
     - 141
     - `S <https://github.com/google/automl/tree/master/efficientdet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/efficientdet/efficientdet_lite2/pretrained/2023-04-25/efficientdet-lite2.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/efficientdet_lite2.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/efficientdet_lite2_profiler_results_compiled.html>`_
     - 448x448x3
     - 5.93
     - 6.84
   * - nanodet_repvgg  |star|
     - 29.3
     - 28.54
     - 1204
     - 1204
     - `S <https://github.com/RangiLyu/nanodet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/nanodet/nanodet_repvgg/pretrained/2024-11-01/nanodet.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/nanodet_repvgg.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/nanodet_repvgg_profiler_results_compiled.html>`_
     - 416x416x3
     - 6.74
     - 11.28
   * - nanodet_repvgg_a12
     - 33.73
     - 31.16
     - 439
     - 439
     - `S <https://github.com/Megvii-BaseDetection/YOLOX>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/nanodet/nanodet_repvgg_a12/pretrained/2024-01-31/nanodet_repvgg_a12_640x640.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/nanodet_repvgg_a12.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/nanodet_repvgg_a12_profiler_results_compiled.html>`_
     - 640x640x3
     - 5.13
     - 28.23
   * - nanodet_repvgg_a1_640
     - 33.28
     - 32.86
     - 411
     - 411
     - `S <https://github.com/RangiLyu/nanodet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/nanodet/nanodet_repvgg_a1_640/pretrained/2024-01-25/nanodet_repvgg_a1_640.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/nanodet_repvgg_a1_640.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/nanodet_repvgg_a1_640_profiler_results_compiled.html>`_
     - 640x640x3
     - 10.79
     - 42.8
   * - ssd_mobilenet_v1  |star|
     - 23.19
     - 22.05
     - 1015
     - 1015
     - `S <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/ssd/ssd_mobilenet_v1/pretrained/2023-07-18/ssd_mobilenet_v1.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/ssd_mobilenet_v1.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/ssd_mobilenet_v1_profiler_results_compiled.html>`_
     - 300x300x3
     - 6.79
     - 2.5
   * - ssd_mobilenet_v2
     - 24.18
     - 23.06
     - 784
     - 784
     - `S <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/ssd/ssd_mobilenet_v2/pretrained/2025-01-15/ssd_mobilenet_v2.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/ssd_mobilenet_v2.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/ssd_mobilenet_v2_profiler_results_compiled.html>`_
     - 300x300x3
     - 4.46
     - 1.52
   * - tiny_yolov3
     - 14.66
     - 13.69
     - 1044
     - 1044
     - `S <https://github.com/Tianxiaomo/pytorch-YOLOv4>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/tiny_yolov3/pretrained/2021-07-11/tiny_yolov3.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/tiny_yolov3.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/tiny_yolov3_profiler_results_compiled.html>`_
     - 416x416x3
     - 8.85
     - 5.58
   * - tiny_yolov4
     - 19.18
     - 17.74
     - 1472
     - 1472
     - `S <https://github.com/Tianxiaomo/pytorch-YOLOv4>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/tiny_yolov4/pretrained/2023-07-18/tiny_yolov4.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/tiny_yolov4.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/tiny_yolov4_profiler_results_compiled.html>`_
     - 416x416x3
     - 6.05
     - 6.92
   * - yolov10b
     - 52.0
     - 50.54
     - 43
     - 98
     - `S <https://github.com/THU-MIG/yolov10>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov10b/pretrained/2024-07-02/yolov10b.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov10b.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov10b_profiler_results_compiled.html>`_
     - 640x640x3
     - 20.15
     - 92.09
   * - yolov10n
     - 38.5
     - 36.77
     - 194
     - 566
     - `S <https://github.com/THU-MIG/yolov10>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov10n/pretrained/2024-05-31/yolov10n.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov10n.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov10n_profiler_results_compiled.html>`_
     - 640x640x3
     - 2.3
     - 6.8
   * - yolov10s
     - 45.86
     - 44.87
     - 117
     - 283
     - `S <https://github.com/THU-MIG/yolov10>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov10s/pretrained/2024-05-31/yolov10s.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov10s.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov10s_profiler_results_compiled.html>`_
     - 640x640x3
     - 7.2
     - 21.7
   * - yolov10x
     - 53.7
     - 50.86
     - 18
     - 34
     - `S <https://github.com/THU-MIG/yolov10>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov10x/pretrained/2024-07-02/yolov10x.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov10x.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov10x_profiler_results_compiled.html>`_
     - 640x640x3
     - 31.72
     - 160.56
   * - yolov11l
     - 52.8
     - 51.62
     - 37
     - 82
     - `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov11l/2024-10-02/yolo11l.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov11l.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov11l_profiler_results_compiled.html>`_
     - 640x640x3
     - 25.3
     - 87.17
   * - yolov11m |rocket|
     - 51.1
     - 50.07
     - 50
     - 102
     - `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov11m/2024-10-02/yolo11m.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov11m.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov11m_profiler_results_compiled.html>`_
     - 640x640x3
     - 20.1
     - 68.1
   * - yolov11n
     - 39.0
     - 37.68
     - 185
     - 538
     - `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov11n/2024-10-02/yolo11n.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov11n.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov11n_profiler_results_compiled.html>`_
     - 640x640x3
     - 2.6
     - 6.55
   * - yolov11s
     - 46.3
     - 45.11
     - 111
     - 303
     - `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov11s/2024-10-02/yolo11s.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov11s.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov11s_profiler_results_compiled.html>`_
     - 640x640x3
     - 9.4
     - 21.6
   * - yolov11x
     - 54.1
     - 52.98
     - 16
     - 29
     - `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov11x/2024-10-02/yolo11x.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov11x.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov11x_profiler_results_compiled.html>`_
     - 640x640x3
     - 56.9
     - 195.29
   * - yolov3
     - 38.42
     - 38.18
     - 37
     - 69
     - `S <https://github.com/AlexeyAB/darknet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov3/pretrained/2021-08-16/yolov3.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov3.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov3_profiler_results_compiled.html>`_
     - 608x608x3
     - 68.79
     - 158.10
   * - yolov3_416
     - 37.73
     - 37.28
     - 47
     - 97
     - `S <https://github.com/AlexeyAB/darknet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov3_416/pretrained/2021-08-16/yolov3_416.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov3_416.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov3_416_profiler_results_compiled.html>`_
     - 416x416x3
     - 61.92
     - 65.94
   * - yolov3_gluon
     - 37.28
     - 35.76
     - 37
     - 65
     - `S <https://cv.gluon.ai/model_zoo/detection.html>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov3_gluon/pretrained/2023-07-18/yolov3_gluon.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov3_gluon.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov3_gluon_profiler_results_compiled.html>`_
     - 608x608x3
     - 68.79
     - 140.7
   * - yolov3_gluon_416
     - 36.27
     - 33.84
     - 49
     - 98
     - `S <https://cv.gluon.ai/model_zoo/detection.html>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov3_gluon_416/pretrained/2023-07-18/yolov3_gluon_416.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov3_gluon_416.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov3_gluon_416_profiler_results_compiled.html>`_
     - 416x416x3
     - 61.92
     - 65.94
   * - yolov4_leaky
     - 42.37
     - 41.06
     - 43
     - 100
     - `S <https://github.com/AlexeyAB/darknet/wiki/YOLOv4-model-zoo>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov4/pretrained/2022-03-17/yolov4.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov4_leaky.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov4_leaky_profiler_results_compiled.html>`_
     - 512x512x3
     - 64.33
     - 91.04
   * - yolov5m
     - 42.59
     - 41.25
     - 156
     - 156
     - `S <https://github.com/ultralytics/yolov5/releases/tag/v2.0>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5m_spp/pretrained/2023-04-25/yolov5m.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov5m.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov5m_profiler_results_compiled.html>`_
     - 640x640x3
     - 21.78
     - 52.17
   * - yolov5m6_6.1
     - 50.68
     - 49.41
     - 33
     - 48
     - `S <https://github.com/ultralytics/yolov5/releases/tag/v6.1>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5m6_6.1/pretrained/2023-04-25/yolov5m6.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov5m6_6.1.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov5m6_6.1_profiler_results_compiled.html>`_
     - 1280x1280x3
     - 35.70
     - 200.04
   * - yolov5m_6.1
     - 44.74
     - 43.16
     - 78
     - 143
     - `S <https://github.com/ultralytics/yolov5/releases/tag/v6.1>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5m_6.1/pretrained/2023-04-25/yolov5m_6.1.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov5m_6.1.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov5m_6.1_profiler_results_compiled.html>`_
     - 640x640x3
     - 21.17
     - 48.96
   * - yolov5m_wo_spp |rocket| |star|
     - 43.06
     - 41.62
     - 242
     - 242
     - `S <https://github.com/ultralytics/yolov5/releases/tag/v2.0>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5m/pretrained/2023-04-25/yolov5m_wo_spp.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov5m_wo_spp.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov5m_wo_spp_profiler_results_compiled.html>`_
     - 640x640x3
     - 22.67
     - 52.88
   * - yolov5s
     - 35.33
     - 34.08
     - 543
     - 543
     - `S <https://github.com/ultralytics/yolov5/releases/tag/v2.0>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5s_spp/pretrained/2023-04-25/yolov5s.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov5s.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov5s_profiler_results_compiled.html>`_
     - 640x640x3
     - 7.46
     - 17.44
   * - yolov5s_c3tr
     - 37.13
     - 35.79
     - 456
     - 456
     - `S <https://github.com/ultralytics/yolov5/tree/v6.0>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5s_c3tr/pretrained/2023-04-25/yolov5s_c3tr.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov5s_c3tr.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov5s_c3tr_profiler_results_compiled.html>`_
     - 640x640x3
     - 10.29
     - 17.02
   * - yolov5s_wo_spp
     - 34.79
     - 33.63
     - 633
     - 633
     - `S <https://github.com/ultralytics/yolov5/releases/tag/v2.0>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5s/pretrained/2023-04-25/yolov5s.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov5s_wo_spp.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov5s_wo_spp_profiler_results_compiled.html>`_
     - 640x640x3
     - 7.85
     - 17.74
   * - yolov5xs_wo_spp
     - 33.18
     - 32.22
     - 1139
     - 1139
     - `S <https://github.com/ultralytics/yolov5/releases/tag/v2.0>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5xs/pretrained/2023-04-25/yolov5xs.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov5xs_wo_spp.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov5xs_wo_spp_profiler_results_compiled.html>`_
     - 512x512x3
     - 7.85
     - 11.36
   * - yolov6n
     - 34.29
     - 32.54
     - 1250
     - 1250
     - `S <https://github.com/meituan/YOLOv6/releases/tag/0.1.0>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov6n/pretrained/2023-05-31/yolov6n.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov6n.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov6n_profiler_results_compiled.html>`_
     - 640x640x3
     - 4.32
     - 11.12
   * - yolov6n_0.2.1
     - 35.16
     - 33.92
     - 1256
     - 1256
     - `S <https://github.com/meituan/YOLOv6/releases/tag/0.2.1>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov6n_0.2.1/pretrained/2023-04-17/yolov6n_0.2.1.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov6n_0.2.1.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov6n_0.2.1_profiler_results_compiled.html>`_
     - 640x640x3
     - 4.33
     - 11.06
   * - yolov6n_0.2.1_nms_core
     - 35.16
     - 33.97
     - 237
     - 237
     - `S <https://github.com/meituan/YOLOv6/releases/tag/0.2.1>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov6n_0.2.1/pretrained/2023-04-17/yolov6n_0.2.1.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov6n_0.2.1_nms_core.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov6n_0.2.1_nms_core_profiler_results_compiled.html>`_
     - 640x640x3
     - 4.32
     - 11.12
   * - yolov7
     - 50.6
     - 47.95
     - 54
     - 99
     - `S <https://github.com/WongKinYiu/yolov7>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov7/pretrained/2023-04-25/yolov7.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov7.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov7_profiler_results_compiled.html>`_
     - 640x640x3
     - 36.91
     - 104.51
   * - yolov7_tiny
     - 37.07
     - 36.14
     - 633
     - 633
     - `S <https://github.com/WongKinYiu/yolov7>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov7_tiny/pretrained/2023-04-25/yolov7_tiny.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov7_tiny.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov7_tiny_profiler_results_compiled.html>`_
     - 640x640x3
     - 6.22
     - 13.74
   * - yolov7e6
     - 55.37
     - 53.14
     - 11
     - 15
     - `S <https://github.com/WongKinYiu/yolov7>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov7e6/pretrained/2023-04-25/yolov7-e6.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov7e6.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov7e6_profiler_results_compiled.html>`_
     - 1280x1280x3
     - 97.20
     - 515.12
   * - yolov8l
     - 52.44
     - 51.25
     - 36
     - 64
     - `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov8l/2023-02-02/yolov8l.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8l.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8l_profiler_results_compiled.html>`_
     - 640x640x3
     - 43.7
     - 165.3
   * - yolov8m |rocket| |star|
     - 49.91
     - 49.26
     - 67
     - 141
     - `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov8m/2023-02-02/yolov8m.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8m.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8m_profiler_results_compiled.html>`_
     - 640x640x3
     - 25.9
     - 78.93
   * - yolov8n
     - 37.02
     - 36.2
     - 1036
     - 1036
     - `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov8n/2023-01-30/yolov8n.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8n.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8n_profiler_results_compiled.html>`_
     - 640x640x3
     - 3.2
     - 8.74
   * - yolov8s
     - 44.58
     - 44.01
     - 491
     - 491
     - `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov8s/2023-02-02/yolov8s.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8s.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8s_profiler_results_compiled.html>`_
     - 640x640x3
     - 11.2
     - 28.6
   * - yolov8x
     - 53.45
     - 52.75
     - 18
     - 29
     - `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov8x/2023-02-02/yolov8x.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8x.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8x_profiler_results_compiled.html>`_
     - 640x640x3
     - 68.2
     - 258
   * - yolov9c
     - 52.6
     - 50.86
     - 44
     - 88
     - `S <https://github.com/WongKinYiu/yolov9>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov9c/pretrained/2024-02-24/yolov9c.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov9c.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov9c_profiler_results_compiled.html>`_
     - 640x640x3
     - 25.3
     - 102.1
   * - yolox_l_leaky  |star|
     - 48.68
     - 46.2
     - 41
     - 74
     - `S <https://github.com/Megvii-BaseDetection/YOLOX>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolox_l_leaky/pretrained/2023-05-31/yolox_l_leaky.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolox_l_leaky.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolox_l_leaky_profiler_results_compiled.html>`_
     - 640x640x3
     - 54.17
     - 155.3
   * - yolox_s_leaky
     - 38.13
     - 37.05
     - 385
     - 385
     - `S <https://github.com/Megvii-BaseDetection/YOLOX>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolox_s_leaky/pretrained/2023-05-31/yolox_s_leaky.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolox_s_leaky.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolox_s_leaky_profiler_results_compiled.html>`_
     - 640x640x3
     - 8.96
     - 26.74
   * - yolox_s_wide_leaky
     - 42.4
     - 40.64
     - 76
     - 131
     - `S <https://github.com/Megvii-BaseDetection/YOLOX>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolox_s_wide_leaky/pretrained/2023-05-31/yolox_s_wide_leaky.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolox_s_wide_leaky.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolox_s_wide_leaky_profiler_results_compiled.html>`_
     - 640x640x3
     - 20.12
     - 59.46
   * - yolox_tiny
     - 32.64
     - 31.39
     - 742
     - 742
     - `S <https://github.com/Megvii-BaseDetection/YOLOX>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolox/yolox_tiny/pretrained/2023-05-31/yolox_tiny.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolox_tiny.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolox_tiny_profiler_results_compiled.html>`_
     - 416x416x3
     - 5.05
     - 6.44
