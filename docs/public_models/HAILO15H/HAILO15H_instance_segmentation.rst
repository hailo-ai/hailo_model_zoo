
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
* All models were compiled using Hailo Dataflow Compiler v3.29.0



.. _Instance Segmentation:

---------------------

COCO
^^^^

.. list-table::
   :widths: 31 9 7 11 9 8 8 8 7 7 7
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
   * - yolact_regnetx_1.6gf   
     - 27.57
     - 0.24
     - 49
     - 72
     - 512x512x3
     - 30.09
     - 125.34
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolact_regnetx_1.6gf/pretrained/2022-11-30/yolact_regnetx_1.6gf.zip>`_
     - `link <https://github.com/dbolya/yolact>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo15h/yolact_regnetx_1.6gf.hef>`_    
   * - yolact_regnetx_800mf   
     - 25.61
     - 0.18
     - 60
     - 84
     - 512x512x3
     - 28.3
     - 116.75
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolact_regnetx_800mf/pretrained/2022-11-30/yolact_regnetx_800mf.zip>`_
     - `link <https://github.com/dbolya/yolact>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo15h/yolact_regnetx_800mf.hef>`_    
   * - yolov5l_seg   
     - 39.78
     - 0.43
     - 34
     - 47
     - 640x640x3
     - 47.89
     - 147.88
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5l/pretrained/2022-10-30/yolov5l-seg.zip>`_
     - `link <https://github.com/ultralytics/yolov5>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo15h/yolov5l_seg.hef>`_    
   * - yolov5m_seg   
     - 37.05
     - 0.37
     - 62
     - 93
     - 640x640x3
     - 32.60
     - 70.94
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5m/pretrained/2022-10-30/yolov5m-seg.zip>`_
     - `link <https://github.com/ultralytics/yolov5>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo15h/yolov5m_seg.hef>`_     
   * - yolov5n_seg  |star| 
     - 23.35
     - 0.29
     - 175
     - 167
     - 640x640x3
     - 1.99
     - 7.1
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5n/pretrained/2022-10-30/yolov5n-seg.zip>`_
     - `link <https://github.com/ultralytics/yolov5>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo15h/yolov5n_seg.hef>`_    
   * - yolov5s_seg   
     - 31.57
     - 0.78
     - 115
     - 161
     - 640x640x3
     - 7.61
     - 26.42
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5s/pretrained/2022-10-30/yolov5s-seg.zip>`_
     - `link <https://github.com/ultralytics/yolov5>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo15h/yolov5s_seg.hef>`_    
   * - yolov8m_seg   
     - 40.6
     - 0.34
     - 45
     - 68
     - 640x640x3
     - 27.3
     - 110.2
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov8/yolov8m/pretrained/2023-03-06/yolov8m-seg.zip>`_
     - `link <https://github.com/ultralytics/ultralytics>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo15h/yolov8m_seg.hef>`_    
   * - yolov8n_seg   
     - 30.32
     - 0.55
     - 202
     - 317
     - 640x640x3
     - 3.4
     - 12.04
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov8/yolov8n/pretrained/2023-03-06/yolov8n-seg.zip>`_
     - `link <https://github.com/ultralytics/ultralytics>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo15h/yolov8n_seg.hef>`_    
   * - yolov8s_seg   
     - 36.63
     - 0.31
     - 96
     - 157
     - 640x640x3
     - 11.8
     - 42.6
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov8/yolov8s/pretrained/2023-03-06/yolov8s-seg.zip>`_
     - `link <https://github.com/ultralytics/ultralytics>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo15h/yolov8s_seg.hef>`_
