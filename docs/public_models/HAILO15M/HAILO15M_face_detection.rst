
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



.. _Face Detection:

--------------

WiderFace
^^^^^^^^^

.. list-table::
   :widths: 31 9 7 11 9 8 8 8 7 7 7
   :header-rows: 1

   * - Network Name
     - mAP
     - HW Accuracy
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled     
   * - lightface_slim  |star| 
     - 39.7
     - 39.2
     - 710
     - 1656
     - 240x320x3
     - 0.26
     - 0.16
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/lightface_slim/2021-07-18/lightface_slim.zip>`_
     - `link <https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo15m/lightface_slim.hef>`_/`nv12 <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo15m/lightface_slim_nv12.hef>`_    
   * - retinaface_mobilenet_v1   
     - 81.27
     - 81.23
     - 49
     - 63
     - 736x1280x3
     - 3.49
     - 25.14
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/retinaface_mobilenet_v1_hd/2023-07-18/retinaface_mobilenet_v1_hd.zip>`_
     - `link <https://github.com/biubug6/Pytorch_Retinaface>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo15m/retinaface_mobilenet_v1.hef>`_    
   * - scrfd_10g   
     - 82.13
     - 82.07
     - 91
     - 131
     - 640x640x3
     - 4.23
     - 26.74
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/scrfd/scrfd_10g/pretrained/2022-09-07/scrfd_10g.zip>`_
     - `link <https://github.com/deepinsight/insightface>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo15m/scrfd_10g.hef>`_    
   * - scrfd_2.5g   
     - 76.59
     - 76.38
     - 212
     - 346
     - 640x640x3
     - 0.82
     - 6.88
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/scrfd/scrfd_2.5g/pretrained/2022-09-07/scrfd_2.5g.zip>`_
     - `link <https://github.com/deepinsight/insightface>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo15m/scrfd_2.5g.hef>`_    
   * - scrfd_500m   
     - 68.98
     - 68.84
     - 238
     - 392
     - 640x640x3
     - 0.63
     - 1.5
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/scrfd/scrfd_500m/pretrained/2022-09-07/scrfd_500m.zip>`_
     - `link <https://github.com/deepinsight/insightface>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo15m/scrfd_500m.hef>`_
