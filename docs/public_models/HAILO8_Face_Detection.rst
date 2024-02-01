
Public Pre-Trained Models
=========================

.. |rocket| image:: ../images/rocket.png
  :width: 18

.. |star| image:: ../images/star.png
  :width: 18

.. _Face Detection:

Face Detection
--------------

WiderFace
^^^^^^^^^

.. list-table::
   :widths: 24 7 12 11 9 8 8 8 7 7 7
   :header-rows: 1

   * - Network Name
     - mAP
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
   * - lightface_slim  |star|
     - 39.7
     - 39.22
     - 240x320x3
     - 0.26
     - 0.16
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/lightface_slim/2021-07-18/lightface_slim.zip>`_
     - `link <https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo8/lightface_slim.hef>`_
     - 4206.18
     - 4206.06
   * - retinaface_mobilenet_v1  |star|
     - 81.27
     - 81.17
     - 736x1280x3
     - 3.49
     - 25.14
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/retinaface_mobilenet_v1_hd/2023-07-18/retinaface_mobilenet_v1_hd.zip>`_
     - `link <https://github.com/biubug6/Pytorch_Retinaface>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo8/retinaface_mobilenet_v1.hef>`_
     - 104.626
     - 104.624
   * - scrfd_10g
     - 82.13
     - 82.03
     - 640x640x3
     - 4.23
     - 26.74
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/scrfd/scrfd_10g/pretrained/2022-09-07/scrfd_10g.zip>`_
     - `link <https://github.com/deepinsight/insightface>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo8/scrfd_10g.hef>`_
     - 292.285
     - 292.278
   * - scrfd_2.5g
     - 76.59
     - 76.32
     - 640x640x3
     - 0.82
     - 6.88
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/scrfd/scrfd_2.5g/pretrained/2022-09-07/scrfd_2.5g.zip>`_
     - `link <https://github.com/deepinsight/insightface>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo8/scrfd_2.5g.hef>`_
     - 694.727
     - 694.64
   * - scrfd_500m
     - 68.98
     - 68.88
     - 640x640x3
     - 0.63
     - 1.5
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/scrfd/scrfd_500m/pretrained/2022-09-07/scrfd_500m.zip>`_
     - `link <https://github.com/deepinsight/insightface>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo8/scrfd_500m.hef>`_
     - 831.076
     - 830.989
