
Public Pre-Trained Models
=========================

.. |rocket| image:: images/rocket.png
  :width: 18

.. |star| image:: images/star.png
  :width: 18
.. _Face Attribute:

Face Attribute
--------------

CELEBA
^^^^^^

.. list-table::
   :widths: 30 7 11 14 9 8 12 8 7 7 7
   :header-rows: 1

   * - Network Name
     - Mean Accuracy
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
   * - face_attr_resnet_v1_18
     - 81.19
     - 81.09
     - 218x178x3
     - 11.74
     - 3
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceAttr/face_attr_resnet_v1_18/2022-06-09/face_attr_resnet_v1_18.zip>`_
     - `link <https://github.com/d-li14/face-attribute-prediction>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo8/face_attr_resnet_v1_18.hef>`_
     - 2928.63
     - 2929.11
