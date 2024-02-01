
Public Pre-Trained Models
=========================

.. |rocket| image:: ../images/rocket.png
  :width: 18

.. |star| image:: ../images/star.png
  :width: 18
.. _Person Attribute:

Person Attribute
----------------

PETA
^^^^

.. list-table::
   :widths: 24 14 12 14 9 8 10 8 7 7 7
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
   * - person_attr_resnet_v1_18
     - 82.5
     - 82.61
     - 224x224x3
     - 11.19
     - 3.64
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/person_attr_resnet_v1_18/pretrained/2022-06-11/person_attr_resnet_v1_18.zip>`_
     - `link <https://github.com/dangweili/pedestrian-attribute-recognition-pytorch>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo8/person_attr_resnet_v1_18.hef>`_
     - 2533.86
     - 2533.77
