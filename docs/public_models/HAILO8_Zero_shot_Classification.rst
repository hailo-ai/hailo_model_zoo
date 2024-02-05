
Public Pre-Trained Models
=========================

.. |rocket| image:: ../images/rocket.png
  :width: 18

.. |star| image:: ../images/star.png
  :width: 18

.. _Zero-shot Classification:

Zero-shot Classification
------------------------

CIFAR100
^^^^^^^^

.. list-table::
   :widths: 30 7 11 14 9 8 12 8 7 7 7
   :header-rows: 1

   * - Network Name
     - Accuracy (top1)
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
   * - clip_resnet_50
     - 42.07
     - 38.57
     - 224x224x3
     - 38.72
     - 11.62
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/clip_resnet_50/pretrained/2023-03-09/clip_resnet_50.zip>`_
     - `link <https://github.com/openai/CLIP>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo8/clip_resnet_50.hef>`_
     - 94.6622
     - 383.319
