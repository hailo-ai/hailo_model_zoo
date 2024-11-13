
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



.. _Face Attribute:

--------------

CELEBA
^^^^^^

.. list-table::
   :widths: 31 9 7 11 9 8 8 8 7 7
   :header-rows: 1

   * - Network Name
     - Mean Accuracy
     - HW Accuracy
     - FPS
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled    
   * - face_attr_resnet_v1_18   
     - 81.19
     - 80.8
     - 0
     - 218x178x3
     - 11.74
     - 3
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceAttr/face_attr_resnet_v1_18/2022-06-09/face_attr_resnet_v1_18.zip>`_
     - `link <https://github.com/d-li14/face-attribute-prediction>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo15h/face_attr_resnet_v1_18.hef>`_
