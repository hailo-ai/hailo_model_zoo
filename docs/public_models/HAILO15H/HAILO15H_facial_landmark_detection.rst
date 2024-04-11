
Hailo15H Facial Landmark Detection
==================================

.. |rocket| image:: ../../images/rocket.png
  :width: 18

.. |star| image:: ../../images/star.png
  :width: 18

Here, we give the full list of publicly pre-trained models supported by the Hailo Model Zoo.

* Benchmark Networks are marked with |rocket|
* Networks available in `TAPPAS <https://github.com/hailo-ai/tappas/>`_ are marked with |star|
* Benchmark and TAPPAS networks run in performance mode
* All models were compiled using Hailo Dataflow Compiler v3.27.0


.. _Facial Landmark Detection:

Facial Landmark Detection
-------------------------

AFLW2k3d
^^^^^^^^

.. list-table::
   :widths: 31 9 7 11 9 8 8 8 7 7 7 7
   :header-rows: 1

   * - Network Name
     - NME
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
   * - tddfa_mobilenet_v1  |star|
     - 3.68
     - 4.05
     - 8936
     - 8941
     - 120x120x3
     - 3.26
     - 0.36
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceLandmarks3d/tddfa/tddfa_mobilenet_v1/pretrained/2021-11-28/tddfa_mobilenet_v1.zip>`_
     - `link <https://github.com/cleardusk/3DDFA_V2>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.11.0/hailo15h/tddfa_mobilenet_v1.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.11.0/hailo15h/tddfa_mobilenet_v1_nv12.hef>`_
