
Public Pre-Trained Models
=========================

.. |rocket| image:: ../../images/rocket.png
  :width: 18

.. |star| image:: ../../images/star.png
  :width: 18

Here, we give the full list of publicly pre-trained models supported by the Hailo Model Zoo.

* Network available in `Hailo Benchmark <https://hailo.ai/products/ai-accelerators/hailo-8l-ai-accelerator-for-ai-light-applications/#hailo8l-benchmarks/>`_ are marked with |rocket|
* Networks available in `TAPPAS <https://github.com/hailo-ai/tappas>`_ are marked with |star|
* Benchmark and TAPPAS  networks run in performance mode
* All models were compiled using Hailo Dataflow Compiler v3.28.0
* Supported tasks:

  * `Person Re-ID`_


.. _Person Re-ID:

Person Re-ID
------------

Market1501
^^^^^^^^^^

.. list-table::
   :widths: 31 9 7 11 9 8 8 8 7 7 7 7
   :header-rows: 1

   * - Network Name
     - rank1
     - Quantized
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
   * - osnet_x1_0
     - 94.43
     - 93.73
     - 106
     - 316
     - 256x128x3
     - 2.19
     - 1.98
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/PersonReID/osnet_x1_0/2022-05-19/osnet_x1_0.zip>`_
     - `link <https://github.com/KaiyangZhou/deep-person-reid>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo8l/osnet_x1_0.hef>`_
   * - repvgg_a0_person_reid_512  |star|
     - 89.9
     - 89.5
     - 3526
     - 3526
     - 256x128x3
     - 7.68
     - 1.78
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/MCPReID/reid/repvgg_a0_person_reid_512/2022-04-18/repvgg_a0_person_reid_512.zip>`_
     - `link <https://github.com/DingXiaoH/RepVGG>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo8l/repvgg_a0_person_reid_512.hef>`_
