
Public Pre-Trained Models
=========================

.. |rocket| image:: ../../images/rocket.png
  :width: 18

.. |star| image:: ../../images/star.png
  :width: 18

Here, we give the full list of publicly pre-trained models supported by the Hailo Model Zoo.

* Network available in `Hailo Benchmark <https://hailo.ai/developer-zone/benchmarks/>`_ are marked with |rocket|
* Networks available in `TAPPAS <https://hailo.ai/developer-zone/tappas-apps-toolkit/>`_ are marked with |star|
* Benchmark, TAPPAS and Recommended networks run in performance mode
* All models were compiled using Hailo Dataflow Compiler v3.27.0
* Supported tasks:

  * `Low Light Enhancement`_


.. _Low Light Enhancement:

Low Light Enhancement
---------------------

LOL
^^^

.. list-table::
   :widths: 31 9 7 11 9 8 8 8 7 7 7 7
   :header-rows: 1

   * - Network Name
     - PSNR
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
   * - zero_dce
     - 16.23
     - 16.24
     - 71
     - 79
     - 400x600x3
     - 0.21
     - 38.2
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/LowLightEnhancement/LOL/zero_dce/pretrained/2023-04-23/zero_dce.zip>`_
     - `link <Internal>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.11.0/hailo15m/zero_dce.hef>`_
     - None
   * - zero_dce_pp
     - 15.95
     - 15.82
     - 54
     - 59
     - 400x600x3
     - 0.02
     - 4.84
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/LowLightEnhancement/LOL/zero_dce_pp/pretrained/2023-07-03/zero_dce_pp.zip>`_
     - `link <Internal>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.11.0/hailo15m/zero_dce_pp.hef>`_
     - None
