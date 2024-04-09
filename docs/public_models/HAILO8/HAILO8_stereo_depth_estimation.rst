
Hailo8 Stereo Depth Estimation
==============================

.. |rocket| image:: ../../images/rocket.png
  :width: 18

.. |star| image:: ../../images/star.png
  :width: 18

Here, we give the full list of publicly pre-trained models supported by the Hailo Model Zoo.

* Network available in `Hailo Benchmark <https://hailo.ai/products/ai-accelerators/hailo-8-ai-accelerator/#hailo8-benchmarks/>`_ are marked with |rocket|
* Networks available in `TAPPAS <https://hailo.ai/developer-zone/tappas-apps-toolkit/>`_ are marked with |star|
* Benchmark and TAPPAS networks run in performance mode
* All models were compiled using Hailo Dataflow Compiler v3.27.0


.. _Stereo Depth Estimation:

Stereo Depth Estimation
-----------------------

N/A
^^^

.. list-table::
   :widths: 31 9 7 11 9 8 8 8 7 7 7 7
   :header-rows: 1

   * - Network Name
     - EPE
     - Quantized
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
   * - stereonet
     - 91.79
     - 89.14
     - 5
     - 3
     - 368x1232x3, 368x1232x3
     - 5.91
     - 126.28
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/DisparityEstimation/stereonet/pretrained/2023-05-31/stereonet.zip>`_
     - `link <https://github.com/nivosco/StereoNet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.11.0/hailo8/stereonet.hef>`_
