
Hailo15M Depth Estimation
=========================

.. |rocket| image:: ../../images/rocket.png
  :width: 18

.. |star| image:: ../../images/star.png
  :width: 18

Here, we give the full list of publicly pre-trained models supported by the Hailo Model Zoo.

* Benchmark Networks are marked with |rocket|
* Networks available in `TAPPAS <https://hailo.ai/developer-zone/tappas-apps-toolkit/>`_ are marked with |star|
* Benchmark and TAPPAS networks run in performance mode
* All models were compiled using Hailo Dataflow Compiler v3.27.0


.. _Depth Estimation:

Depth Estimation
----------------

NYU
^^^

.. list-table::
   :widths: 31 9 7 11 9 8 8 8 7 7 7 7
   :header-rows: 1

   * - Network Name
     - RMSE
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
   * - fast_depth  |star|
     - 0.6
     - 0.62
     - 324
     - 717
     - 224x224x3
     - 1.35
     - 0.74
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/DepthEstimation/indoor/fast_depth/pretrained/2021-10-18/fast_depth.zip>`_
     - `link <https://github.com/dwofk/fast-depth>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.11.0/hailo15m/fast_depth.hef>`_
     - None
   * - scdepthv3
     - 0.48
     - 0.51
     - 153
     - 274
     - 256x320x3
     - 14.8
     - 10.7
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/DepthEstimation/indoor/scdepthv3/pretrained/2023-07-20/scdepthv3.zip>`_
     - `link <https://github.com/JiawangBian/sc_depth_pl/>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.11.0/hailo15m/scdepthv3.hef>`_
     - None
