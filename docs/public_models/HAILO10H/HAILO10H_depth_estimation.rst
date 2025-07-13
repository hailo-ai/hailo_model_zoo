
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
* All models were compiled using Hailo Dataflow Compiler v3.31.0

Link Legend

The following shortcuts are used in the table below to indicate available resources for each model:

* S – Source: Link to the model’s open-source code repository.
* PT – Pretrained: Download the pretrained model file (compressed in ZIP format).
* H, NV, X – Compiled Models: Links to the compiled model in various formats:
            * H: regular HEF with RGBX format
            * NV: HEF with NV12 format
            * X: HEF with RGBX format

* PR – Profiler Report: Download the model’s performance profiling report.



.. _Depth Estimation:

----------------

NYU
^^^

.. list-table::
   :widths: 31 9 7 11 9 8 8 8 9
   :header-rows: 1

   * - Network Name
     - float mAP
     - Hardware mAP
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)        
   * - fast_depth  |star| 
     - 0.6
     - 0.61
     - 2046
     - 2046
     - `S <https://github.com/dwofk/fast-depth>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/DepthEstimation/indoor/fast_depth/pretrained/2021-10-18/fast_depth.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/fast_depth.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/fast_depth_profiler_results_compiled.html>`_
     - 224x224x3
     - 1.35
     - 0.74      
   * - scdepthv3   
     - 0.48
     - 0.48
     - 736
     - 736
     - `S <https://github.com/JiawangBian/sc_depth_pl/>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/DepthEstimation/indoor/scdepthv3/pretrained/2023-07-20/scdepthv3.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/scdepthv3.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/scdepthv3_profiler_results_compiled.html>`_
     - 256x320x3
     - 14.8
     - 10.7
