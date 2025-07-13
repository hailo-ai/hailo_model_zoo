
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



.. _Low Light Enhancement:

---------------------

LOL
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
   * - zero_dce   
     - 16.23
     - 16.22
     - 200
     - 200
     - `S <Internal>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/LowLightEnhancement/LOL/zero_dce/pretrained/2023-04-23/zero_dce.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/zero_dce.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/zero_dce_profiler_results_compiled.html>`_
     - 400x600x3
     - 0.21
     - 38.2    
   * - zero_dce_pp   
     - 15.95
     - 15.9
     - 96
     - 96
     - `S <Internal>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/LowLightEnhancement/LOL/zero_dce_pp/pretrained/2023-07-03/zero_dce_pp.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/zero_dce_pp.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/zero_dce_pp_profiler_results_compiled.html>`_
     - 400x600x3
     - 0.02
     - 4.84
