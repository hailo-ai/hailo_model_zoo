
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
* All models were compiled using Hailo Dataflow Compiler v5.0.0

Link Legend

The following shortcuts are used in the table below to indicate available resources for each model:

* S – Source: Link to the model’s open-source code repository.
* PT – Pretrained: Download the pretrained model file (compressed in ZIP format).
* H, NV, X – Compiled Models: Links to the compiled model in various formats:
            * H: regular HEF with RGB format
            * NV: HEF with NV12 format
            * X: HEF with RGBX format

* PR – Profiler Report: Download the model’s performance profiling report.



.. _zero-shot depth estimation:

--------------------------

Zero-Shot Depth Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 31 9 7 11 9 8 8 8 9
   :header-rows: 1

   * - Network Name
     - float
     - Hardware
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
   * - depth_anything_v2_vits
     - 0.15
     - 0.19
     - 54
     - 132
     - `S <https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/DepthEstimation/Depth_Anything/v2/vits/pretrained/2025-07-09/depth_anything_v2_vits_224X224_sim_hf.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/depth_anything_v2_vits.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/depth_anything_v2_vits_profiler_results_compiled.html>`_
     - 224x224x3
     - 24.2
     - 16.7
   * - depth_anything_vits
     - 0.13
     - 0.16
     - 58
     - 147
     - `S <https://huggingface.co/LiheYoung/depth-anything-small-hf>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/DepthEstimation/Depth_Anything/v1/vits/pretrained/2025-07-09/depth_anything_vits_224X224_sim_hf.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/depth_anything_vits.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/depth_anything_vits_profiler_results_compiled.html>`_
     - 224x224x3
     - 24.2
     - 16.7
