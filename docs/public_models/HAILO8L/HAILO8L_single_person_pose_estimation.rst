
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
* All models were compiled using Hailo Dataflow Compiler v3.33.0

Link Legend

The following shortcuts are used in the table below to indicate available resources for each model:

* S – Source: Link to the model’s open-source code repository.
* PT – Pretrained: Download the pretrained model file (compressed in ZIP format).
* H, NV, X – Compiled Models: Links to the compiled model in various formats:
            * H: regular HEF with RGB format
            * NV: HEF with NV12 format
            * X: HEF with RGBX format

* PR – Profiler Report: Download the model’s performance profiling report.



.. _Single Person Pose Estimation:

-----------------------------

COCO
^^^^

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
   * - mspn_regnetx_800mf  |star|
     - 70.8
     - 0.2
     - 0
     - 795
     - `S <https://github.com/open-mmlab/mmpose>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SinglePersonPoseEstimation/mspn_regnetx_800mf/pretrained/2022-07-12/mspn_regnetx_800mf.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/mspn_regnetx_800mf.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/mspn_regnetx_800mf_profiler_results_compiled.html>`_
     - 256x192x3
     - 7.17
     - 2.94
   * - vit_pose_small
     - 74.2
     - 1.4
     - 1
     - 41
     - `S <https://github.com/ViTAE-Transformer/ViTPose>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SinglePersonPoseEstimation/vit/vit_pose_small/pretrained/2023-11-14/vit_pose_small.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/vit_pose_small.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/vit_pose_small_profiler_results_compiled.html>`_
     - 256x192x3
     - 24.29
     - 17.17
   * - vit_pose_small_bn
     - 72.0
     - 1.0
     - 1
     - 70
     - `S <https://github.com/ViTAE-Transformer/ViTPose>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SinglePersonPoseEstimation/vit/vit_pose_small_bn/pretrained/2023-07-20/vit_pose_small_bn.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/vit_pose_small_bn.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/vit_pose_small_bn_profiler_results_compiled.html>`_
     - 256x192x3
     - 24.32
     - 17.17
