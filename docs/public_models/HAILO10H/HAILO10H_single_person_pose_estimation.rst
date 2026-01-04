


Public Models
=============

All models were compiled using Hailo Dataflow Compiler v5.2.0.

|

Single Person Pose Estimation
=============================

|

Link Legend
-----------

|

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - **Key / Icon**
     - **Description**
   * - ⭐
     - Networks used by `Hailo-apps <https://github.com/hailo-ai/hailo-apps-infra>`_.
   * - **S**
     - Source – Link to the model’s open-source repository.
   * - **PT**
     - Pretrained – Download the pretrained model file (ZIP format).
   * - **HEF, NV12, RGBX**
     - Compiled Models – Links to models in various formats:
       - **HEF:** RGB format
       - **NV12:** NV12 format
       - **RGBX:** RGBX format
   * - **PR**
     - Profiler Report – Download the model’s performance profiling report.

|

Coco
----

|

.. list-table::
   :header-rows: 1
   :widths: 31 9 7 11 9 8 8 8 9

   
   * - Network Name
     - float AP
     - Hardware AP
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
   
   
   
   
   
   
   

   * - mspn_regnetx_800mf
     - 70.8
     - 69.9
     - 2033
     - 2033
     - | `S <https://github.com/open-mmlab/mmpose>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SinglePersonPoseEstimation/mspn_regnetx_800mf/pretrained/2022-07-12/mspn_regnetx_800mf.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/mspn_regnetx_800mf_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/mspn_regnetx_800mf.hef>`_
     - 256x192x3
     - 7.17
     - 2.94
   
   
   
   
   
   
   

   * - vit_pose_small
     - 74.2
     - 73.0
     - 67.7
     - 213
     - | `S <https://github.com/ViTAE-Transformer/ViTPose>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SinglePersonPoseEstimation/vit/vit_pose_small/pretrained/2023-11-14/vit_pose_small.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/vit_pose_small_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/vit_pose_small.hef>`_
     - 256x192x3
     - 24.29
     - 17.17
   
   
   
   
   
   
   

   * - vit_pose_small_bn
     - 72.0
     - 70.9
     - 113
     - 350
     - | `S <https://github.com/ViTAE-Transformer/ViTPose>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SinglePersonPoseEstimation/vit/vit_pose_small_bn/pretrained/2023-07-20/vit_pose_small_bn.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/vit_pose_small_bn_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/vit_pose_small_bn.hef>`_
     - 256x192x3
     - 24.32
     - 17.17