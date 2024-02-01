
Public Pre-Trained Models
=========================

.. |rocket| image:: ../images/rocket.png
  :width: 18

.. |star| image:: ../images/star.png
  :width: 18

.. _Single Person Pose Estimation:

Single Person Pose Estimation
-----------------------------

COCO
^^^^

.. list-table::
   :widths: 24 8 9 18 9 8 9 8 7 7 7
   :header-rows: 1

   * - Network Name
     - AP
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
   * - mspn_regnetx_800mf  |star|
     - 70.8
     - 70.3
     - 256x192x3
     - 7.17
     - 2.94
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SinglePersonPoseEstimation/mspn_regnetx_800mf/pretrained/2022-07-12/mspn_regnetx_800mf.zip>`_
     - `link <https://github.com/open-mmlab/mmpose>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo8/mspn_regnetx_800mf.hef>`_
     - 1843.36
     - 1840.82
   * - vit_pose_small
     - 74.16
     - 71.6
     - 256x192x3
     - 24.29
     - 17.17
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SinglePersonPoseEstimation/vit/vit_pose_small/pretrained/2023-11-14/vit_pose_small.zip>`_
     - `link <https://github.com/ViTAE-Transformer/ViTPose>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo8/vit_pose_small.hef>`_
     - 32.9208
     - 154.658
   * - vit_pose_small_bn
     - 72.01
     - 70.81
     - 256x192x3
     - 24.32
     - 17.17
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SinglePersonPoseEstimation/vit/vit_pose_small_bn/pretrained/2023-07-20/vit_pose_small_bn.zip>`_
     - `link <https://github.com/ViTAE-Transformer/ViTPose>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo8/vit_pose_small_bn.hef>`_
     - 60.9302
     - 247.454
