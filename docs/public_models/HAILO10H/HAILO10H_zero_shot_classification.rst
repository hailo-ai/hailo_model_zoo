
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
* All models were compiled using Hailo Dataflow Compiler v5.1.0

Link Legend

The following shortcuts are used in the table below to indicate available resources for each model:

* S – Source: Link to the model’s open-source code repository.
* PT – Pretrained: Download the pretrained model file (compressed in ZIP format).
* H, NV, X – Compiled Models: Links to the compiled model in various formats:
            * H: regular HEF with RGBX format
            * NV: HEF with NV12 format
            * X: HEF with RGBX format

* PR – Profiler Report: Download the model’s performance profiling report.



.. _Zero-shot Classification:

------------------------

CIFAR100
^^^^^^^^

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
   * - clip_resnet_50_image_encoder
     - 37.7
     - 33.3
     - 176
     - 528
     - `S <https://github.com/openai/CLIP>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_resnet_50/image_encoder/pretrained/2023-03-09/clip_resnet_50.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/clip_resnet_50_image_encoder.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/clip_resnet_50_image_encoder_profiler_results_compiled.html>`_
     - 224x224x3
     - 38.72
     - 11.62
   * - clip_resnet_50x4_image_encoder
     - 50.2
     - 50.1
     - 56
     - 132
     - `S <https://github.com/openai/CLIP>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_resnet_50x4/image_encoder/pretrained/2023-03-09/clip_resnet_50x4.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/clip_resnet_50x4_image_encoder.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/clip_resnet_50x4_image_encoder_profiler_results_compiled.html>`_
     - 288x288x3
     - 87.0
     - 41.3
   * - clip_vit_b_16_image_encoder
     - 67.8
     - 67.0
     - 11
     - 16
     - `S <https://github.com/openai/CLIP>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_base_patch16_224/image_encoder/pretrained/2023-03-09/clip_vit_b_16.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/clip_vit_b_16_image_encoder.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/clip_vit_b_16_image_encoder_profiler_results_compiled.html>`_
     - 224x224x3
     - 86
     - 35.1
   * - clip_vit_b_32_16b_image_encoder
     - 62.4
     - 59.5
     - 27
     - 96
     - `S <https://github.com/openai/CLIP>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_base_patch32_224/image_encoder/pretrained/2023-03-09/clip_vit_b_32.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/clip_vit_b_32_16b_image_encoder.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/clip_vit_b_32_16b_image_encoder_profiler_results_compiled.html>`_
     - 224x224x3
     - 87.8
     - 8.8
   * - clip_vit_l_14_336_16b_image_encoder
     - 72.0
     - 66.8
     - 3
     - 5
     - `S <https://huggingface.co/openai/clip-vit-large-patch14-336>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_large_patch14_336/image_encoder/pretrained/2025-01-13/clip_vit_l_14_336.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/clip_vit_l_14_336_16b_image_encoder.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/clip_vit_l_14_336_16b_image_encoder_profiler_results_compiled.html>`_
     - 336x336x3
     - 304.16
     - 382.9
   * - clip_vit_l_14_laion2B_16b_image_encoder
     - 76.6
     - 74.6
     - 4
     - 7
     - `S <https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_large_patch14_laion2B/image_encoder/pretrained/2024-09-23/CLIP-ViT-L-14-laion2B-s32B-b82K_with_projection_op15_sim.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/clip_vit_l_14_laion2B_16b_image_encoder.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/clip_vit_l_14_laion2B_16b_image_encoder_profiler_results_compiled.html>`_
     - 224x224x3
     - 304.16
     - 164.43
   * - siglip2_b_32_256_image_encoder
     - 70.8
     - 66.9
     - 28
     - 62
     - `S <https://huggingface.co/google/siglip2-base-patch32-256>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/siglip/siglip2_base_patch32_256/image_encoder/pretrained/2025-05-21/siglip2-base-patch32-256_vision_encoder.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/siglip2_b_32_256_image_encoder.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/siglip2_b_32_256_image_encoder_profiler_results_compiled.html>`_
     - 256x256x3
     - 93.9
     - 11.5
.. list-table::
   :header-rows: 1

   * - Network Name
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
     - Profile Report
   * - clip_vit_b_32_image_encoder
     - 67
     - 340
     - `S <https://github.com/openai/CLIP>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_base_patch32_224/image_encoder/pretrained/2023-03-09/clip_vit_b_32.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/clip_vit_b_32_image_encoder.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/clip_vit_b_32_image_encoder_profiler_results_compiled.html>`_
     - 224x224x3
     - 87.8
     - 8.8
   * - siglip2_b_16_image_encoder
     - 0
     - 0
     - `S <https://huggingface.co/google/siglip2-base-patch16-224>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/siglip/siglip2_base_patch16_224/image_encoder/pretrained/2025-05-06/siglip2-base-patch16-224_vision_encoder.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/siglip2_b_16_image_encoder.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/siglip2_b_16_image_encoder_profiler_results_compiled.html>`_
     - 224x224x3
     - 92.1
     - 35.4
