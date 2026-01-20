


Public Models
=============

All models were compiled using Hailo Dataflow Compiler v5.2.0.

|

Zero-Shot Classification
========================

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

Cifar100
--------

|

.. list-table::
   :header-rows: 1
   :widths: 31 9 7 11 9 8 8 8 9

   
   * - Network Name
     - float Accuracy (top1)
     - Hardware Accuracy (top1)
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
   
   
   
   
   
   
   

   * - clip_resnet_50_image_encoder
     - 42.1
     - 41.4
     - 137
     - 296
     - | `S <https://github.com/openai/CLIP>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_resnet_50/image_encoder/pretrained/2023-03-09/clip_resnet_50.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/clip_resnet_50_image_encoder_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/clip_resnet_50_image_encoder.hef>`_
     - 224x224x3
     - 38.72
     - 11.62
   
   
   
   
   
   
   

   * - clip_resnet_50x4_image_encoder
     - 50.3
     - 51.1
     - 63.1
     - 157
     - | `S <https://github.com/openai/CLIP>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_resnet_50x4/image_encoder/pretrained/2023-03-09/clip_resnet_50x4.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/clip_resnet_50x4_image_encoder_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/clip_resnet_50x4_image_encoder.hef>`_
     - 288x288x3
     - 87.0
     - 41.3
   
   
   
   
   
   
   

   * - clip_vit_b_16_image_encoder
     - 68.6
     - 65.0
     - 57.1
     - 169
     - | `S <https://github.com/openai/CLIP>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_base_patch16_224/image_encoder/pretrained/2023-03-09/clip_vit_b_16.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/clip_vit_b_16_image_encoder_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/clip_vit_b_16_image_encoder.hef>`_
     - 224x224x3
     - 86
     - 35.1
   
   
   
   
   
   
   

   * - clip_vit_b_32_image_encoder
     - 65.3
     - 64.0
     - 71.8
     - 335
     - | `S <https://github.com/openai/CLIP>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_base_patch32_224/image_encoder/pretrained/2023-03-09/clip_vit_b_32.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/clip_vit_b_32_image_encoder_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/clip_vit_b_32_image_encoder.hef>`_ `RGBX <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/clip_vit_b_32_image_encoder_rgbx.hef>`_
     - 224x224x3
     - 87.8
     - 8.8
   
   
   
   
   
   
   

   * - clip_vit_l_14_336_image_encoder
     - 77.2
     - 71.7
     - 6.72
     - 12.4
     - | `S <https://huggingface.co/openai/clip-vit-large-patch14-336>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_large_patch14_336/image_encoder/pretrained/2025-01-13/clip_vit_l_14_336.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/clip_vit_l_14_336_image_encoder_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/clip_vit_l_14_336_image_encoder.hef>`_
     - 336x336x3
     - 304.16
     - 162.36
   
   
   
   
   
   
   

   * - clip_vit_l_14_laion2B_image_encoder
     - 78.6
     - 77.7
     - 15.4
     - 40.8
     - | `S <https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_large_patch14_laion2B/image_encoder/pretrained/2024-09-23/CLIP-ViT-L-14-laion2B-s32B-b82K_with_projection_op15_sim.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/clip_vit_l_14_laion2B_image_encoder_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/clip_vit_l_14_laion2B_image_encoder.hef>`_
     - 224x224x3
     - 304.16
     - 162.36
   
   
   
   
   
   
   

   * - siglip2_b_32_256_image_encoder
     - 74.7
     - 70.1
     - 63.4
     - 220
     - | `S <https://huggingface.co/google/siglip2-base-patch32-256>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/siglip/siglip2_base_patch32_256/image_encoder/pretrained/2025-05-21/siglip2-base-patch32-256_vision_encoder.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/siglip2_b_32_256_image_encoder_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/siglip2_b_32_256_image_encoder.hef>`_
     - 256x256x3
     - 93.9
     - 11.5
   
   
   
   
   
   
   

   * - siglip_b_16_image_encoder
     - 72.2
     - 71.8
     - 39.6
     - 88.1
     - | `S <https://huggingface.co/google/siglip-base-patch16-224>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/siglip/siglip_base_patch16_224/image_encoder/pretrained/2025-03-16/siglip_base_patch16_224_vision_encoder.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/siglip_b_16_image_encoder_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/siglip_b_16_image_encoder.hef>`_
     - 224x224x3
     - 92.1
     - 35.5
   
   
   
   
   
   
   

   * - siglip_l_16_256_image_encoder
     - 82.0
     - 79.3
     - 14.7
     - 32.1
     - | `S <https://huggingface.co/google/siglip-large-patch16-256>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/siglip/siglip_large_patch16_256/image_encoder/pretrained/2025-03-25/siglip_large_patch16_256_vision_encoder.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/siglip_l_16_256_image_encoder_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/siglip_l_16_256_image_encoder.hef>`_
     - 256x256x3
     - 315
     - 163.1
   
   
   
   
   
   
   

   * - tinyclip_vit_39m_16_text_19m_yfcc15m_image_encoder
     - 67.9
     - 65.2
     - 86.3
     - 271
     - | `S <https://huggingface.co/wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/tinyclip/tinyclip_vit_39m_16_text_19m_yfcc15m_image_encoder/pretrained/2025-07-21/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M_image_encoder.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/tinyclip_vit_39m_16_text_19m_yfcc15m_image_encoder_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/tinyclip_vit_39m_16_text_19m_yfcc15m_image_encoder.hef>`_
     - 224x224x3
     - 39
     - 16.02
   
   
   
   
   
   
   

   * - tinyclip_vit_40m_32_text_19m_laion400m_image_encoder
     - 69.7
     - 68.0
     - 127
     - 495
     - | `S <https://huggingface.co/wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/tinyclip/tinyclip_vit_40m_32_text_19m_laion400m_image_encoder/pretrained/2025-07-21/TinyCLIP-ViT-40M-32-Text-19M-LAION400M_image_encoder.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/tinyclip_vit_40m_32_text_19m_laion400m_image_encoder_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/tinyclip_vit_40m_32_text_19m_laion400m_image_encoder.hef>`_
     - 224x224x3
     - 40
     - 4
   
   
   
   
   
   
   

   * - tinyclip_vit_61m_32_text_29m_laion400m_image_encoder
     - 72.8
     - 70.0
     - 98.2
     - 362
     - | `S <https://huggingface.co/wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/tinyclip/tinyclip_vit_61m_32_text_29m_laion400m_image_encoder/pretrained/2025-07-21/TinyCLIP-ViT-61M-32-Text-29M-LAION400M_image_encoder.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/tinyclip_vit_61m_32_text_29m_laion400m_image_encoder_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/tinyclip_vit_61m_32_text_29m_laion400m_image_encoder.hef>`_
     - 224x224x3
     - 61
     - 6.18
   
   
   
   
   
   
   

   * - tinyclip_vit_8m_16_text_3m_yfcc15m_image_encoder
     - 42.0
     - 40.9
     - 149
     - 521
     - | `S <https://huggingface.co/wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/tinyclip/tinyclip_vit_8m_16_text_3m_yfcc15m_image_encoder/pretrained/2025-07-21/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M_image_encoder.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/tinyclip_vit_8m_16_text_3m_yfcc15m_image_encoder_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/tinyclip_vit_8m_16_text_3m_yfcc15m_image_encoder.hef>`_
     - 224x224x3
     - 8
     - 3.6
