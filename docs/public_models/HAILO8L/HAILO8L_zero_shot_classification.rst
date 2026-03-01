


Public Models
=============

All models were compiled using Hailo Dataflow Compiler v2.18.0.

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
     - Source – Link to the model's open-source repository.
   * - **PT**
     - Pretrained – Download the pretrained model file (ZIP format).
   * - **HEF, NV12, RGBX**
     - Compiled Models – Links to models in various formats:
       - **HEF:** RGB format
       - **NV12:** NV12 format
       - **RGBX:** RGBX format
   * - **PR**
     - Profiler Report – Download the model's performance profiling report.

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
     - 40.0
     - 57.9
     - 156
     - | `S <https://github.com/openai/CLIP>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_resnet_50/image_encoder/pretrained/2023-03-09/clip_resnet_50.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/clip_resnet_50_image_encoder_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/clip_resnet_50_image_encoder.hef>`_
     - 224x224x3
     - 38.72
     - 11.62
   
   
   
   
   
   
   

   * - clip_resnet_50x4_image_encoder
     - 50.3
     - 50.4
     - 26.6
     - 56.6
     - | `S <https://github.com/openai/CLIP>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_resnet_50x4/image_encoder/pretrained/2023-03-09/clip_resnet_50x4.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/clip_resnet_50x4_image_encoder_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/clip_resnet_50x4_image_encoder.hef>`_
     - 288x288x3
     - 87.0
     - 41.3
   
   
   
   
   
   
   

   * - clip_vit_b_16_image_encoder
     - 68.6
     - 65.0
     - 20.4
     - 52.9
     - | `S <https://github.com/openai/CLIP>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_base_patch16_224/image_encoder/pretrained/2023-03-09/clip_vit_b_16.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/clip_vit_b_16_image_encoder_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/clip_vit_b_16_image_encoder.hef>`_
     - 224x224x3
     - 86
     - 35.1
   
   
   
   
   
   
   

   * - clip_vit_b_32_image_encoder⭐
     - 65.3
     - 62.3
     - 30.1
     - 130
     - | `S <https://github.com/openai/CLIP>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_base_patch32_224/image_encoder/pretrained/2023-03-09/clip_vit_b_32.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/clip_vit_b_32_image_encoder_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/clip_vit_b_32_image_encoder.hef>`_
     - 224x224x3
     - 87.8
     - 8.8
   
   
   
   
   
   
   

   * - siglip2_b_32_256_image_encoder
     - 74.7
     - 73.0
     - 10.9
     - 21.0
     - | `S <https://huggingface.co/google/siglip2-base-patch32-256>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/siglip/siglip2_base_patch32_256/image_encoder/pretrained/2025-05-21/siglip2-base-patch32-256_vision_encoder.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/siglip2_b_32_256_image_encoder_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/siglip2_b_32_256_image_encoder.hef>`_
     - 256x256x3
     - 93.9
     - 11.5
   
   
   
   
   
   
   

   * - siglip_b_16_image_encoder
     - 72.2
     - 68.9
     - 7.8
     - 14.5
     - | `S <https://huggingface.co/google/siglip-base-patch16-224>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/siglip/siglip_base_patch16_224/image_encoder/pretrained/2025-03-16/siglip_base_patch16_224_vision_encoder.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/siglip_b_16_image_encoder_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/siglip_b_16_image_encoder.hef>`_
     - 224x224x3
     - 92.1
     - 35.5
   
   
   
   
   
   
   

   * - tinyclip_vit_39m_16_text_19m_yfcc15m_image_encoder
     - 67.9
     - 64.5
     - 28.7
     - 95.7
     - | `S <https://huggingface.co/wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/tinyclip/tinyclip_vit_39m_16_text_19m_yfcc15m_image_encoder/pretrained/2025-07-21/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M_image_encoder.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/tinyclip_vit_39m_16_text_19m_yfcc15m_image_encoder_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/tinyclip_vit_39m_16_text_19m_yfcc15m_image_encoder.hef>`_
     - 224x224x3
     - 39
     - 16.02
   
   
   
   
   
   
   

   * - tinyclip_vit_40m_32_text_19m_laion400m_image_encoder
     - 69.7
     - 65.7
     - 35.2
     - 98.6
     - | `S <https://huggingface.co/wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/tinyclip/tinyclip_vit_40m_32_text_19m_laion400m_image_encoder/pretrained/2025-07-21/TinyCLIP-ViT-40M-32-Text-19M-LAION400M_image_encoder.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/tinyclip_vit_40m_32_text_19m_laion400m_image_encoder_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/tinyclip_vit_40m_32_text_19m_laion400m_image_encoder.hef>`_
     - 224x224x3
     - 40
     - 4
   
   
   
   
   
   
   

   * - tinyclip_vit_61m_32_text_29m_laion400m_image_encoder
     - 72.8
     - 67.8
     - 44.2
     - 215
     - | `S <https://huggingface.co/wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/tinyclip/tinyclip_vit_61m_32_text_29m_laion400m_image_encoder/pretrained/2025-07-21/TinyCLIP-ViT-61M-32-Text-29M-LAION400M_image_encoder.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/tinyclip_vit_61m_32_text_29m_laion400m_image_encoder_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/tinyclip_vit_61m_32_text_29m_laion400m_image_encoder.hef>`_
     - 224x224x3
     - 61
     - 6.18
   
   
   
   
   
   
   

   * - tinyclip_vit_8m_16_text_3m_yfcc15m_image_encoder
     - 42.0
     - 38.5
     - 84.6
     - 340
     - | `S <https://huggingface.co/wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/tinyclip/tinyclip_vit_8m_16_text_3m_yfcc15m_image_encoder/pretrained/2025-07-21/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M_image_encoder.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/tinyclip_vit_8m_16_text_3m_yfcc15m_image_encoder_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/tinyclip_vit_8m_16_text_3m_yfcc15m_image_encoder.hef>`_
     - 224x224x3
     - 8
     - 3.6