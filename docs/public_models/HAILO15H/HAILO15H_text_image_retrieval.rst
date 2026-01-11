


Public Models
=============

All models were compiled using Hailo Dataflow Compiler v5.2.0.

|

Text_Image_Retrieval
====================

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

Text Image Retrieval
--------------------

|

.. list-table::
   :header-rows: 1
   :widths: 31 9 7 11 9 8 8 8 9

   
   * - Network Name
     - float Retrieval@10
     - Hardware Retrieval@10
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
   
   
   
   
   
   
   

   * - clip_resnet_50_text_encoder
     - 88.8
     - 86.3
     - 45.1
     - 96.7
     - | `S <https://huggingface.co/timm/resnet50_clip.openai>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_resnet_50/text_encoder/pretrained/2024-09-16/clip_text_encoder_resnet50.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/clip_resnet_50_text_encoder_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/clip_resnet_50_text_encoder.hef>`_
     - 1x77x512
     - 37.8
     - 6.0
   
   
   
   
   
   
   

   * - clip_resnet_50x4_text_encoder
     - 91.2
     - 89.8
     - 37.4
     - 73.4
     - | `S <https://huggingface.co/timm/resnet50x4_clip.openai>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_resnet_50x4/text_encoder/pretrained/2024-09-16/clip_text_encoder_resnet50x4.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/clip_resnet_50x4_text_encoder_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/clip_resnet_50x4_text_encoder.hef>`_
     - 1x77x640
     - 59.1
     - 9.3
   
   
   
   
   
   
   

   * - clip_vit_b_16_text_encoder
     - 90.9
     - 90.4
     - 47.4
     - 103
     - | `S <https://huggingface.co/openai/clip-vit-base-patch16>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_base_patch16_224/text_encoder/pretrained/2024-12-04/clip_text_encoder_vitb_16_sim.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/clip_vit_b_16_text_encoder_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/clip_vit_b_16_text_encoder.hef>`_
     - 1x77x512
     - 37.8
     - 6.0
   
   
   
   
   
   
   

   * - clip_vit_b_32_text_encoder
     - 90.6
     - 89.5
     - 50.0
     - 99.6
     - | `S <https://huggingface.co/openai/clip-vit-base-patch32>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_base_patch32_224/text_encoder/pretrained/2024-12-04/clip_text_encoder_vitb_32_sim.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/clip_vit_b_32_text_encoder_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/clip_vit_b_32_text_encoder.hef>`_
     - 1x77x512
     - 37.8
     - 6.0
   
   
   
   
   
   
   

   * - clip_vit_l_14_laion2B_text_encoder
     - 94.7
     - 94.5
     - 29.9
     - 55.2
     - | `S <https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_large_patch14_laion2B/text_encoder/pretrained/2024-09-24/clip_text_encoder_vit_l_14_laion2B.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/clip_vit_l_14_laion2B_text_encoder_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/clip_vit_l_14_laion2B_text_encoder.hef>`_
     - 1x77x768
     - 78.87
     - 13.85
   
   
   
   
   
   
   

   * - clip_vit_l_14_text_encoder
     - 91.8
     - 91.1
     - 28.8
     - 49.4
     - | `S <https://huggingface.co/openai/clip-vit-large-patch14>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_large_patch14_224/text_encoder/pretrained/2024-08-25/clip_text_encoder_vit_large.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/clip_vit_l_14_text_encoder_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/clip_vit_l_14_text_encoder.hef>`_
     - 1x77x768
     - 59.1
     - 13.85
   
   
   
   
   
   
   

   * - siglip2_b_16_text_encoder
     - 97.4
     - 97.1
     - 34.8
     - 67.6
     - | `S <https://huggingface.co/google/siglip2-base-patch16-224>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/siglip/siglip2_base_patch16_224/text_encoder/pretrained/2025-05-12/siglip2_base_patch16_224_text_encoder.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/siglip2_b_16_text_encoder_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/siglip2_b_16_text_encoder.hef>`_
     - 1x64x768
     - 85.6
     - 11.1
   
   
   
   
   
   
   

   * - siglip2_b_32_256_text_encoder
     - 96.1
     - 96.7
     - 34.0
     - 68.3
     - | `S <https://huggingface.co/google/siglip2-base-patch32-256>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/siglip/siglip2_base_patch32_256/text_encoder/pretrained/2025-05-21/siglip2_base_patch32_256_text_encoder.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/siglip2_b_32_256_text_encoder_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/siglip2_b_32_256_text_encoder.hef>`_
     - 1x64x768
     - 85.6
     - 11.0
   
   
   
   
   
   
   

   * - siglip2_l_16_256_text_encoder
     - 97.0
     - 96.7
     - 8.18
     - 16.0
     - | `S <https://huggingface.co/google/siglip2-large-patch16-256>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/siglip/siglip2_large_patch16_256/text_encoder/pretrained/2025-05-12/siglip_large_patch16_256_text_encoder.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/siglip2_l_16_256_text_encoder_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/siglip2_l_16_256_text_encoder.hef>`_
     - 1x64x1024
     - 303.3
     - 39.2
   
   
   
   
   
   
   

   * - siglip_b_16_text_encoder
     - 96.2
     - 96.2
     - 34.4
     - 63.2
     - | `S <https://huggingface.co/google/siglip-base-patch16-224>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/siglip/siglip_base_patch16_224/text_encoder/pretrained/2025-04-02/siglip_base_patch16_224_text_encoder.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/siglip_b_16_text_encoder_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/siglip_b_16_text_encoder.hef>`_
     - 1x64x768
     - 85.6
     - 11.1
   
   
   
   
   
   
   

   * - siglip_l_16_256_text_encoder
     - 96.7
     - 96.7
     - 6.45
     - 10.2
     - | `S <https://huggingface.co/google/siglip-large-patch16-256>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/siglip/siglip_large_patch16_256/text_encoder/pretrained/2025-04-06/siglip_large_patch16_256_text_encoder.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/siglip_l_16_256_text_encoder_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/siglip_l_16_256_text_encoder.hef>`_
     - 1x64x1024
     - 303.3
     - 39.2
   
   
   
   
   
   
   

   * - tinyclip_vit_39m_16_text_19m_yfcc15m_text_encoder
     - 94.0
     - 94.2
     - 93.9
     - 198
     - | `S <https://huggingface.co/wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/tinyclip/tinyclip_vit_39m_16_text_19m_yfcc15m_text_encoder/pretrained/2025-07-21/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M_text_encoder.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/tinyclip_vit_39m_16_text_19m_yfcc15m_text_encoder_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/tinyclip_vit_39m_16_text_19m_yfcc15m_text_encoder.hef>`_
     - 1x77x512
     - 19
     - 3
   
   
   
   
   
   
   

   * - tinyclip_vit_40m_32_text_19m_laion400m_text_encoder
     - 91.1
     - 89.7
     - 94.6
     - 206
     - | `S <https://huggingface.co/wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/tinyclip/tinyclip_vit_40m_32_text_19m_laion400m_text_encoder/pretrained/2025-07-21/TinyCLIP-ViT-40M-32-Text-19M-LAION400M_text_encoder.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/tinyclip_vit_40m_32_text_19m_laion400m_text_encoder_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/tinyclip_vit_40m_32_text_19m_laion400m_text_encoder.hef>`_
     - 1x77x512
     - 19
     - 3
   
   
   
   
   
   
   

   * - tinyclip_vit_61m_32_text_29m_laion400m_text_encoder
     - 93.8
     - 90.3
     - 55.5
     - 125
     - | `S <https://huggingface.co/wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/tinyclip/tinyclip_vit_61m_32_text_29m_laion400m_text_encoder/pretrained/2025-07-21/TinyCLIP-ViT-61M-32-Text-29M-LAION400M_text_encoder.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/tinyclip_vit_61m_32_text_29m_laion400m_text_encoder_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/tinyclip_vit_61m_32_text_29m_laion400m_text_encoder.hef>`_
     - 1x77x512
     - 29
     - 4.5
   
   
   
   
   
   
   

   * - tinyclip_vit_8m_16_text_3m_yfcc15m_text_encoder
     - 84.4
     - 83.7
     - 392
     - 1133
     - | `S <https://huggingface.co/wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/tinyclip/tinyclip_vit_8m_16_text_3m_yfcc15m_text_encoder/pretrained/2025-07-21/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M_text_encoder.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/tinyclip_vit_8m_16_text_3m_yfcc15m_text_encoder_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/tinyclip_vit_8m_16_text_3m_yfcc15m_text_encoder.hef>`_
     - 1x77x512
     - 3
     - 11.59