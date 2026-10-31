


Public Models
=============

All models were compiled using Hailo Dataflow Compiler v2.18.0.

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
     - 83.9
     - 32.3
     - 114
     - | `S <https://huggingface.co/timm/resnet50_clip.openai>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_resnet_50/text_encoder/pretrained/2024-09-16/clip_text_encoder_resnet50.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/clip_resnet_50_text_encoder_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/clip_resnet_50_text_encoder.hef>`_
     - 1x77x512
     - 37.8
     - 6.0
   
   
   
   
   
   
   

   * - clip_resnet_50x4_text_encoder
     - 91.2
     - 89.9
     - 21.4
     - 79.0
     - | `S <https://huggingface.co/timm/resnet50x4_clip.openai>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_resnet_50x4/text_encoder/pretrained/2024-09-16/clip_text_encoder_resnet50x4.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/clip_resnet_50x4_text_encoder_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/clip_resnet_50x4_text_encoder.hef>`_
     - 1x77x640
     - 59.1
     - 9.3
   
   
   
   
   
   
   

   * - clip_vit_b_16_text_encoder
     - 90.9
     - 89.8
     - 33.6
     - 129
     - | `S <https://huggingface.co/openai/clip-vit-base-patch16>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_base_patch16_224/text_encoder/pretrained/2024-12-04/clip_text_encoder_vitb_16_sim.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/clip_vit_b_16_text_encoder_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/clip_vit_b_16_text_encoder.hef>`_
     - 1x77x512
     - 37.8
     - 6.0
   
   
   
   
   
   
   

   * - clip_vit_b_32_text_encoder⭐
     - 90.6
     - 89.3
     - 32.4
     - 137
     - | `S <https://huggingface.co/openai/clip-vit-base-patch32>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_base_patch32_224/text_encoder/pretrained/2024-12-04/clip_text_encoder_vitb_32_sim.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/clip_vit_b_32_text_encoder_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/clip_vit_b_32_text_encoder.hef>`_
     - 1x77x512
     - 37.8
     - 6.0
   
   
   
   
   
   
   

   * - siglip2_b_32_256_text_encoder
     - 96.1
     - 96.2
     - 16.6
     - 81.9
     - | `S <https://huggingface.co/google/siglip2-base-patch32-256>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/siglip/siglip2_base_patch32_256/text_encoder/pretrained/2025-05-21/siglip2_base_patch32_256_text_encoder.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/siglip2_b_32_256_text_encoder_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/siglip2_b_32_256_text_encoder.hef>`_
     - 8x8x768
     - 85.6
     - 11.0
   
   
   
   
   
   
   

   * - siglip_b_16_text_encoder
     - 96.2
     - 96.0
     - 16.5
     - 81.9
     - | `S <https://huggingface.co/google/siglip-base-patch16-224>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/siglip/siglip_base_patch16_224/text_encoder/pretrained/2025-04-02/siglip_base_patch16_224_text_encoder.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/siglip_b_16_text_encoder_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/siglip_b_16_text_encoder.hef>`_
     - 8x8x768
     - 85.6
     - 11.1
   
   
   
   
   
   
   

   * - tinyclip_vit_39m_16_text_19m_yfcc15m_text_encoder
     - 94.0
     - 94.0
     - 68.3
     - 247
     - | `S <https://huggingface.co/wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/tinyclip/tinyclip_vit_39m_16_text_19m_yfcc15m_text_encoder/pretrained/2025-07-21/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M_text_encoder.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/tinyclip_vit_39m_16_text_19m_yfcc15m_text_encoder_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/tinyclip_vit_39m_16_text_19m_yfcc15m_text_encoder.hef>`_
     - 1x77x512
     - 19
     - 3
   
   
   
   
   
   
   

   * - tinyclip_vit_40m_32_text_19m_laion400m_text_encoder
     - 91.1
     - 90.0
     - 68.9
     - 229
     - | `S <https://huggingface.co/wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/tinyclip/tinyclip_vit_40m_32_text_19m_laion400m_text_encoder/pretrained/2025-07-21/TinyCLIP-ViT-40M-32-Text-19M-LAION400M_text_encoder.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/tinyclip_vit_40m_32_text_19m_laion400m_text_encoder_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/tinyclip_vit_40m_32_text_19m_laion400m_text_encoder.hef>`_
     - 1x77x512
     - 19
     - 3
   
   
   
   
   
   
   

   * - tinyclip_vit_61m_32_text_29m_laion400m_text_encoder
     - 93.8
     - 90.0
     - 44.7
     - 178
     - | `S <https://huggingface.co/wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/tinyclip/tinyclip_vit_61m_32_text_29m_laion400m_text_encoder/pretrained/2025-07-21/TinyCLIP-ViT-61M-32-Text-29M-LAION400M_text_encoder.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/tinyclip_vit_61m_32_text_29m_laion400m_text_encoder_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/tinyclip_vit_61m_32_text_29m_laion400m_text_encoder.hef>`_
     - 1x77x512
     - 29
     - 4.5
   
   
   
   
   
   
   

   * - tinyclip_vit_8m_16_text_3m_yfcc15m_text_encoder
     - 84.4
     - 84.2
     - 358
     - 1665
     - | `S <https://huggingface.co/wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/tinyclip/tinyclip_vit_8m_16_text_3m_yfcc15m_text_encoder/pretrained/2025-07-21/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M_text_encoder.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/tinyclip_vit_8m_16_text_3m_yfcc15m_text_encoder_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/tinyclip_vit_8m_16_text_3m_yfcc15m_text_encoder.hef>`_
     - 1x77x512
     - 3
     - 382