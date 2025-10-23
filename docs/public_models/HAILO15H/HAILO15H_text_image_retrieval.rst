


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

Text Image Retrieval
====================

.. list-table::
   :header-rows: 1
   :widths: 31 9 7 11 9 8 8 8 9

   * - Network Name
     - float mAP
     - Hardware mAP
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
   
   
   
   

   * - clip_resnet_50x4_text_encoder 
     - 91.2
     - 89.8
     - 25.1
     - 47.2
     - |
       `S <https://huggingface.co/timm/resnet50x4_clip.openai>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_resnet_50x4/text_encoder/pretrained/2024-09-16/clip_text_encoder_resnet50x4.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/clip_resnet_50x4_text_encoder.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/clip_resnet_50x4_text_encoder_profiler_results_compiled.html>`_
     - 1x77x640
     - 59.1
     - 9.3
   
   
   
   

   * - clip_vit_b_16_text_encoder 
     - 90.9
     - 90.1
     - 35.6
     - 72.0
     - |
       `S <https://huggingface.co/openai/clip-vit-base-patch16>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_base_patch16_224/text_encoder/pretrained/2024-12-04/clip_text_encoder_vitb_16_sim.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/clip_vit_b_16_text_encoder.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/clip_vit_b_16_text_encoder_profiler_results_compiled.html>`_
     - 1x77x512
     - 37.8
     - 6.0
   
   
   
   

   * - clip_vit_b_32_text_encoder 
     - 90.6
     - 88.9
     - 37.7
     - 85.0
     - |
       `S <https://huggingface.co/openai/clip-vit-base-patch32>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_base_patch32_224/text_encoder/pretrained/2024-12-04/clip_text_encoder_vitb_32_sim.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/clip_vit_b_32_text_encoder.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/clip_vit_b_32_text_encoder_profiler_results_compiled.html>`_
     - 1x77x512
     - 37.8
     - 6.0
   
   
   
   

   * - clip_vit_l_14_laion2B_text_encoder 
     - 94.7
     - 94.5
     - 24.4
     - 45.8
     - |
       `S <https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_large_patch14_laion2B/text_encoder/pretrained/2024-09-24/clip_text_encoder_vit_l_14_laion2B.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/clip_vit_l_14_laion2B_text_encoder.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/clip_vit_l_14_laion2B_text_encoder_profiler_results_compiled.html>`_
     - 1x77x768
     - 78.87
     - 13.85
   
   
   
   

   * - clip_vit_l_14_text_encoder 
     - 91.8
     - 91.2
     - 19.6
     - 38.5
     - |
       `S <https://huggingface.co/openai/clip-vit-large-patch14>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_large_patch14_224/text_encoder/pretrained/2024-08-25/clip_text_encoder_vit_large.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/clip_vit_l_14_text_encoder.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/clip_vit_l_14_text_encoder_profiler_results_compiled.html>`_
     - 1x77x768
     - 59.1
     - 13.85
   
   
   
   

   * - siglip2_b_16_text_encoder 
     - 97.4
     - 97.1
     - 23.0
     - 49.9
     - |
       `S <https://huggingface.co/google/siglip2-base-patch16-224>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/siglip/siglip2_base_patch16_224/text_encoder/pretrained/2025-05-12/siglip2_base_patch16_224_text_encoder.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/siglip2_b_16_text_encoder.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/siglip2_b_16_text_encoder_profiler_results_compiled.html>`_
     - 1x64x768
     - 85.6
     - 11.1
   
   
   
   

   * - siglip2_b_32_256_text_encoder 
     - 96.1
     - 96.7
     - 23.5
     - 50.9
     - |
       `S <https://huggingface.co/google/siglip2-base-patch32-256>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/siglip/siglip2_base_patch32_256/text_encoder/pretrained/2025-05-21/siglip2_base_patch32_256_text_encoder.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/siglip2_b_32_256_text_encoder.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/siglip2_b_32_256_text_encoder_profiler_results_compiled.html>`_
     - 1x64x768
     - 85.6
     - 11.0
   
   
   
   

   * - tinyclip_vit_39m_16_text_19m_yfcc15m_text_encoder 
     - 94.0
     - 94.2
     - 92.1
     - 207
     - |
       `S <https://huggingface.co/wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/tinyclip/tinyclip_vit_39m_16_text_19m_yfcc15m_text_encoder/pretrained/2025-07-21/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M_text_encoder.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/tinyclip_vit_39m_16_text_19m_yfcc15m_text_encoder.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/tinyclip_vit_39m_16_text_19m_yfcc15m_text_encoder_profiler_results_compiled.html>`_
     - 1x77x512
     - 19
     - 3
   
   
   
   

   * - tinyclip_vit_40m_32_text_19m_laion400m_text_encoder 
     - 91.1
     - 89.9
     - 91.0
     - 201
     - |
       `S <https://huggingface.co/wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/tinyclip/tinyclip_vit_40m_32_text_19m_laion400m_text_encoder/pretrained/2025-07-21/TinyCLIP-ViT-40M-32-Text-19M-LAION400M_text_encoder.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/tinyclip_vit_40m_32_text_19m_laion400m_text_encoder.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/tinyclip_vit_40m_32_text_19m_laion400m_text_encoder_profiler_results_compiled.html>`_
     - 1x77x512
     - 19
     - 3
   
   
   
   

   * - tinyclip_vit_61m_32_text_29m_laion400m_text_encoder 
     - 93.8
     - 91.8
     - 49.1
     - 101
     - |
       `S <https://huggingface.co/wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/tinyclip/tinyclip_vit_61m_32_text_29m_laion400m_text_encoder/pretrained/2025-07-21/TinyCLIP-ViT-61M-32-Text-29M-LAION400M_text_encoder.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/tinyclip_vit_61m_32_text_29m_laion400m_text_encoder.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/tinyclip_vit_61m_32_text_29m_laion400m_text_encoder_profiler_results_compiled.html>`_
     - 1x77x512
     - 29
     - 4.5

