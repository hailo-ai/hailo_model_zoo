
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
            * H: regular HEF with RGBX format
            * NV: HEF with NV12 format
            * X: HEF with RGBX format

* PR – Profiler Report: Download the model’s performance profiling report.



.. _Text_image_Retrieval:

--------------------

N/A
^^^

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
   * - clip_resnet_50x4_text_encoder   
     - 89.7
     - 88.2
     - 26
     - 52
     - `S <https://huggingface.co/timm/resnet50x4_clip.openai>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_resnet_50x4/text_encoder/pretrained/2024-09-16/clip_text_encoder_resnet50x4.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo10h/clip_resnet_50x4_text_encoder.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo10h/clip_resnet_50x4_text_encoder_profiler_results_compiled.html>`_
     - 1x77x640
     - 59.1
     - 9.3    
   * - clip_vit_b_16_text_encoder   
     - 89.6
     - 88.3
     - 39
     - 86
     - `S <https://huggingface.co/openai/clip-vit-base-patch16>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_base_patch16_224/text_encoder/pretrained/2024-12-04/clip_text_encoder_vitb_16_sim.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo10h/clip_vit_b_16_text_encoder.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo10h/clip_vit_b_16_text_encoder_profiler_results_compiled.html>`_
     - 1x77x512
     - 37.8
     - 6.0    
   * - clip_vit_b_32_text_encoder   
     - 89.4
     - 88.2
     - 41
     - 90
     - `S <https://huggingface.co/openai/clip-vit-base-patch32>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_base_patch32_224/text_encoder/pretrained/2024-12-04/clip_text_encoder_vitb_32_sim.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo10h/clip_vit_b_32_text_encoder.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo10h/clip_vit_b_32_text_encoder_profiler_results_compiled.html>`_
     - 1x77x512
     - 37.8
     - 6.0    
   * - clip_vit_l_14_laion2B_text_encoder   
     - 94.3
     - 93.9
     - 23
     - 49
     - `S <https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_large_patch14_laion2B/text_encoder/pretrained/2024-09-24/clip_text_encoder_vit_l_14_laion2B.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo10h/clip_vit_l_14_laion2B_text_encoder.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo10h/clip_vit_l_14_laion2B_text_encoder_profiler_results_compiled.html>`_
     - 1x77x768
     - 78.87
     - 13.85    
   * - clip_vit_l_14_text_encoder   
     - 91.0
     - 90.2
     - 20
     - 41
     - `S <https://huggingface.co/openai/clip-vit-large-patch14>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_large_patch14_224/text_encoder/pretrained/2024-08-25/clip_text_encoder_vit_large.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo10h/clip_vit_l_14_text_encoder.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo10h/clip_vit_l_14_text_encoder_profiler_results_compiled.html>`_
     - 1x77x768
     - 59.1
     - 13.85    
   * - siglip2_b_32_256_text_encoder   
     - 96.5
     - 96.1
     - 22
     - 50
     - `S <https://huggingface.co/google/siglip2-base-patch32-256>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/siglip/siglip2_base_patch32_256/text_encoder/pretrained/2025-05-21/siglip2_base_patch32_256_text_encoder.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo10h/siglip2_b_32_256_text_encoder.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo10h/siglip2_b_32_256_text_encoder_profiler_results_compiled.html>`_
     - 1x64x768
     - 85.6
     - 11.0    
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
   * - siglip2_b_16_text_encoder   
     - 97.4
     - 97.4
     - 23
     - 45
     - `S <https://huggingface.co/google/siglip2-base-patch16-224>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/siglip/siglip2_base_patch16_224/text_encoder/pretrained/2025-05-12/siglip2_base_patch16_224_text_encoder.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo10h/siglip2_b_16_text_encoder.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo10h/siglip2_b_16_text_encoder_profiler_results_compiled.html>`_
     - 1x64x768
     - 85.6
     - 11.1
