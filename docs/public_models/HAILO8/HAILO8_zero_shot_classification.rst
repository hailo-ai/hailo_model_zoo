
Public Pre-Trained Models
=========================

.. |rocket| image:: ../../images/rocket.png
  :width: 18

.. |star| image:: ../../images/star.png
  :width: 18

Here, we give the full list of publicly pre-trained models supported by the Hailo Model Zoo.

* Network available in `Hailo Benchmark <https://hailo.ai/products/ai-accelerators/hailo-8-ai-accelerator/#hailo8-benchmarks/>`_ are marked with |rocket|
* Networks available in `TAPPAS <https://github.com/hailo-ai/tappas>`_ are marked with |star|
* Benchmark and TAPPAS  networks run in performance mode
* All models were compiled using Hailo Dataflow Compiler v3.31.0

Link Legend

The following shortcuts are used in the table below to indicate available resources for each model:

* S – Source: Link to the model’s open-source code repository.
* PT – Pretrained: Download the pretrained model file (compressed in ZIP format).
* H, NV, X – Compiled Models: Links to the compiled model in various formats:
            * H: regular HEF with RGB format
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
   * - clip_resnet_50x4_image_encoder   
     - 
     - 0
     - 0
     - 0
     - `S <https://github.com/openai/CLIP>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_resnet_50x4/image_encoder/pretrained/2023-03-09/clip_resnet_50x4.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/clip_resnet_50x4_image_encoder.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/clip_resnet_50x4_image_encoder_profiler_results_compiled.html>`_
     - 288x288x3
     - 87.0
     - 41.3  
   * - clip_vit_b_32_image_encoder   
     - 65.3
     - 62.01
     - 34
     - 187
     - `S <https://github.com/openai/CLIP>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_base_patch32_224/image_encoder/pretrained/2023-03-09/clip_vit_b_32.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/clip_vit_b_32_image_encoder.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/clip_vit_b_32_image_encoder_profiler_results_compiled.html>`_
     - 224x224x3
     - 87.8
     - 8.8    
   * - tinyclip_vit_39m_16_text_19m_yfcc15m_image_encoder   
     - 67.9
     - 64.3
     - 42
     - 175
     - `S <https://huggingface.co/wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/tinyclip/tinyclip_vit_39m_16_text_19m_yfcc15m_image_encoder/pretrained/2025-07-21/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M_image_encoder.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/tinyclip_vit_39m_16_text_19m_yfcc15m_image_encoder.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/tinyclip_vit_39m_16_text_19m_yfcc15m_image_encoder_profiler_results_compiled.html>`_
     - 224x224x3
     - 39
     - 16.02    
   * - tinyclip_vit_61m_32_text_29m_laion400m_image_encoder   
     - 72.8
     - 67.5
     - 48
     - 287
     - `S <https://huggingface.co/wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/tinyclip/tinyclip_vit_61m_32_text_29m_laion400m_image_encoder/pretrained/2025-07-21/TinyCLIP-ViT-61M-32-Text-29M-LAION400M_image_encoder.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/tinyclip_vit_61m_32_text_29m_laion400m_image_encoder.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/tinyclip_vit_61m_32_text_29m_laion400m_image_encoder_profiler_results_compiled.html>`_
     - 224x224x3
     - 61
     - 6.18                    
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
   * - clip_vit_b_16_image_encoder   
     - 
     - 0
     - 0
     - 0
     - `S <https://github.com/openai/CLIP>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_base_patch16_224/image_encoder/pretrained/2023-03-09/clip_vit_b_16.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/clip_vit_b_16_image_encoder.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/clip_vit_b_16_image_encoder_profiler_results_compiled.html>`_
     - 224x224x3
     - 86
     - 35.1    
   * - clip_vit_b_32_16b_image_encoder   
     - 
     - 0
     - 0
     - 0
     - `S <https://github.com/openai/CLIP>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_base_patch32_224/image_encoder/pretrained/2023-03-09/clip_vit_b_32.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/clip_vit_b_32_16b_image_encoder.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/clip_vit_b_32_16b_image_encoder_profiler_results_compiled.html>`_
     - 224x224x3
     - 87.8
     - 8.8    
   * - clip_vit_l_14_336_16b_image_encoder   
     - 
     - 0
     - 0
     - 0
     - `S <https://huggingface.co/openai/clip-vit-large-patch14-336>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_large_patch14_336/image_encoder/pretrained/2025-01-13/clip_vit_l_14_336.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/clip_vit_l_14_336_16b_image_encoder.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/clip_vit_l_14_336_16b_image_encoder_profiler_results_compiled.html>`_
     - 336x336x3
     - 304.16
     - 382.9    
   * - clip_vit_l_14_laion2B_16b_image_encoder   
     - 
     - 0
     - 0
     - 0
     - `S <https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_large_patch14_laion2B/image_encoder/pretrained/2024-09-23/CLIP-ViT-L-14-laion2B-s32B-b82K_with_projection_op15_sim.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/clip_vit_l_14_laion2B_16b_image_encoder.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/clip_vit_l_14_laion2B_16b_image_encoder_profiler_results_compiled.html>`_
     - 224x224x3
     - 304.16
     - 164.43    
   * - tinyclip_vit_8m_16_text_3m_yfcc15m_image_encoder   
     - 
     - 0
     - 0
     - 0
     - `S <https://huggingface.co/wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/tinyclip/tinyclip_vit_8m_16_text_3m_yfcc15m_image_encoder/pretrained/2025-07-21/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M_image_encoder.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/tinyclip_vit_8m_16_text_3m_yfcc15m_image_encoder.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/tinyclip_vit_8m_16_text_3m_yfcc15m_image_encoder_profiler_results_compiled.html>`_
     - 224x224x3
     - 8
     - 3.6
