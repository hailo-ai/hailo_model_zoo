
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
* All models were compiled using Hailo Dataflow Compiler v3.31.0

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
   :header-rows: 1

   * - Network Name
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
   * - clip_resnet_50_image_encoder   
     - 0
     - 0
     - `S <https://github.com/openai/CLIP>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_resnet_50/image_encoder/pretrained/2023-03-09/clip_resnet_50.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/clip_resnet_50_image_encoder.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/clip_resnet_50_image_encoder_profiler_results_compiled.html>`_
     - 224x224x3
     - 38.72
     - 11.62    
   * - clip_resnet_50x4_image_encoder   
     - 0
     - 0
     - `S <https://github.com/openai/CLIP>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_resnet_50x4/image_encoder/pretrained/2023-03-09/clip_resnet_50x4.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/clip_resnet_50x4_image_encoder.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/clip_resnet_50x4_image_encoder_profiler_results_compiled.html>`_
     - 288x288x3
     - 87.0
     - 41.3    
   * - clip_vit_b_16_image_encoder   
     - 0
     - 0
     - `S <https://github.com/openai/CLIP>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_base_patch16_224/image_encoder/pretrained/2023-03-09/clip_vit_b_16.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/clip_vit_b_16_image_encoder.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/clip_vit_b_16_image_encoder_profiler_results_compiled.html>`_
     - 224x224x3
     - 86
     - 35.1    
   * - clip_vit_b_32_16b_image_encoder   
     - 0
     - 0
     - `S <https://github.com/openai/CLIP>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_base_patch32_224/image_encoder/pretrained/2023-03-09/clip_vit_b_32.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/clip_vit_b_32_16b_image_encoder.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/clip_vit_b_32_16b_image_encoder_profiler_results_compiled.html>`_
     - 224x224x3
     - 87.8
     - 8.8    
   * - clip_vit_b_32_image_encoder   
     - 0
     - 0
     - `S <https://github.com/openai/CLIP>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_base_patch32_224/image_encoder/pretrained/2023-03-09/clip_vit_b_32.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/clip_vit_b_32_image_encoder.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/clip_vit_b_32_image_encoder_profiler_results_compiled.html>`_
     - 224x224x3
     - 87.8
     - 8.8    
   * - clip_vit_l_14_336_16b_image_encoder   
     - 0
     - 0
     - `S <https://huggingface.co/openai/clip-vit-large-patch14-336>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_large_patch14_336/image_encoder/pretrained/2025-01-13/clip_vit_l_14_336.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/clip_vit_l_14_336_16b_image_encoder.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/clip_vit_l_14_336_16b_image_encoder_profiler_results_compiled.html>`_
     - 336x336x3
     - 304.16
     - 382.9    
   * - clip_vit_l_14_laion2B_16b_image_encoder   
     - 0
     - 0
     - `S <https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_large_patch14_laion2B/image_encoder/pretrained/2024-09-23/CLIP-ViT-L-14-laion2B-s32B-b82K_with_projection_op15_sim.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/clip_vit_l_14_laion2B_16b_image_encoder.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/clip_vit_l_14_laion2B_16b_image_encoder_profiler_results_compiled.html>`_
     - 224x224x3
     - 304.16
     - 164.43
