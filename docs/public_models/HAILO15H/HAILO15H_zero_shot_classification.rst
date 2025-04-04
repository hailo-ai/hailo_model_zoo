
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



.. _Zero-shot Classification:

------------------------

CIFAR100
^^^^^^^^

.. list-table::
   :widths: 31 9 7 11 9 8 8 8 7 7 7 7
   :header-rows: 1

   * - Network Name
     - Accuracy (top1)
     - HW Accuracy
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
     - Profile Html
   * - clip_resnet_50
     - 42.07
     - 40.09
     - 169
     - 516
     - 224x224x3
     - 38.72
     - 11.62
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/clip_resnet_50/pretrained/2023-03-09/clip_resnet_50.zip>`_
     - `link <https://github.com/openai/CLIP>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/clip_resnet_50.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/clip_resnet_50_profiler_results_compiled.html>`_
   * - clip_resnet_50x4
     - 50.31
     - 48.45
     - 51
     - 107
     - 288x288x3
     - 87.0
     - 41.3
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/clip_resnet_50x4/pretrained/2023-03-09/clip_resnet_50x4.zip>`_
     - `link <https://github.com/openai/CLIP>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/clip_resnet_50x4.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/clip_resnet_50x4_profiler_results_compiled.html>`_
.. list-table::
   :header-rows: 1

   * - Network Name
     - Accuracy (top1)
     - HW Accuracy
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
     - Profile Html
   * - clip_vit_l_14_laion2B_16b
     - 78.6
     - 75.7
     - 4
     - 7
     - 224x224x3
     - 304.16
     - 164.43
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/clip_vit_l_14/pretrained/2024-09-23/CLIP-ViT-L-14-laion2B-s32B-b82K_with_projection_op15_sim.zip>`_
     - `link <https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/clip_vit_l_14_laion2B_16b.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/clip_vit_l_14_laion2B_16b_profiler_results_compiled.html>`_
   * - clip_vit_l_14_336_16b
     - 77.2
     - 75.7
     - 3
     - 4
     - 336x336x3
     - 304.16
     - 382
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/clip_vit_l_14_336/pretrained/2025-01-13/vl14_336_op15.zip>`_
     - `link <https://huggingface.co/openai/clip-vit-large-patch14-336>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/clip_vit_l_14_336_16b.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/clip_vit_l_14_336_16b_profiler_results_compiled.html>`_
   * - clip_vit_b_32_16b
     - 65.2
     - 62.7
     - 24
     - 77
     - 224x224x3
     - 87.8
     - 8.8
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/clip_vit_b_32/pretrained/2023-03-09/clip_vit_b_32.zip>`_
     - `link <https://github.com/openai/CLIP>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/clip_vit_b_32_16b.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/clip_vit_b_32_16b_profiler_results_compiled.html>`_
   * - clip_text_encoder_vitb_16
     - 90.9
     - 90.7
     - 24
     - 77
     - 224x224x3
     - 25
     - 39
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/clip/vitb_16/pretrained/2024-12-04/clip_text_encoder_vitb_16_sim.zip>`_
     - `link <https://huggingface.co/openai/clip-vit-base-patch16>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/clip_text_encoder_vitb_16.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/clip_text_encoder_vitb_16_profiler_results_compiled.html>`_
   * - clip_vit_b_16
     - 68.58
     - 66.3
     - 9
     - 13
     - 224x224x3
     - 25
     - 39
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/clip_vit_b_16/pretrained/2023-03-09/clip_vit_b_16.zip>`_
     - `link <https://github.com/openai/CLIP>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/clip_vit_b_16.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/clip_vit_b_16_profiler_results_compiled.html>`_
   * - clip_text_encoder_vit_l_14_laion2B
     - 94.7
     - 94
     - 18
     - 31
     - 1x77x768
     - 78.8
     - 13.8
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/clip/vit_l_14_laion2B/pretrained/2024-09-24/clip-vit-l-14-laion2b-s32b-b82k_text_op15.zip>`_
     - `link <https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/clip_text_encoder_vit_l_14_laion2B.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/clip_text_encoder_vit_l_14_laion2B_profiler_results_compiled.html>`_
   * - clip_text_encoder_vitb_32
     - 90.6
     - 88.8
     - 26
     - 44
     - 1x77x512
     - 37.8
     - 6
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/clip/vitb_32/pretrained/2024-12-04/clip_text_encoder_vitb_32_sim.zip>`_
     - `link <https://huggingface.co/openai/clip-vit-base-patch32>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/clip_text_encoder_vitb_32.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/clip_text_encoder_vitb_32_profiler_results_compiled.html>`_
   * - clip_vit_b_32
     - 65.2
     - 62.8
     - 73
     - 358
     - 224x224x3
     - 87.8
     - 8.8
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/clip_vit_b_32/pretrained/2023-03-09/clip_vit_b_32.zip>`_
     - `link <https://github.com/openai/CLIP>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/clip_vit_b_32.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/clip_vit_b_32_profiler_results_compiled.html>`_

