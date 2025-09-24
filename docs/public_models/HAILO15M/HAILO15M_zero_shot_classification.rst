
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
* All models were compiled using Hailo Dataflow Compiler v3.30.0



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
     - 98
     - 207
     - 224x224x3
     - 38.72
     - 11.62
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/clip_resnet_50/pretrained/2023-03-09/clip_resnet_50.zip>`_
     - `link <https://github.com/openai/CLIP>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo15m/clip_resnet_50.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo15m/clip_resnet_50_profiler_results_compiled.html>`_    
   * - clip_resnet_50x4   
     - 50.3
     - 48.8
     - 33
     - 62
     - 288x288x3
     - 87.0
     - 41.3
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/clip_resnet_50x4/pretrained/2023-03-09/clip_resnet_50x4.zip>`_
     - `link <https://github.com/openai/CLIP>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo15m/clip_resnet_50x4.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo15m/clip_resnet_50x4_profiler_results_compiled.html>`_    
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
     - Profile Html    
   * - clip_vit_l_14_laion2B_16b   
     - 0
     - 0
     - 224x224x3
     - 304.16
     - 164.43
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/clip_vit_l_14/pretrained/2024-09-23/CLIP-ViT-L-14-laion2B-s32B-b82K_with_projection_op15_sim.zip>`_
     - `link <https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo15m/clip_vit_l_14_laion2B_16b.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo15m/clip_vit_l_14_laion2B_16b_profiler_results_compiled.html>`_
