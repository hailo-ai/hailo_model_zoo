

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
* All models were compiled using Hailo Dataflow Compiler v3.29.0



.. _Text Image Retrievaln:

--------------

ImageNet
^^^^^^^^

.. list-table::
   :widths: 31 9 7 11 9 8 8 8 7 7 7
   :header-rows: 1

   * - Network Name
     - Accuracy (top1)
     - Quantized
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled    
   * - clip_text_encoder_resnet50x4 
     - 91.2
     - 88.3
     - 19
     - 32
     - 1x77x640
     - 59.1
     - 9.3
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/clip/resnet_50x4/2024-09-16/clip_text_encoder_resnet50x4.zip>`_
     - `link <https://huggingface.co/timm/resnet50x4_clip.openai>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo15h/clip_text_encoder_resnet50x4.hef>`_    
   * - clip_text_encoder_vit_large   
     - 91.93
     - 89.8
     - 16
     - 25
     - 1x77x768
     - 59.1
     - 13.85
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/clip/vit_large/2024-08-25/clip_text_encoder_vit_large.zip>`_
     - `link <https://huggingface.co/openai/clip-vit-large-patch14>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo15h/clip_text_encoder_vit_large.hef>`_  
