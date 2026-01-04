


Public Models
=============

All models were compiled using Hailo Dataflow Compiler v5.2.0.

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
   
   
   
   
   
   
   

   * - clip_resnet_50x4_image_encoder
     - 50.3
     - 49.9
     - 27.5
     - 45.8
     - | `S <https://github.com/openai/CLIP>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_resnet_50x4/image_encoder/pretrained/2023-03-09/clip_resnet_50x4.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15l/clip_resnet_50x4_image_encoder_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15l/clip_resnet_50x4_image_encoder.hef>`_
     - 288x288x3
     - 87.0
     - 41.3
   
   
   
   
   
   
   

   * - clip_vit_b_32_image_encoder
     - 65.3
     - 63.6
     - 50.5
     - 197
     - | `S <https://github.com/openai/CLIP>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_base_patch32_224/image_encoder/pretrained/2023-03-09/clip_vit_b_32.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15l/clip_vit_b_32_image_encoder_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15l/clip_vit_b_32_image_encoder.hef>`_
     - 224x224x3
     - 87.8
     - 8.8