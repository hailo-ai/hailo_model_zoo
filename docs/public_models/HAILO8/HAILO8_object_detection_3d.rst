


Public Models
=============

All models were compiled using Hailo Dataflow Compiler v2.18.0.

|

Object Detection 3D
===================

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

Nuscenes 2019
-------------

|








.. list-table::
   :header-rows: 1
   :widths: 31 9 7 11 9 8 8 8 9

   * - Network Name
     - float 
     - Hardware 
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
   
   
   
   
   
   
   

   * - petrv2_repvggB0_backbone_pp_800x320
     - 
     - 
     - 573
     - 573
     - | `S <https://github.com/megvii-research/petr>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection3d/Detection3d-Nuscenes/petrv2/pretrained/2024-09-30/petrv2_repvggB0_BN1d_2d_backbone_800x320_pp.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/petrv2_repvggB0_backbone_pp_800x320_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/petrv2_repvggB0_backbone_pp_800x320.hef>`_
     - 320x800x3
     - 13.39
     - 31.19
   
   
   
   
   
   
   

   * - petrv2_repvggB0_transformer_pp_800x320
     - 25.9
     - 23.3
     - 26.4
     - 33.3
     - | `S <https://github.com/megvii-research/petr>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection3d/Detection3d-Nuscenes/petrv2/pretrained/2024-08-13/petrv2_repvggB0_BN1d_2d_transformer_800x320_pp.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/petrv2_repvggB0_transformer_pp_800x320_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/petrv2_repvggB0_transformer_pp_800x320.hef>`_
     - 12x250x1280
     - 6.7
     - 11.7