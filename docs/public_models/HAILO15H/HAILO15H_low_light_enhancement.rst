


Public Models
=============

All models were compiled using Hailo Dataflow Compiler v5.2.0.

|

Low Light Enhancement
=====================

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

Lol
---

|

.. list-table::
   :header-rows: 1
   :widths: 31 9 7 11 9 8 8 8 9

   
   * - Network Name
     - float PSNR
     - Hardware PSNR
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
   
   
   
   
   
   
   

   * - zero_dce
     - 16.2
     - 16.2
     - 200
     - 200
     - | `S <Internal>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/LowLightEnhancement/LOL/zero_dce/pretrained/2023-04-23/zero_dce.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/zero_dce_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/zero_dce.hef>`_
     - 400x600x3
     - 0.21
     - 38.2
   
   
   
   
   
   
   

   * - zero_dce_pp
     - 16.0
     - 15.9
     - 96.3
     - 96.3
     - | `S <Internal>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/LowLightEnhancement/LOL/zero_dce_pp/pretrained/2023-07-03/zero_dce_pp.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/zero_dce_pp_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/zero_dce_pp.hef>`_
     - 400x600x3
     - 0.02
     - 4.84