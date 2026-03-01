


Public Models
=============

All models were compiled using Hailo Dataflow Compiler v5.2.0.

|

Image Denoising
===============

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

Bsd68
-----

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
   
   
   
   
   
   
   

   * - dncnn3
     - 31.5
     - 31.3
     - 50.0
     - 50.0
     - | `S <https://github.com/cszn/KAIR>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ImageDenoising/dncnn3/2023-06-15/dncnn3.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/dncnn3_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/dncnn3.hef>`_
     - 321x481x1
     - 0.66
     - 205.26

|

Cbsd68
------

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
   
   
   
   
   
   
   

   * - dncnn_color_blind
     - 33.9
     - 33.0
     - 50.0
     - 50.0
     - | `S <https://github.com/cszn/KAIR>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ImageDenoising/dncnn_color_blind/2023-06-25/dncnn_color_blind.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/dncnn_color_blind_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/dncnn_color_blind.hef>`_
     - 321x481x3
     - 0.66
     - 205.97