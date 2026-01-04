


Public Models
=============

All models were compiled using Hailo Dataflow Compiler v5.2.0.

|

Text Detection
==============

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

Icdar15 Detection
-----------------

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
   
   
   
   
   
   
   

   * - paddle_ocr_v5_mobile_detection
     - 4.59
     - 4.36
     - 22.2
     - 40.7
     - | `S <https://github.com/PaddlePaddle/PaddleOCR/tree/main>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/OCR/PaddleOCR/v5/2025-08-10/PP-OCRv5_mobile_det_544x960_sim.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/paddle_ocr_v5_mobile_detection_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/paddle_ocr_v5_mobile_detection.hef>`_
     - 544x960x3
     - 1.2
     - 6.5