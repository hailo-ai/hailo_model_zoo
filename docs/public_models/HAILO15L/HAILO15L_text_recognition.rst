


Public Models
=============

* System host: Intel® Core™ i5-9400 CPU @ 2.90GHz
* Hailo Dataflow Compiler Version v5.5.0
* Measurement conditions: Measuring from the SoC, room temperature

|

Text Recognition
================

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

Icdar15 Recognition
-------------------

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
   
   
   
   
   
   
   

   * - paddle_ocr_v5_mobile_recognition
     - 62.6
     - 59.7
     - 108
     - 247
     - | `S <https://github.com/PaddlePaddle/PaddleOCR/tree/main>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/OCR/PaddleOCR/v5/2025-08-10/PP-OCRv5_mobile_rec_48x320_sim.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo15l/paddle_ocr_v5_mobile_recognition_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo15l/paddle_ocr_v5_mobile_recognition.hef>`_
     - 48x320x3
     - 4.1
     - 1.5
   
   
   
   
   
   
   

   * - paddle_ocr_v6_medium_recognition
     - 83.2
     - 80.7
     - 65.3
     - 155
     - | `S <https://github.com/PaddlePaddle/PaddleOCR/tree/main>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/OCR/PaddleOCR/v6/2026-07-22/PP-OCRv6_medium_rec_48x320_sim.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo15l/paddle_ocr_v6_medium_recognition_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo15l/paddle_ocr_v6_medium_recognition.hef>`_
     - 48x320x3
     - 19
     - 11.37
   
   
   
   
   
   
   

   * - paddle_ocr_v6_small_recognition
     - 78.6
     - 77.3
     - 191
     - 515
     - | `S <https://github.com/PaddlePaddle/PaddleOCR/tree/main>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/OCR/PaddleOCR/v6/2026-07-22/PP-OCRv6_small_rec_48x320_sim.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo15l/paddle_ocr_v6_small_recognition_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo15l/paddle_ocr_v6_small_recognition.hef>`_
     - 48x320x3
     - 5.2
     - 2.18
   
   
   
   
   
   
   

   * - paddle_ocr_v6_tiny_recognition
     - 63.7
     - 56.9
     - 637
     - 1786
     - | `S <https://github.com/PaddlePaddle/PaddleOCR/tree/main>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/OCR/PaddleOCR/v6/2026-07-22/PP-OCRv6_tiny_rec_48x320_sim.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo15l/paddle_ocr_v6_tiny_recognition_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo15l/paddle_ocr_v6_tiny_recognition.hef>`_
     - 48x320x3
     - 1.1
     - 0.37