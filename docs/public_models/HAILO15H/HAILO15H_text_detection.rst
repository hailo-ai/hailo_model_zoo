


Public Models
=============

* System host: Intel® Core™ i5-9400 CPU @ 2.90GHz
* Hailo Dataflow Compiler Version v5.5.0
* Measurement conditions: Measuring from the SoC, room temperature

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
   
   
   
   
   
   
   

   * - paddle_ocr_v5_mobile_detection⭐
     - 4.59
     - 4.26
     - 22.5
     - 41.0
     - | `S <https://github.com/PaddlePaddle/PaddleOCR/tree/main>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/OCR/PaddleOCR/v5/2025-08-10/PP-OCRv5_mobile_det_544x960_sim.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo15h/paddle_ocr_v5_mobile_detection_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo15h/paddle_ocr_v5_mobile_detection.hef>`_
     - 544x960x3
     - 1.2
     - 6.5
   
   
   
   
   
   
   

   * - paddle_ocr_v6_medium_detection
     - 7.13
     - 7.4
     - 19.0
     - 39.2
     - | `S <https://github.com/PaddlePaddle/PaddleOCR/tree/main>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/OCR/PaddleOCR/v6/2026-08-04/PP-OCRv6_medium_det_544x960_sim.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo15h/paddle_ocr_v6_medium_detection_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo15h/paddle_ocr_v6_medium_detection.hef>`_
     - 544x960x3
     - 15.49
     - 88.97
   
   
   
   
   
   
   

   * - paddle_ocr_v6_small_detection
     - 5.98
     - 6.01
     - 50.6
     - 97.9
     - | `S <https://github.com/PaddlePaddle/PaddleOCR/tree/main>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/OCR/PaddleOCR/v6/2026-08-04/PP-OCRv6_small_det_544x960_sim.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo15h/paddle_ocr_v6_small_detection_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo15h/paddle_ocr_v6_small_detection.hef>`_
     - 544x960x3
     - 2.45
     - 11.48
   
   
   
   
   
   
   

   * - paddle_ocr_v6_tiny_detection
     - 3.92
     - 3.94
     - 84.5
     - 178
     - | `S <https://github.com/PaddlePaddle/PaddleOCR/tree/main>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/OCR/PaddleOCR/v6/2026-08-04/PP-OCRv6_tiny_det_544x960_sim.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo15h/paddle_ocr_v6_tiny_detection_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo15h/paddle_ocr_v6_tiny_detection.hef>`_
     - 544x960x3
     - 0.43
     - 3.57