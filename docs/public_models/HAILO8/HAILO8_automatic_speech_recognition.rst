


Public Models
=============

All models were compiled using Hailo Dataflow Compiler v2.18.0.

|

Automatic Speech Recognition
============================

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




|








.. list-table::
   :header-rows: 1
   :widths: 31 11 9 8 8 8 9

   * - Network Name
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
   
   
   
   
   
   
   

   * - whisper_base_5s_encoder
     - 40.0
     - 84.7
     - | `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/whisper_base_5s_encoder_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/whisper_base_5s_encoder.hef>`_
     - 1x500x80
     - 19.85
     - 10.73
   
   
   
   
   
   
   

   * - whisper_base_5s_no_kqs_decoder
     - 48.4
     - 227
     - | `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/whisper_base_5s_no_kqs_decoder_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/whisper_base_5s_no_kqs_decoder.hef>`_
     - 1x250x512
     - 51.87
     - 3.99
   
   
   
   
   
   
   

   * - whisper_tiny_10s_encoder
     - 52.4
     - 103
     - | `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/whisper_tiny_10s_encoder_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/whisper_tiny_10s_encoder.hef>`_
     - 1x1000x80
     - 7.65
     - 9.26
   
   
   
   
   
   
   

   * - whisper_tiny_10s_no_kqs_decoder
     - 79.3
     - 271
     - | `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/whisper_tiny_10s_no_kqs_decoder_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/whisper_tiny_10s_no_kqs_decoder.hef>`_
     - 1x500x384
     - 29.45
     - 3.09