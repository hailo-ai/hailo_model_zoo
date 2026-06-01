


Public Models
=============

* System host: Intel® Core™ i5-9400 CPU @ 2.90GHz
* Hailo Dataflow Compiler Version v2.19.0
* Measurement conditions: PCIe Gen 3 x 4 lanes, room temperature

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
     - 41.0
     - 105
     - | `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/whisper_base_5s_encoder_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/whisper_base_5s_encoder.hef>`_
     - 1x500x80
     - 19.85
     - 10.73
   
   
   
   
   
   
   

   * - whisper_base_5s_no_kqs_decoder
     - 41.8
     - 147
     - | `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/whisper_base_5s_no_kqs_decoder_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/whisper_base_5s_no_kqs_decoder.hef>`_
     - 1x250x512
     - 51.87
     - 3.99
   
   
   
   
   
   
   

   * - whisper_tiny_10s_encoder
     - 55.4
     - 136
     - | `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/whisper_tiny_10s_encoder_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/whisper_tiny_10s_encoder.hef>`_
     - 1x1000x80
     - 7.65
     - 9.26
   
   
   
   
   
   
   

   * - whisper_tiny_10s_no_kqs_decoder
     - 44.7
     - 109
     - | `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/whisper_tiny_10s_no_kqs_decoder_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/whisper_tiny_10s_no_kqs_decoder.hef>`_
     - 1x500x384
     - 29.45
     - 3.09