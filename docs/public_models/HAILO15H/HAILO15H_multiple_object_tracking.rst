


Public Models
=============

* System host: Intel® Core™ i5-9400 CPU @ 2.90GHz
* Hailo Dataflow Compiler Version v5.5.0
* Measurement conditions: Measuring from the SoC, room temperature

|

Multiple Object Tracking
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

Mot16
-----

|








.. list-table::
   :header-rows: 1
   :widths: 31 9 7 11 9 8 8 8 9

   * - Network Name
     - float MOTA
     - Hardware MOTA
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
   
   
   
   
   
   
   

   * - bytetrack_nano_mot17
     - 70.0
     - 69.3
     - 46.0
     - 69.2
     - | `S <https://github.com/ifzhang/ByteTrack>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/MultipleObjectTracking/bytetrack_nano_mot17/pretrained/2026-09-27/bytetrack_nano_mot17.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo15h/bytetrack_nano_mot17_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo15h/bytetrack_nano_mot17.hef>`_
     - 608x1088x3
     - 0.90
     - 4.12
   
   
   
   
   
   
   

   * - bytetrack_s_mot17
     - 79.5
     - 78.1
     - 133
     - 213
     - | `S <https://github.com/ifzhang/ByteTrack>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/MultipleObjectTracking/bytetrack_s_mot17/pretrained/2026-09-27/bytetrack_s_mot17.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo15h/bytetrack_s_mot17_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo15h/bytetrack_s_mot17.hef>`_
     - 608x1088x3
     - 8.94
     - 43.21
   
   
   
   
   
   
   

   * - bytetrack_tiny_mot17
     - 77.1
     - 75.8
     - 197
     - 359
     - | `S <https://github.com/ifzhang/ByteTrack>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/MultipleObjectTracking/bytetrack_tiny_mot17/pretrained/2026-09-27/bytetrack_tiny_mot17.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo15h/bytetrack_tiny_mot17_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo15h/bytetrack_tiny_mot17.hef>`_
     - 608x1088x3
     - 5.03
     - 24.60