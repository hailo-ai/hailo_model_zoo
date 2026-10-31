


Public Models
=============

* System host: Intel® Core™ i5-9400 CPU @ 2.90GHz
* Hailo Dataflow Compiler Version v5.5.0
* Measurement conditions: Measuring from the SoC, room temperature

|

Age/Gender Classification
=========================

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

Lagenda
-------

|








.. list-table::
   :header-rows: 1
   :widths: 31 9 7 11 9 8 8 9

   * - Network Name
     - float 
     - Hardware 
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - OPS (G)
   
   
   
   
   
   
   

   * - mivolo_d1
     - 91.7
     - 91.8
     - 17.1
     - 40.1
     - | `S <https://github.com/WildChlamydia/MiVOLO>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/AgeGender/mivolo_d1/2026-07-23/mivolo_d1.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo15h/mivolo_d1_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo15h/mivolo_d1.hef>`_
     - 224x224x6
     - 27.3