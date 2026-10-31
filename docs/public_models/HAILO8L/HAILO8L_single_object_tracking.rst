


Public Models
=============

* System host: Intel® Core™ i5-9400 CPU @ 2.90GHz
* Hailo Dataflow Compiler Version v2.20.0
* Measurement conditions: PCIe Gen 3 x 4 lanes, room temperature

|

Single Object Tracking
======================

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

Lasot
-----

|








.. list-table::
   :header-rows: 1
   :widths: 31 9 7 11 9 8 8 8 9

   * - Network Name
     - float Success-plot AUC
     - Hardware Success-plot AUC
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
   
   
   
   
   
   
   

   * - stark_lightning_repvgg_a0
     - 56.9
     - 55.2
     - 224
     - 553
     - | `S <https://github.com/researchmm/Stark>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SingleObjectTracking/stark_lightning_repvgg_a0/pretrained/2026-08-03/stark_lightning_repvgg_a0.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8l/stark_lightning_repvgg_a0_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8l/stark_lightning_repvgg_a0.hef>`_
     - 320x320x3
     - 4.62
     - 3.48