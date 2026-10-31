


Public Models
=============

* System host: Intel® Core™ i5-9400 CPU @ 2.90GHz
* Hailo Dataflow Compiler Version v5.5.0
* Measurement conditions: PCIe Gen 3 x 4 lanes, room temperature

|

Text_Classification
===================

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

Text Classification
-------------------

|








.. list-table::
   :header-rows: 1
   :widths: 31 9 7 11 9 8 8 8 9

   * - Network Name
     - float Accuracy
     - Hardware Accuracy
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
   
   
   
   
   
   
   

   * - bert_base_uncased
     - 63.7
     - 60.8
     - 16.5
     - 18.2
     - | `S <https://huggingface.co/google-bert/bert-base-uncased>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/LanguageModeling/bert_base_uncased/pretrained/2026-01-19/bert_base_uncased.zip>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/bert_base_uncased.hef>`_
     - 512x768
     - 109.1
     - 121.4