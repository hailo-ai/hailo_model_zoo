


Public Models
=============

* System host: Intel® Core™ i5-9400 CPU @ 2.90GHz
* Hailo Dataflow Compiler Version v2.19.0
* Measurement conditions: PCIe Gen 3 x 4 lanes, room temperature

|

Sentence_Embedding_Generation
=============================

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

Mteb
----

|








.. list-table::
   :header-rows: 1
   :widths: 31 9 7 11 9 8 8 8 9

   * - Network Name
     - float Retrieval@10
     - Hardware Retrieval@10
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
   
   
   
   
   
   
   

   * - all_minilm_l6_v2
     - 61.7
     - 61.6
     - 156
     - 742
     - | `S <https://github.com/huggingface/sentence-transformers/>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/all_minilm_l6_v2/2026-02-24/all_minilm_l6_v2.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/all_minilm_l6_v2_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/all_minilm_l6_v2.hef>`_
     - 16x8x384
     - 10.6
     - 2.9
   
   
   
   
   
   
   

   * - all_minilm_l6_v2_v2a
     - 96.4
     - 94.7
     - 155
     - 751
     - | `S <https://github.com/huggingface/sentence-transformers/>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/all_minilm_l6_v2/2026-02-24/all_minilm_l6_v2.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/all_minilm_l6_v2_v2a_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/all_minilm_l6_v2_v2a.hef>`_
     - 16x8x384
     - 10.6
     - 2.9