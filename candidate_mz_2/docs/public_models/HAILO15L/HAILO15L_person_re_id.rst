


Public Models
=============

All models were compiled using Hailo Dataflow Compiler v5.2.0.

|

Person Re-Id
============

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

Market1501
----------

|

.. list-table::
   :header-rows: 1
   :widths: 31 9 7 11 9 8 8 8 9

   
   * - Network Name
     - float rank1
     - Hardware rank1
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
   
   
   
   
   
   
   

   * - osnet_x1_0
     - 94.4
     - 93.9
     - 168
     - 304
     - | `S <https://github.com/KaiyangZhou/deep-person-reid>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/PersonReID/osnet_x1_0/2022-05-19/osnet_x1_0.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15l/osnet_x1_0_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15l/osnet_x1_0.hef>`_
     - 256x128x3
     - 2.19
     - 1.98
   
   
   
   
   
   
   

   * - repvgg_a0_person_reid_512
     - 89.9
     - 89.4
     - 611
     - 1764
     - | `S <https://github.com/DingXiaoH/RepVGG>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/MCPReID/reid/repvgg_a0_person_reid_512/2022-04-18/repvgg_a0_person_reid_512.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15l/repvgg_a0_person_reid_512_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15l/repvgg_a0_person_reid_512.hef>`_
     - 256x128x3
     - 7.68
     - 1.78