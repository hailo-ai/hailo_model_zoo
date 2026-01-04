


Public Models
=============

All models were compiled using Hailo Dataflow Compiler v5.2.0.

|

Face Attribute
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

Celeba
------

|

.. list-table::
   :header-rows: 1
   :widths: 31 9 7 11 9 8 8 8 9

   
   * - Network Name
     - float Mean Accuracy
     - Hardware Mean Accuracy
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
   
   
   
   
   
   
   

   * - face_attr_resnet_v1_18
     - 81.2
     - 80.8
     - 2926
     - 2926
     - | `S <https://github.com/d-li14/face-attribute-prediction>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceAttr/face_attr_resnet_v1_18/2022-06-09/face_attr_resnet_v1_18.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/face_attr_resnet_v1_18_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/face_attr_resnet_v1_18.hef>`_ `RGBX <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15h/face_attr_resnet_v1_18_rgbx.hef>`_
     - 218x178x3
     - 11.74
     - 3