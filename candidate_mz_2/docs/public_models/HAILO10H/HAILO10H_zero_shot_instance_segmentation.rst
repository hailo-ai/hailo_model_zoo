


Public Models
=============

All models were compiled using Hailo Dataflow Compiler v5.2.0.

|

Zero-Shot Instance Segmentation
===============================

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

Coco
----

|

.. list-table::
   :header-rows: 1
   :widths: 31 9 7 11 9 8 8 8 9

   
   * - Network Name
     - float AR1000
     - Hardware AR1000
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
   
   
   
   
   
   
   

   * - fast_sam_s
     - 40.1
     - 38.9
     - 35.6
     - 40.8
     - | `S <https://github.com/CASIA-IVA-Lab/FastSAM>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SegmentAnything/coco/fast_sam/fast_sam_s/pretrained/2023-03-06/fast_sam_s.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/fast_sam_s_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/fast_sam_s.hef>`_
     - 640x640x3
     - 11.1
     - 42.4