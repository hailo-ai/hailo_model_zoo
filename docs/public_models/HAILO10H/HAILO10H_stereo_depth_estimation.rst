


Public Models
=============

All models were compiled using Hailo Dataflow Compiler v5.2.0.

|

Stereo Depth Estimation
=======================

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

Kitti Stereo 2015
-----------------

|

.. list-table::
   :header-rows: 1
   :widths: 31 9 7 11 9 8 8 8 9

   
   * - Network Name
     - float EPE
     - Hardware EPE
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
   
   
   
   
   
   
   

   * - stereonet
     - 8.22
     - 10.3
     - 10.0
     - 10.3
     - | `S <https://github.com/nivosco/StereoNet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/DisparityEstimation/stereonet/pretrained/2023-05-31/stereonet.zip>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/stereonet.hef>`_
     - 368x1232x3
     - 623.1
     - 112.2
