


Public Models
=============

All models were compiled using Hailo Dataflow Compiler v5.2.0.

|

Zero-Shot Depth Estimation
==========================

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

Zero Shot Depth Estimation
--------------------------

|

.. list-table::
   :header-rows: 1
   :widths: 31 9 7 11 9 8 8 8 9

   
   * - Network Name
     - float AbsRel
     - Hardware AbsRel
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
   
   
   
   
   
   
   

   * - depth_anything_v2_vits
     - 0.15
     - 0.19
     - 51.4
     - 119
     - | `S <https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/DepthEstimation/Depth_Anything/v2/vits/pretrained/2025-07-09/depth_anything_v2_vits_224X224_sim_hf.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/depth_anything_v2_vits_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/depth_anything_v2_vits.hef>`_
     - 224x224x3
     - 24.2
     - 16.7
   
   
   
   
   
   
   

   * - depth_anything_vits
     - 0.13
     - 0.16
     - 53.2
     - 118
     - | `S <https://huggingface.co/LiheYoung/depth-anything-small-hf>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/DepthEstimation/Depth_Anything/v1/vits/pretrained/2025-07-09/depth_anything_vits_224X224_sim_hf.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/depth_anything_vits_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/depth_anything_vits.hef>`_
     - 224x224x3
     - 24.2
     - 16.7