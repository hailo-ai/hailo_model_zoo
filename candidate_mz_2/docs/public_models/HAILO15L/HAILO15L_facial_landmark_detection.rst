


Public Models
=============

All models were compiled using Hailo Dataflow Compiler v5.2.0.

|

Facial Landmark Detection
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

Face Landmark
-------------

|

.. list-table::
   :header-rows: 1
   :widths: 31 9 7 11 9 8 8 8 9

   
   * - Network Name
     - float NME
     - Hardware NME
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
   
   
   
   
   
   
   

   * - face_landmarks_lite
     - 
     - 
     - 1103
     - 2769
     - | `S <https://github.com/google-ai-edge/mediapipe>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceLandmarks3d/mediapipe/face_landmarks_lite/pretrained/2025-02-04/face_landmarks_lite.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15l/face_landmarks_lite_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15l/face_landmarks_lite.hef>`_
     - 192x192x3
     - 0.6
     - 0.07

|

Aflw2K3D
--------

|

.. list-table::
   :header-rows: 1
   :widths: 31 9 7 11 9 8 8 8 9

   
   * - Network Name
     - float NME
     - Hardware NME
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
   
   
   
   
   
   
   

   * - tddfa_mobilenet_v1
     - 3.68
     - 4.03
     - 952
     - 2963
     - | `S <https://github.com/cleardusk/3DDFA_V2>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceLandmarks3d/tddfa/tddfa_mobilenet_v1/pretrained/2025-03-18/tddfa_mobilenet_v1.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15l/tddfa_mobilenet_v1_profiler_results_compiled.html>`_
         `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo15l/tddfa_mobilenet_v1.hef>`_
     - 120x120x3
     - 3.26
     - 0.36