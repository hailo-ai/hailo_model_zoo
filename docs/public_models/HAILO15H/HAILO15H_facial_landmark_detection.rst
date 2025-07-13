
Public Pre-Trained Models
=========================

.. |rocket| image:: ../../images/rocket.png
  :width: 18

.. |star| image:: ../../images/star.png
  :width: 18

Here, we give the full list of publicly pre-trained models supported by the Hailo Model Zoo.

* Benchmark Networks are marked with |rocket|
* Networks available in `TAPPAS <https://github.com/hailo-ai/tappas>`_ are marked with |star|
* Benchmark and TAPPAS  networks run in performance mode
* All models were compiled using Hailo Dataflow Compiler v3.31.0

Link Legend

The following shortcuts are used in the table below to indicate available resources for each model:

* S – Source: Link to the model’s open-source code repository.
* PT – Pretrained: Download the pretrained model file (compressed in ZIP format).
* H, NV, X – Compiled Models: Links to the compiled model in various formats:
            * H: regular HEF with RGBX format
            * NV: HEF with NV12 format
            * X: HEF with RGBX format

* PR – Profiler Report: Download the model’s performance profiling report.



.. _Facial Landmark Detection:

-------------------------

AFLW2k3d
^^^^^^^^

.. list-table::
   :widths: 31 9 7 11 9 8 8 8 9
   :header-rows: 1

   * - Network Name
     - float mAP
     - Hardware mAP
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)       
   * - tddfa_mobilenet_v1  |star| 
     - 3.68
     - 4.04
     - 10300
     - 10268
     - `S <https://github.com/cleardusk/3DDFA_V2>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceLandmarks3d/tddfa/tddfa_mobilenet_v1/pretrained/2025-03-18/tddfa_mobilenet_v1.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/tddfa_mobilenet_v1.hef>`_/`NV <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/tddfa_mobilenet_v1_nv12.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/tddfa_mobilenet_v1_profiler_results_compiled.html>`_
     - 120x120x3
     - 3.26
     - 0.36

Face Landmark
^^^^^^^^^^^^^
    
.. list-table::
   :header-rows: 1

   * - Network Name
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G) 
   * - face_landmarks_lite   
     - 0
     - 0
     - `S <https://github.com/google-ai-edge/mediapipe>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceLandmarks3d/mediapipe/face_landmarks_lite/pretrained/2025-02-04/face_landmarks_lite.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/face_landmarks_lite.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/face_landmarks_lite_profiler_results_compiled.html>`_
     - 192x192x3
     - 0.6
     - 0.07
