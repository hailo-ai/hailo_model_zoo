


Public Models
=============

All models were compiled using Hailo Dataflow Compiler v2.18.0.

|

Face Detection
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

Widerface
---------

|








.. list-table::
   :header-rows: 1
   :widths: 31 9 7 11 9 8 8 8 9

   * - Network Name
     - float mAP
     - Hardware mAP
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
   
   
   
   
   
   
   

   * - lightface_slim
     - 39.7
     - 39.3
     - 3622
     - 3612
     - | `S <https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/lightface_slim/2021-07-18/lightface_slim.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/lightface_slim_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/lightface_slim.hef>`_
     - 240x320x3
     - 0.26
     - 0.16
   
   
   
   
   
   
   

   * - retinaface_mobilenet_v1
     - 81.3
     - 81.2
     - 75.6
     - 104
     - | `S <https://github.com/biubug6/Pytorch_Retinaface>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/retinaface_mobilenet_v1_hd/2023-07-18/retinaface_mobilenet_v1_hd.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/retinaface_mobilenet_v1_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/retinaface_mobilenet_v1.hef>`_
     - 736x1280x3
     - 3.49
     - 25.14
   
   
   
   
   
   
   

   * - scrfd_10g⭐
     - 82.1
     - 82.1
     - 149
     - 149
     - | `S <https://github.com/deepinsight/insightface>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/scrfd/scrfd_10g/pretrained/2022-09-07/scrfd_10g.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/scrfd_10g_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/scrfd_10g.hef>`_
     - 640x640x3
     - 4.23
     - 26.74
   
   
   
   
   
   
   

   * - scrfd_2.5g⭐
     - 76.6
     - 76.4
     - 311
     - 311
     - | `S <https://github.com/deepinsight/insightface>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/scrfd/scrfd_2.5g/pretrained/2022-09-07/scrfd_2.5g.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/scrfd_2.5g_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/scrfd_2.5g.hef>`_
     - 640x640x3
     - 0.82
     - 6.88
   
   
   
   
   
   
   

   * - scrfd_500m
     - 69.0
     - 68.7
     - 588
     - 588
     - | `S <https://github.com/deepinsight/insightface>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/scrfd/scrfd_500m/pretrained/2022-09-07/scrfd_500m.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/scrfd_500m_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/scrfd_500m.hef>`_
     - 640x640x3
     - 0.63
     - 1.5