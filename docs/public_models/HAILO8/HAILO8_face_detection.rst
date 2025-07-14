
Public Pre-Trained Models
=========================

.. |rocket| image:: ../../images/rocket.png
  :width: 18

.. |star| image:: ../../images/star.png
  :width: 18

Here, we give the full list of publicly pre-trained models supported by the Hailo Model Zoo.

* Network available in `Hailo Benchmark <https://hailo.ai/products/ai-accelerators/hailo-8-ai-accelerator/#hailo8-benchmarks/>`_ are marked with |rocket|
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



.. _Face Detection:

--------------

WiderFace
^^^^^^^^^

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
   * - lightface_slim  |star| 
     - 39.7
     - 39.27
     - 4205
     - 4205
     - `S <https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/lightface_slim/2021-07-18/lightface_slim.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/lightface_slim.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/lightface_slim_profiler_results_compiled.html>`_
     - 240x320x3
     - 0.26
     - 0.16    
   * - retinaface_mobilenet_v1   
     - 81.27
     - 81.23
     - 163
     - 163
     - `S <https://github.com/biubug6/Pytorch_Retinaface>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/retinaface_mobilenet_v1_hd/2023-07-18/retinaface_mobilenet_v1_hd.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/retinaface_mobilenet_v1.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/retinaface_mobilenet_v1_profiler_results_compiled.html>`_
     - 736x1280x3
     - 3.49
     - 25.14    
   * - scrfd_10g   
     - 82.13
     - 82.12
     - 440
     - 440
     - `S <https://github.com/deepinsight/insightface>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/scrfd/scrfd_10g/pretrained/2022-09-07/scrfd_10g.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/scrfd_10g.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/scrfd_10g_profiler_results_compiled.html>`_
     - 640x640x3
     - 4.23
     - 26.74    
   * - scrfd_2.5g   
     - 76.59
     - 76.39
     - 1058
     - 1058
     - `S <https://github.com/deepinsight/insightface>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/scrfd/scrfd_2.5g/pretrained/2022-09-07/scrfd_2.5g.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/scrfd_2.5g.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/scrfd_2.5g_profiler_results_compiled.html>`_
     - 640x640x3
     - 0.82
     - 6.88    
   * - scrfd_500m   
     - 68.98
     - 68.73
     - 1332
     - 1332
     - `S <https://github.com/deepinsight/insightface>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/scrfd/scrfd_500m/pretrained/2022-09-07/scrfd_500m.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/scrfd_500m.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/scrfd_500m_profiler_results_compiled.html>`_
     - 640x640x3
     - 0.63
     - 1.5
