


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
* All models were compiled using Hailo Dataflow Compiler v5.1.0

Link Legend

The following shortcuts are used in the table below to indicate available resources for each model:

* S – Source: Link to the model’s open-source code repository.
* PT – Pretrained: Download the pretrained model file (compressed in ZIP format).
* H, NV, X – Compiled Models: Links to the compiled model in various formats:
            * H: regular HEF with RGBX format
            * NV: HEF with NV12 format
            * X: HEF with RGBX format

* PR – Profiler Report: Download the model’s performance profiling report.

Nyu
===

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
   
   
   
   

   * - fast_depth_nv12_fhd 
     - 0.61
     - 0.63
     - 96.2
     - 96.2
     - |
       `S <https://github.com/dwofk/fast-depth>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/DepthEstimation/indoor/fast_depth/pretrained/2021-10-18/fast_depth.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/fast_depth_nv12_fhd.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/fast_depth_nv12_fhd_profiler_results_compiled.html>`_
     - 540x1920x3
     - 1.35
     - 0.77



Widerface
=========

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
   
   
   
   

   * - lightface_slim_nv12_fhd 
     - 39.0
     - 38.3
     - 96.2
     - 96.1
     - |
       `S <https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/lightface_slim/2021-07-18/lightface_slim.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/lightface_slim_nv12_fhd.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/lightface_slim_nv12_fhd_profiler_results_compiled.html>`_
     - 540x1920x3
     - 0.26
     - 0.16
   
   
   
   

   * - scrfd_10g_nv12_fhd 
     - 81.4
     - 81.3
     - 96.2
     - 96.2
     - |
       `S <https://github.com/deepinsight/insightface>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/scrfd/scrfd_10g/pretrained/2022-09-07/scrfd_10g.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/scrfd_10g_nv12_fhd.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/scrfd_10g_nv12_fhd_profiler_results_compiled.html>`_
     - 540x1920x3
     - 4.23
     - 26.74



Coco
====

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
   
   
   
   

   * - yolov5m_wo_spp_nv12_fhd 
     - 43.2
     - 41.4
     - 53.2
     - 67.9
     - |
       `S <https://github.com/ultralytics/yolov5/releases/tag/v2.0>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5m/pretrained/2023-04-25/yolov5m_wo_spp.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/yolov5m_wo_spp_nv12_fhd.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/yolov5m_wo_spp_nv12_fhd_profiler_results_compiled.html>`_
     - 540x1920x3
     - 22.67
     - 52.98
   
   
   
   

   * - yolov5n_seg_nv12_fhd 
     - 23.4
     - 23.0
     - 77.6
     - 82.4
     - |
       `S <https://github.com/ultralytics/yolov5>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5n/pretrained/2022-10-30/yolov5n-seg.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/yolov5n_seg_nv12_fhd.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/yolov5n_seg_nv12_fhd_profiler_results_compiled.html>`_
     - 540x1920x3
     - 1.99
     - 7.1
   
   
   
   

   * - yolov5s_personface_nv12_fhd 
     - 47.2
     - 45.9
     - 67.2
     - 79.1
     - |
       `S <https://github.com/ultralytics/yolov5/releases/tag/v2.0>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/MCPReID/personface_detector/yolov5s_personface/2023-04-25/yolov5s_personface.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/yolov5s_personface_nv12_fhd.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/yolov5s_personface_nv12_fhd_profiler_results_compiled.html>`_
     - 540x1920x3
     - 7.25
     - 16.76

