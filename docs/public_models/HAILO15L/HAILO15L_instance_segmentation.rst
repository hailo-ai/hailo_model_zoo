


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
   
   
   
   

   * - yolov5l_seg 
     - 39.5
     - 
     - 14.2
     - 15.3
     - |
       `S <https://github.com/ultralytics/yolov5>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5l/pretrained/2022-10-30/yolov5l-seg.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/yolov5l_seg.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/yolov5l_seg_profiler_results_compiled.html>`_
     - 640x640x3
     - 47.89
     - 147.88
   
   
   
   

   * - yolov5m_seg 
     - 36.9
     - 
     - 29.1
     - 32.7
     - |
       `S <https://github.com/ultralytics/yolov5>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5m/pretrained/2022-10-30/yolov5m-seg.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/yolov5m_seg.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/yolov5m_seg_profiler_results_compiled.html>`_
     - 640x640x3
     - 32.60
     - 70.94
   
   
   
   

   * - yolov5n_seg 
     - 23.2
     - 
     - 121
     - 144
     - |
       `S <https://github.com/ultralytics/yolov5>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5n/pretrained/2022-10-30/yolov5n-seg.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/yolov5n_seg.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/yolov5n_seg_profiler_results_compiled.html>`_
     - 640x640x3
     - 1.99
     - 7.1
   
   
   
   

   * - yolov5s_seg 
     - 31.3
     - 30.7
     - 59.9
     - 68.3
     - |
       `S <https://github.com/ultralytics/yolov5>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5s/pretrained/2022-10-30/yolov5s-seg.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/yolov5s_seg.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/yolov5s_seg_profiler_results_compiled.html>`_
     - 640x640x3
     - 7.61
     - 26.42
   
   
   
   

   * - yolov8n_seg 
     - 30.0
     - 29.6
     - 112
     - 140
     - |
       `S <https://github.com/ultralytics/ultralytics>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov8/yolov8n/pretrained/2023-03-06/yolov8n-seg.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/yolov8n_seg.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/yolov8n_seg_profiler_results_compiled.html>`_
     - 640x640x3
     - 3.4
     - 12.04
   
   
   
   

   * - yolov8s_seg 
     - 36.5
     - 
     - 49.1
     - 58.9
     - |
       `S <https://github.com/ultralytics/ultralytics>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov8/yolov8s/pretrained/2023-03-06/yolov8s-seg.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/yolov8s_seg.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/yolov8s_seg_profiler_results_compiled.html>`_
     - 640x640x3
     - 11.8
     - 42.6

