


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
   
   
   
   

   * - yolov5m_vehicles 
     - 46.1
     - 
     - 30.9
     - 34.9
     - |
       `S <https://github.com/ultralytics/yolov5/releases/tag/v2.0>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/LPR/vehicle_detector/yolov5m_vehicles/2023-04-25/yolov5m_vehicles.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/yolov5m_vehicles.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/yolov5m_vehicles_profiler_results_compiled.html>`_
     - 1080x1920x3
     - 21.47
     - 51.19
   
   
   
   

   * - yolov5s_personface 
     - 46.9
     - 45.6
     - 88.8
     - 109
     - |
       `S <https://github.com/ultralytics/yolov5/releases/tag/v2.0>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/MCPReID/personface_detector/yolov5s_personface/2023-04-25/yolov5s_personface.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/yolov5s_personface.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/yolov5s_personface_profiler_results_compiled.html>`_
     - 640x640x3
     - 7.25
     - 16.71

