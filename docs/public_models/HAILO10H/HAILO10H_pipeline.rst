


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

Celeba
======

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
   
   
   
   

   * - face_attr_resnet_v1_18_rgbx 
     - 81.2
     - 80.8
     - 2369
     - 2369
     - |
       `S <https://github.com/d-li14/face-attribute-prediction>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceAttr/face_attr_resnet_v1_18/2022-06-09/face_attr_resnet_v1_18.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/face_attr_resnet_v1_18_rgbx.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/face_attr_resnet_v1_18_rgbx_profiler_results_compiled.html>`_
     - 218x178x4
     - 11.74
     - 3



Lpr Net Dataset
===============

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
   
   
   
   

   * - lprnet_yuy2 
     - 54.2
     - 54.0
     - 308
     - 308
     - |
       `S <>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/LPR/ocr/lprnet_304x75/2022-05-01/lprnet_304x75.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/lprnet_yuy2.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/lprnet_yuy2_profiler_results_compiled.html>`_
     - 75x304x2
     - 7.14
     - 37.01



Peta
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
   
   
   
   

   * - person_attr_resnet_v1_18_rgbx 
     - 82.5
     - 82.6
     - 2352
     - 2239
     - |
       `S <https://github.com/dangweili/pedestrian-attribute-recognition-pytorch>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/person_attr_resnet_v1_18/pretrained/2022-06-11/person_attr_resnet_v1_18.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/person_attr_resnet_v1_18_rgbx.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/person_attr_resnet_v1_18_rgbx_profiler_results_compiled.html>`_
     - 224x224x4
     - 11.19
     - 3.64



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
   
   
   
   

   * - retinaface_mobilenet_v1_rgbx 
     - 81.3
     - 81.2
     - 98.8
     - 135
     - |
       `S <https://github.com/biubug6/Pytorch_Retinaface>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/retinaface_mobilenet_v1_hd/2023-07-18/retinaface_mobilenet_v1_hd.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/retinaface_mobilenet_v1_rgbx.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/retinaface_mobilenet_v1_rgbx_profiler_results_compiled.html>`_
     - 736x1280x4
     - 3.49
     - 25.14



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
   
   
   
   

   * - tiny_yolov4_license_plates 
     - 74.1
     - 74.0
     - 1188
     - 1189
     - |
       `S <https://github.com/ultralytics/yolov5/releases/tag/v2.0>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/LPR/lp_detector/tiny_yolov4_license_plates/2021-12-23/tiny_yolov4_license_plates.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/tiny_yolov4_license_plates.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/tiny_yolov4_license_plates_profiler_results_compiled.html>`_
     - 416x416x3
     - 5.87
     - 6.8
   
   
   
   

   * - tiny_yolov4_license_plates_yuy2 
     - 74.1
     - 74.4
     - 1248
     - 1248
     - |
       `S <https://github.com/ultralytics/yolov5/releases/tag/v2.0>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/LPR/lp_detector/tiny_yolov4_license_plates/2021-12-23/tiny_yolov4_license_plates.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/tiny_yolov4_license_plates_yuy2.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/tiny_yolov4_license_plates_yuy2_profiler_results_compiled.html>`_
     - 416x416x2
     - 5.87
     - 6.8
   
   
   
   

   * - yolov5m_vehicles 
     - 46.1
     - 43.4
     - 51.5
     - 66.9
     - |
       `S <https://github.com/ultralytics/yolov5/releases/tag/v2.0>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/LPR/vehicle_detector/yolov5m_vehicles/2023-04-25/yolov5m_vehicles.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/yolov5m_vehicles.hef>`_
       
       `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/yolov5m_vehicles_rgbx.hef>`_
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/yolov5m_vehicles_profiler_results_compiled.html>`_
     - 1080x1920x3
     - 21.47
     - 51.19
   
   
   
   

   * - yolov5m_vehicles_yuy2 
     - 46.1
     - 43.4
     - 51.4
     - 66.7
     - |
       `S <https://github.com/ultralytics/yolov5/releases/tag/v2.0>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/LPR/vehicle_detector/yolov5m_vehicles/2023-04-25/yolov5m_vehicles.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/yolov5m_vehicles_yuy2.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/yolov5m_vehicles_yuy2_profiler_results_compiled.html>`_
     - 1080x1920x2
     - 21.47
     - 51.19
   
   
   
   

   * - yolov5m_wo_spp_yuy2 
     - 43.0
     - 41.7
     - 73.5
     - 102
     - |
       `S <https://github.com/ultralytics/yolov5/releases/tag/v2.0>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5m/pretrained/2023-04-25/yolov5m_wo_spp.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/yolov5m_wo_spp_yuy2.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/yolov5m_wo_spp_yuy2_profiler_results_compiled.html>`_
     - 720x1280x2
     - 22.67
     - 52.89
   
   
   
   

   * - yolov5s_personface 
     - 47.6
     - 46.0
     - 242
     - 237
     - |
       `S <https://github.com/ultralytics/yolov5/releases/tag/v2.0>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/MCPReID/personface_detector/yolov5s_personface/2023-04-25/yolov5s_personface.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/yolov5s_personface.hef>`_
       
       `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/yolov5s_personface_rgbx.hef>`_
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/yolov5s_personface_profiler_results_compiled.html>`_
     - 640x640x3
     - 7.25
     - 16.71
   
   
   
   

   * - yolov5s_personface_rgbx 
     - 47.6
     - 45.7
     - 230
     - 449
     - |
       `S <https://github.com/ultralytics/yolov5/releases/tag/v2.0>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/MCPReID/personface_detector/yolov5s_personface/2023-04-25/yolov5s_personface.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/yolov5s_personface_rgbx.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15h/yolov5s_personface_rgbx_profiler_results_compiled.html>`_
     - 640x640x4
     - 7.25
     - 16.71

