
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
* All models were compiled using Hailo Dataflow Compiler v3.30.0



.. _Object Detection 3D:

-------------------

nuScenes 2019
^^^^^^^^^^^^^

.. list-table::
   :widths: 31 9 7 11 9 8 8 8 7 7 7 7
   :header-rows: 1

   * - Network Name
     - mAP
     - HW Accuracy
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
     - Profile Html    
   * - petrv2_repvggB0_transformer_pp_800x320   
     - 25.87
     - 23.39
     - 0
     - 0
     - 12x250x1280, 12x250x256
     - 6.7
     - 11.7
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection3d/Detection3d-Nuscenes/petrv2/pretrained/2024-08-13/petrv2_repvggB0_BN1d_2d_transformer_800x320_pp.zip>`_
     - `link <https://github.com/megvii-research/petr>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo15m/petrv2_repvggB0_transformer_pp_800x320.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo15m/petrv2_repvggB0_transformer_pp_800x320_profiler_results_compiled.html>`_    
.. list-table::
   :header-rows: 1

   * - Network Name
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
     - Profile Html    
   * - petrv2_repvggB0_backbone_pp_800x320   
     - 0
     - 0
     - 320x800x3
     - 13.39
     - 31.19
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection3d/Detection3d-Nuscenes/petrv2/pretrained/2024-09-30/petrv2_repvggB0_BN1d_2d_backbone_800x320_pp.zip>`_
     - `link <https://github.com/megvii-research/petr>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo15m/petrv2_repvggB0_backbone_pp_800x320.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo15m/petrv2_repvggB0_backbone_pp_800x320_profiler_results_compiled.html>`_
