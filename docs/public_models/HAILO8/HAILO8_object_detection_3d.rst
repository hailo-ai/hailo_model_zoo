
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
* All models were compiled using Hailo Dataflow Compiler v3.29.0



.. _Object Detection_3D:

----------------

nuScenes 2019
^^^^

.. list-table::
   :widths: 31 9 7 11 9 8 8 8 7 7 7
   :header-rows: 1

   * - Network Name
     - mAP
     - Quantized
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
   * - petrv2_repvggB0_backbone_pp_800x320
     - NA
     - NA
     - 390
     - 390
     - 320x800x3
     - 13.3
     - 31.9
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection3d/Detection3d-Nuscenes/petrv2/pretrained/2024-08-13/petrv2_repvggB0_BN1d_2d_backbone_800x320_pp.zip>`_
     - `link <https://github.com/megvii-research/petr>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo8/petrv2_repvggB0_backbone_pp_800x320.hef>`_
   * - petrv2_repvggB0_transformer_pp_800x320
     - 25
     - 2.2
     - 35
     - 35
     - 12x250x1280, 12x250x256
     - 6.7
     - 11.7
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection3d/Detection3d-Nuscenes/petrv2/pretrained/2024-08-13/petrv2_repvggB0_BN1d_2d_transformer_800x320_pp.zip>`_
     - `link <https://github.com/megvii-research/petr>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo8/petrv2_repvggB0_transformer_pp_800x320.hef>`_
