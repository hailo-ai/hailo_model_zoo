
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
* All models were compiled using Hailo Dataflow Compiler v5.0.0

Link Legend

The following shortcuts are used in the table below to indicate available resources for each model:

* S – Source: Link to the model’s open-source code repository.
* PT – Pretrained: Download the pretrained model file (compressed in ZIP format).
* H, NV, X – Compiled Models: Links to the compiled model in various formats:
            * H: regular HEF with RGB format
            * NV: HEF with NV12 format
            * X: HEF with RGBX format

* PR – Profiler Report: Download the model’s performance profiling report.



.. _Face Recognition:

----------------

LFW
^^^

.. list-table::
   :header-rows: 1

   * - Network Name
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
   * - arcface_mobilefacenet  |star|
     - 4254
     - 4254
     - `S <https://github.com/deepinsight/insightface>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceRecognition/arcface/arcface_mobilefacenet/pretrained/2022-08-24/arcface_mobilefacenet.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo15h/arcface_mobilefacenet.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo15h/arcface_mobilefacenet_profiler_results_compiled.html>`_
     - 112x112x3
     - 2.04
     - 0.88
   * - arcface_r50
     - 201
     - 530
     - `S <https://github.com/deepinsight/insightface>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceRecognition/arcface/arcface_r50/pretrained/2022-08-24/arcface_r50.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo15h/arcface_r50.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo15h/arcface_r50_profiler_results_compiled.html>`_
     - 112x112x3
     - 31.0
     - 12.6

N/A
^^^

.. list-table::
   :header-rows: 1

   * - Network Name
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
   * - lprnet  |star|
     - 307
     - 307
     - `S <N/A>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/LPR/ocr/lprnet/2022-03-09/lprnet.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo15h/lprnet.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo15h/lprnet_profiler_results_compiled.html>`_
     - 75x300x3
     - 7.14
     - 36.54
