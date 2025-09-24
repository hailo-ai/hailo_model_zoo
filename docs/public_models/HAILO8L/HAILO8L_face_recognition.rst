
Public Pre-Trained Models
=========================

.. |rocket| image:: ../../images/rocket.png
  :width: 18

.. |star| image:: ../../images/star.png
  :width: 18

Here, we give the full list of publicly pre-trained models supported by the Hailo Model Zoo.

* Network available in `Hailo Benchmark <https://hailo.ai/products/ai-accelerators/hailo-8l-ai-accelerator-for-ai-light-applications/#hailo8l-benchmarks/>`_ are marked with |rocket|
* Networks available in `TAPPAS <https://github.com/hailo-ai/tappas>`_ are marked with |star|
* Benchmark and TAPPAS  networks run in performance mode
* All models were compiled using Hailo Dataflow Compiler v3.33.0

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
   * - arcface_mobilefacenet  |star|
     - 99.4
     - 0.0
     - 0
     - 1666
     - `S <https://github.com/deepinsight/insightface>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceRecognition/arcface/arcface_mobilefacenet/pretrained/2022-08-24/arcface_mobilefacenet.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/arcface_mobilefacenet.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/arcface_mobilefacenet_profiler_results_compiled.html>`_
     - 112x112x3
     - 2.04
     - 0.88
   * - arcface_r50
     - 99.7
     - 0.0
     - 0
     - 92
     - `S <https://github.com/deepinsight/insightface>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceRecognition/arcface/arcface_r50/pretrained/2022-08-24/arcface_r50.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/arcface_r50.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/arcface_r50_profiler_results_compiled.html>`_
     - 112x112x3
     - 31.0
     - 12.6
