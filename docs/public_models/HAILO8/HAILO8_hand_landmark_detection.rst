
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



.. _Hand Landmark detection:

-----------------------

Hand Landmark
^^^^^^^^^^^^^

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
   * - hand_landmark_lite
     - 3091
     - 3091
     - 224x224x3
     - 1.01
     - 0.3
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HandLandmark/hand_landmark_lite/2023-07-18/hand_landmark_lite.zip>`_
     - `link <https://github.com/google/mediapipe>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/hand_landmark_lite.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/hand_landmark_lite_profiler_results_compiled.html>`_
