
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
* All models were compiled using Hailo Dataflow Compiler v3.31.0



.. _Image Denoising:

---------------

BSD68
^^^^^

.. list-table::
   :widths: 31 9 7 11 9 8 8 8 7 7 7 7
   :header-rows: 1

   * - Network Name
     - PSNR
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
   * - dncnn3
     - 31.46
     - 31.3
     - 49
     - 49
     - 321x481x1
     - 0.66
     - 205.26
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ImageDenoising/dncnn3/2023-06-15/dncnn3.zip>`_
     - `link <https://github.com/cszn/KAIR>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/dncnn3.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/dncnn3_profiler_results_compiled.html>`_

CBSD68
^^^^^^

.. list-table::
   :widths: 31 9 7 11 9 8 8 8 7 7 7 7
   :header-rows: 1

   * - Network Name
     - PSNR
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
   * - dncnn_color_blind
     - 33.87
     - 32.98
     - 49
     - 49
     - 321x481x3
     - 0.66
     - 205.97
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ImageDenoising/dncnn_color_blind/2023-06-25/dncnn_color_blind.zip>`_
     - `link <https://github.com/cszn/KAIR>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/dncnn_color_blind.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/dncnn_color_blind_profiler_results_compiled.html>`_
