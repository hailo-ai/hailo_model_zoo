
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



.. _zero-shot instance segmentation:

-------------------------------

COCO
^^^^

.. list-table::
   :widths: 31 9 7 11 9 8 8 8 7 7 7 7
   :header-rows: 1

   * - Network Name
     - AR1000
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
   * - fast_sam_s   
     - 40.1
     - 39.01
     - 12
     - 13
     - 640x640x3
     - 11.1
     - 42.4
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SegmentAnything/coco/fast_sam/fast_sam_s/pretrained/2023-03-06/fast_sam_s.zip>`_
     - `link <https://github.com/CASIA-IVA-Lab/FastSAM>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo15m/fast_sam_s.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo15m/fast_sam_s_profiler_results_compiled.html>`_
