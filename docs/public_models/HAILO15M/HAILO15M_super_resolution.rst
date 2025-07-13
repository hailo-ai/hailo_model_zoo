
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



.. _Super Resolution:

----------------

BSD100
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
   * - espcn_x2   
     - 31.22
     - 30.77
     - 1636
     - 1636
     - 156x240x1
     - 0.02
     - 1.6
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SuperResolution/espcn/espcn_x2/2022-08-02/espcn_x2.zip>`_
     - `link <https://github.com/Lornatang/ESPCN-PyTorch>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo15m/espcn_x2.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo15m/espcn_x2_profiler_results_compiled.html>`_    
   * - espcn_x3   
     - 28.29
     - 28.12
     - 1924
     - 1924
     - 104x160x1
     - 0.02
     - 0.76
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SuperResolution/espcn/espcn_x3/2022-08-02/espcn_x3.zip>`_
     - `link <https://github.com/Lornatang/ESPCN-PyTorch>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo15m/espcn_x3.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo15m/espcn_x3_profiler_results_compiled.html>`_    
   * - espcn_x4   
     - 26.83
     - 26.65
     - 1907
     - 1908
     - 78x120x1
     - 0.02
     - 0.46
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SuperResolution/espcn/espcn_x4/2022-08-02/espcn_x4.zip>`_
     - `link <https://github.com/Lornatang/ESPCN-PyTorch>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo15m/espcn_x4.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo15m/espcn_x4_profiler_results_compiled.html>`_

N/A
^^^

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
   * - real_esrgan_x2   
     - 28.27
     - 27.66
     - 1
     - 0
     - 512x512x3
     - 16.7
     - 2350
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SuperResolution/Real-ESRGAN/Real_ESRGAN_x2/pretrained/2024-10-31/RealESRGAN_x2_sim.zip>`_
     - `link <https://github.com/ai-forever/Real-ESRGAN>`_
     - `rgbx <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo15m/real_esrgan_x2.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo15m/real_esrgan_x2_profiler_results_compiled.html>`_
