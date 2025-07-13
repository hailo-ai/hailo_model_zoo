
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

Link Legend

The following shortcuts are used in the table below to indicate available resources for each model:

* S – Source: Link to the model’s open-source code repository.
* PT – Pretrained: Download the pretrained model file (compressed in ZIP format).
* H, NV, X – Compiled Models: Links to the compiled model in various formats:
            * H: regular HEF with RGBX format
            * NV: HEF with NV12 format
            * X: HEF with RGBX format

* PR – Profiler Report: Download the model’s performance profiling report.



.. _Super Resolution:

----------------

BSD100
^^^^^^

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
   * - espcn_x2   
     - 31.22
     - 30.77
     - 1637
     - 1637
     - `S <https://github.com/Lornatang/ESPCN-PyTorch>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SuperResolution/espcn/espcn_x2/2022-08-02/espcn_x2.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/espcn_x2.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/espcn_x2_profiler_results_compiled.html>`_
     - 156x240x1
     - 0.02
     - 1.6    
   * - espcn_x3   
     - 28.29
     - 28.12
     - 1924
     - 1924
     - `S <https://github.com/Lornatang/ESPCN-PyTorch>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SuperResolution/espcn/espcn_x3/2022-08-02/espcn_x3.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/espcn_x3.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/espcn_x3_profiler_results_compiled.html>`_
     - 104x160x1
     - 0.02
     - 0.76    
   * - espcn_x4   
     - 26.83
     - 26.65
     - 1908
     - 1908
     - `S <https://github.com/Lornatang/ESPCN-PyTorch>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SuperResolution/espcn/espcn_x4/2022-08-02/espcn_x4.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/espcn_x4.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/espcn_x4_profiler_results_compiled.html>`_
     - 78x120x1
     - 0.02
     - 0.46

N/A
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
   * - real_esrgan_x2   
     - 28.27
     - 27.66
     - 2
     - 2
     - `S <https://github.com/ai-forever/Real-ESRGAN>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SuperResolution/Real-ESRGAN/Real_ESRGAN_x2/pretrained/2024-10-31/RealESRGAN_x2_sim.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/real_esrgan_x2.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/real_esrgan_x2_profiler_results_compiled.html>`_
     - 512x512x3
     - 16.7
     - 2350
