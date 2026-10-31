


Public Models
=============

* System host: Intel® Core™ i5-9400 CPU @ 2.90GHz
* Hailo Dataflow Compiler Version v5.5.0
* Measurement conditions: PCIe Gen 3 x 4 lanes, room temperature

|

Super Resolution
================

|

Link Legend
-----------

|

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - **Key / Icon**
     - **Description**
   * - ⭐
     - Networks used by `Hailo-apps <https://github.com/hailo-ai/hailo-apps-infra>`_.
   * - **S**
     - Source – Link to the model's open-source repository.
   * - **PT**
     - Pretrained – Download the pretrained model file (ZIP format).
   * - **HEF, NV12, RGBX**
     - Compiled Models – Links to models in various formats:
       - **HEF:** RGB format
       - **NV12:** NV12 format
       - **RGBX:** RGBX format
   * - **PR**
     - Profiler Report – Download the model's performance profiling report.

|

Bsd100
------

|








.. list-table::
   :header-rows: 1
   :widths: 31 9 7 11 9 8 8 8 9

   * - Network Name
     - float PSNR
     - Hardware PSNR
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
   
   
   
   
   
   
   

   * - espcn_x2
     - 31.2
     - 30.8
     - 1637
     - 1638
     - | `S <https://github.com/Lornatang/ESPCN-PyTorch>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SuperResolution/espcn/espcn_x2/2022-08-02/espcn_x2.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/espcn_x2_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/espcn_x2.hef>`_
     - 156x240x1
     - 0.02
     - 1.6
   
   
   
   
   
   
   

   * - espcn_x3
     - 28.3
     - 28.1
     - 1926
     - 1925
     - | `S <https://github.com/Lornatang/ESPCN-PyTorch>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SuperResolution/espcn/espcn_x3/2022-08-02/espcn_x3.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/espcn_x3_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/espcn_x3.hef>`_
     - 104x160x1
     - 0.02
     - 0.76
   
   
   
   
   
   
   

   * - espcn_x4
     - 26.8
     - 26.7
     - 1909
     - 1910
     - | `S <https://github.com/Lornatang/ESPCN-PyTorch>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SuperResolution/espcn/espcn_x4/2022-08-02/espcn_x4.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/espcn_x4_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/espcn_x4.hef>`_
     - 78x120x1
     - 0.02
     - 0.46

|

Div2K
-----

|








.. list-table::
   :header-rows: 1
   :widths: 31 9 7 11 9 8 8 8 9

   * - Network Name
     - float PSNR
     - Hardware PSNR
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
   
   
   
   
   
   
   

   * - real_esrgan_x2⭐
     - 28.3
     - 28.0
     - 3.54
     - 
     - | `S <https://github.com/ai-forever/Real-ESRGAN>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SuperResolution/Real-ESRGAN/Real_ESRGAN_x2/pretrained/2024-10-31/RealESRGAN_x2_sim.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/real_esrgan_x2_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/real_esrgan_x2.hef>`_
     - 512x512x3
     - 16.7
     - 2350
   
   
   
   
   
   
   

   * - span_x2
     - 36.2
     - 34.7
     - 3.09
     - 
     - | `S <https://github.com/hongyuanyu/SPAN>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SuperResolution/span/span_x2/pretrained/2026-09-16/span_x2.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/span_x2_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/span_x2.hef>`_
     - 1024x1024x3
     - 0.41
     - 860.37
   
   
   
   
   
   
   

   * - span_x4
     - 30.5
     - 30.0
     - 26.5
     - 
     - | `S <https://github.com/hongyuanyu/SPAN>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SuperResolution/span/span_x4/pretrained/2026-09-16/span_x4.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/span_x4_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/span_x4.hef>`_
     - 512x512x3
     - 0.43
     - 223.25
   
   
   
   
   
   
   

   * - spanf_x4
     - 30.3
     - 29.5
     - 64.8
     - 67.3
     - | `S <https://github.com/Amazingren/NTIRE2025_ESR>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SuperResolution/span/spanf_x4/pretrained/2026-09-16/spanf_x4.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/spanf_x4_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/spanf_x4.hef>`_
     - 512x512x3
     - 0.15
     - 77.59