
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



.. _Classification:

--------------

ImageNet
^^^^^^^^

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
   * - cas_vit_m   
     - 81.2
     - 81.06
     - 56
     - 134
     - `S <https://github.com/Tianfang-Zhang/CAS-ViT>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/cas_vit_m/pretrained/2024-09-03/cas_vit_m.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/cas_vit_m.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/cas_vit_m_profiler_results_compiled.html>`_
     - 384x384x3
     - 12.42
     - 10.89    
   * - cas_vit_s   
     - 79.93
     - 79.84
     - 86
     - 219
     - `S <https://github.com/Tianfang-Zhang/CAS-ViT>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/cas_vit_s/pretrained/2024-08-13/cas_vit_s.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/cas_vit_s.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/cas_vit_s_profiler_results_compiled.html>`_
     - 384x384x3
     - 5.5
     - 5.4    
   * - cas_vit_t   
     - 81.9
     - 81.61
     - 37
     - 82
     - `S <https://github.com/Tianfang-Zhang/CAS-ViT>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/cas_vit_t/pretrained/2024-09-03/cas_vit_t.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/cas_vit_t.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/cas_vit_t_profiler_results_compiled.html>`_
     - 384x384x3
     - 21.76
     - 20.85    
   * - davit_tiny   
     - 82.7
     - 82.29
     - 15
     - 28
     - `S <https://huggingface.co/timm/davit_tiny.msft_in1k>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/davit_tiny/pretrained/2024-10-01/davit_tiny.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/davit_tiny.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/davit_tiny_profiler_results_compiled.html>`_
     - 224x224x3
     - 28.36
     - 9.1    
   * - deit_base   
     - 80.93
     - 80.25
     - 43
     - 119
     - `S <https://github.com/facebookresearch/deit>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/deit_base/pretrained/2024-05-21/deit_base.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/deit_base.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/deit_base_profiler_results_compiled.html>`_
     - 224x224x3
     - 80.26
     - 35.22    
   * - deit_small   
     - 78.25
     - 77.61
     - 103
     - 375
     - `S <https://github.com/facebookresearch/deit>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/deit_small/pretrained/2024-05-21/deit_small.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/deit_small.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/deit_small_profiler_results_compiled.html>`_
     - 224x224x3
     - 20.52
     - 9.4    
   * - deit_tiny   
     - 69.07
     - 68.72
     - 124
     - 427
     - `S <https://github.com/facebookresearch/deit>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/deit_tiny/pretrained/2024-05-21/deit_tiny.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/deit_tiny.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/deit_tiny_profiler_results_compiled.html>`_
     - 224x224x3
     - 5.3
     - 2.57    
   * - efficientformer_l1   
     - 79.13
     - 76.55
     - 87
     - 165
     - `S <https://github.com/snap-research/EfficientFormer/tree/main>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientformer_l1/pretrained/2024-08-11/efficientformer_l1.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/efficientformer_l1.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/efficientformer_l1_profiler_results_compiled.html>`_
     - 224x224x3
     - 12.3
     - 2.6    
   * - efficientnet_l   
     - 80.47
     - 79.28
     - 126
     - 237
     - `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_l/pretrained/2023-07-18/efficientnet_l.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/efficientnet_l.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/efficientnet_l_profiler_results_compiled.html>`_
     - 300x300x3
     - 10.55
     - 19.4    
   * - efficientnet_lite0   
     - 74.99
     - 73.84
     - 2215
     - 2215
     - `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite0/pretrained/2023-07-18/efficientnet_lite0.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/efficientnet_lite0.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/efficientnet_lite0_profiler_results_compiled.html>`_
     - 224x224x3
     - 4.63
     - 0.78    
   * - efficientnet_lite1   
     - 76.67
     - 76.27
     - 998
     - 998
     - `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite1/pretrained/2023-07-18/efficientnet_lite1.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/efficientnet_lite1.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/efficientnet_lite1_profiler_results_compiled.html>`_
     - 240x240x3
     - 5.39
     - 1.22    
   * - efficientnet_lite2   
     - 77.46
     - 76.69
     - 495
     - 495
     - `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite2/pretrained/2023-07-18/efficientnet_lite2.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/efficientnet_lite2.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/efficientnet_lite2_profiler_results_compiled.html>`_
     - 260x260x3
     - 6.06
     - 1.74    
   * - efficientnet_lite3   
     - 79.29
     - 78.66
     - 165
     - 370
     - `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite3/pretrained/2023-07-18/efficientnet_lite3.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/efficientnet_lite3.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/efficientnet_lite3_profiler_results_compiled.html>`_
     - 280x280x3
     - 8.16
     - 2.8    
   * - efficientnet_lite4   
     - 80.79
     - 80.06
     - 136
     - 341
     - `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite4/pretrained/2023-07-18/efficientnet_lite4.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/efficientnet_lite4.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/efficientnet_lite4_profiler_results_compiled.html>`_
     - 300x300x3
     - 12.95
     - 5.10    
   * - efficientnet_m   
     - 78.91
     - 78.46
     - 664
     - 664
     - `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_m/pretrained/2023-07-18/efficientnet_m.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/efficientnet_m.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/efficientnet_m_profiler_results_compiled.html>`_
     - 240x240x3
     - 6.87
     - 7.32    
   * - efficientnet_s   
     - 77.63
     - 77.24
     - 903
     - 903
     - `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_s/pretrained/2023-07-18/efficientnet_s.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/efficientnet_s.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/efficientnet_s_profiler_results_compiled.html>`_
     - 224x224x3
     - 5.41
     - 4.72    
   * - fastvit_sa12   
     - 79.8
     - 76.81
     - 275
     - 966
     - `S <https://github.com/apple/ml-fastvit/tree/main>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/fastvit_sa12/pretrained/2023-08-21/fastvit_sa12.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/fastvit_sa12.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/fastvit_sa12_profiler_results_compiled.html>`_
     - 224x224x3
     - 11.99
     - 3.59    
   * - hardnet39ds   
     - 73.43
     - 72.92
     - 557
     - 1644
     - `S <https://github.com/PingoLH/Pytorch-HarDNet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/hardnet39ds/pretrained/2021-07-20/hardnet39ds.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/hardnet39ds.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/hardnet39ds_profiler_results_compiled.html>`_
     - 224x224x3
     - 3.48
     - 0.86    
   * - hardnet68   
     - 75.47
     - 75.25
     - 209
     - 534
     - `S <https://github.com/PingoLH/Pytorch-HarDNet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/hardnet68/pretrained/2021-07-20/hardnet68.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/hardnet68.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/hardnet68_profiler_results_compiled.html>`_
     - 224x224x3
     - 17.56
     - 8.5    
   * - inception_v1   
     - 69.74
     - 69.55
     - 1307
     - 1307
     - `S <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/inception_v1/pretrained/2023-07-18/inception_v1.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/inception_v1.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/inception_v1_profiler_results_compiled.html>`_
     - 224x224x3
     - 6.62
     - 3    
   * - levit128   
     - 78.4
     - 76.57
     - 209
     - 826
     - `S <https://github.com/facebookresearch/LeViT>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/levit_128/pretrained/2024-07-10/LeViT_128_simp.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/levit128.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/levit128_profiler_results_compiled.html>`_
     - 224x224x3
     - 9.2
     - 0.8    
   * - levit192   
     - 79.7
     - 77.63
     - 227
     - 903
     - `S <https://github.com/facebookresearch/LeViT>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/levit_192/pretrained/2024-07-10/LeViT_192_simp.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/levit192.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/levit192_profiler_results_compiled.html>`_
     - 224x224x3
     - 10.9
     - 1.3    
   * - levit256   
     - 81.4
     - 79.09
     - 170
     - 665
     - `S <https://github.com/facebookresearch/LeViT>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/levit_256/2024-05-13/levit-256.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/levit256.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/levit256_profiler_results_compiled.html>`_
     - 224x224x3
     - 18.9
     - 2.3    
   * - levit384   
     - 82.3
     - 78.94
     - 119
     - 454
     - `S <https://github.com/facebookresearch/LeViT>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/levit_384/pretrained/2024-07-10/LeViT_384_simp.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/levit384.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/levit384_profiler_results_compiled.html>`_
     - 224x224x3
     - 39.1
     - 4.7    
   * - mobilenet_v1   
     - 70.97
     - 70.3
     - 4155
     - 4156
     - `S <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v1/pretrained/2023-07-18/mobilenet_v1.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/mobilenet_v1.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/mobilenet_v1_profiler_results_compiled.html>`_
     - 224x224x3
     - 4.22
     - 1.14    
   * - mobilenet_v2_1.0   
     - 71.78
     - 70.96
     - 3454
     - 3454
     - `S <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v2_1.0/pretrained/2025-01-15/mobilenet_v2_1.0.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/mobilenet_v2_1.0.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/mobilenet_v2_1.0_profiler_results_compiled.html>`_
     - 224x224x3
     - 3.49
     - 0.62    
   * - mobilenet_v2_1.4   
     - 74.18
     - 73.26
     - 1813
     - 1813
     - `S <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v2_1.4/pretrained/2021-07-11/mobilenet_v2_1.4.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/mobilenet_v2_1.4.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/mobilenet_v2_1.4_profiler_results_compiled.html>`_
     - 224x224x3
     - 6.09
     - 1.18    
   * - mobilenet_v3   
     - 72.21
     - 71.75
     - 3298
     - 3298
     - `S <https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v3/pretrained/2023-07-18/mobilenet_v3.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/mobilenet_v3.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/mobilenet_v3_profiler_results_compiled.html>`_
     - 224x224x3
     - 4.07
     - 2    
   * - mobilenet_v3_large_minimalistic   
     - 72.12
     - 70.6
     - 4894
     - 4894
     - `S <https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v3_large_minimalistic/pretrained/2021-07-11/mobilenet_v3_large_minimalistic.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/mobilenet_v3_large_minimalistic.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/mobilenet_v3_large_minimalistic_profiler_results_compiled.html>`_
     - 224x224x3
     - 3.91
     - 0.42    
   * - regnetx_1.6gf   
     - 77.05
     - 76.66
     - 2709
     - 2709
     - `S <https://github.com/facebookresearch/pycls>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/regnetx_1.6gf/pretrained/2021-07-11/regnetx_1.6gf.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/regnetx_1.6gf.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/regnetx_1.6gf_profiler_results_compiled.html>`_
     - 224x224x3
     - 9.17
     - 3.22    
   * - regnetx_800mf   
     - 75.16
     - 74.86
     - 4482
     - 4471
     - `S <https://github.com/facebookresearch/pycls>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/regnetx_800mf/pretrained/2021-07-11/regnetx_800mf.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/regnetx_800mf.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/regnetx_800mf_profiler_results_compiled.html>`_
     - 224x224x3
     - 7.24
     - 1.6    
   * - repghost_1_0x   
     - 73.03
     - 72.19
     - 307
     - 1256
     - `S <https://github.com/ChengpengChen/RepGhost>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/repghost/repghostnet_1_0x/pretrained/2023-04-03/repghostnet_1_0x.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/repghost_1_0x.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/repghost_1_0x_profiler_results_compiled.html>`_
     - 224x224x3
     - 4.1
     - 0.28    
   * - repghost_2_0x   
     - 77.18
     - 76.86
     - 190
     - 677
     - `S <https://github.com/ChengpengChen/RepGhost>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/repghost/repghostnet_2_0x/pretrained/2023-04-03/repghostnet_2_0x.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/repghost_2_0x.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/repghost_2_0x_profiler_results_compiled.html>`_
     - 224x224x3
     - 9.8
     - 1.04    
   * - repvgg_a1   
     - 74.4
     - 72.18
     - 2018
     - 2018
     - `S <https://github.com/DingXiaoH/RepVGG>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/repvgg/repvgg_a1/pretrained/2022-10-02/RepVGG-A1.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/repvgg_a1.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/repvgg_a1_profiler_results_compiled.html>`_
     - 224x224x3
     - 12.79
     - 4.7    
   * - repvgg_a2   
     - 76.52
     - 74.43
     - 288
     - 583
     - `S <https://github.com/DingXiaoH/RepVGG>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/repvgg/repvgg_a2/pretrained/2022-10-02/RepVGG-A2.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/repvgg_a2.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/repvgg_a2_profiler_results_compiled.html>`_
     - 224x224x3
     - 25.5
     - 10.2    
   * - resmlp12_relu   
     - 75.27
     - 74.89
     - 89
     - 311
     - `S <https://github.com/rwightman/pytorch-image-models/>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resmlp12_relu/pretrained/2022-03-03/resmlp12_relu.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/resmlp12_relu.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/resmlp12_relu_profiler_results_compiled.html>`_
     - 224x224x3
     - 15.77
     - 6.04    
   * - resnet_v1_18   
     - 71.27
     - 70.79
     - 2708
     - 2708
     - `S <https://github.com/yhhhli/BRECQ>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v1_18/pretrained/2022-04-19/resnet_v1_18.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/resnet_v1_18.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/resnet_v1_18_profiler_results_compiled.html>`_
     - 224x224x3
     - 11.68
     - 3.64    
   * - resnet_v1_34   
     - 72.7
     - 72.22
     - 363
     - 1040
     - `S <https://github.com/tensorflow/models/tree/master/research/slim>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v1_34/pretrained/2025-01-15/resnet_v1_34.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/resnet_v1_34.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/resnet_v1_34_profiler_results_compiled.html>`_
     - 224x224x3
     - 21.79
     - 7.34        
   * - resnet_v1_50 |rocket| |star| 
     - 75.21
     - 74.69
     - 320
     - 1019
     - `S <https://github.com/tensorflow/models/tree/master/research/slim>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v1_50/pretrained/2025-01-15/resnet_v1_50.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/resnet_v1_50.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/resnet_v1_50_profiler_results_compiled.html>`_
     - 224x224x3
     - 25.53
     - 6.98    
   * - resnext26_32x4d   
     - 76.17
     - 75.96
     - 375
     - 839
     - `S <https://github.com/osmr/imgclsmob/tree/master/pytorch>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnext26_32x4d/pretrained/2023-09-18/resnext26_32x4d.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/resnext26_32x4d.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/resnext26_32x4d_profiler_results_compiled.html>`_
     - 224x224x3
     - 15.37
     - 4.96    
   * - resnext50_32x4d   
     - 79.3
     - 78.35
     - 259
     - 727
     - `S <https://github.com/osmr/imgclsmob/tree/master/pytorch>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnext50_32x4d/pretrained/2023-07-18/resnext50_32x4d.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/resnext50_32x4d.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/resnext50_32x4d_profiler_results_compiled.html>`_
     - 224x224x3
     - 24.99
     - 8.48    
   * - squeezenet_v1.1   
     - 59.85
     - 59.35
     - 4251
     - 4255
     - `S <https://github.com/osmr/imgclsmob/tree/master/pytorch>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/squeezenet_v1.1/pretrained/2023-07-18/squeezenet_v1.1.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/squeezenet_v1.1.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/squeezenet_v1.1_profiler_results_compiled.html>`_
     - 224x224x3
     - 1.24
     - 0.78    
   * - swin_small   
     - 83.13
     - 80.03
     - 19
     - 53
     - `S <https://huggingface.co/microsoft/swin-small-patch4-window7-224>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/swin_small/pretrained/2024-08-01/swin_small_classifier.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/swin_small.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/swin_small_profiler_results_compiled.html>`_
     - 224x224x3
     - 50
     - 17.6    
   * - swin_tiny   
     - 81.3
     - 79.33
     - 37
     - 98
     - `S <https://huggingface.co/microsoft/swin-tiny-patch4-window7-224>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/swin_tiny/pretrained/2024-08-01/swin_tiny_classifier.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/swin_tiny.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/swin_tiny_profiler_results_compiled.html>`_
     - 224x224x3
     - 29
     - 9.1    
   * - vit_base   
     - 84.5
     - 83.44
     - 53
     - 175
     - `S <https://github.com/rwightman/pytorch-image-models>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_base/pretrained/2024-04-03/vit_base_patch16_224_ops17.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/vit_base.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/vit_base_profiler_results_compiled.html>`_
     - 224x224x3
     - 86.5
     - 35.188      
   * - vit_base_bn |rocket|  
     - 79.98
     - 79.24
     - 64
     - 207
     - `S <https://github.com/rwightman/pytorch-image-models>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_base_bn/pretrained/2023-01-25/vit_base.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/vit_base_bn.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/vit_base_bn_profiler_results_compiled.html>`_
     - 224x224x3
     - 86.5
     - 35.188    
   * - vit_small   
     - 81.5
     - 80.27
     - 112
     - 398
     - `S <https://github.com/rwightman/pytorch-image-models>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_small/pretrained/2024-04-03/vit_small_patch16_224_ops17.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/vit_small.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/vit_small_profiler_results_compiled.html>`_
     - 224x224x3
     - 21.12
     - 8.62    
   * - vit_small_bn   
     - 78.12
     - 77.26
     - 153
     - 547
     - `S <https://github.com/rwightman/pytorch-image-models>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_small_bn/pretrained/2022-08-08/vit_small.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/vit_small_bn.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/vit_small_bn_profiler_results_compiled.html>`_
     - 224x224x3
     - 21.12
     - 8.62    
   * - vit_tiny   
     - 75.51
     - 74.18
     - 123
     - 432
     - `S <https://github.com/rwightman/pytorch-image-models>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_tiny/pretrained/2024-04-03/vit_tiny_patch16_224_ops17.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/vit_tiny.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/vit_tiny_profiler_results_compiled.html>`_
     - 224x224x3
     - 5.73
     - 2.2    
   * - vit_tiny_bn   
     - 68.95
     - 67.33
     - 341
     - 1581
     - `S <https://github.com/rwightman/pytorch-image-models>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_tiny_bn/pretrained/2023-08-29/vit_tiny_bn.zip>`_ `X <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/vit_tiny_bn.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/vit_tiny_bn_profiler_results_compiled.html>`_
     - 224x224x3
     - 5.73
     - 2.2
