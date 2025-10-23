


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
* All models were compiled using Hailo Dataflow Compiler v5.1.0

Link Legend

The following shortcuts are used in the table below to indicate available resources for each model:

* S – Source: Link to the model’s open-source code repository.
* PT – Pretrained: Download the pretrained model file (compressed in ZIP format).
* H, NV, X – Compiled Models: Links to the compiled model in various formats:
            * H: regular HEF with RGBX format
            * NV: HEF with NV12 format
            * X: HEF with RGBX format

* PR – Profiler Report: Download the model’s performance profiling report.

Imagenet
========

.. list-table::
   :header-rows: 1
   :widths: 31 9 7 11 9 8 8 8 9

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
     - 81.0
     - 33.6
     - 45.8
     - |
       `S <https://github.com/Tianfang-Zhang/CAS-ViT>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/cas_vit_m/pretrained/2024-09-03/cas_vit_m.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/cas_vit_m.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/cas_vit_m_profiler_results_compiled.html>`_
     - 384x384x3
     - 12.42
     - 10.89
   
   
   
   

   * - cas_vit_s 
     - 79.8
     - 79.6
     - 46.0
     - 65.5
     - |
       `S <https://github.com/Tianfang-Zhang/CAS-ViT>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/cas_vit_s/pretrained/2024-08-13/cas_vit_s.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/cas_vit_s.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/cas_vit_s_profiler_results_compiled.html>`_
     - 384x384x3
     - 5.5
     - 5.4
   
   
   
   

   * - cas_vit_t 
     - 81.9
     - 81.6
     - 24.4
     - 32.0
     - |
       `S <https://github.com/Tianfang-Zhang/CAS-ViT>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/cas_vit_t/pretrained/2024-09-03/cas_vit_t.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/cas_vit_t.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/cas_vit_t_profiler_results_compiled.html>`_
     - 384x384x3
     - 21.76
     - 20.85
   
   
   
   

   * - efficientnet_l 
     - 80.5
     - 79.3
     - 65.1
     - 95.2
     - |
       `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_l/pretrained/2023-07-18/efficientnet_l.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/efficientnet_l.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/efficientnet_l_profiler_results_compiled.html>`_
     - 300x300x3
     - 10.55
     - 19.4
   
   
   
   

   * - efficientnet_lite0 
     - 75.0
     - 74.1
     - 261
     - 461
     - |
       `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite0/pretrained/2023-07-18/efficientnet_lite0.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/efficientnet_lite0.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/efficientnet_lite0_profiler_results_compiled.html>`_
     - 224x224x3
     - 4.63
     - 0.78
   
   
   
   

   * - efficientnet_lite1 
     - 76.5
     - 76.0
     - 190
     - 342
     - |
       `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite1/pretrained/2023-07-18/efficientnet_lite1.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/efficientnet_lite1.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/efficientnet_lite1_profiler_results_compiled.html>`_
     - 240x240x3
     - 5.39
     - 1.22
   
   
   
   

   * - efficientnet_lite2 
     - 77.5
     - 
     - 99.6
     - 147
     - |
       `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite2/pretrained/2023-07-18/efficientnet_lite2.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/efficientnet_lite2.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/efficientnet_lite2_profiler_results_compiled.html>`_
     - 260x260x3
     - 6.06
     - 1.74
   
   
   
   

   * - efficientnet_lite4 
     - 80.8
     - 
     - 53.3
     - 78.5
     - |
       `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite4/pretrained/2023-07-18/efficientnet_lite4.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/efficientnet_lite4.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/efficientnet_lite4_profiler_results_compiled.html>`_
     - 300x300x3
     - 12.95
     - 5.10
   
   
   
   

   * - efficientnet_m 
     - 78.8
     - 
     - 112
     - 170
     - |
       `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_m/pretrained/2023-07-18/efficientnet_m.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/efficientnet_m.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/efficientnet_m_profiler_results_compiled.html>`_
     - 240x240x3
     - 6.87
     - 7.32
   
   
   
   

   * - fastvit_sa12 
     - 79.7
     - 
     - 108
     - 191
     - |
       `S <https://github.com/apple/ml-fastvit/tree/main>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/fastvit_sa12/pretrained/2023-08-21/fastvit_sa12.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/fastvit_sa12.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/fastvit_sa12_profiler_results_compiled.html>`_
     - 224x224x3
     - 11.99
     - 3.59
   
   
   
   

   * - hardnet39ds 
     - 73.3
     - 72.9
     - 314
     - 605
     - |
       `S <https://github.com/PingoLH/Pytorch-HarDNet>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/hardnet39ds/pretrained/2021-07-20/hardnet39ds.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/hardnet39ds.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/hardnet39ds_profiler_results_compiled.html>`_
     - 224x224x3
     - 3.48
     - 0.86
   
   
   
   

   * - hardnet68 
     - 75.3
     - 75.1
     - 98.9
     - 167
     - |
       `S <https://github.com/PingoLH/Pytorch-HarDNet>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/hardnet68/pretrained/2021-07-20/hardnet68.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/hardnet68.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/hardnet68_profiler_results_compiled.html>`_
     - 224x224x3
     - 17.56
     - 8.5
   
   
   
   

   * - inception_v1 
     - 69.7
     - 
     - 249
     - 409
     - |
       `S <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/inception_v1/pretrained/2023-07-18/inception_v1.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/inception_v1.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/inception_v1_profiler_results_compiled.html>`_
     - 224x224x3
     - 6.62
     - 3
   
   
   
   

   * - mobilenet_v1 
     - 70.8
     - 70.1
     - 459
     - 815
     - |
       `S <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v1/pretrained/2023-07-18/mobilenet_v1.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/mobilenet_v1.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/mobilenet_v1_profiler_results_compiled.html>`_
     - 224x224x3
     - 4.22
     - 1.14
   
   
   
   

   * - mobilenet_v2_1.0 
     - 71.6
     - 
     - 351
     - 630
     - |
       `S <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v2_1.0/pretrained/2025-01-15/mobilenet_v2_1.0.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/mobilenet_v2_1.0.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/mobilenet_v2_1.0_profiler_results_compiled.html>`_
     - 224x224x3
     - 3.49
     - 0.62
   
   
   
   

   * - mobilenet_v3 
     - 72.0
     - 71.8
     - 315
     - 586
     - |
       `S <https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v3/pretrained/2023-07-18/mobilenet_v3.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/mobilenet_v3.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/mobilenet_v3_profiler_results_compiled.html>`_
     - 224x224x3
     - 4.07
     - 2
   
   
   
   

   * - regnetx_1.6gf 
     - 76.8
     - 76.4
     - 257
     - 485
     - |
       `S <https://github.com/facebookresearch/pycls>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/regnetx_1.6gf/pretrained/2021-07-11/regnetx_1.6gf.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/regnetx_1.6gf.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/regnetx_1.6gf_profiler_results_compiled.html>`_
     - 224x224x3
     - 9.17
     - 3.22
   
   
   
   

   * - regnetx_800mf 
     - 75.0
     - 74.6
     - 381
     - 802
     - |
       `S <https://github.com/facebookresearch/pycls>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/regnetx_800mf/pretrained/2021-07-11/regnetx_800mf.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/regnetx_800mf.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/regnetx_800mf_profiler_results_compiled.html>`_
     - 224x224x3
     - 7.24
     - 1.6
   
   
   
   

   * - repghost_1_0x 
     - 73.0
     - 
     - 213
     - 436
     - |
       `S <https://github.com/ChengpengChen/RepGhost>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/repghost/repghostnet_1_0x/pretrained/2023-04-03/repghostnet_1_0x.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/repghost_1_0x.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/repghost_1_0x_profiler_results_compiled.html>`_
     - 224x224x3
     - 4.1
     - 0.28
   
   
   
   

   * - repghost_2_0x 
     - 77.2
     - 
     - 123
     - 209
     - |
       `S <https://github.com/ChengpengChen/RepGhost>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/repghost/repghostnet_2_0x/pretrained/2023-04-03/repghostnet_2_0x.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/repghost_2_0x.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/repghost_2_0x_profiler_results_compiled.html>`_
     - 224x224x3
     - 9.8
     - 1.04
   
   
   
   

   * - repvgg_a1 
     - 74.4
     - 
     - 261
     - 492
     - |
       `S <https://github.com/DingXiaoH/RepVGG>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/repvgg/repvgg_a1/pretrained/2022-10-02/RepVGG-A1.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/repvgg_a1.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/repvgg_a1_profiler_results_compiled.html>`_
     - 224x224x3
     - 12.79
     - 4.7
   
   
   
   

   * - repvgg_a2 
     - 76.4
     - 
     - 126
     - 203
     - |
       `S <https://github.com/DingXiaoH/RepVGG>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/repvgg/repvgg_a2/pretrained/2022-10-02/RepVGG-A2.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/repvgg_a2.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/repvgg_a2_profiler_results_compiled.html>`_
     - 224x224x3
     - 25.5
     - 10.2
   
   
   
   

   * - resmlp12_relu 
     - 74.9
     - 
     - 107
     - 255
     - |
       `S <https://github.com/rwightman/pytorch-image-models/>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resmlp12_relu/pretrained/2022-03-03/resmlp12_relu.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/resmlp12_relu.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/resmlp12_relu_profiler_results_compiled.html>`_
     - 224x224x3
     - 15.77
     - 6.04
   
   
   
   

   * - resnet_v1_18 
     - 71.1
     - 70.6
     - 319
     - 556
     - |
       `S <https://github.com/yhhhli/BRECQ>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v1_18/pretrained/2022-04-19/resnet_v1_18.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/resnet_v1_18.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/resnet_v1_18_profiler_results_compiled.html>`_
     - 224x224x3
     - 11.68
     - 3.64
   
   
   
   

   * - resnet_v1_34 
     - 72.6
     - 
     - 150
     - 268
     - |
       `S <https://github.com/tensorflow/models/tree/master/research/slim>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v1_34/pretrained/2025-01-15/resnet_v1_34.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/resnet_v1_34.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/resnet_v1_34_profiler_results_compiled.html>`_
     - 224x224x3
     - 21.79
     - 7.34
   
   
   
   

   * - resnet_v1_50 
     - 75.2
     - 74.5
     - 150
     - 322
     - |
       `S <https://github.com/tensorflow/models/tree/master/research/slim>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v1_50/pretrained/2025-01-15/resnet_v1_50.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/resnet_v1_50.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/resnet_v1_50_profiler_results_compiled.html>`_
     - 224x224x3
     - 25.53
     - 6.98
   
   
   
   

   * - squeezenet_v1.1 
     - 59.6
     - 59.1
     - 600
     - 937
     - |
       `S <https://github.com/osmr/imgclsmob/tree/master/pytorch>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/squeezenet_v1.1/pretrained/2023-07-18/squeezenet_v1.1.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/squeezenet_v1.1.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/squeezenet_v1.1_profiler_results_compiled.html>`_
     - 224x224x3
     - 1.24
     - 0.78
   
   
   
   

   * - vit_small_bn 
     - 77.9
     - 77.2
     - 104
     - 231
     - |
       `S <https://github.com/rwightman/pytorch-image-models>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_small_bn/pretrained/2022-08-08/vit_small.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/vit_small_bn.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/vit_small_bn_profiler_results_compiled.html>`_
     - 224x224x3
     - 21.12
     - 8.62
   
   
   
   

   * - vit_tiny_bn 
     - 68.5
     - 67.1
     - 215
     - 484
     - |
       `S <https://github.com/rwightman/pytorch-image-models>`_
       `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_tiny_bn/pretrained/2023-08-29/vit_tiny_bn.zip>`_
       `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/vit_tiny_bn.hef>`_
       
       
       `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo15l/vit_tiny_bn_profiler_results_compiled.html>`_
     - 224x224x3
     - 5.73
     - 2.2

