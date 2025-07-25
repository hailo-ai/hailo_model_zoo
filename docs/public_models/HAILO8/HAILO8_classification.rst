
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
* All models were compiled using Hailo Dataflow Compiler v3.32.0

Link Legend

The following shortcuts are used in the table below to indicate available resources for each model:

* S – Source: Link to the model’s open-source code repository.
* PT – Pretrained: Download the pretrained model file (compressed in ZIP format).
* H, NV, X – Compiled Models: Links to the compiled model in various formats:
            * H: regular HEF with RGB format
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
     - 81.01
     - 80.81
     - 47
     - 157
     - `S <https://github.com/Tianfang-Zhang/CAS-ViT>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/cas_vit_m/pretrained/2024-09-03/cas_vit_m.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/cas_vit_m.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/cas_vit_m_profiler_results_compiled.html>`_
     - 384x384x3
     - 12.42
     - 10.89
   * - cas_vit_s
     - 79.65
     - 79.37
     - 71
     - 225
     - `S <https://github.com/Tianfang-Zhang/CAS-ViT>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/cas_vit_s/pretrained/2024-08-13/cas_vit_s.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/cas_vit_s.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/cas_vit_s_profiler_results_compiled.html>`_
     - 384x384x3
     - 5.5
     - 5.4
   * - cas_vit_t
     - 81.52
     - 81.14
     - 31
     - 95
     - `S <https://github.com/Tianfang-Zhang/CAS-ViT>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/cas_vit_t/pretrained/2024-09-03/cas_vit_t.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/cas_vit_t.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/cas_vit_t_profiler_results_compiled.html>`_
     - 384x384x3
     - 21.76
     - 20.85
   * - deit_base
     - 79.9
     - 78.86
     - 28
     - 107
     - `S <https://github.com/facebookresearch/deit>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/deit_base/pretrained/2024-05-21/deit_base.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/deit_base.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/deit_base_profiler_results_compiled.html>`_
     - 224x224x3
     - 80.26
     - 35.22
   * - deit_small
     - 77.18
     - 76.11
     - 69
     - 309
     - `S <https://github.com/facebookresearch/deit>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/deit_small/pretrained/2024-05-21/deit_small.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/deit_small.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/deit_small_profiler_results_compiled.html>`_
     - 224x224x3
     - 20.52
     - 9.4
   * - deit_tiny
     - 68.57
     - 68.07
     - 89
     - 406
     - `S <https://github.com/facebookresearch/deit>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/deit_tiny/pretrained/2024-05-21/deit_tiny.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/deit_tiny.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/deit_tiny_profiler_results_compiled.html>`_
     - 224x224x3
     - 5.3
     - 2.57
   * - efficientformer_l1
     - 76.46
     - 73.79
     - 93
     - 274
     - `S <https://github.com/snap-research/EfficientFormer/tree/main>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientformer_l1/pretrained/2024-08-11/efficientformer_l1.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/efficientformer_l1.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/efficientformer_l1_profiler_results_compiled.html>`_
     - 224x224x3
     - 12.3
     - 2.6
   * - efficientnet_l
     - 79.28
     - 78.09
     - 221
     - 221
     - `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_l/pretrained/2023-07-18/efficientnet_l.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/efficientnet_l.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/efficientnet_l_profiler_results_compiled.html>`_
     - 300x300x3
     - 10.55
     - 19.4
   * - efficientnet_lite0
     - 73.84
     - 72.69
     - 1952
     - 1952
     - `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite0/pretrained/2023-07-18/efficientnet_lite0.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/efficientnet_lite0.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/efficientnet_lite0_profiler_results_compiled.html>`_
     - 224x224x3
     - 4.63
     - 0.78
   * - efficientnet_lite1
     - 76.25
     - 75.84
     - 1668
     - 1668
     - `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite1/pretrained/2023-07-18/efficientnet_lite1.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/efficientnet_lite1.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/efficientnet_lite1_profiler_results_compiled.html>`_
     - 240x240x3
     - 5.39
     - 1.22
   * - efficientnet_lite2
     - 76.67
     - 75.88
     - 977
     - 977
     - `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite2/pretrained/2023-07-18/efficientnet_lite2.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/efficientnet_lite2.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/efficientnet_lite2_profiler_results_compiled.html>`_
     - 260x260x3
     - 6.06
     - 1.74
   * - efficientnet_lite3
     - 78.66
     - 78.03
     - 512
     - 512
     - `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite3/pretrained/2023-07-18/efficientnet_lite3.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/efficientnet_lite3.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/efficientnet_lite3_profiler_results_compiled.html>`_
     - 280x280x3
     - 8.16
     - 2.8
   * - efficientnet_lite4
     - 80.08
     - 79.36
     - 273
     - 273
     - `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite4/pretrained/2023-07-18/efficientnet_lite4.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/efficientnet_lite4.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/efficientnet_lite4_profiler_results_compiled.html>`_
     - 300x300x3
     - 12.95
     - 5.10
   * - efficientnet_m
     - 78.45
     - 77.98
     - 984
     - 984
     - `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_m/pretrained/2023-07-18/efficientnet_m.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/efficientnet_m.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/efficientnet_m_profiler_results_compiled.html>`_
     - 240x240x3
     - 6.87
     - 7.32
   * - efficientnet_s
     - 77.24
     - 76.85
     - 1232
     - 1232
     - `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_s/pretrained/2023-07-18/efficientnet_s.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/efficientnet_s.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/efficientnet_s_profiler_results_compiled.html>`_
     - 224x224x3
     - 5.41
     - 4.72
   * - fastvit_sa12
     - 76.56
     - 73.32
     - 1113
     - 1113
     - `S <https://github.com/apple/ml-fastvit/tree/main>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/fastvit_sa12/pretrained/2023-08-21/fastvit_sa12.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/fastvit_sa12.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/fastvit_sa12_profiler_results_compiled.html>`_
     - 224x224x3
     - 11.99
     - 3.59
   * - hardnet39ds
     - 73.01
     - 72.59
     - 1975
     - 1975
     - `S <https://github.com/PingoLH/Pytorch-HarDNet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/hardnet39ds/pretrained/2021-07-20/hardnet39ds.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/hardnet39ds.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/hardnet39ds_profiler_results_compiled.html>`_
     - 224x224x3
     - 3.48
     - 0.86
   * - hardnet68
     - 75.22
     - 74.98
     - 137
     - 400
     - `S <https://github.com/PingoLH/Pytorch-HarDNet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/hardnet68/pretrained/2021-07-20/hardnet68.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/hardnet68.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/hardnet68_profiler_results_compiled.html>`_
     - 224x224x3
     - 17.56
     - 8.5
   * - inception_v1
     - 69.56
     - 69.39
     - 2454
     - 2454
     - `S <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/inception_v1/pretrained/2023-07-18/inception_v1.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/inception_v1.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/inception_v1_profiler_results_compiled.html>`_
     - 224x224x3
     - 6.62
     - 3
   * - mobilenet_v1
     - 70.35
     - 69.72
     - 3305
     - 3305
     - `S <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v1/pretrained/2023-07-18/mobilenet_v1.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/mobilenet_v1.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/mobilenet_v1_profiler_results_compiled.html>`_
     - 224x224x3
     - 4.22
     - 1.14
   * - mobilenet_v2_1.0
     - 70.96
     - 70.15
     - 2597
     - 2597
     - `S <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v2_1.0/pretrained/2025-01-15/mobilenet_v2_1.0.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/mobilenet_v2_1.0.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/mobilenet_v2_1.0_profiler_results_compiled.html>`_
     - 224x224x3
     - 3.49
     - 0.62
   * - mobilenet_v2_1.4
     - 73.21
     - 72.24
     - 1652
     - 1652
     - `S <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v2_1.4/pretrained/2021-07-11/mobilenet_v2_1.4.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/mobilenet_v2_1.4.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/mobilenet_v2_1.4_profiler_results_compiled.html>`_
     - 224x224x3
     - 6.09
     - 1.18
   * - mobilenet_v3
     - 71.77
     - 71.33
     - 3697
     - 3697
     - `S <https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v3/pretrained/2023-07-18/mobilenet_v3.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/mobilenet_v3.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/mobilenet_v3_profiler_results_compiled.html>`_
     - 224x224x3
     - 4.07
     - 2
   * - regnetx_1.6gf
     - 76.68
     - 76.32
     - 2551
     - 2551
     - `S <https://github.com/facebookresearch/pycls>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/regnetx_1.6gf/pretrained/2021-07-11/regnetx_1.6gf.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/regnetx_1.6gf.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/regnetx_1.6gf_profiler_results_compiled.html>`_
     - 224x224x3
     - 9.17
     - 3.22
   * - regnetx_800mf
     - 74.91
     - 74.65
     - 3498
     - 3498
     - `S <https://github.com/facebookresearch/pycls>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/regnetx_800mf/pretrained/2021-07-11/regnetx_800mf.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/regnetx_800mf.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/regnetx_800mf_profiler_results_compiled.html>`_
     - 224x224x3
     - 7.24
     - 1.6
   * - repghost_1_0x
     - 72.24
     - 71.45
     - 242
     - 1074
     - `S <https://github.com/ChengpengChen/RepGhost>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/repghost/repghostnet_1_0x/pretrained/2023-04-03/repghostnet_1_0x.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/repghost_1_0x.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/repghost_1_0x_profiler_results_compiled.html>`_
     - 224x224x3
     - 4.1
     - 0.28
   * - repghost_2_0x
     - 76.91
     - 76.64
     - 140
     - 623
     - `S <https://github.com/ChengpengChen/RepGhost>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/repghost/repghostnet_2_0x/pretrained/2023-04-03/repghostnet_2_0x.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/repghost_2_0x.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/repghost_2_0x_profiler_results_compiled.html>`_
     - 224x224x3
     - 9.8
     - 1.04
   * - repvgg_a1
     - 72.32
     - 70.24
     - 2707
     - 2707
     - `S <https://github.com/DingXiaoH/RepVGG>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/repvgg/repvgg_a1/pretrained/2022-10-02/RepVGG-A1.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/repvgg_a1.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/repvgg_a1_profiler_results_compiled.html>`_
     - 224x224x3
     - 12.79
     - 4.7
   * - repvgg_a2
     - 74.42
     - 72.32
     - 1132
     - 1132
     - `S <https://github.com/DingXiaoH/RepVGG>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/repvgg/repvgg_a2/pretrained/2022-10-02/RepVGG-A2.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/repvgg_a2.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/repvgg_a2_profiler_results_compiled.html>`_
     - 224x224x3
     - 25.5
     - 10.2
   * - resmlp12_relu
     - 74.95
     - 74.64
     - 1429
     - 1429
     - `S <https://github.com/rwightman/pytorch-image-models/>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resmlp12_relu/pretrained/2022-03-03/resmlp12_relu.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/resmlp12_relu.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/resmlp12_relu_profiler_results_compiled.html>`_
     - 224x224x3
     - 15.77
     - 6.04
   * - resnet_v1_18
     - 71.07
     - 70.88
     - 2718
     - 2718
     - `S <https://github.com/yhhhli/BRECQ>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v1_18/pretrained/2022-04-19/resnet_v1_18.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/resnet_v1_18.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/resnet_v1_18_profiler_results_compiled.html>`_
     - 224x224x3
     - 11.68
     - 3.64
   * - resnet_v1_34
     - 72.23
     - 71.76
     - 1505
     - 1505
     - `S <https://github.com/tensorflow/models/tree/master/research/slim>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v1_34/pretrained/2025-01-15/resnet_v1_34.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/resnet_v1_34.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/resnet_v1_34_profiler_results_compiled.html>`_
     - 224x224x3
     - 21.79
     - 7.34
   * - resnet_v1_50 |rocket| |star|
     - 74.62
     - 74.04
     - 1372
     - 1372
     - `S <https://github.com/tensorflow/models/tree/master/research/slim>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v1_50/pretrained/2025-01-15/resnet_v1_50.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/resnet_v1_50.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/resnet_v1_50_profiler_results_compiled.html>`_
     - 224x224x3
     - 25.53
     - 6.98
   * - resnext26_32x4d
     - 75.89
     - 75.61
     - 1630
     - 1630
     - `S <https://github.com/osmr/imgclsmob/tree/master/pytorch>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnext26_32x4d/pretrained/2023-09-18/resnext26_32x4d.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/resnext26_32x4d.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/resnext26_32x4d_profiler_results_compiled.html>`_
     - 224x224x3
     - 15.37
     - 4.96
   * - resnext50_32x4d
     - 78.35
     - 77.39
     - 761
     - 761
     - `S <https://github.com/osmr/imgclsmob/tree/master/pytorch>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnext50_32x4d/pretrained/2023-07-18/resnext50_32x4d.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/resnext50_32x4d.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/resnext50_32x4d_profiler_results_compiled.html>`_
     - 224x224x3
     - 24.99
     - 8.48
   * - squeezenet_v1.1
     - 59.34
     - 58.84
     - 3034
     - 3034
     - `S <https://github.com/osmr/imgclsmob/tree/master/pytorch>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/squeezenet_v1.1/pretrained/2023-07-18/squeezenet_v1.1.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/squeezenet_v1.1.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/squeezenet_v1.1_profiler_results_compiled.html>`_
     - 224x224x3
     - 1.24
     - 0.78
   * - swin_small
     - 79.96
     - 76.8
     - 16
     - 51
     - `S <https://huggingface.co/microsoft/swin-small-patch4-window7-224>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/swin_small/pretrained/2024-08-01/swin_small_classifier.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/swin_small.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/swin_small_profiler_results_compiled.html>`_
     - 224x224x3
     - 50
     - 17.6
   * - swin_tiny
     - 79.25
     - 77.2
     - 28
     - 81
     - `S <https://huggingface.co/microsoft/swin-tiny-patch4-window7-224>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/swin_tiny/pretrained/2024-08-01/swin_tiny_classifier.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/swin_tiny.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/swin_tiny_profiler_results_compiled.html>`_
     - 224x224x3
     - 29
     - 9.1
   * - vit_base
     - 83.13
     - 81.77
     - 29
     - 108
     - `S <https://github.com/rwightman/pytorch-image-models>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_base/pretrained/2024-04-03/vit_base_patch16_224_ops17.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/vit_base.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/vit_base_profiler_results_compiled.html>`_
     - 224x224x3
     - 86.5
     - 35.188
   * - vit_base_bn |rocket|
     - 78.57
     - 77.16
     - 40
     - 167
     - `S <https://github.com/rwightman/pytorch-image-models>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_base_bn/pretrained/2023-01-25/vit_base.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/vit_base_bn.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/vit_base_bn_profiler_results_compiled.html>`_
     - 224x224x3
     - 86.5
     - 35.188
   * - vit_small
     - 79.96
     - 78.41
     - 70
     - 309
     - `S <https://github.com/rwightman/pytorch-image-models>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_small/pretrained/2024-04-03/vit_small_patch16_224_ops17.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/vit_small.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/vit_small_profiler_results_compiled.html>`_
     - 224x224x3
     - 21.12
     - 8.62
   * - vit_small_bn
     - 77.24
     - 76.36
     - 131
     - 640
     - `S <https://github.com/rwightman/pytorch-image-models>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_small_bn/pretrained/2022-08-08/vit_small.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/vit_small_bn.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/vit_small_bn_profiler_results_compiled.html>`_
     - 224x224x3
     - 21.12
     - 8.62
   * - vit_tiny
     - 73.82
     - 72.13
     - 89
     - 406
     - `S <https://github.com/rwightman/pytorch-image-models>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_tiny/pretrained/2024-04-03/vit_tiny_patch16_224_ops17.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/vit_tiny.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/vit_tiny_profiler_results_compiled.html>`_
     - 224x224x3
     - 5.73
     - 2.2
   * - vit_tiny_bn
     - 67.28
     - 65.62
     - 208
     - 1081
     - `S <https://github.com/rwightman/pytorch-image-models>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_tiny_bn/pretrained/2023-08-29/vit_tiny_bn.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/vit_tiny_bn.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/vit_tiny_bn_profiler_results_compiled.html>`_
     - 224x224x3
     - 5.73
     - 2.2

