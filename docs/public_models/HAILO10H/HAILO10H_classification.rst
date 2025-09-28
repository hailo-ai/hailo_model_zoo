
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
     - 81.0
     - 80.8
     - 56
     - 154
     - `S <https://github.com/Tianfang-Zhang/CAS-ViT>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/cas_vit_m/pretrained/2024-09-03/cas_vit_m.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/cas_vit_m.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/cas_vit_m_profiler_results_compiled.html>`_
     - 384x384x3
     - 12.42
     - 10.89
   * - cas_vit_s
     - 79.8
     - 79.7
     - 85
     - 215
     - `S <https://github.com/Tianfang-Zhang/CAS-ViT>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/cas_vit_s/pretrained/2024-08-13/cas_vit_s.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/cas_vit_s.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/cas_vit_s_profiler_results_compiled.html>`_
     - 384x384x3
     - 5.5
     - 5.4
   * - cas_vit_t
     - 81.6
     - 81.3
     - 43
     - 109
     - `S <https://github.com/Tianfang-Zhang/CAS-ViT>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/cas_vit_t/pretrained/2024-09-03/cas_vit_t.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/cas_vit_t.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/cas_vit_t_profiler_results_compiled.html>`_
     - 384x384x3
     - 21.76
     - 20.85
   * - davit_tiny
     - 82.3
     - 81.9
     - 14
     - 24
     - `S <https://huggingface.co/timm/davit_tiny.msft_in1k>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/davit_tiny/pretrained/2024-10-01/davit_tiny.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/davit_tiny.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/davit_tiny_profiler_results_compiled.html>`_
     - 224x224x3
     - 28.36
     - 9.1
   * - deit_base
     - 80.2
     - 79.5
     - 43
     - 120
     - `S <https://github.com/facebookresearch/deit>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/deit_base/pretrained/2024-05-21/deit_base.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/deit_base.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/deit_base_profiler_results_compiled.html>`_
     - 224x224x3
     - 80.26
     - 35.22
   * - deit_small
     - 77.6
     - 77.0
     - 105
     - 359
     - `S <https://github.com/facebookresearch/deit>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/deit_small/pretrained/2024-05-21/deit_small.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/deit_small.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/deit_small_profiler_results_compiled.html>`_
     - 224x224x3
     - 20.52
     - 9.4
   * - deit_tiny
     - 68.6
     - 68.1
     - 133
     - 433
     - `S <https://github.com/facebookresearch/deit>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/deit_tiny/pretrained/2024-05-21/deit_tiny.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/deit_tiny.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/deit_tiny_profiler_results_compiled.html>`_
     - 224x224x3
     - 5.3
     - 2.57
   * - efficientformer_l1
     - 76.4
     - 73.7
     - 102
     - 202
     - `S <https://github.com/snap-research/EfficientFormer/tree/main>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientformer_l1/pretrained/2024-08-11/efficientformer_l1.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/efficientformer_l1.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/efficientformer_l1_profiler_results_compiled.html>`_
     - 224x224x3
     - 12.3
     - 2.6
   * - efficientnet_l
     - 79.3
     - 78.1
     - 126
     - 235
     - `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_l/pretrained/2023-07-18/efficientnet_l.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/efficientnet_l.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/efficientnet_l_profiler_results_compiled.html>`_
     - 300x300x3
     - 10.55
     - 19.4
   * - efficientnet_lite0
     - 73.8
     - 72.6
     - 2215
     - 2215
     - `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite0/pretrained/2023-07-18/efficientnet_lite0.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/efficientnet_lite0.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/efficientnet_lite0_profiler_results_compiled.html>`_
     - 224x224x3
     - 4.63
     - 0.78
   * - efficientnet_lite1
     - 76.3
     - 75.9
     - 998
     - 998
     - `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite1/pretrained/2023-07-18/efficientnet_lite1.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/efficientnet_lite1.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/efficientnet_lite1_profiler_results_compiled.html>`_
     - 240x240x3
     - 5.39
     - 1.22
   * - efficientnet_lite2
     - 76.8
     - 76.1
     - 212
     - 503
     - `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite2/pretrained/2023-07-18/efficientnet_lite2.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/efficientnet_lite2.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/efficientnet_lite2_profiler_results_compiled.html>`_
     - 260x260x3
     - 6.06
     - 1.74
   * - efficientnet_lite4
     - 80.0
     - 79.2
     - 137
     - 330
     - `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite4/pretrained/2023-07-18/efficientnet_lite4.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/efficientnet_lite4.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/efficientnet_lite4_profiler_results_compiled.html>`_
     - 300x300x3
     - 12.95
     - 5.10
   * - efficientnet_m
     - 78.5
     - 78.1
     - 664
     - 664
     - `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_m/pretrained/2023-07-18/efficientnet_m.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/efficientnet_m.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/efficientnet_m_profiler_results_compiled.html>`_
     - 240x240x3
     - 6.87
     - 7.32
   * - efficientnet_s
     - 76.9
     - 76.2
     - 903
     - 903
     - `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_s/pretrained/2023-07-18/efficientnet_s.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/efficientnet_s.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/efficientnet_s_profiler_results_compiled.html>`_
     - 224x224x3
     - 5.41
     - 4.72
   * - fastvit_sa12
     - 76.8
     - 73.8
     - 283
     - 947
     - `S <https://github.com/apple/ml-fastvit/tree/main>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/fastvit_sa12/pretrained/2023-08-21/fastvit_sa12.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/fastvit_sa12.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/fastvit_sa12_profiler_results_compiled.html>`_
     - 224x224x3
     - 11.99
     - 3.59
   * - hardnet39ds
     - 73.0
     - 72.6
     - 568
     - 1656
     - `S <https://github.com/PingoLH/Pytorch-HarDNet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/hardnet39ds/pretrained/2021-07-20/hardnet39ds.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/hardnet39ds.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/hardnet39ds_profiler_results_compiled.html>`_
     - 224x224x3
     - 3.48
     - 0.86
   * - hardnet68
     - 75.3
     - 75.1
     - 218
     - 560
     - `S <https://github.com/PingoLH/Pytorch-HarDNet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/hardnet68/pretrained/2021-07-20/hardnet68.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/hardnet68.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/hardnet68_profiler_results_compiled.html>`_
     - 224x224x3
     - 17.56
     - 8.5
   * - inception_v1
     - 69.5
     - 69.3
     - 1307
     - 1307
     - `S <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/inception_v1/pretrained/2023-07-18/inception_v1.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/inception_v1.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/inception_v1_profiler_results_compiled.html>`_
     - 224x224x3
     - 6.62
     - 3
   * - levit128
     - 76.5
     - 74.6
     - 226
     - 870
     - `S <https://github.com/facebookresearch/LeViT>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/levit_128/pretrained/2024-07-10/LeViT_128_simp.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/levit128.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/levit128_profiler_results_compiled.html>`_
     - 224x224x3
     - 9.2
     - 0.8
   * - levit192
     - 77.3
     - 74.9
     - 240
     - 926
     - `S <https://github.com/facebookresearch/LeViT>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/levit_192/pretrained/2024-07-10/LeViT_192_simp.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/levit192.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/levit192_profiler_results_compiled.html>`_
     - 224x224x3
     - 10.9
     - 1.3
   * - levit256
     - 79.2
     - 77.0
     - 176
     - 675
     - `S <https://github.com/facebookresearch/LeViT>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/levit_256/2024-05-13/levit-256.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/levit256.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/levit256_profiler_results_compiled.html>`_
     - 224x224x3
     - 18.9
     - 2.3
   * - levit384
     - 78.9
     - 75.5
     - 119
     - 455
     - `S <https://github.com/facebookresearch/LeViT>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/levit_384/pretrained/2024-07-10/LeViT_384_simp.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/levit384.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/levit384_profiler_results_compiled.html>`_
     - 224x224x3
     - 39.1
     - 4.7
   * - mobilenet_v1
     - 70.3
     - 69.6
     - 3676
     - 3677
     - `S <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v1/pretrained/2023-07-18/mobilenet_v1.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/mobilenet_v1.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/mobilenet_v1_profiler_results_compiled.html>`_
     - 224x224x3
     - 4.22
     - 1.14
   * - mobilenet_v2_1.0
     - 71.0
     - 70.2
     - 3454
     - 3454
     - `S <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v2_1.0/pretrained/2025-01-15/mobilenet_v2_1.0.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/mobilenet_v2_1.0.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/mobilenet_v2_1.0_profiler_results_compiled.html>`_
     - 224x224x3
     - 3.49
     - 0.62
   * - mobilenet_v2_1.4
     - 73.2
     - 72.2
     - 1794
     - 1794
     - `S <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v2_1.4/pretrained/2021-07-11/mobilenet_v2_1.4.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/mobilenet_v2_1.4.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/mobilenet_v2_1.4_profiler_results_compiled.html>`_
     - 224x224x3
     - 6.09
     - 1.18
   * - mobilenet_v3
     - 71.8
     - 71.4
     - 2879
     - 2879
     - `S <https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v3/pretrained/2023-07-18/mobilenet_v3.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/mobilenet_v3.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/mobilenet_v3_profiler_results_compiled.html>`_
     - 224x224x3
     - 4.07
     - 2
   * - mobilenet_v3_large_minimalistic
     - 70.6
     - 69.1
     - 3540
     - 3540
     - `S <https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v3_large_minimalistic/pretrained/2021-07-11/mobilenet_v3_large_minimalistic.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/mobilenet_v3_large_minimalistic.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/mobilenet_v3_large_minimalistic_profiler_results_compiled.html>`_
     - 224x224x3
     - 3.91
     - 0.42
   * - regnetx_1.6gf
     - 76.6
     - 76.2
     - 2741
     - 2740
     - `S <https://github.com/facebookresearch/pycls>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/regnetx_1.6gf/pretrained/2021-07-11/regnetx_1.6gf.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/regnetx_1.6gf.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/regnetx_1.6gf_profiler_results_compiled.html>`_
     - 224x224x3
     - 9.17
     - 3.22
   * - regnetx_800mf
     - 74.8
     - 74.4
     - 4506
     - 4510
     - `S <https://github.com/facebookresearch/pycls>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/regnetx_800mf/pretrained/2021-07-11/regnetx_800mf.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/regnetx_800mf.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/regnetx_800mf_profiler_results_compiled.html>`_
     - 224x224x3
     - 7.24
     - 1.6
   * - repghost_1_0x
     - 72.4
     - 71.8
     - 332
     - 1195
     - `S <https://github.com/ChengpengChen/RepGhost>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/repghost/repghostnet_1_0x/pretrained/2023-04-03/repghostnet_1_0x.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/repghost_1_0x.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/repghost_1_0x_profiler_results_compiled.html>`_
     - 224x224x3
     - 4.1
     - 0.28
   * - repghost_2_0x
     - 76.9
     - 76.6
     - 200
     - 705
     - `S <https://github.com/ChengpengChen/RepGhost>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/repghost/repghostnet_2_0x/pretrained/2023-04-03/repghostnet_2_0x.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/repghost_2_0x.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/repghost_2_0x_profiler_results_compiled.html>`_
     - 224x224x3
     - 9.8
     - 1.04
   * - repvgg_a1
     - 72.2
     - 70.0
     - 2018
     - 2018
     - `S <https://github.com/DingXiaoH/RepVGG>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/repvgg/repvgg_a1/pretrained/2022-10-02/RepVGG-A1.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/repvgg_a1.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/repvgg_a1_profiler_results_compiled.html>`_
     - 224x224x3
     - 12.79
     - 4.7
   * - repvgg_a2
     - 74.5
     - 72.5
     - 289
     - 637
     - `S <https://github.com/DingXiaoH/RepVGG>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/repvgg/repvgg_a2/pretrained/2022-10-02/RepVGG-A2.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/repvgg_a2.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/repvgg_a2_profiler_results_compiled.html>`_
     - 224x224x3
     - 25.5
     - 10.2
   * - resmlp12_relu
     - 74.9
     - 74.5
     - 90
     - 313
     - `S <https://github.com/rwightman/pytorch-image-models/>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resmlp12_relu/pretrained/2022-03-03/resmlp12_relu.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/resmlp12_relu.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/resmlp12_relu_profiler_results_compiled.html>`_
     - 224x224x3
     - 15.77
     - 6.04
   * - resnet_v1_18
     - 70.8
     - 70.3
     - 2708
     - 2708
     - `S <https://github.com/yhhhli/BRECQ>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v1_18/pretrained/2022-04-19/resnet_v1_18.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/resnet_v1_18.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/resnet_v1_18_profiler_results_compiled.html>`_
     - 224x224x3
     - 11.68
     - 3.64
   * - resnet_v1_34
     - 72.3
     - 71.9
     - 352
     - 1012
     - `S <https://github.com/tensorflow/models/tree/master/research/slim>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v1_34/pretrained/2025-01-15/resnet_v1_34.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/resnet_v1_34.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/resnet_v1_34_profiler_results_compiled.html>`_
     - 224x224x3
     - 21.79
     - 7.34
   * - resnet_v1_50 |rocket| |star|
     - 74.6
     - 74.0
     - 313
     - 1044
     - `S <https://github.com/tensorflow/models/tree/master/research/slim>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v1_50/pretrained/2025-01-15/resnet_v1_50.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/resnet_v1_50.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/resnet_v1_50_profiler_results_compiled.html>`_
     - 224x224x3
     - 25.53
     - 6.98
   * - resnext26_32x4d
     - 75.9
     - 75.6
     - 803
     - 803
     - `S <https://github.com/osmr/imgclsmob/tree/master/pytorch>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnext26_32x4d/pretrained/2023-09-18/resnext26_32x4d.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/resnext26_32x4d.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/resnext26_32x4d_profiler_results_compiled.html>`_
     - 224x224x3
     - 15.37
     - 4.96
   * - resnext50_32x4d
     - 78.4
     - 77.5
     - 263
     - 731
     - `S <https://github.com/osmr/imgclsmob/tree/master/pytorch>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnext50_32x4d/pretrained/2023-07-18/resnext50_32x4d.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/resnext50_32x4d.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/resnext50_32x4d_profiler_results_compiled.html>`_
     - 224x224x3
     - 24.99
     - 8.48
   * - squeezenet_v1.1
     - 59.3
     - 58.8
     - 4308
     - 4308
     - `S <https://github.com/osmr/imgclsmob/tree/master/pytorch>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/squeezenet_v1.1/pretrained/2023-07-18/squeezenet_v1.1.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/squeezenet_v1.1.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/squeezenet_v1.1_profiler_results_compiled.html>`_
     - 224x224x3
     - 1.24
     - 0.78
   * - swin_small
     - 80.0
     - 76.9
     - 19
     - 53
     - `S <https://huggingface.co/microsoft/swin-small-patch4-window7-224>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/swin_small/pretrained/2024-08-01/swin_small_classifier.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/swin_small.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/swin_small_profiler_results_compiled.html>`_
     - 224x224x3
     - 50
     - 17.6
   * - swin_tiny
     - 79.4
     - 77.5
     - 38
     - 98
     - `S <https://huggingface.co/microsoft/swin-tiny-patch4-window7-224>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/swin_tiny/pretrained/2024-08-01/swin_tiny_classifier.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/swin_tiny.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/swin_tiny_profiler_results_compiled.html>`_
     - 224x224x3
     - 29
     - 9.1
   * - vit_base
     - 83.6
     - 82.7
     - 48
     - 135
     - `S <https://github.com/rwightman/pytorch-image-models>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_base/pretrained/2024-04-03/vit_base_patch16_224_ops17.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/vit_base.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/vit_base_profiler_results_compiled.html>`_
     - 224x224x3
     - 86.5
     - 35.188
   * - vit_base_bn |rocket|
     - 79.1
     - 78.2
     - 65
     - 210
     - `S <https://github.com/rwightman/pytorch-image-models>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_base_bn/pretrained/2023-01-25/vit_base.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/vit_base_bn.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/vit_base_bn_profiler_results_compiled.html>`_
     - 224x224x3
     - 86.5
     - 35.188
   * - vit_small
     - 80.5
     - 79.5
     - 113
     - 406
     - `S <https://github.com/rwightman/pytorch-image-models>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_small/pretrained/2024-04-03/vit_small_patch16_224_ops17.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/vit_small.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/vit_small_profiler_results_compiled.html>`_
     - 224x224x3
     - 21.12
     - 8.62
   * - vit_small_bn
     - 77.4
     - 76.7
     - 156
     - 550
     - `S <https://github.com/rwightman/pytorch-image-models>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_small_bn/pretrained/2022-08-08/vit_small.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/vit_small_bn.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/vit_small_bn_profiler_results_compiled.html>`_
     - 224x224x3
     - 21.12
     - 8.62
   * - vit_tiny
     - 74.5
     - 73.5
     - 137
     - 458
     - `S <https://github.com/rwightman/pytorch-image-models>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_tiny/pretrained/2024-04-03/vit_tiny_patch16_224_ops17.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/vit_tiny.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/vit_tiny_profiler_results_compiled.html>`_
     - 224x224x3
     - 5.73
     - 2.2
   * - vit_tiny_bn
     - 67.4
     - 65.8
     - 346
     - 1591
     - `S <https://github.com/rwightman/pytorch-image-models>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_tiny_bn/pretrained/2023-08-29/vit_tiny_bn.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/vit_tiny_bn.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/vit_tiny_bn_profiler_results_compiled.html>`_
     - 224x224x3
     - 5.73
     - 2.2
.. list-table::
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
   * - efficientnet_lite3
     - 297
     - 371
     - `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite3/pretrained/2023-07-18/efficientnet_lite3.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/efficientnet_lite3.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/efficientnet_lite3_profiler_results_compiled.html>`_
     - 280x280x3
     - 8.16
     - 2.8
