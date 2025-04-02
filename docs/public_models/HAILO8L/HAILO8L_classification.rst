
Public Pre-Trained Models
=========================

.. |rocket| image:: ../../images/rocket.png
  :width: 18

.. |star| image:: ../../images/star.png
  :width: 18

Here, we give the full list of publicly pre-trained models supported by the Hailo Model Zoo.

* Network available in `Hailo Benchmark <https://hailo.ai/products/ai-accelerators/hailo-8l-ai-accelerator-for-ai-light-applications/#hailo8l-benchmarks/>`_ are marked with |rocket|
* Networks available in `TAPPAS <https://github.com/hailo-ai/tappas>`_ are marked with |star|
* Benchmark and TAPPAS  networks run in performance mode
* All models were compiled using Hailo Dataflow Compiler v3.31.0



.. _Classification:

--------------

ImageNet
^^^^^^^^

.. list-table::
   :widths: 31 9 7 11 9 8 8 8 7 7 7 7
   :header-rows: 1

   * - Network Name
     - Accuracy (top1)
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
   * - cas_vit_m
     - 81.2
     - 81.01
     - 38
     - 103
     - 384x384x3
     - 12.42
     - 10.89
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/cas_vit_m/pretrained/2024-09-03/cas_vit_m.zip>`_
     - `link <https://github.com/Tianfang-Zhang/CAS-ViT>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/cas_vit_m.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/cas_vit_m_profiler_results_compiled.html>`_
   * - cas_vit_s
     - 79.93
     - 79.65
     - 55
     - 153
     - 384x384x3
     - 5.5
     - 5.4
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/cas_vit_s/pretrained/2024-08-13/cas_vit_s.zip>`_
     - `link <https://github.com/Tianfang-Zhang/CAS-ViT>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/cas_vit_s.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/cas_vit_s_profiler_results_compiled.html>`_
   * - cas_vit_t
     - 81.9
     - 81.52
     - 27
     - 63
     - 384x384x3
     - 21.76
     - 20.85
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/cas_vit_t/pretrained/2024-09-03/cas_vit_t.zip>`_
     - `link <https://github.com/Tianfang-Zhang/CAS-ViT>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/cas_vit_t.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/cas_vit_t_profiler_results_compiled.html>`_
   * - deit_base
     - 80.93
     - 79.9
     - 20
     - 58
     - 224x224x3
     - 80.26
     - 35.22
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/deit_base/pretrained/2024-05-21/deit_base.zip>`_
     - `link <https://github.com/facebookresearch/deit>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/deit_base.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/deit_base_profiler_results_compiled.html>`_
   * - deit_small
     - 78.25
     - 77.18
     - 53
     - 203
     - 224x224x3
     - 20.52
     - 9.4
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/deit_small/pretrained/2024-05-21/deit_small.zip>`_
     - `link <https://github.com/facebookresearch/deit>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/deit_small.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/deit_small_profiler_results_compiled.html>`_
   * - deit_tiny
     - 69.07
     - 68.57
     - 68
     - 256
     - 224x224x3
     - 5.3
     - 2.57
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/deit_tiny/pretrained/2024-05-21/deit_tiny.zip>`_
     - `link <https://github.com/facebookresearch/deit>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/deit_tiny.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/deit_tiny_profiler_results_compiled.html>`_
   * - efficientformer_l1
     - 79.13
     - 76.46
     - 79
     - 204
     - 224x224x3
     - 12.3
     - 2.6
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientformer_l1/pretrained/2024-08-11/efficientformer_l1.zip>`_
     - `link <https://github.com/snap-research/EfficientFormer/tree/main>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/efficientformer_l1.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/efficientformer_l1_profiler_results_compiled.html>`_
   * - efficientnet_l
     - 80.47
     - 79.28
     - 78
     - 162
     - 300x300x3
     - 10.55
     - 19.4
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_l/pretrained/2023-07-18/efficientnet_l.zip>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/efficientnet_l.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/efficientnet_l_profiler_results_compiled.html>`_
   * - efficientnet_lite0
     - 74.99
     - 73.84
     - 1057
     - 1057
     - 224x224x3
     - 4.63
     - 0.78
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite0/pretrained/2023-07-18/efficientnet_lite0.zip>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/efficientnet_lite0.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/efficientnet_lite0_profiler_results_compiled.html>`_
   * - efficientnet_lite1
     - 76.67
     - 76.25
     - 159
     - 477
     - 240x240x3
     - 5.39
     - 1.22
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite1/pretrained/2023-07-18/efficientnet_lite1.zip>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/efficientnet_lite1.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/efficientnet_lite1_profiler_results_compiled.html>`_
   * - efficientnet_lite2
     - 77.46
     - 76.67
     - 108
     - 274
     - 260x260x3
     - 6.06
     - 1.74
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite2/pretrained/2023-07-18/efficientnet_lite2.zip>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/efficientnet_lite2.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/efficientnet_lite2_profiler_results_compiled.html>`_
   * - efficientnet_lite3
     - 79.29
     - 78.66
     - 111
     - 305
     - 280x280x3
     - 8.16
     - 2.8
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite3/pretrained/2023-07-18/efficientnet_lite3.zip>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/efficientnet_lite3.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/efficientnet_lite3_profiler_results_compiled.html>`_
   * - efficientnet_lite4
     - 80.79
     - 80.08
     - 73
     - 195
     - 300x300x3
     - 12.95
     - 5.10
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite4/pretrained/2023-07-18/efficientnet_lite4.zip>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/efficientnet_lite4.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/efficientnet_lite4_profiler_results_compiled.html>`_
   * - efficientnet_m
     - 78.91
     - 78.45
     - 155
     - 408
     - 240x240x3
     - 6.87
     - 7.32
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_m/pretrained/2023-07-18/efficientnet_m.zip>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/efficientnet_m.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/efficientnet_m_profiler_results_compiled.html>`_
   * - efficientnet_s
     - 77.63
     - 77.24
     - 548
     - 548
     - 224x224x3
     - 5.41
     - 4.72
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_s/pretrained/2023-07-18/efficientnet_s.zip>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/efficientnet_s.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/efficientnet_s_profiler_results_compiled.html>`_
   * - fastvit_sa12
     - 79.8
     - 76.56
     - 138
     - 535
     - 224x224x3
     - 11.99
     - 3.59
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/fastvit_sa12/pretrained/2023-08-21/fastvit_sa12.zip>`_
     - `link <https://github.com/apple/ml-fastvit/tree/main>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/fastvit_sa12.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/fastvit_sa12_profiler_results_compiled.html>`_
   * - hardnet39ds
     - 73.43
     - 73.01
     - 277
     - 933
     - 224x224x3
     - 3.48
     - 0.86
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/hardnet39ds/pretrained/2021-07-20/hardnet39ds.zip>`_
     - `link <https://github.com/PingoLH/Pytorch-HarDNet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/hardnet39ds.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/hardnet39ds_profiler_results_compiled.html>`_
   * - hardnet68
     - 75.47
     - 75.22
     - 115
     - 298
     - 224x224x3
     - 17.56
     - 8.5
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/hardnet68/pretrained/2021-07-20/hardnet68.zip>`_
     - `link <https://github.com/PingoLH/Pytorch-HarDNet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/hardnet68.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/hardnet68_profiler_results_compiled.html>`_
   * - inception_v1
     - 69.74
     - 69.56
     - 309
     - 882
     - 224x224x3
     - 6.62
     - 3
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/inception_v1/pretrained/2023-07-18/inception_v1.zip>`_
     - `link <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/inception_v1.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/inception_v1_profiler_results_compiled.html>`_
   * - levit128
     - 78.4
     - 75.94
     - 102
     - 494
     - 224x224x3
     - 9.2
     - 0.8
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/levit_128/pretrained/2024-07-10/LeViT_128_simp.zip>`_
     - `link <https://github.com/facebookresearch/LeViT>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/levit128.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/levit128_profiler_results_compiled.html>`_
   * - levit192
     - 79.7
     - 76.97
     - 98
     - 491
     - 224x224x3
     - 10.9
     - 1.3
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/levit_192/pretrained/2024-07-10/LeViT_192_simp.zip>`_
     - `link <https://github.com/facebookresearch/LeViT>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/levit192.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/levit192_profiler_results_compiled.html>`_
   * - levit256
     - 81.4
     - 78.95
     - 78
     - 389
     - 224x224x3
     - 18.9
     - 2.3
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/levit_256/2024-05-13/levit-256.zip>`_
     - `link <https://github.com/facebookresearch/LeViT>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/levit256.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/levit256_profiler_results_compiled.html>`_
   * - levit384
     - 82.3
     - 78.78
     - 51
     - 232
     - 224x224x3
     - 39.1
     - 4.7
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/levit_384/pretrained/2024-07-10/LeViT_384_simp.zip>`_
     - `link <https://github.com/facebookresearch/LeViT>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/levit384.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/levit384_profiler_results_compiled.html>`_
   * - mobilenet_v1
     - 70.97
     - 70.35
     - 2565
     - 2565
     - 224x224x3
     - 4.22
     - 1.14
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v1/pretrained/2023-07-18/mobilenet_v1.zip>`_
     - `link <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/mobilenet_v1.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/mobilenet_v1_profiler_results_compiled.html>`_
   * - mobilenet_v2_1.0
     - 71.78
     - 70.96
     - 1738
     - 1738
     - 224x224x3
     - 3.49
     - 0.62
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v2_1.0/pretrained/2025-01-15/mobilenet_v2_1.0.zip>`_
     - `link <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/mobilenet_v2_1.0.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/mobilenet_v2_1.0_profiler_results_compiled.html>`_
   * - mobilenet_v2_1.4
     - 74.18
     - 73.21
     - 1098
     - 1098
     - 224x224x3
     - 6.09
     - 1.18
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v2_1.4/pretrained/2021-07-11/mobilenet_v2_1.4.zip>`_
     - `link <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/mobilenet_v2_1.4.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/mobilenet_v2_1.4_profiler_results_compiled.html>`_
   * - mobilenet_v3
     - 72.21
     - 71.77
     - 1936
     - 1936
     - 224x224x3
     - 4.07
     - 2
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v3/pretrained/2023-07-18/mobilenet_v3.zip>`_
     - `link <https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/mobilenet_v3.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/mobilenet_v3_profiler_results_compiled.html>`_
   * - mobilenet_v3_large_minimalistic
     - 72.12
     - 70.59
     - 2778
     - 2778
     - 224x224x3
     - 3.91
     - 0.42
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v3_large_minimalistic/pretrained/2021-07-11/mobilenet_v3_large_minimalistic.zip>`_
     - `link <https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/mobilenet_v3_large_minimalistic.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/mobilenet_v3_large_minimalistic_profiler_results_compiled.html>`_
   * - regnetx_1.6gf
     - 77.05
     - 76.68
     - 1339
     - 1339
     - 224x224x3
     - 9.17
     - 3.22
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/regnetx_1.6gf/pretrained/2021-07-11/regnetx_1.6gf.zip>`_
     - `link <https://github.com/facebookresearch/pycls>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/regnetx_1.6gf.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/regnetx_1.6gf_profiler_results_compiled.html>`_
   * - regnetx_800mf
     - 75.16
     - 74.91
     - 2450
     - 2450
     - 224x224x3
     - 7.24
     - 1.6
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/regnetx_800mf/pretrained/2021-07-11/regnetx_800mf.zip>`_
     - `link <https://github.com/facebookresearch/pycls>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/regnetx_800mf.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/regnetx_800mf_profiler_results_compiled.html>`_
   * - repghost_1_0x
     - 73.03
     - 72.24
     - 189
     - 804
     - 224x224x3
     - 4.1
     - 0.28
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/repghost/repghostnet_1_0x/pretrained/2023-04-03/repghostnet_1_0x.zip>`_
     - `link <https://github.com/ChengpengChen/RepGhost>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/repghost_1_0x.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/repghost_1_0x_profiler_results_compiled.html>`_
   * - repghost_2_0x
     - 77.18
     - 76.91
     - 111
     - 426
     - 224x224x3
     - 9.8
     - 1.04
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/repghost/repghostnet_2_0x/pretrained/2023-04-03/repghostnet_2_0x.zip>`_
     - `link <https://github.com/ChengpengChen/RepGhost>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/repghost_2_0x.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/repghost_2_0x_profiler_results_compiled.html>`_
   * - repvgg_a1
     - 74.4
     - 72.32
     - 1346
     - 1346
     - 224x224x3
     - 12.79
     - 4.7
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/repvgg/repvgg_a1/pretrained/2022-10-02/RepVGG-A1.zip>`_
     - `link <https://github.com/DingXiaoH/RepVGG>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/repvgg_a1.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/repvgg_a1_profiler_results_compiled.html>`_
   * - repvgg_a2
     - 76.52
     - 74.42
     - 145
     - 335
     - 224x224x3
     - 25.5
     - 10.2
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/repvgg/repvgg_a2/pretrained/2022-10-02/RepVGG-A2.zip>`_
     - `link <https://github.com/DingXiaoH/RepVGG>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/repvgg_a2.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/repvgg_a2_profiler_results_compiled.html>`_
   * - resmlp12_relu
     - 75.27
     - 74.95
     - 45
     - 192
     - 224x224x3
     - 15.77
     - 6.04
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resmlp12_relu/pretrained/2022-03-03/resmlp12_relu.zip>`_
     - `link <https://github.com/rwightman/pytorch-image-models/>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/resmlp12_relu.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/resmlp12_relu_profiler_results_compiled.html>`_
   * - resnet_v1_18
     - 71.27
     - 71.07
     - 1281
     - 1281
     - 224x224x3
     - 11.68
     - 3.64
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v1_18/pretrained/2022-04-19/resnet_v1_18.zip>`_
     - `link <https://github.com/yhhhli/BRECQ>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/resnet_v1_18.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/resnet_v1_18_profiler_results_compiled.html>`_
   * - resnet_v1_34
     - 72.7
     - 72.23
     - 162
     - 484
     - 224x224x3
     - 21.79
     - 7.34
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v1_34/pretrained/2025-01-15/resnet_v1_34.zip>`_
     - `link <https://github.com/tensorflow/models/tree/master/research/slim>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/resnet_v1_34.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/resnet_v1_34_profiler_results_compiled.html>`_
   * - resnet_v1_50 |rocket| |star|
     - 75.21
     - 74.62
     - 138
     - 501
     - 224x224x3
     - 25.53
     - 6.98
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v1_50/pretrained/2025-01-15/resnet_v1_50.zip>`_
     - `link <https://github.com/tensorflow/models/tree/master/research/slim>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/resnet_v1_50.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/resnet_v1_50_profiler_results_compiled.html>`_
   * - resnext26_32x4d
     - 76.17
     - 75.89
     - 201
     - 505
     - 224x224x3
     - 15.37
     - 4.96
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnext26_32x4d/pretrained/2023-09-18/resnext26_32x4d.zip>`_
     - `link <https://github.com/osmr/imgclsmob/tree/master/pytorch>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/resnext26_32x4d.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/resnext26_32x4d_profiler_results_compiled.html>`_
   * - resnext50_32x4d
     - 79.3
     - 78.35
     - 126
     - 348
     - 224x224x3
     - 24.99
     - 8.48
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnext50_32x4d/pretrained/2023-07-18/resnext50_32x4d.zip>`_
     - `link <https://github.com/osmr/imgclsmob/tree/master/pytorch>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/resnext50_32x4d.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/resnext50_32x4d_profiler_results_compiled.html>`_
   * - squeezenet_v1.1
     - 59.85
     - 59.34
     - 2514
     - 2514
     - 224x224x3
     - 1.24
     - 0.78
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/squeezenet_v1.1/pretrained/2023-07-18/squeezenet_v1.1.zip>`_
     - `link <https://github.com/osmr/imgclsmob/tree/master/pytorch>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/squeezenet_v1.1.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/squeezenet_v1.1_profiler_results_compiled.html>`_
   * - swin_small
     - 83.13
     - 79.96
     - 12
     - 31
     - 224x224x3
     - 50
     - 17.6
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/swin_small/pretrained/2024-08-01/swin_small_classifier.zip>`_
     - `link <https://huggingface.co/microsoft/swin-small-patch4-window7-224>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/swin_small.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/swin_small_profiler_results_compiled.html>`_
   * - swin_tiny
     - 81.3
     - 79.25
     - 23
     - 58
     - 224x224x3
     - 29
     - 9.1
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/swin_tiny/pretrained/2024-08-01/swin_tiny_classifier.zip>`_
     - `link <https://huggingface.co/microsoft/swin-tiny-patch4-window7-224>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/swin_tiny.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/swin_tiny_profiler_results_compiled.html>`_
   * - vit_base
     - 84.5
     - 83.13
     - 20
     - 58
     - 224x224x3
     - 86.5
     - 35.188
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_base/pretrained/2024-04-03/vit_base_patch16_224_ops17.zip>`_
     - `link <https://github.com/rwightman/pytorch-image-models>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/vit_base.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/vit_base_profiler_results_compiled.html>`_
   * - vit_base_bn |rocket|
     - 79.98
     - 78.57
     - 35
     - 121
     - 224x224x3
     - 86.5
     - 35.188
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_base_bn/pretrained/2023-01-25/vit_base.zip>`_
     - `link <https://github.com/rwightman/pytorch-image-models>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/vit_base_bn.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/vit_base_bn_profiler_results_compiled.html>`_
   * - vit_small
     - 81.5
     - 79.96
     - 54
     - 205
     - 224x224x3
     - 21.12
     - 8.62
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_small/pretrained/2024-04-03/vit_small_patch16_224_ops17.zip>`_
     - `link <https://github.com/rwightman/pytorch-image-models>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/vit_small.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/vit_small_profiler_results_compiled.html>`_
   * - vit_small_bn
     - 78.12
     - 77.24
     - 97
     - 414
     - 224x224x3
     - 21.12
     - 8.62
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_small_bn/pretrained/2022-08-08/vit_small.zip>`_
     - `link <https://github.com/rwightman/pytorch-image-models>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/vit_small_bn.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/vit_small_bn_profiler_results_compiled.html>`_
   * - vit_tiny
     - 75.51
     - 73.82
     - 67
     - 256
     - 224x224x3
     - 5.73
     - 2.2
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_tiny/pretrained/2024-04-03/vit_tiny_patch16_224_ops17.zip>`_
     - `link <https://github.com/rwightman/pytorch-image-models>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/vit_tiny.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/vit_tiny_profiler_results_compiled.html>`_
   * - vit_tiny_bn
     - 68.95
     - 67.28
     - 193
     - 950
     - 224x224x3
     - 5.73
     - 2.2
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_tiny_bn/pretrained/2023-08-29/vit_tiny_bn.zip>`_
     - `link <https://github.com/rwightman/pytorch-image-models>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/vit_tiny_bn.hef>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/vit_tiny_bn_profiler_results_compiled.html>`_
