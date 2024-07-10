
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
* All models were compiled using Hailo Dataflow Compiler v3.28.0
* Supported tasks:

  * `Classification`_


.. _Classification:

Classification
--------------

ImageNet
^^^^^^^^

.. list-table::
   :widths: 31 9 7 11 9 8 8 8 7 7
   :header-rows: 1

   * - Network Name
     - Accuracy (top1)
     - Quantized
     - FPS
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
   * - deit_base
     - 80.93
     - 80.51
     - 31
     - 224x224x3
     - 80.26
     - 35.22
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/deit_base/pretrained/2024-05-21/deit_base.zip>`_
     - `link <https://github.com/facebookresearch/deit>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15h/deit_base.hef>`_
   * - deit_small
     - 76.25
     - 75.15
     - 43
     - 224x224x3
     - 20.52
     - 9.4
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/deit_small/pretrained/2024-05-21/deit_small.zip>`_
     - `link <https://github.com/facebookresearch/deit>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15h/deit_small.hef>`_
   * - deit_tiny
     - 69.07
     - 65.65
     - 68
     - 224x224x3
     - 5.3
     - 2.57
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/deit_tiny/pretrained/2024-05-21/deit_tiny.zip>`_
     - `link <https://github.com/facebookresearch/deit>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15h/deit_tiny.hef>`_
   * - efficientnet_l
     - 80.46
     - 79.36
     - 0
     - 300x300x3
     - 10.55
     - 19.4
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_l/pretrained/2023-07-18/efficientnet_l.zip>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15h/efficientnet_l.hef>`_
   * - efficientnet_lite0
     - 74.99
     - 73.81
     - 1007
     - 224x224x3
     - 4.63
     - 0.78
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite0/pretrained/2023-07-18/efficientnet_lite0.zip>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15h/efficientnet_lite0.hef>`_
   * - efficientnet_lite1
     - 76.68
     - 76.21
     - 273
     - 240x240x3
     - 5.39
     - 1.22
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite1/pretrained/2023-07-18/efficientnet_lite1.zip>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15h/efficientnet_lite1.hef>`_
   * - efficientnet_lite2
     - 77.45
     - 76.74
     - 179
     - 260x260x3
     - 6.06
     - 1.74
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite2/pretrained/2023-07-18/efficientnet_lite2.zip>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15h/efficientnet_lite2.hef>`_
   * - efficientnet_lite3
     - 79.29
     - 78.42
     - 0
     - 280x280x3
     - 8.16
     - 2.8
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite3/pretrained/2023-07-18/efficientnet_lite3.zip>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15h/efficientnet_lite3.hef>`_
   * - efficientnet_lite4
     - 80.79
     - 80.19
     - 98
     - 300x300x3
     - 12.95
     - 5.10
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite4/pretrained/2023-07-18/efficientnet_lite4.zip>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15h/efficientnet_lite4.hef>`_
   * - efficientnet_m
     - 78.91
     - 78.43
     - 166
     - 240x240x3
     - 6.87
     - 7.32
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_m/pretrained/2023-07-18/efficientnet_m.zip>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15h/efficientnet_m.hef>`_
   * - efficientnet_s
     - 77.64
     - 77.32
     - 0
     - 224x224x3
     - 5.41
     - 4.72
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_s/pretrained/2023-07-18/efficientnet_s.zip>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15h/efficientnet_s.hef>`_
   * - fastvit_sa12
     - 76.8
     - 73.9
     - 189
     - 224x224x3
     - 11.99
     - 3.59
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/fastvit_sa12/pretrained/2023-08-21/fastvit_sa12.zip>`_
     - `link <https://github.com/apple/ml-fastvit/tree/main>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15h/fastvit_sa12.hef>`_
   * - hardnet39ds
     - 73.43
     - 73.03
     - 348
     - 224x224x3
     - 3.48
     - 0.86
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/hardnet39ds/pretrained/2021-07-20/hardnet39ds.zip>`_
     - `link <https://github.com/PingoLH/Pytorch-HarDNet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15h/hardnet39ds.hef>`_
   * - hardnet68
     - 75.47
     - 75.22
     - 149
     - 224x224x3
     - 17.56
     - 8.5
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/hardnet68/pretrained/2021-07-20/hardnet68.zip>`_
     - `link <https://github.com/PingoLH/Pytorch-HarDNet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15h/hardnet68.hef>`_
   * - inception_v1
     - 69.74
     - 69.54
     - 336
     - 224x224x3
     - 6.62
     - 3
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/inception_v1/pretrained/2023-07-18/inception_v1.zip>`_
     - `link <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15h/inception_v1.hef>`_
   * - mobilenet_v1
     - 70.97
     - 70.26
     - 1516
     - 224x224x3
     - 4.22
     - 1.14
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v1/pretrained/2023-07-18/mobilenet_v1.zip>`_
     - `link <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15h/mobilenet_v1.hef>`_
   * - mobilenet_v2_1.4
     - 74.18
     - 73.18
     - 579
     - 224x224x3
     - 6.09
     - 1.18
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v2_1.4/pretrained/2021-07-11/mobilenet_v2_1.4.zip>`_
     - `link <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15h/mobilenet_v2_1.4.hef>`_
   * - mobilenet_v3
     - 72.21
     - 71.73
     - 373
     - 224x224x3
     - 4.07
     - 2
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v3/pretrained/2023-07-18/mobilenet_v3.zip>`_
     - `link <https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15h/mobilenet_v3.hef>`_
   * - regnetx_1.6gf
     - 77.05
     - 76.75
     - 362
     - 224x224x3
     - 9.17
     - 3.22
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/regnetx_1.6gf/pretrained/2021-07-11/regnetx_1.6gf.zip>`_
     - `link <https://github.com/facebookresearch/pycls>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15h/regnetx_1.6gf.hef>`_
   * - repghost_1_0x
     - 73.03
     - 72.35
     - 228
     - 224x224x3
     - 4.1
     - 0.28
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/repghost/repghostnet_1_0x/pretrained/2023-04-03/repghostnet_1_0x.zip>`_
     - `link <https://github.com/ChengpengChen/RepGhost>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15h/repghost_1_0x.hef>`_
   * - repghost_2_0x
     - 77.18
     - 76.82
     - 138
     - 224x224x3
     - 9.8
     - 1.04
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/repghost/repghostnet_2_0x/pretrained/2023-04-03/repghostnet_2_0x.zip>`_
     - `link <https://github.com/ChengpengChen/RepGhost>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15h/repghost_2_0x.hef>`_
   * - repvgg_a2
     - 76.52
     - 74.52
     - 248
     - 224x224x3
     - 25.5
     - 10.2
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/repvgg/repvgg_a2/pretrained/2022-10-02/RepVGG-A2.zip>`_
     - `link <https://github.com/DingXiaoH/RepVGG>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15h/repvgg_a2.hef>`_
   * - resmlp12_relu
     - 75.26
     - 74.32
     - 94
     - 224x224x3
     - 15.77
     - 6.04
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resmlp12_relu/pretrained/2022-03-03/resmlp12_relu.zip>`_
     - `link <https://github.com/rwightman/pytorch-image-models/>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15h/resmlp12_relu.hef>`_
   * - resnet_v1_34
     - 72.7
     - 72.22
     - 260
     - 224x224x3
     - 21.79
     - 7.34
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v1_34/pretrained/2021-07-11/resnet_v1_34.zip>`_
     - `link <https://github.com/tensorflow/models/tree/master/research/slim>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15h/resnet_v1_34.hef>`_
   * - resnet_v1_50 |rocket| |star|
     - 75.12
     - 74.47
     - 282
     - 224x224x3
     - 25.53
     - 6.98
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v1_50/pretrained/2021-07-11/resnet_v1_50.zip>`_
     - `link <https://github.com/tensorflow/models/tree/master/research/slim>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15h/resnet_v1_50.hef>`_
   * - resnext26_32x4d
     - 76.18
     - 75.93
     - 342
     - 224x224x3
     - 15.37
     - 4.96
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnext26_32x4d/pretrained/2023-09-18/resnext26_32x4d.zip>`_
     - `link <https://github.com/osmr/imgclsmob/tree/master/pytorch>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15h/resnext26_32x4d.hef>`_
   * - resnext50_32x4d
     - 79.31
     - 78.41
     - 204
     - 224x224x3
     - 24.99
     - 8.48
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnext50_32x4d/pretrained/2023-07-18/resnext50_32x4d.zip>`_
     - `link <https://github.com/osmr/imgclsmob/tree/master/pytorch>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15h/resnext50_32x4d.hef>`_
   * - vit_base
     - 84.55
     - 83.12
     - 0
     - 224x224x3
     - 86.5
     - 35.188
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_base/pretrained/2024-04-03/vit_base_patch16_224_ops17.zip>`_
     - `link <https://github.com/rwightman/pytorch-image-models>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15h/vit_base.hef>`_
   * - vit_base_bn |rocket|
     - 79.98
     - 78.58
     - 59
     - 224x224x3
     - 86.5
     - 35.188
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_base_bn/pretrained/2023-01-25/vit_base.zip>`_
     - `link <https://github.com/rwightman/pytorch-image-models>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15h/vit_base_bn.hef>`_
   * - vit_small
     - 81.54
     - 79.84
     - 0
     - 224x224x3
     - 21.12
     - 8.62
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_small/pretrained/2024-04-03/vit_small_patch16_224_ops17.zip>`_
     - `link <https://github.com/rwightman/pytorch-image-models>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15h/vit_small.hef>`_
   * - vit_small_bn
     - 78.12
     - 77.22
     - 125
     - 224x224x3
     - 21.12
     - 8.62
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_small_bn/pretrained/2022-08-08/vit_small.zip>`_
     - `link <https://github.com/rwightman/pytorch-image-models>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15h/vit_small_bn.hef>`_
   * - vit_tiny
     - 75.51
     - 73.49
     - 0
     - 224x224x3
     - 5.73
     - 2.2
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_tiny/pretrained/2024-04-03/vit_tiny_patch16_224_ops17.zip>`_
     - `link <https://github.com/rwightman/pytorch-image-models>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15h/vit_tiny.hef>`_
   * - vit_tiny_bn
     - 68.95
     - 67.15
     - 233
     - 224x224x3
     - 5.73
     - 2.2
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_tiny_bn/pretrained/2023-08-29/vit_tiny_bn.zip>`_
     - `link <https://github.com/rwightman/pytorch-image-models>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15h/vit_tiny_bn.hef>`_
