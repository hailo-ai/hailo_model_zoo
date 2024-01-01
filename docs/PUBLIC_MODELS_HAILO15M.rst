
Public Pre-Trained Models
=========================

.. |rocket| image:: images/rocket.png
  :width: 18

.. |star| image:: images/star.png
  :width: 18

Here, we give the full list of publicly pre-trained models supported by the Hailo Model Zoo.

* Network available in `Hailo Benchmark <https://hailo.ai/developer-zone/benchmarks/>`_ are marked with |rocket|
* Networks available in `TAPPAS <https://hailo.ai/developer-zone/tappas-apps-toolkit/>`_ are marked with |star|
* Benchmark, TAPPAS and Recommended networks run in performance mode
* All models were compiled using Hailo Dataflow Compiler v3.26.0
* Supported tasks:

  * `Classification`_
  * `Object Detection`_
  * `Semantic Segmentation`_
  * `Pose Estimation`_
  * `Single Person Pose Estimation`_
  * `Face Detection`_
  * `Instance Segmentation`_
  * `Depth Estimation`_
  * `Facial Landmark Detection`_
  * `Person Re-ID`_
  * `Super Resolution`_
  * `Face Recognition`_
  * `Person Attribute`_
  * `Face Attribute`_
  * `Zero-shot Classification`_
  * `Stereo Depth Estimation`_
  * `Low Light Enhancement`_
  * `Image Denoising`_
  * `Hand Landmark detection`_


.. _Classification:
Classification
--------------

ImageNet
^^^^^^^^

.. list-table::
   :widths: 31 9 7 11 9 8 8 8 7 7 7 7
   :header-rows: 1

   * - Network Name
     - Accuracy (top1)
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Latency
   * - efficientnet_l
     - 80.46
     - 79.36
     - 300x300x3
     - 10.55
     - 19.4
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_l/pretrained/2023-07-18/efficientnet_l.zip>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/efficientnet_l.hef>`_
     - 57.9951
     - 98.1713
     - {}
   * - efficientnet_lite0
     - 74.99
     - 73.81
     - 224x224x3
     - 4.63
     - 0.78
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite0/pretrained/2023-07-18/efficientnet_lite0.zip>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/efficientnet_lite0.hef>`_
     - {}
     - {}
     - {}
   * - efficientnet_lite1
     - 76.68
     - 76.21
     - 240x240x3
     - 5.39
     - 1.22
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite1/pretrained/2023-07-18/efficientnet_lite1.zip>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/efficientnet_lite1.hef>`_
     - {}
     - {}
     - {}
   * - efficientnet_lite2
     - 77.45
     - 76.74
     - 260x260x3
     - 6.06
     - 1.74
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite2/pretrained/2023-07-18/efficientnet_lite2.zip>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/efficientnet_lite2.hef>`_
     - {}
     - {}
     - {}
   * - efficientnet_lite3
     - 79.29
     - 78.33
     - 280x280x3
     - 8.16
     - 2.8
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite3/pretrained/2023-07-18/efficientnet_lite3.zip>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/efficientnet_lite3.hef>`_
     - {}
     - {}
     - {}
   * - efficientnet_lite4
     - 80.79
     - 79.99
     - 300x300x3
     - 12.95
     - 5.10
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite4/pretrained/2023-07-18/efficientnet_lite4.zip>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/efficientnet_lite4.hef>`_
     - 73.7215
     - 151.255
     - {}
   * - efficientnet_m |rocket|
     - 78.91
     - 78.63
     - 240x240x3
     - 6.87
     - 7.32
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_m/pretrained/2023-07-18/efficientnet_m.zip>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/efficientnet_m.hef>`_
     - 127.623
     - 253.787
     - {}
   * - efficientnet_s
     - 77.64
     - 77.32
     - 224x224x3
     - 5.41
     - 4.72
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_s/pretrained/2023-07-18/efficientnet_s.zip>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/efficientnet_s.hef>`_
     - {}
     - {}
     - {}
   * - hardnet39ds
     - 73.43
     - 72.92
     - 224x224x3
     - 3.48
     - 0.86
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/hardnet39ds/pretrained/2021-07-20/hardnet39ds.zip>`_
     - `link <https://github.com/PingoLH/Pytorch-HarDNet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/hardnet39ds.hef>`_
     - 282.524
     - 794.832
     - {}
   * - hardnet68
     - 75.47
     - 75.04
     - 224x224x3
     - 17.56
     - 8.5
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/hardnet68/pretrained/2021-07-20/hardnet68.zip>`_
     - `link <https://github.com/PingoLH/Pytorch-HarDNet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/hardnet68.hef>`_
     - 124.168
     - 258.157
     - {}
   * - inception_v1
     - 69.74
     - 69.54
     - 224x224x3
     - 6.62
     - 3
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/inception_v1/pretrained/2023-07-18/inception_v1.zip>`_
     - `link <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/inception_v1.hef>`_
     - 237.247
     - 522.84
     - {}
   * - mobilenet_v1
     - 70.97
     - 70.26
     - 224x224x3
     - 4.22
     - 1.14
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v1/pretrained/2023-07-18/mobilenet_v1.zip>`_
     - `link <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/mobilenet_v1.hef>`_
     - 1427.43
     - 1427.4
     - 1.93967
   * - mobilenet_v2_1.0 |rocket|
     - 71.78
     - 71.0
     - 224x224x3
     - 3.49
     - 0.62
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v2_1.0/pretrained/2021-07-11/mobilenet_v2_1.0.zip>`_
     - `link <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/mobilenet_v2_1.0.hef>`_
     - 351.244
     - 821.737
     - {}
   * - mobilenet_v2_1.4
     - 74.18
     - 73.18
     - 224x224x3
     - 6.09
     - 1.18
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v2_1.4/pretrained/2021-07-11/mobilenet_v2_1.4.zip>`_
     - `link <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/mobilenet_v2_1.4.hef>`_
     - 268.706
     - 638.46
     - {}
   * - mobilenet_v3
     - 72.21
     - 71.73
     - 224x224x3
     - 4.07
     - 2
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v3/pretrained/2023-07-18/mobilenet_v3.zip>`_
     - `link <https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/mobilenet_v3.hef>`_
     - 324.885
     - 778.084
     - {}
   * - mobilenet_v3_large_minimalistic
     - 72.11
     - 70.96
     - 224x224x3
     - 3.91
     - 0.42
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v3_large_minimalistic/pretrained/2021-07-11/mobilenet_v3_large_minimalistic.zip>`_
     - `link <https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/mobilenet_v3_large_minimalistic.hef>`_
     - 447.24
     - 1248.59
     - {}
   * - regnetx_1.6gf
     - 77.05
     - 76.75
     - 224x224x3
     - 9.17
     - 3.22
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/regnetx_1.6gf/pretrained/2021-07-11/regnetx_1.6gf.zip>`_
     - `link <https://github.com/facebookresearch/pycls>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/regnetx_1.6gf.hef>`_
     - 303.337
     - 775.166
     - {}
   * - regnetx_800mf
     - 75.16
     - 74.84
     - 224x224x3
     - 7.24
     - 1.6
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/regnetx_800mf/pretrained/2021-07-11/regnetx_800mf.zip>`_
     - `link <https://github.com/facebookresearch/pycls>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/regnetx_800mf.hef>`_
     - 413.96
     - 1107.48
     - {}
   * - repvgg_a1
     - 74.4
     - 72.4
     - 224x224x3
     - 12.79
     - 4.7
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/repvgg/repvgg_a1/pretrained/2022-10-02/RepVGG-A1.zip>`_
     - `link <https://github.com/DingXiaoH/RepVGG>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/repvgg_a1.hef>`_
     - 321.519
     - 719.195
     - {}
   * - repvgg_a2
     - 76.52
     - 74.52
     - 224x224x3
     - 25.5
     - 10.2
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/repvgg/repvgg_a2/pretrained/2022-10-02/RepVGG-A2.zip>`_
     - `link <https://github.com/DingXiaoH/RepVGG>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/repvgg_a2.hef>`_
     - 177.889
     - 322.874
     - {}
   * - resmlp12_relu
     - 75.26
     - 74.32
     - 224x224x3
     - 15.77
     - 6.04
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resmlp12_relu/pretrained/2022-03-03/resmlp12_relu.zip>`_
     - `link <https://github.com/rwightman/pytorch-image-models/>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/resmlp12_relu.hef>`_
     - 87.3561
     - 308.33
     - {}
   * - resnet_v1_18
     - 71.26
     - 71.06
     - 224x224x3
     - 11.68
     - 3.64
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v1_18/pretrained/2022-04-19/resnet_v1_18.zip>`_
     - `link <https://github.com/yhhhli/BRECQ>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/resnet_v1_18.hef>`_
     - 364.787
     - 846.451
     - {}
   * - resnet_v1_34
     - 72.7
     - 72.14
     - 224x224x3
     - 21.79
     - 7.34
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v1_34/pretrained/2021-07-11/resnet_v1_34.zip>`_
     - `link <https://github.com/tensorflow/models/tree/master/research/slim>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/resnet_v1_34.hef>`_
     - 184.44
     - 390.499
     - {}
   * - resnet_v1_50 |rocket| |star|
     - 75.12
     - 74.47
     - 224x224x3
     - 25.53
     - 6.98
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v1_50/pretrained/2021-07-11/resnet_v1_50.zip>`_
     - `link <https://github.com/tensorflow/models/tree/master/research/slim>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/resnet_v1_50.hef>`_
     - 224.794
     - 565.76
     - {}
   * - resnext26_32x4d
     - 76.18
     - 75.78
     - 224x224x3
     - 15.37
     - 4.96
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnext26_32x4d/pretrained/2023-09-18/resnext26_32x4d.zip>`_
     - `link <https://github.com/osmr/imgclsmob/tree/master/pytorch>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/resnext26_32x4d.hef>`_
     - 245.252
     - 500.296
     - {}
   * - resnext50_32x4d
     - 79.31
     - 78.21
     - 224x224x3
     - 24.99
     - 8.48
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnext50_32x4d/pretrained/2023-07-18/resnext50_32x4d.zip>`_
     - `link <https://github.com/osmr/imgclsmob/tree/master/pytorch>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/resnext50_32x4d.hef>`_
     - 142.988
     - 311.237
     - {}
   * - squeezenet_v1.1
     - 59.85
     - 59.4
     - 224x224x3
     - 1.24
     - 0.78
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/squeezenet_v1.1/pretrained/2023-07-18/squeezenet_v1.1.zip>`_
     - `link <https://github.com/osmr/imgclsmob/tree/master/pytorch>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/squeezenet_v1.1.hef>`_
     - 651.115
     - 1276.19
     - {}
   * - vit_base_bn
     - 79.98
     - 78.58
     - 224x224x3
     - 86.5
     - 34.25
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_base/pretrained/2023-01-25/vit_base.zip>`_
     - `link <https://github.com/rwightman/pytorch-image-models>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/vit_base_bn.hef>`_
     - 42.9047
     - 101.898
     - {}
   * - vit_small_bn
     - 78.12
     - 77.02
     - 224x224x3
     - 21.12
     - 8.62
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_small/pretrained/2022-08-08/vit_small.zip>`_
     - `link <https://github.com/rwightman/pytorch-image-models>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/vit_small_bn.hef>`_
     - 93.3559
     - 267.595
     - {}
   * - vit_tiny_bn
     - 68.95
     - 66.75
     - 224x224x3
     - 5.73
     - 2.2
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_tiny/pretrained/2023-08-29/vit_tiny_bn.zip>`_
     - `link <https://github.com/rwightman/pytorch-image-models>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/vit_tiny_bn.hef>`_
     - 154.715
     - 556.287
     - {}

.. _Object Detection:
Object Detection
----------------

COCO
^^^^

.. list-table::
   :widths: 33 8 7 12 8 8 8 7 7 7 7 7
   :header-rows: 1

   * - Network Name
     - mAP
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Latency
   * - centernet_resnet_v1_18_postprocess
     - 26.3
     - 23.31
     - 512x512x3
     - 14.22
     - 31.21
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/centernet/centernet_resnet_v1_18/pretrained/2023-07-18/centernet_resnet_v1_18.zip>`_
     - `link <https://cv.gluon.ai/model_zoo/detection.html>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/centernet_resnet_v1_18_postprocess.hef>`_
     - 86.5256
     - 116.281
     - {}
   * - centernet_resnet_v1_50_postprocess
     - 31.78
     - 29.23
     - 512x512x3
     - 30.07
     - 56.92
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/centernet/centernet_resnet_v1_50_postprocess/pretrained/2023-07-18/centernet_resnet_v1_50_postprocess.zip>`_
     - `link <https://cv.gluon.ai/model_zoo/detection.html>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/centernet_resnet_v1_50_postprocess.hef>`_
     - 53.3531
     - 71.9656
     - {}
   * - damoyolo_tinynasL20_T
     - 42.8
     - 41.7
     - 640x640x3
     - 11.35
     - 18.02
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/damoyolo_tinynasL20_T/pretrained/2022-12-19/damoyolo_tinynasL20_T.zip>`_
     - `link <https://github.com/tinyvision/DAMO-YOLO>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/damoyolo_tinynasL20_T.hef>`_
     - 100.067
     - 169.343
     - {}
   * - damoyolo_tinynasL25_S
     - 46.53
     - 46.04
     - 640x640x3
     - 16.25
     - 37.64
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/damoyolo_tinynasL25_S/pretrained/2022-12-19/damoyolo_tinynasL25_S.zip>`_
     - `link <https://github.com/tinyvision/DAMO-YOLO>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/damoyolo_tinynasL25_S.hef>`_
     - 60.8048
     - 97.3912
     - {}
   * - damoyolo_tinynasL35_M
     - 49.7
     - 47.9
     - 640x640x3
     - 33.98
     - 61.64
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/damoyolo_tinynasL35_M/pretrained/2022-12-19/damoyolo_tinynasL35_M.zip>`_
     - `link <https://github.com/tinyvision/DAMO-YOLO>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/damoyolo_tinynasL35_M.hef>`_
     - 40.5352
     - 63.23
     - {}
   * - detr_resnet_v1_18_bn
     - 33.91
     - 30.36
     - 800x800x3
     - 32.42
     - 58.97
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/detr/detr_r18/detr_resnet_v1_18/2022-09-18/detr_resnet_v1_18_bn.zip>`_
     - `link <https://github.com/facebookresearch/detr>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/detr_resnet_v1_18_bn.hef>`_
     - 18.9067
     - 30.5393
     - {}
   * - efficientdet_lite0
     - 27.32
     - 26.49
     - 320x320x3
     - 3.56
     - 1.94
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/efficientdet/efficientdet_lite0/pretrained/2023-04-25/efficientdet-lite0.zip>`_
     - `link <https://github.com/google/automl/tree/master/efficientdet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/efficientdet_lite0.hef>`_
     - {}
     - {}
     - {}
   * - efficientdet_lite1
     - 32.27
     - 31.72
     - 384x384x3
     - 4.73
     - 4
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/efficientdet/efficientdet_lite1/pretrained/2023-04-25/efficientdet-lite1.zip>`_
     - `link <https://github.com/google/automl/tree/master/efficientdet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/efficientdet_lite1.hef>`_
     - {}
     - {}
     - {}
   * - efficientdet_lite2
     - 35.95
     - 34.67
     - 448x448x3
     - 5.93
     - 6.84
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/efficientdet/efficientdet_lite2/pretrained/2023-04-25/efficientdet-lite2.zip>`_
     - `link <https://github.com/google/automl/tree/master/efficientdet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/efficientdet_lite2.hef>`_
     - {}
     - {}
     - {}
   * - nanodet_repvgg  |star|
     - 29.3
     - 28.53
     - 416x416x3
     - 6.74
     - 11.28
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/nanodet/nanodet_repvgg/pretrained/2022-02-07/nanodet.zip>`_
     - `link <https://github.com/RangiLyu/nanodet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/nanodet_repvgg.hef>`_
     - 192.699
     - 311.163
     - {}
   * - nanodet_repvgg_a12
     - 33.73
     - 31.33
     - 640x640x3
     - 5.13
     - 28.23
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/nanodet/nanodet_repvgg_a12/pretrained/2023-05-31/nanodet_repvgg_a12_640x640.zip>`_
     - `link <https://github.com/Megvii-BaseDetection/YOLOX>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/nanodet_repvgg_a12.hef>`_
     - 109.333
     - 159.891
     - {}
   * - nanodet_repvgg_a1_640
     - 33.28
     - 32.88
     - 640x640x3
     - 10.79
     - 42.8
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/nanodet/nanodet_repvgg_a1_640/pretrained/2022-07-19/nanodet_repvgg_a1_640.zip>`_
     - `link <https://github.com/RangiLyu/nanodet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/nanodet_repvgg_a1_640.hef>`_
     - 85.5934
     - 117.759
     - {}
   * - ssd_mobilenet_v1 |rocket| |star|
     - 23.19
     - 22.29
     - 300x300x3
     - 6.79
     - 2.5
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/ssd/ssd_mobilenet_v1/pretrained/2023-07-18/ssd_mobilenet_v1.zip>`_
     - `link <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/ssd_mobilenet_v1.hef>`_
     - 182.389
     - 354.052
     - {}
   * - ssd_mobilenet_v2
     - 24.15
     - 22.94
     - 300x300x3
     - 4.46
     - 1.52
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/ssd/ssd_mobilenet_v2/pretrained/2023-03-16/ssd_mobilenet_v2.zip>`_
     - `link <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/ssd_mobilenet_v2.hef>`_
     - 136.849
     - 257.357
     - {}
   * - tiny_yolov3
     - 14.66
     - 14.41
     - 416x416x3
     - 8.85
     - 5.58
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/tiny_yolov3/pretrained/2021-07-11/tiny_yolov3.zip>`_
     - `link <https://github.com/Tianxiaomo/pytorch-YOLOv4>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/tiny_yolov3.hef>`_
     - 269.057
     - 455.898
     - {}
   * - tiny_yolov4
     - 19.18
     - 17.73
     - 416x416x3
     - 6.05
     - 6.92
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/tiny_yolov4/pretrained/2023-07-18/tiny_yolov4.zip>`_
     - `link <https://github.com/Tianxiaomo/pytorch-YOLOv4>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/tiny_yolov4.hef>`_
     - 273.062
     - 430.406
     - {}
   * - yolov3  |star|
     - 38.42
     - 38.37
     - 608x608x3
     - 68.79
     - 158.10
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov3/pretrained/2021-08-16/yolov3.zip>`_
     - `link <https://github.com/AlexeyAB/darknet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/yolov3.hef>`_
     - 20.7228
     - 24.7059
     - {}
   * - yolov3_416
     - 37.73
     - 37.53
     - 416x416x3
     - 61.92
     - 65.94
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov3_416/pretrained/2021-08-16/yolov3_416.zip>`_
     - `link <https://github.com/AlexeyAB/darknet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/yolov3_416.hef>`_
     - 35.1757
     - 50.8807
     - {}
   * - yolov3_gluon |rocket| |star|
     - 37.28
     - 35.64
     - 608x608x3
     - 68.79
     - 158.1
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov3_gluon/pretrained/2023-07-18/yolov3_gluon.zip>`_
     - `link <https://cv.gluon.ai/model_zoo/detection.html>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/yolov3_gluon.hef>`_
     - 22.0821
     - 27.6056
     - {}
   * - yolov3_gluon_416  |star|
     - 36.27
     - 34.92
     - 416x416x3
     - 61.92
     - 65.94
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov3_gluon_416/pretrained/2023-07-18/yolov3_gluon_416.zip>`_
     - `link <https://cv.gluon.ai/model_zoo/detection.html>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/yolov3_gluon_416.hef>`_
     - 34.5337
     - 50.1277
     - {}
   * - yolov4_leaky  |star|
     - 42.37
     - 41.08
     - 512x512x3
     - 64.33
     - 91.04
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov4/pretrained/2022-03-17/yolov4.zip>`_
     - `link <https://github.com/AlexeyAB/darknet/wiki/YOLOv4-model-zoo>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/yolov4_leaky.hef>`_
     - 29.1715
     - 39.9617
     - {}
   * - yolov5m
     - 42.59
     - 41.19
     - 640x640x3
     - 21.78
     - 52.17
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5m_spp/pretrained/2023-04-25/yolov5m.zip>`_
     - `link <https://github.com/ultralytics/yolov5/releases/tag/v2.0>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/yolov5m.hef>`_
     - 53.809
     - 75.5971
     - {}
   * - yolov5m6_6.1
     - 50.67
     - 48.97
     - 1280x1280x3
     - 35.70
     - 200.04
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5m6_6.1/pretrained/2023-04-25/yolov5m6.zip>`_
     - `link <https://github.com/ultralytics/yolov5/releases/tag/v6.1>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/yolov5m6_6.1.hef>`_
     - 16.3575
     - 18.8832
     - {}
   * - yolov5m_6.1
     - 44.8
     - 43.36
     - 640x640x3
     - 21.17
     - 48.96
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5m_6.1/pretrained/2023-04-25/yolov5m_6.1.zip>`_
     - `link <https://github.com/ultralytics/yolov5/releases/tag/v6.1>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/yolov5m_6.1.hef>`_
     - 53.8092
     - 75.5973
     - {}
   * - yolov5m_wo_spp |rocket|
     - 43.06
     - 40.76
     - 640x640x3
     - 22.67
     - 52.88
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5m/pretrained/2023-04-25/yolov5m_wo_spp.zip>`_
     - `link <https://github.com/ultralytics/yolov5/releases/tag/v2.0>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/yolov5m_wo_spp_60p.hef>`_
     - 67.6233
     - 94.4975
     - {}
   * - yolov5s  |star|
     - 35.33
     - 33.98
     - 640x640x3
     - 7.46
     - 17.44
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5s_spp/pretrained/2023-04-25/yolov5s.zip>`_
     - `link <https://github.com/ultralytics/yolov5/releases/tag/v2.0>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/yolov5s.hef>`_
     - 117.162
     - 180.261
     - {}
   * - yolov5s_c3tr
     - 37.13
     - 35.63
     - 640x640x3
     - 10.29
     - 17.02
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5s_c3tr/pretrained/2023-04-25/yolov5s_c3tr.zip>`_
     - `link <https://github.com/ultralytics/yolov5/tree/v6.0>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/yolov5s_c3tr.hef>`_
     - 100.892
     - 167.908
     - {}
   * - yolov5xs_wo_spp
     - 33.18
     - 32.2
     - 512x512x3
     - 7.85
     - 11.36
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5xs/pretrained/2023-04-25/yolov5xs.zip>`_
     - `link <https://github.com/ultralytics/yolov5/releases/tag/v2.0>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/yolov5xs_wo_spp.hef>`_
     - 178.885
     - 319.815
     - {}
   * - yolov5xs_wo_spp_nms_core
     - 32.57
     - 31.06
     - 512x512x3
     - 7.85
     - 11.36
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5xs/pretrained/2022-05-10/yolov5xs_wo_spp_nms.zip>`_
     - `link <https://github.com/ultralytics/yolov5/releases/tag/v2.0>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/yolov5xs_wo_spp_nms_core.hef>`_
     - 178.878
     - 319.704
     - {}
   * - yolov6n
     - 34.28
     - 32.18
     - 640x640x3
     - 4.32
     - 11.12
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov6n/pretrained/2023-05-31/yolov6n.zip>`_
     - `link <https://github.com/meituan/YOLOv6/releases/tag/0.1.0>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/yolov6n.hef>`_
     - 168.961
     - 285.641
     - {}
   * - yolov6n_0.2.1
     - 35.16
     - 33.66
     - 640x640x3
     - 4.33
     - 11.06
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov6n_0.2.1/pretrained/2023-04-17/yolov6n_0.2.1.zip>`_
     - `link <https://github.com/meituan/YOLOv6/releases/tag/0.2.1>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/yolov6n_0.2.1.hef>`_
     - 181.87
     - 320.321
     - {}
   * - yolov7
     - 50.59
     - 47.89
     - 640x640x3
     - 36.91
     - 104.51
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov7/pretrained/2023-04-25/yolov7.zip>`_
     - `link <https://github.com/WongKinYiu/yolov7>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/yolov7.hef>`_
     - 28.8124
     - 37.7975
     - {}
   * - yolov7_tiny
     - 37.07
     - 35.97
     - 640x640x3
     - 6.22
     - 13.74
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov7_tiny/pretrained/2023-04-25/yolov7_tiny.zip>`_
     - `link <https://github.com/WongKinYiu/yolov7>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/yolov7_tiny.hef>`_
     - 128.247
     - 193.335
     - {}
   * - yolov7e6
     - 55.37
     - 53.47
     - 1280x1280x3
     - 97.20
     - 515.12
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov7e6/pretrained/2023-04-25/yolov7-e6.zip>`_
     - `link <https://github.com/WongKinYiu/yolov7>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/yolov7e6.hef>`_
     - 6.17829
     - 6.53669
     - {}
   * - yolov8l
     - 52.44
     - 51.78
     - 640x640x3
     - 43.7
     - 165.3
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov8l/2023-02-02/yolov8l.zip>`_
     - `link <https://github.com/ultralytics/ultralytics>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/yolov8l.hef>`_
     - 18.7227
     - 23.979
     - {}
   * - yolov8m
     - 49.91
     - 49.11
     - 640x640x3
     - 25.9
     - 78.93
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov8m/2023-02-02/yolov8m.zip>`_
     - `link <https://github.com/ultralytics/ultralytics>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/yolov8m.hef>`_
     - 39.082
     - 55.9632
     - {}
   * - yolov8n
     - 37.02
     - 36.32
     - 640x640x3
     - 3.2
     - 8.74
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov8n/2023-01-30/yolov8n.zip>`_
     - `link <https://github.com/ultralytics/ultralytics>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/yolov8n.hef>`_
     - 163.514
     - 311.781
     - {}
   * - yolov8s
     - 44.58
     - 43.98
     - 640x640x3
     - 11.2
     - 28.6
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov8s/2023-02-02/yolov8s.zip>`_
     - `link <https://github.com/ultralytics/ultralytics>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/yolov8s.hef>`_
     - 82.4367
     - 129.367
     - {}
   * - yolov8x
     - 53.45
     - 52.75
     - 640x640x3
     - 68.2
     - 258
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov8x/2023-02-02/yolov8x.zip>`_
     - `link <https://github.com/ultralytics/ultralytics>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/yolov8x.hef>`_
     - 10.9973
     - 13.0795
     - {}
   * - yolox_l_leaky  |star|
     - 48.69
     - 46.71
     - 640x640x3
     - 54.17
     - 155.3
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolox_l_leaky/pretrained/2023-05-31/yolox_l_leaky.zip>`_
     - `link <https://github.com/Megvii-BaseDetection/YOLOX>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/yolox_l_leaky.hef>`_
     - 17.9951
     - 21.8031
     - {}
   * - yolox_s_leaky
     - 38.12
     - 37.27
     - 640x640x3
     - 8.96
     - 26.74
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolox_s_leaky/pretrained/2023-05-31/yolox_s_leaky.zip>`_
     - `link <https://github.com/Megvii-BaseDetection/YOLOX>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/yolox_s_leaky.hef>`_
     - 87.0709
     - 129.368
     - {}
   * - yolox_s_wide_leaky
     - 42.4
     - 40.97
     - 640x640x3
     - 20.12
     - 59.46
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolox_s_wide_leaky/pretrained/2023-05-31/yolox_s_wide_leaky.zip>`_
     - `link <https://github.com/Megvii-BaseDetection/YOLOX>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/yolox_s_wide_leaky.hef>`_
     - 53.5291
     - 70.4935
     - {}
   * - yolox_tiny
     - 32.64
     - 31.39
     - 416x416x3
     - 5.05
     - 6.44
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolox/yolox_tiny/pretrained/2023-05-31/yolox_tiny.zip>`_
     - `link <https://github.com/Megvii-BaseDetection/YOLOX>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/yolox_tiny.hef>`_
     - 182.502
     - 362.678
     - {}

VisDrone
^^^^^^^^

.. list-table::
   :widths: 31 7 9 12 9 8 9 8 7 7 7 7
   :header-rows: 1

   * - Network Name
     - mAP
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Latency
   * - ssd_mobilenet_v1_visdrone  |star|
     - 2.37
     - 2.22
     - 300x300x3
     - 5.64
     - 2.3
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-Visdrone/ssd/ssd_mobilenet_v1_visdrone/pretrained/2023-07-18/ssd_mobilenet_v1_visdrone.zip>`_
     - `link <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/ssd_mobilenet_v1_visdrone.hef>`_
     - 244.12
     - 527.101
     - {}

.. _Semantic Segmentation:
Semantic Segmentation
---------------------

Cityscapes
^^^^^^^^^^

.. list-table::
   :widths: 31 7 9 12 9 8 9 8 7 7 7 7
   :header-rows: 1

   * - Network Name
     - mIoU
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Latency
   * - fcn8_resnet_v1_18  |star|
     - 69.41
     - 69.21
     - 1024x1920x3
     - 11.20
     - 142.82
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Cityscapes/fcn8_resnet_v1_18/pretrained/2023-06-22/fcn8_resnet_v1_18.zip>`_
     - `link <https://mmsegmentation.readthedocs.io/en/latest>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/fcn8_resnet_v1_18.hef>`_
     - 18.8156
     - 21.0842
     - {}
   * - segformer_b0_bn
     - 69.81
     - 68.01
     - 512x1024x3
     - 3.72
     - 27.2
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Cityscapes/segformer_b0_512x1024_bn/pretrained/2023-09-04/segformer_b0_512x1024_bn.zip>`_
     - `link <https://github.com/NVlabs/SegFormer>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/segformer_b0_bn.hef>`_
     - {}
     - {}
     - {}
   * - stdc1 |rocket|
     - 74.57
     - 73.92
     - 1024x1920x3
     - 8.27
     - 126.47
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Cityscapes/stdc1/pretrained/2023-06-12/stdc1.zip>`_
     - `link <https://mmsegmentation.readthedocs.io/en/latest>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/stdc1.hef>`_
     - 15.5433
     - 18.1757
     - {}

Oxford-IIIT Pet
^^^^^^^^^^^^^^^

.. list-table::
   :widths: 31 7 9 12 9 8 9 8 7 7 7 7
   :header-rows: 1

   * - Network Name
     - mIoU
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Latency
   * - unet_mobilenet_v2
     - 77.32
     - 77.02
     - 256x256x3
     - 10.08
     - 28.88
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Oxford_Pet/unet_mobilenet_v2/pretrained/2022-02-03/unet_mobilenet_v2.zip>`_
     - `link <https://www.tensorflow.org/tutorials/images/segmentation>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/unet_mobilenet_v2.hef>`_
     - 133.439
     - 227.59
     - {}

Pascal VOC
^^^^^^^^^^

.. list-table::
   :widths: 36 7 9 12 9 8 9 8 7 7 7 7
   :header-rows: 1

   * - Network Name
     - mIoU
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Latency
   * - deeplab_v3_mobilenet_v2
     - 76.05
     - 74.8
     - 513x513x3
     - 2.10
     - 17.65
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Pascal/deeplab_v3_mobilenet_v2_dilation/pretrained/2023-08-22/deeplab_v3_mobilenet_v2_dilation.zip>`_
     - `link <https://github.com/bonlime/keras-deeplab-v3-plus>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/deeplab_v3_mobilenet_v2.hef>`_
     - 42.9866
     - 56.8301
     - {}
   * - deeplab_v3_mobilenet_v2_wo_dilation
     - 71.46
     - 71.26
     - 513x513x3
     - 2.10
     - 3.21
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Pascal/deeplab_v3_mobilenet_v2/pretrained/2021-08-12/deeplab_v3_mobilenet_v2.zip>`_
     - `link <https://github.com/tensorflow/models/tree/master/research/deeplab>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/deeplab_v3_mobilenet_v2_wo_dilation.hef>`_
     - 87.5364
     - 119.235
     - {}

.. _Pose Estimation:
Pose Estimation
---------------

COCO
^^^^

.. list-table::
   :widths: 24 8 9 18 9 8 9 8 7 7 7 7
   :header-rows: 1

   * - Network Name
     - AP
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Latency
   * - centerpose_regnetx_1.6gf_fpn  |star|
     - 53.54
     - 52.84
     - 640x640x3
     - 14.28
     - 64.58
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/PoseEstimation/centerpose_regnetx_1.6gf_fpn/pretrained/2022-03-23/centerpose_regnetx_1.6gf_fpn.zip>`_
     - `link <https://github.com/tensorboy/centerpose>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/centerpose_regnetx_1.6gf_fpn.hef>`_
     - 46.2622
     - 63.1189
     - {}
   * - centerpose_regnetx_800mf
     - 44.07
     - 42.97
     - 512x512x3
     - 12.31
     - 86.12
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/PoseEstimation/centerpose_regnetx_800mf/pretrained/2021-07-11/centerpose_regnetx_800mf.zip>`_
     - `link <https://github.com/tensorboy/centerpose>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/centerpose_regnetx_800mf.hef>`_
     - 57.078
     - 75.5737
     - {}
   * - centerpose_repvgg_a0  |star|
     - 39.17
     - 37.17
     - 416x416x3
     - 11.71
     - 28.27
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/PoseEstimation/centerpose_repvgg_a0/pretrained/2021-09-26/centerpose_repvgg_a0.zip>`_
     - `link <https://github.com/tensorboy/centerpose>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/centerpose_repvgg_a0.hef>`_
     - 91.8911
     - 137.367
     - {}

.. _Single Person Pose Estimation:
Single Person Pose Estimation
-----------------------------

COCO
^^^^

.. list-table::
   :widths: 24 8 9 18 9 8 9 8 7 7 7 7
   :header-rows: 1

   * - Network Name
     - AP
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Latency
   * - mspn_regnetx_800mf  |star|
     - 70.8
     - 70.3
     - 256x192x3
     - 7.17
     - 2.94
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SinglePersonPoseEstimation/mspn_regnetx_800mf/pretrained/2022-07-12/mspn_regnetx_800mf.zip>`_
     - `link <https://github.com/open-mmlab/mmpose>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/mspn_regnetx_800mf.hef>`_
     - 239.436
     - 543.199
     - {}
   * - vit_pose_small
     - 74.16
     - 71.6
     - 256x192x3
     - 24.29
     - 17.17
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SinglePersonPoseEstimation/vit/vit_pose_small/pretrained/2023-11-14/vit_pose_small.zip>`_
     - `link <https://github.com/ViTAE-Transformer/ViTPose>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/vit_pose_small.hef>`_
     - 25.0889
     - 77.8054
     - {}
   * - vit_pose_small_bn
     - 72.01
     - 70.81
     - 256x192x3
     - 24.32
     - 17.17
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SinglePersonPoseEstimation/vit/vit_pose_small_bn/pretrained/2023-07-20/vit_pose_small_bn.zip>`_
     - `link <https://github.com/ViTAE-Transformer/ViTPose>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/vit_pose_small_bn.hef>`_
     - 73.2665
     - 194.156
     - {}

.. _Face Detection:
Face Detection
--------------

WiderFace
^^^^^^^^^

.. list-table::
   :widths: 24 7 12 11 9 8 8 8 7 7 7 7
   :header-rows: 1

   * - Network Name
     - mAP
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Latency
   * - lightface_slim  |star|
     - 39.7
     - 39.22
     - 240x320x3
     - 0.26
     - 0.16
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/lightface_slim/2021-07-18/lightface_slim.zip>`_
     - `link <https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/lightface_slim.hef>`_
     - 649.138
     - 1535.33
     - {}
   * - retinaface_mobilenet_v1  |star|
     - 81.27
     - 81.17
     - 736x1280x3
     - 3.49
     - 25.14
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/retinaface_mobilenet_v1_hd/2023-07-18/retinaface_mobilenet_v1_hd.zip>`_
     - `link <https://github.com/biubug6/Pytorch_Retinaface>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/retinaface_mobilenet_v1.hef>`_
     - 50.0765
     - 65.3995
     - {}
   * - scrfd_10g
     - 82.13
     - 82.03
     - 640x640x3
     - 4.23
     - 26.74
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/scrfd/scrfd_10g/pretrained/2022-09-07/scrfd_10g.zip>`_
     - `link <https://github.com/deepinsight/insightface>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/scrfd_10g.hef>`_
     - 90.3432
     - 131.564
     - {}
   * - scrfd_2.5g
     - 76.59
     - 76.32
     - 640x640x3
     - 0.82
     - 6.88
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/scrfd/scrfd_2.5g/pretrained/2022-09-07/scrfd_2.5g.zip>`_
     - `link <https://github.com/deepinsight/insightface>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/scrfd_2.5g.hef>`_
     - 206.764
     - 320.546
     - {}
   * - scrfd_500m
     - 68.98
     - 68.88
     - 640x640x3
     - 0.63
     - 1.5
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/scrfd/scrfd_500m/pretrained/2022-09-07/scrfd_500m.zip>`_
     - `link <https://github.com/deepinsight/insightface>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/scrfd_500m.hef>`_
     - 231.127
     - 407.787
     - {}

.. _Instance Segmentation:
Instance Segmentation
---------------------

COCO
^^^^

.. list-table::
   :widths: 34 7 7 11 9 8 8 8 7 7 7 7
   :header-rows: 1

   * - Network Name
     - mAP
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Latency
   * - yolact_regnetx_1.6gf
     - 27.57
     - 27.27
     - 512x512x3
     - 30.09
     - 125.34
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolact_regnetx_1.6gf/pretrained/2022-11-30/yolact_regnetx_1.6gf.zip>`_
     - `link <https://github.com/dbolya/yolact>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/yolact_regnetx_1.6gf.hef>`_
     - 34.3425
     - 43.6273
     - {}
   * - yolact_regnetx_800mf
     - 25.61
     - 25.5
     - 512x512x3
     - 28.3
     - 116.75
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolact_regnetx_800mf/pretrained/2022-11-30/yolact_regnetx_800mf.zip>`_
     - `link <https://github.com/dbolya/yolact>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/yolact_regnetx_800mf.hef>`_
     - 40.7949
     - 50.4228
     - {}
   * - yolov5l_seg
     - 39.78
     - 39.09
     - 640x640x3
     - 47.89
     - 147.88
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5l/pretrained/2022-10-30/yolov5l-seg.zip>`_
     - `link <https://github.com/ultralytics/yolov5>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/yolov5l_seg.hef>`_
     - 20.3549
     - 24.6877
     - {}
   * - yolov5m_seg
     - 37.05
     - 36.32
     - 640x640x3
     - 32.60
     - 70.94
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5m/pretrained/2022-10-30/yolov5m-seg.zip>`_
     - `link <https://github.com/ultralytics/yolov5>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/yolov5m_seg.hef>`_
     - 43.6168
     - 56.6371
     - {}
   * - yolov5n_seg  |star|
     - 23.35
     - 22.75
     - 640x640x3
     - 1.99
     - 7.1
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5n/pretrained/2022-10-30/yolov5n-seg.zip>`_
     - `link <https://github.com/ultralytics/yolov5>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/yolov5n_seg.hef>`_
     - 148.879
     - 174.184
     - {}
   * - yolov5s_seg
     - 31.57
     - 30.49
     - 640x640x3
     - 7.61
     - 26.42
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov5/yolov5s/pretrained/2022-10-30/yolov5s-seg.zip>`_
     - `link <https://github.com/ultralytics/yolov5>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/yolov5s_seg.hef>`_
     - 82.509
     - 113.995
     - {}
   * - yolov8m_seg
     - 40.6
     - 39.88
     - 640x640x3
     - 27.3
     - 110.2
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov8/yolov8m/pretrained/2023-03-06/yolov8m-seg.zip>`_
     - `link <https://github.com/ultralytics/ultralytics>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/yolov8m_seg.hef>`_
     - 29.2613
     - 39.2299
     - {}
   * - yolov8n_seg
     - 30.32
     - 29.68
     - 640x640x3
     - 3.4
     - 12.04
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov8/yolov8n/pretrained/2023-03-06/yolov8n-seg.zip>`_
     - `link <https://github.com/ultralytics/ultralytics>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/yolov8n_seg.hef>`_
     - 132.766
     - 227.347
     - {}
   * - yolov8s_seg
     - 36.63
     - 36.03
     - 640x640x3
     - 11.8
     - 42.6
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolov8/yolov8s/pretrained/2023-03-06/yolov8s-seg.zip>`_
     - `link <https://github.com/ultralytics/ultralytics>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/yolov8s_seg.hef>`_
     - 63.7904
     - 90.8048
     - {}

.. _Depth Estimation:
Depth Estimation
----------------

NYU
^^^

.. list-table::
   :widths: 34 7 7 11 9 8 8 8 7 7 7 7
   :header-rows: 1

   * - Network Name
     - RMSE
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Latency
   * - fast_depth  |star|
     - 0.6
     - 0.62
     - 224x224x3
     - 1.35
     - 0.74
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/DepthEstimation/indoor/fast_depth/pretrained/2021-10-18/fast_depth.zip>`_
     - `link <https://github.com/dwofk/fast-depth>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/fast_depth.hef>`_
     - 317.237
     - 768.32
     - {}
   * - scdepthv3
     - 0.48
     - 0.51
     - 256x320x3
     - 14.8
     - 10.7
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/DepthEstimation/indoor/scdepthv3/pretrained/2023-07-20/scdepthv3.zip>`_
     - `link <https://github.com/JiawangBian/sc_depth_pl/>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/scdepthv3.hef>`_
     - 157.349
     - 276.312
     - {}

.. _Facial Landmark Detection:
Facial Landmark Detection
-------------------------

AFLW2k3d
^^^^^^^^

.. list-table::
   :widths: 28 8 8 16 9 8 8 8 7 7 7 7
   :header-rows: 1

   * - Network Name
     - NME
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Latency
   * - tddfa_mobilenet_v1  |star|
     - 3.68
     - 4.05
     - 120x120x3
     - 3.26
     - 0.36
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceLandmarks3d/tddfa/tddfa_mobilenet_v1/pretrained/2021-11-28/tddfa_mobilenet_v1.zip>`_
     - `link <https://github.com/cleardusk/3DDFA_V2>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/tddfa_mobilenet_v1.hef>`_
     - 4482.73
     - 4484.42
     - {}

.. _Person Re-ID:
Person Re-ID
------------

Market1501
^^^^^^^^^^

.. list-table::
   :widths: 28 8 9 13 9 8 8 8 7 7 7 7
   :header-rows: 1

   * - Network Name
     - rank1
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Latency
   * - osnet_x1_0
     - 94.43
     - 93.63
     - 256x128x3
     - 2.19
     - 1.98
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/PersonReID/osnet_x1_0/2022-05-19/osnet_x1_0.zip>`_
     - `link <https://github.com/KaiyangZhou/deep-person-reid>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/osnet_x1_0.hef>`_
     - 110.173
     - 317.055
     - {}
   * - repvgg_a0_person_reid_512  |star|
     - 89.9
     - 89.3
     - 256x128x3
     - 7.68
     - 1.78
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/MCPReID/reid/repvgg_a0_person_reid_512/2022-04-18/repvgg_a0_person_reid_512.zip>`_
     - `link <https://github.com/DingXiaoH/RepVGG>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/repvgg_a0_person_reid_512.hef>`_
     - 2632.33
     - 2632.12
     - 1.17645

.. _Super Resolution:
Super Resolution
----------------

BSD100
^^^^^^

.. list-table::
   :widths: 32 8 7 11 9 8 8 8 7 7 7 7
   :header-rows: 1

   * - Network Name
     - PSNR
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Latency
   * - espcn_x2
     - 31.4
     - 30.3
     - 156x240x1
     - 0.02
     - 1.6
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SuperResolution/espcn/espcn_x2/2022-08-02/espcn_x2.zip>`_
     - `link <https://github.com/Lornatang/ESPCN-PyTorch>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/espcn_x2.hef>`_
     - 1746.31
     - 1745.98
     - 0.621769
   * - espcn_x3
     - 28.41
     - 28.06
     - 104x160x1
     - 0.02
     - 0.76
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SuperResolution/espcn/espcn_x3/2022-08-02/espcn_x3.zip>`_
     - `link <https://github.com/Lornatang/ESPCN-PyTorch>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/espcn_x3.hef>`_
     - 1925.37
     - 1924.98
     - 0.569555
   * - espcn_x4
     - 26.83
     - 26.58
     - 78x120x1
     - 0.02
     - 0.46
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SuperResolution/espcn/espcn_x4/2022-08-02/espcn_x4.zip>`_
     - `link <https://github.com/Lornatang/ESPCN-PyTorch>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/espcn_x4.hef>`_
     - 1908.98
     - 1908.56
     - 0.559387

.. _Face Recognition:
Face Recognition
----------------

LFW
^^^

.. list-table::
   :widths: 12 7 12 14 9 8 10 8 7 7 7 7
   :header-rows: 1

   * - Network Name
     - lfw verification accuracy
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Latency
   * - arcface_mobilefacenet
     - 99.43
     - 99.41
     - 112x112x3
     - 2.04
     - 0.88
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceRecognition/arcface/arcface_mobilefacenet/pretrained/2022-08-24/arcface_mobilefacenet.zip>`_
     - `link <https://github.com/deepinsight/insightface>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/arcface_mobilefacenet.hef>`_
     - 444.238
     - 1190.37
     - {}
   * - arcface_r50
     - 99.72
     - 99.71
     - 112x112x3
     - 31.0
     - 12.6
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceRecognition/arcface/arcface_r50/pretrained/2022-08-24/arcface_r50.zip>`_
     - `link <https://github.com/deepinsight/insightface>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/arcface_r50.hef>`_
     - 112.901
     - 234.881
     - {}

.. _Person Attribute:
Person Attribute
----------------

PETA
^^^^

.. list-table::
   :widths: 24 14 12 14 9 8 10 8 7 7 7 7
   :header-rows: 1

   * - Network Name
     - Mean Accuracy
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Latency
   * - person_attr_resnet_v1_18
     - 82.5
     - 82.61
     - 224x224x3
     - 11.19
     - 3.64
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/person_attr_resnet_v1_18/pretrained/2022-06-11/person_attr_resnet_v1_18.zip>`_
     - `link <https://github.com/dangweili/pedestrian-attribute-recognition-pytorch>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/person_attr_resnet_v1_18.hef>`_
     - 364.245
     - 816.634
     - {}

.. _Face Attribute:
Face Attribute
--------------

CELEBA
^^^^^^

.. list-table::
   :widths: 30 7 11 14 9 8 12 8 7 7 7 7
   :header-rows: 1

   * - Network Name
     - Mean Accuracy
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Latency
   * - face_attr_resnet_v1_18
     - 81.19
     - 81.09
     - 218x178x3
     - 11.74
     - 3
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceAttr/face_attr_resnet_v1_18/2022-06-09/face_attr_resnet_v1_18.zip>`_
     - `link <https://github.com/d-li14/face-attribute-prediction>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/face_attr_resnet_v1_18.hef>`_
     - 366.06
     - 884.264
     - {}

.. _Zero-shot Classification:
Zero-shot Classification
------------------------

CIFAR100
^^^^^^^^

.. list-table::
   :widths: 30 7 11 14 9 8 12 8 7 7 7 7
   :header-rows: 1

   * - Network Name
     - Accuracy (top1)
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Latency
   * - clip_resnet_50
     - 42.07
     - 38.57
     - 224x224x3
     - 38.72
     - 11.62
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/clip_resnet_50/pretrained/2023-03-09/clip_resnet_50.zip>`_
     - `link <https://github.com/openai/CLIP>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/clip_resnet_50.hef>`_
     - {}
     -
     - {}

.. _Stereo Depth Estimation:
Stereo Depth Estimation
-----------------------

N/A
^^^

.. list-table::
   :widths: 30 7 11 14 9 8 12 8 7 7 7 7
   :header-rows: 1

   * - Network Name
     - EPE
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Latency
   * - stereonet
     - 91.79
     - 89.14
     - 368X1232X3, 368X1232X3
     - 5.91
     - 126.28
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/DisparityEstimation/stereonet/pretrained/2023-05-31/stereonet.zip>`_
     - `link <https://github.com/nivosco/StereoNet>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/stereonet.hef>`_
     - {}
     - {}
     - {}

.. _Low Light Enhancement:
Low Light Enhancement
---------------------

LOL
^^^

.. list-table::
   :widths: 30 7 11 14 9 8 12 8 7 7 7 7
   :header-rows: 1

   * - Network Name
     - PSNR
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Latency
   * - zero_dce
     - 16.23
     - 16.24
     - 400x600x3
     - 0.21
     - 38.2
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/LowLightEnhancement/LOL/zero_dce/pretrained/2023-04-23/zero_dce.zip>`_
     - `link <Internal>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/zero_dce.hef>`_
     - 71.5373
     - 79.9602
     - {}
   * - zero_dce_pp
     - 15.95
     - 15.82
     - 400x600x3
     - 0.02
     - 4.84
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/LowLightEnhancement/LOL/zero_dce_pp/pretrained/2023-07-03/zero_dce_pp.zip>`_
     - `link <Internal>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/zero_dce_pp.hef>`_
     - 52.7195
     - 58.1534
     - {}

.. _Image Denoising:
Image Denoising
---------------

BSD68
^^^^^

.. list-table::
   :widths: 30 7 11 14 9 8 12 8 7 7 7 7
   :header-rows: 1

   * - Network Name
     - PSNR
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Latency
   * - dncnn3
     - 31.46
     - 31.26
     - 321x481x1
     - 0.66
     - 205.26
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ImageDenoising/dncnn3/2023-06-15/dncnn3.zip>`_
     - `link <https://github.com/cszn/KAIR>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/dncnn3.hef>`_
     - 20.5436
     - 21.0863
     - {}

CBSD68
^^^^^^

.. list-table::
   :widths: 30 7 11 14 9 8 12 8 7 7 7 7
   :header-rows: 1

   * - Network Name
     - PSNR
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Latency
   * - dncnn_color_blind
     - 33.87
     - 32.97
     - 321x481x3
     - 0.66
     - 205.97
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ImageDenoising/dncnn_color_blind/2023-06-25/dncnn_color_blind.zip>`_
     - `link <https://github.com/cszn/KAIR>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/dncnn_color_blind.hef>`_
     - 20.5436
     - 21.0838
     - {}

.. _Hand Landmark detection:
Hand Landmark detection
-----------------------

Hand Landmark
^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Network Name
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Latency
   * - hand_landmark_lite
     - 224x224x3
     - 1.01
     - 0.3
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HandLandmark/hand_landmark_lite/2023-07-18/hand_landmark_lite.zip>`_
     - `link <https://github.com/google/mediapipe>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo15m/hand_landmark_lite.hef>`_
     - 386.777
     - 1,101.96
     - {}
