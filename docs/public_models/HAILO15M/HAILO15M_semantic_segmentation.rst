
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

  * `Semantic Segmentation`_


.. _Semantic Segmentation:

Semantic Segmentation
---------------------

Cityscapes
^^^^^^^^^^

.. list-table::
   :widths: 31 9 7 11 9 8 8 8 7 7 7
   :header-rows: 1

   * - Network Name
     - mIoU
     - Quantized
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
   * - fcn8_resnet_v1_18
     - 69.41
     - 69.21
     - 18
     - 20
     - 1024x1920x3
     - 11.20
     - 142.82
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Cityscapes/fcn8_resnet_v1_18/pretrained/2023-06-22/fcn8_resnet_v1_18.zip>`_
     - `link <https://mmsegmentation.readthedocs.io/en/latest>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15m/fcn8_resnet_v1_18.hef>`_
   * - segformer_b0_bn
     - 69.81
     - 68.01
     - 9
     - 14
     - 512x1024x3
     - 3.72
     - 35.76
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Cityscapes/segformer_b0_512x1024_bn/pretrained/2023-09-04/segformer_b0_512x1024_bn.zip>`_
     - `link <https://github.com/NVlabs/SegFormer>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15m/segformer_b0_bn.hef>`_
   * - stdc1 |rocket|
     - 74.57
     - 73.92
     - 15
     - 18
     - 1024x1920x3
     - 8.27
     - 126.47
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Cityscapes/stdc1/pretrained/2023-06-12/stdc1.zip>`_
     - `link <https://mmsegmentation.readthedocs.io/en/latest>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15m/stdc1.hef>`_

Oxford-IIIT Pet
^^^^^^^^^^^^^^^

.. list-table::
   :widths: 31 9 7 11 9 8 8 8 7 7 7
   :header-rows: 1

   * - Network Name
     - mIoU
     - Quantized
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
   * - unet_mobilenet_v2
     - 77.32
     - 77.02
     - 133
     - 227
     - 256x256x3
     - 10.08
     - 28.88
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Oxford_Pet/unet_mobilenet_v2/pretrained/2022-02-03/unet_mobilenet_v2.zip>`_
     - `link <https://www.tensorflow.org/tutorials/images/segmentation>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15m/unet_mobilenet_v2.hef>`_

Pascal VOC
^^^^^^^^^^

.. list-table::
   :widths: 31 9 7 11 9 8 8 8 7 7 7
   :header-rows: 1

   * - Network Name
     - mIoU
     - Quantized
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
   * - deeplab_v3_mobilenet_v2
     - 76.05
     - 74.8
     - 42
     - 56
     - 513x513x3
     - 2.10
     - 17.65
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Pascal/deeplab_v3_mobilenet_v2_dilation/pretrained/2023-08-22/deeplab_v3_mobilenet_v2_dilation.zip>`_
     - `link <https://github.com/bonlime/keras-deeplab-v3-plus>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15m/deeplab_v3_mobilenet_v2.hef>`_
   * - deeplab_v3_mobilenet_v2_wo_dilation
     - 71.46
     - 71.26
     - 87
     - 119
     - 513x513x3
     - 2.10
     - 3.21
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Pascal/deeplab_v3_mobilenet_v2/pretrained/2021-08-12/deeplab_v3_mobilenet_v2.zip>`_
     - `link <https://github.com/tensorflow/models/tree/master/research/deeplab>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo15m/deeplab_v3_mobilenet_v2_wo_dilation.hef>`_
