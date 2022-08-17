
Public Pre-Trained Models
=========================

.. |rocket| image:: images/rocket.png
  :width: 18

.. |star| image:: images/star.png
  :width: 18

Here, we give the full list of publicly pre-trained models supported by the Hailo Model Zoo.

* FLOPs in the table are counted as MAC operations.
* Network available in `Hailo Benchmark <https://hailo.ai/developer-zone/benchmarks/>`_ are marked with |rocket|
* Networks available in `TAPPAS <https://hailo.ai/developer-zone/tappas-apps-toolkit/>`_ are marked with |star|
* Supported tasks:

  * `Classification`_
  * `Object Detection`_
  * `Semantic Segmentation`_
  * `Pose Estimation`_
  * `Face Detection`_
  * `Instance Segmentation`_
  * `Depth Estimation`_
  * `Facial Landmark`_
  * `Person Re-ID`_
  * `Person Attribute`_
  * `Face Attribute`_
  * `Hand Landmark Detection`_
  * `Palm Detection`_
  
 
.. _Classification:

Classification
--------------

ImageNet
^^^^^^^^

.. list-table::
   :widths: 31 9 7 11 9 8 8 8 7
   :header-rows: 1

   * - Network Name
     - Accuracy (top1)
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - FLOPs (G)
     - Pretrained
     - Source
     - Compiled    
   * - efficientnet_l  
     - 80.464
     - 79.06
     - 300x300x3
     - 10.55
     - 9.70
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_l/pretrained/2021-07-11/efficientnet_l.zip>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/efficientnet_l.hef>`_    
   * - efficientnet_lite0  
     - 74.992
     - 74.24
     - 224x224x3
     - 4.63
     - 0.39
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite0/pretrained/2021-07-11/efficientnet_lite0.zip>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/efficientnet_lite0.hef>`_    
   * - efficientnet_lite1  
     - 76.68
     - 76.19
     - 240x240x3
     - 5.39
     - 0.61
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite1/pretrained/2021-07-11/efficientnet_lite1.zip>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/efficientnet_lite1.hef>`_    
   * - efficientnet_lite2  
     - 77.454
     - 76.24
     - 260x260x3
     - 6.06
     - 0.87
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite2/pretrained/2021-07-11/efficientnet_lite2.zip>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/efficientnet_lite2.hef>`_    
   * - efficientnet_lite3  
     - 79.29
     - 78.58
     - 280x280x3
     - 8.16
     - 1.40
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite3/pretrained/2021-07-11/efficientnet_lite3.zip>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/efficientnet_lite3.hef>`_    
   * - efficientnet_lite4  
     - 80.786
     - 80.01
     - 300x300x3
     - 12.95
     - 2.58
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite4/pretrained/2021-07-11/efficientnet_lite4.zip>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/efficientnet_lite4.hef>`_      
   * - efficientnet_m |rocket| 
     - 78.908
     - 78.28
     - 240x240x3
     - 6.87
     - 3.68
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_m/pretrained/2021-07-11/efficientnet_m.zip>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/efficientnet_m.hef>`_    
   * - efficientnet_s  
     - 77.638
     - 76.74
     - 224x224x3
     - 5.41
     - 2.36
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_s/pretrained/2021-07-11/efficientnet_s.zip>`_
     - `link <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/efficientnet_s.hef>`_    
   * - hardnet39ds  
     - 73.434
     - 71.91
     - 224x224x3
     - 3.48
     - 0.43
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/hardnet39ds/pretrained/2021-07-20/hardnet39ds.zip>`_
     - `link <https://github.com/PingoLH/Pytorch-HarDNet>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/hardnet39ds.hef>`_    
   * - hardnet68  
     - 75.472
     - 75.04
     - 224x224x3
     - 17.56
     - 4.25
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/hardnet68/pretrained/2021-07-20/hardnet68.zip>`_
     - `link <https://github.com/PingoLH/Pytorch-HarDNet>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/hardnet68.hef>`_    
   * - inception_v1  
     - 69.742
     - 69.3
     - 224x224x3
     - 6.62
     - 1.50
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/inception_v1/pretrained/2021-07-11/inception_v1.zip>`_
     - `link <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/inception_v1.hef>`_    
   * - mobilenet_v1  
     - 70.97
     - 70.13
     - 224x224x3
     - 4.22
     - 0.57
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v1/pretrained/2021-07-11/mobilenet_v1.zip>`_
     - `link <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/mobilenet_v1.hef>`_      
   * - mobilenet_v2_1.0 |rocket| 
     - 71.778
     - 71.07
     - 224x224x3
     - 3.49
     - 0.31
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v2_1.0/pretrained/2021-07-11/mobilenet_v2_1.0.zip>`_
     - `link <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/mobilenet_v2_1.0.hef>`_    
   * - mobilenet_v2_1.4  
     - 74.178
     - 73.22
     - 224x224x3
     - 6.09
     - 0.59
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v2_1.4/pretrained/2021-07-11/mobilenet_v2_1.4.zip>`_
     - `link <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/mobilenet_v2_1.4.hef>`_    
   * - mobilenet_v3  
     - 72.208
     - 71.73
     - 224x224x3
     - 4.07
     - 1.00
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v3/pretrained/2021-07-11/mobilenet_v3.zip>`_
     - `link <https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/mobilenet_v3.hef>`_      
   * - mobilenet_v3_large_minimalistic |rocket| 
     - 72.11
     - 71.24
     - 224x224x3
     - 3.91
     - 0.21
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v3_large_minimalistic/pretrained/2021-07-11/mobilenet_v3_large_minimalistic.zip>`_
     - `link <https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/mobilenet_v3_large_minimalistic.hef>`_    
   * - regnetx_1.6gf  
     - 77.048
     - 76.44
     - 224x224x3
     - 9.17
     - 1.61
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/regnetx_1.6gf/pretrained/2021-07-11/regnetx_1.6gf.zip>`_
     - `link <https://github.com/facebookresearch/pycls>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/regnetx_1.6gf.hef>`_      
   * - regnetx_800mf |rocket| 
     - 75.156
     - 74.51
     - 224x224x3
     - 7.24
     - 0.80
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/regnetx_800mf/pretrained/2021-07-11/regnetx_800mf.zip>`_
     - `link <https://github.com/facebookresearch/pycls>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/regnetx_800mf.hef>`_    
   * - regnety_200mf  
     - 70.38
     - 69.91
     - 224x224x3
     - 3.15
     - 0.20
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/regnety_200mf/pretrained/2021-07-11/regnety_200mf.zip>`_
     - `link <https://github.com/facebookresearch/pycls>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/regnety_200mf.hef>`_    
   * - resmlp12_relu  
     - 75.264
     - 74.06
     - 224x224x3
     - 15.77
     - 3.02
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resmlp12_relu/pretrained/2022-03-03/resmlp12_relu.zip>`_
     - `link <https://github.com/rwightman/pytorch-image-models/>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/resmlp12_relu.hef>`_    
   * - resnet_v1_18  
     - 71.258
     - 71.01
     - 224x224x3
     - 11.68
     - 1.82
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v1_18/pretrained/2022-04-19/resnet_v1_18.zip>`_
     - `link <https://github.com/yhhhli/BRECQ>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/resnet_v1_18.hef>`_    
   * - resnet_v1_34  
     - 72.698
     - 72.02
     - 224x224x3
     - 21.79
     - 3.67
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v1_34/pretrained/2021-07-11/resnet_v1_34.zip>`_
     - `link <https://github.com/tensorflow/models/tree/master/research/slim>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/resnet_v1_34.hef>`_       
   * - resnet_v1_50 |rocket| |star|
     - 75.124
     - 74.69
     - 224x224x3
     - 25.53
     - 3.49
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v1_50/pretrained/2021-07-11/resnet_v1_50.zip>`_
     - `link <https://github.com/tensorflow/models/tree/master/research/slim>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/resnet_v1_50.hef>`_    
   * - resnet_v2_18  
     - 69.568
     - 68.19
     - 224x224x3
     - 11.68
     - 1.82
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v2_18/pretrained/2021-07-11/resnet_v2_18.zip>`_
     - `link <https://github.com/onnx/models/tree/master/vision/classification/resnet>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/resnet_v2_18.hef>`_    
   * - resnet_v2_34  
     - 73.072
     - 72.72
     - 224x224x3
     - 21.79
     - 3.67
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v2_34/pretrained/2021-07-11/resnet_v2_34.zip>`_
     - `link <https://github.com/onnx/models/tree/master/vision/classification/resnet>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/resnet_v2_34.hef>`_    
   * - resnext26_32x4d  
     - 76.178
     - 75.78
     - 224x224x3
     - 15.37
     - 2.48
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnext26_32x4d/pretrained/2021-07-11/resnext26_32x4d.zip>`_
     - `link <https://github.com/osmr/imgclsmob/tree/master/pytorch>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/resnext26_32x4d.hef>`_    
   * - resnext50_32x4d  
     - 79.306
     - 78.39
     - 224x224x3
     - 24.99
     - 4.24
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnext50_32x4d/pretrained/2021-07-11/resnext50_32x4d.zip>`_
     - `link <https://github.com/osmr/imgclsmob/tree/master/pytorch>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/resnext50_32x4d.hef>`_    
   * - shufflenet_g8_w1  
     - 66.302
     - 65.44
     - 224x224x3
     - 2.46
     - 0.18
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/shufflenet_g8_w1/pretrained/2021-07-11/shufflenet_g8_w1.zip>`_
     - `link <https://github.com/osmr/imgclsmob/tree/master/pytorch>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/shufflenet_g8_w1.hef>`_      
   * - squeezenet_v1.1 |rocket| 
     - 59.848
     - 58.99
     - 224x224x3
     - 1.24
     - 0.39
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/squeezenet_v1.1/pretrained/2021-07-11/squeezenet_v1.1.zip>`_
     - `link <https://github.com/osmr/imgclsmob/tree/master/pytorch>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/squeezenet_v1.1.hef>`_
 
.. _Object Detection:

Object Detection
----------------

COCO
^^^^

.. list-table::
   :widths: 33 8 7 12 8 8 8 7 7
   :header-rows: 1

   * - Network Name
     - mAP
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - FLOPs (G)
     - Pretrained
     - Source
     - Compiled    
   * - centernet_resnet_v1_18_postprocess  
     - 26.289
     - 24.72
     - 512x512x3
     - 14.22
     - 15.63
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/centernet/centernet_resnet_v1_18/pretrained/2021-07-11/centernet_resnet_v1_18.zip>`_
     - `link <https://cv.gluon.ai/model_zoo/detection.html>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/centernet_resnet_v1_18_postprocess.hef>`_    
   * - centernet_resnet_v1_50_postprocess  
     - 31.778
     - 30.08
     - 512x512x3
     - 30.07
     - 28.46
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/centernet/centernet_resnet_v1_50_postprocess/pretrained/2021-07-11/centernet_resnet_v1_50_postprocess.zip>`_
     - `link <https://cv.gluon.ai/model_zoo/detection.html>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/centernet_resnet_v1_50_postprocess.hef>`_    
   * - efficientdet_lite0  
     - 27.435
     - 26.27
     - 320x320x3
     - 3.56
     - 0.99
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/efficientdet/efficientdet_lite0/pretrained/2022-06-14/efficientdet-lite0.zip>`_
     - `link <https://github.com/google/automl/tree/master/efficientdet>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/efficientdet_lite0.hef>`_    
   * - efficientdet_lite1  
     - 32.458
     - 31.69
     - 384x384x3
     - 4.73
     - 2
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/efficientdet/efficientdet_lite1/pretrained/2022-06-26/efficientdet-lite1.zip>`_
     - `link <https://github.com/google/automl/tree/master/efficientdet>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/efficientdet_lite1.hef>`_    
   * - efficientdet_lite2  
     - 36.157
     - 34.99
     - 448x448x3
     - 5.93
     - 3.42
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/efficientdet/efficientdet_lite2/pretrained/2022-06-26/efficientdet-lite2.zip>`_
     - `link <https://github.com/google/automl/tree/master/efficientdet>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/efficientdet_lite2.hef>`_    
   * - nanodet_repvgg  
     - 29.3
     - 28.53
     - 416x416x3
     - 6.74
     - 5.64
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/nanodet/nanodet_repvgg/pretrained/2022-02-07/nanodet.zip>`_
     - `link <https://github.com/RangiLyu/nanodet>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/nanodet_repvgg.hef>`_    
   * - nanodet_repvgg_a1_640  
     - 33.28
     - 32.88
     - 640x640x3
     - 10.79
     - 21.4
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/nanodet/nanodet_repvgg_a1_640/pretrained/2022-07-19/nanodet_repvgg_a1_640.zip>`_
     - `link <https://github.com/RangiLyu/nanodet>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/nanodet_repvgg_a1_640.hef>`_    
   * - ssd_mobiledet_dsp  
     - 28.9
     - 28.17
     - 320x320x3
     - 7.07
     - 2.83
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/ssd/ssd_mobiledet_dsp/pretrained/2021-07-11/ssd_mobiledet_dsp.zip>`_
     - `link <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/ssd_mobiledet_dsp.hef>`_       
   * - ssd_mobilenet_v1 |rocket| |star|
     - 23.172
     - 22.29
     - 300x300x3
     - 6.79
     - 1.25
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/ssd/ssd_mobilenet_v1/pretrained/2021-07-11/ssd_mobilenet_v1.zip>`_
     - `link <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/ssd_mobilenet_v1.hef>`_    
   * - ssd_mobilenet_v1_hd  
     - 17.657
     - 16.41
     - 720x1280x3
     - 6.81
     - 12.26
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/ssd/ssd_mobilenet_v1_hd/pretrained/2021-07-11/ssd_mobilenet_v1_hd.zip>`_
     - `link <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/ssd_mobilenet_v1_hd.hef>`_      
   * - ssd_mobilenet_v2 |rocket| 
     - 24.154
     - 23.28
     - 300x300x3
     - 4.46
     - 0.76
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/ssd/ssd_mobilenet_v2/pretrained/2021-07-11/ssd_mobilenet_v2.zip>`_
     - `link <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/ssd_mobilenet_v2.hef>`_    
   * - ssd_resnet_v1_18  
     - 17.664
     - 17.41
     - 300x300x3
     - 16.62
     - 3.54
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/ssd/ssd_resnet_v1_18/pretrained/2021-07-11/ssd_resnet_v1_18.zip>`_
     - `link <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/ssd_resnet_v1_18.hef>`_      
   * - tiny_yolov3 |rocket| 
     - 14.36
     - 13.45
     - 416x416x3
     - 8.85
     - 2.79
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/tiny_yolov3/pretrained/2021-07-11/tiny_yolov3.zip>`_
     - `link <https://github.com/Tianxiaomo/pytorch-YOLOv4>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/tiny_yolov3.hef>`_    
   * - tiny_yolov4  
     - 19.177
     - 17.73
     - 416x416x3
     - 6.05
     - 3.46
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/tiny_yolov4/pretrained/2021-07-11/tiny_yolov4.zip>`_
     - `link <https://github.com/Tianxiaomo/pytorch-YOLOv4>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/tiny_yolov4.hef>`_     
   * - yolov3  |star|
     - 38.421
     - 37.22
     - 608x608x3
     - 68.79
     - 79.17
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov3/pretrained/2021-08-16/yolov3.zip>`_
     - `link <https://github.com/AlexeyAB/darknet>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/yolov3.hef>`_    
   * - yolov3_416  
     - 37.733
     - 35.86
     - 416x416x3
     - 61.92
     - 32.97
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov3_416/pretrained/2021-08-16/yolov3_416.zip>`_
     - `link <https://github.com/AlexeyAB/darknet>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/yolov3_416.hef>`_       
   * - yolov3_gluon |rocket| |star|
     - 37.277
     - 35.64
     - 608x608x3
     - 68.79
     - 79.17
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov3_gluon/pretrained/2021-07-11/yolov3_gluon.zip>`_
     - `link <https://cv.gluon.ai/model_zoo/detection.html>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/yolov3_gluon.hef>`_       
   * - yolov3_gluon_416 |rocket| |star|
     - 36.274
     - 35.23
     - 416x416x3
     - 61.92
     - 32.97
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov3_gluon_416/pretrained/2021-07-11/yolov3_gluon_416.zip>`_
     - `link <https://cv.gluon.ai/model_zoo/detection.html>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/yolov3_gluon_416.hef>`_       
   * - yolov4_leaky |rocket| |star|
     - 42.37
     - 41.47
     - 512x512x3
     - 64.33
     - 45.60
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov4/pretrained/2022-03-17/yolov4.zip>`_
     - `link <https://github.com/AlexeyAB/darknet/wiki/YOLOv4-model-zoo>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/yolov4_leaky.hef>`_    
   * - yolov5l  
     - 46.012
     - 44.01
     - 640x640x3
     - 48.54
     - 60.78
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5l_spp/pretrained/2022-02-03/yolov5l.zip>`_
     - `link <https://github.com/ultralytics/yolov5/releases/tag/v2.0>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/yolov5l.hef>`_    
   * - yolov5m  
     - 42.591
     - 40.93
     - 640x640x3
     - 21.78
     - 26.14
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5m_spp/pretrained/2022-01-02/yolov5m.zip>`_
     - `link <https://github.com/ultralytics/yolov5/releases/tag/v2.0>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/yolov5m.hef>`_      
   * - yolov5m_wo_spp |rocket| 
     - 42.46
     - 40.66
     - 640x640x3
     - 22.67
     - 26.49
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5m/pretrained/2022-04-19/yolov5m_wo_spp.zip>`_
     - `link <https://github.com/ultralytics/yolov5/releases/tag/v2.0>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/yolov5m_wo_spp_60p.hef>`_    
   * - yolov5s  
     - 35.331
     - 33.78
     - 640x640x3
     - 7.46
     - 8.72
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5s_spp/pretrained/2022-01-02/yolov5s.zip>`_
     - `link <https://github.com/ultralytics/yolov5/releases/tag/v2.0>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/yolov5s.hef>`_    
   * - yolov5s_personface  
     - 47.7
     - 46.1
     - 640x640x3
     - 7.25
     - 8.38
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/MCPReID/personface_detector/yolov5s_personface/2022-04-01/yolov5s_personface.zip>`_
     - `link <https://github.com/ultralytics/yolov5/releases/tag/v2.0>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/yolov5s_personface.hef>`_    
   * - yolov5s_wo_spp  
     - 34.469
     - 33.26
     - 640x640x3
     - 7.85
     - 8.87
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5s/pretrained/2021-07-11/yolov5s.zip>`_
     - `link <https://github.com/ultralytics/yolov5/releases/tag/v2.0>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/yolov5s_wo_spp.hef>`_    
   * - yolov5xs_wo_spp  
     - 32.783
     - 31.43
     - 512x512x3
     - 7.85
     - 5.68
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5xs/pretrained/2021-07-11/yolov5xs.zip>`_
     - `link <https://github.com/ultralytics/yolov5/releases/tag/v2.0>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/yolov5xs_wo_spp.hef>`_    
   * - yolov5xs_wo_spp_nms  
     - 32.567
     - 30.87
     - 512x512x3
     - 7.85
     - 5.68
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5xs/pretrained/2022-05-10/yolov5xs_wo_spp_nms.zip>`_
     - `link <https://github.com/ultralytics/yolov5/releases/tag/v2.0>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/yolov5xs_wo_spp_nms.hef>`_    
   * - yolov7  
     - 49.719
     - 47.52
     - 640x640x3
     - 36.91
     - 52.34
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov7/pretrained/2022-07-10/yolov7.zip>`_
     - `link <https://github.com/WongKinYiu/yolov7>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/yolov7.hef>`_    
   * - yolov7_tiny  
     - 36.491
     - 35.39
     - 640x640x3
     - 6.22
     - 6.87
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov7_tiny/pretrained/2022-07-10/yolov7_tiny.zip>`_
     - `link <https://github.com/WongKinYiu/yolov7>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/yolov7_tiny.hef>`_     
   * - yolox_l_leaky  |star|
     - 48.684
     - 47.08
     - 640x640x3
     - 54.17
     - 77.74
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolox_l_leaky/pretrained/2021-09-23/yolox_l_leaky.zip>`_
     - `link <https://github.com/Megvii-BaseDetection/YOLOX>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/yolox_l_leaky.hef>`_    
   * - yolox_s_leaky  
     - 38.126
     - 37.47
     - 640x640x3
     - 8.96
     - 13.37
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolox_s_leaky/pretrained/2021-09-12/yolox_s_leaky.zip>`_
     - `link <https://github.com/Megvii-BaseDetection/YOLOX>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/yolox_s_leaky.hef>`_    
   * - yolox_s_wide_leaky  
     - 42.401
     - 41.01
     - 640x640x3
     - 20.12
     - 29.73
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolox_s_wide_leaky/pretrained/2021-09-12/yolox_s_wide_leaky.zip>`_
     - `link <https://github.com/Megvii-BaseDetection/YOLOX>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/yolox_s_wide_leaky.hef>`_    
   * - yolox_tiny  
     - 32.64
     - 31.32
     - 416x416x3
     - 5.05
     - 3.22
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolox/yolox_tiny/pretrained/2022-06-01/yolox_tiny.zip>`_
     - `link <https://github.com/Megvii-BaseDetection/YOLOX>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/yolox_tiny.hef>`_    
   * - yolox_tiny_leaky  
     - 30.269
     - 29.64
     - 416x416x3
     - 5.05
     - 3.22
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolox_tiny_leaky/pretrained/2021-08-12/yolox_tiny_leaky.zip>`_
     - `link <https://github.com/Megvii-BaseDetection/YOLOX>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/yolox_tiny_leaky.hef>`_

VisDrone
^^^^^^^^

.. list-table::
   :widths: 31 7 9 12 9 8 9 8 7
   :header-rows: 1

   * - Network Name
     - mAP
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - FLOPs (G)
     - Pretrained
     - Source
     - Compiled     
   * - ssd_mobilenet_v1_visdrone  |star|
     - 2.181
     - 2.16
     - 300x300x3
     - 5.64
     - 1.15
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-Visdrone/ssd/ssd_mobilenet_v1_visdrone/pretrained/2021-07-11/ssd_mobilenet_v1_visdrone.zip>`_
     - `link <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/ssd_mobilenet_v1_visdrone.hef>`_
 
.. _Semantic Segmentation:

Semantic Segmentation
---------------------

Cityscapes
^^^^^^^^^^

.. list-table::
   :widths: 23 8 9 18 9 8 9 8 8
   :header-rows: 1

   * - Network Name
     - mIoU
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - FLOPs (G)
     - Pretrained
     - Source
     - Compiled     
   * - fcn16_resnet_v1_18  |star|
     - 66.827
     - 66.57
     - 1024x1920x3
     - 11.19
     - 71.26
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Cityscapes/fcn16_resnet_v1_18/pretrained/2022-02-07/fcn16_resnet_v1_18.zip>`_
     - `link <https://mmsegmentation.readthedocs.io/en/latest>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/fcn16_resnet_v1_18.hef>`_      
   * - fcn8_resnet_v1_18 |rocket| 
     - 68.753
     - 68.51
     - 1024x1920x3
     - 11.20
     - 71.51
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Cityscapes/fcn8_resnet_v1_18/pretrained/2022-02-09/fcn8_resnet_v1_18.zip>`_
     - `link <https://mmsegmentation.readthedocs.io/en/latest>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/fcn8_resnet_v1_18.hef>`_    
   * - fcn8_resnet_v1_22  
     - 67.545
     - 67.39
     - 1920x1024x3
     - 15.12
     - 150.04
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Cityscapes/fcn8_resnet_v1_22/pretrained/2021-07-11/fcn8_resnet_v1_22.zip>`_
     - `link <https://cv.gluon.ai/model_zoo/segmentation.html>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/fcn8_resnet_v1_22.hef>`_    
   * - stdc1  
     - 74.566
     - 73.49
     - 1024x1920x3
     - 8.27
     - 63.34
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Cityscapes/stdc1/pretrained/2022-03-17/stdc1.zip>`_
     - `link <https://mmsegmentation.readthedocs.io/en/latest>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/stdc1.hef>`_

Oxford-IIIT Pet
^^^^^^^^^^^^^^^

.. list-table::
   :widths: 24 8 9 18 9 8 9 8 7
   :header-rows: 1

   * - Network Name
     - mIoU
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - FLOPs (G)
     - Pretrained
     - Source
     - Compiled      
   * - unet_mobilenet_v2 |rocket| 
     - 77.319
     - 76.42
     - 256x256x3
     - 10.08
     - 14.44
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Oxford_Pet/unet_mobilenet_v2/pretrained/2022-02-03/unet_mobilenet_v2.zip>`_
     - `link <https://www.tensorflow.org/tutorials/images/segmentation>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/unet_mobilenet_v2.hef>`_

Pascal VOC
^^^^^^^^^^

.. list-table::
   :widths: 34 7 7 11 9 8 8 8 7
   :header-rows: 1

   * - Network Name
     - mIoU
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - FLOPs (G)
     - Pretrained
     - Source
     - Compiled      
   * - deeplab_v3_mobilenet_v2 |rocket| 
     - 76.045
     - 74.8
     - 513x513x3
     - 2.10
     - 8.91
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Pascal/deeplab_v3_mobilenet_v2_dilation/pretrained/2021-09-26/deeplab_v3_mobilenet_v2_dilation.zip>`_
     - `link <https://github.com/bonlime/keras-deeplab-v3-plus>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/deeplab_v3_mobilenet_v2.hef>`_    
   * - deeplab_v3_mobilenet_v2_wo_dilation  
     - 71.461
     - 70.25
     - 513x513x3
     - 2.10
     - 1.64
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Pascal/deeplab_v3_mobilenet_v2/pretrained/2021-08-12/deeplab_v3_mobilenet_v2.zip>`_
     - `link <https://github.com/tensorflow/models/tree/master/research/deeplab>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/deeplab_v3_mobilenet_v2_wo_dilation.hef>`_
 
.. _Pose Estimation:

Pose Estimation
---------------

COCO
^^^^

.. list-table::
   :widths: 28 8 8 16 9 8 8 8 7
   :header-rows: 1

   * - Network Name
     - AP
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - FLOPs (G)
     - Pretrained
     - Source
     - Compiled     
   * - centerpose_regnetx_1.6gf_fpn  |star|
     - 53.542
     - 46.94
     - 640x640x3
     - 14.28
     - 32.38
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/PoseEstimation/centerpose_regnetx_1.6gf_fpn/pretrained/2022-03-23/centerpose_regnetx_1.6gf_fpn.zip>`_
     - `link <https://github.com/tensorboy/centerpose>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/centerpose_regnetx_1.6gf_fpn.hef>`_    
   * - centerpose_regnetx_800mf  
     - 44.074
     - 42.23
     - 512x512x3
     - 12.31
     - 43.06
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/PoseEstimation/centerpose_regnetx_800mf/pretrained/2021-07-11/centerpose_regnetx_800mf.zip>`_
     - `link <https://github.com/tensorboy/centerpose>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/centerpose_regnetx_800mf.hef>`_     
   * - centerpose_repvgg_a0  |star|
     - 39.167
     - 36.89
     - 416x416x3
     - 11.71
     - 14.15
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/PoseEstimation/centerpose_repvgg_a0/pretrained/2021-09-26/centerpose_repvgg_a0.zip>`_
     - `link <https://github.com/tensorboy/centerpose>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/centerpose_repvgg_a0.hef>`_
 
.. _Face Detection:

Face Detection
--------------

WiderFace
^^^^^^^^^

.. list-table::
   :widths: 28 8 9 13 9 8 8 8 7
   :header-rows: 1

   * - Network Name
     - mAP
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - FLOPs (G)
     - Pretrained
     - Source
     - Compiled     
   * - lightface_slim  |star|
     - 39.705
     - 39.36
     - 240x320x3
     - 0.26
     - 0.08
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/lightface_slim/2021-07-18/lightface_slim.zip>`_
     - `link <https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/lightface_slim.hef>`_     
   * - retinaface_mobilenet_v1  |star|
     - 81.272
     - 81.17
     - 736x1280x3
     - 3.49
     - 12.57
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/retinaface_mobilenet_v1_hd/2021-07-18/retinaface_mobilenet_v1_hd.zip>`_
     - `link <https://github.com/biubug6/Pytorch_Retinaface>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/retinaface_mobilenet_v1.hef>`_
 
.. _Instance Segmentation:

Instance Segmentation
---------------------

COCO
^^^^

.. list-table::
   :widths: 32 8 7 11 9 8 8 8 7
   :header-rows: 1

   * - Network Name
     - mAP
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - FLOPs (G)
     - Pretrained
     - Source
     - Compiled    
   * - yolact_mobilenet_v1  
     - 17.783
     - 17.15
     - 512x512x3
     - 19.11
     - 51.92
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolact_mobilenet_v1/pretrained/2021-01-12/yolact_mobilenet_v1.zip>`_
     - `link <https://github.com/dbolya/yolact>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/yolact_mobilenet_v1.hef>`_     
   * - yolact_regnetx_800mf_20classes  |star|
     - 22.865
     - 22.56
     - 512x512x3
     - 21.97
     - 51.47
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolact_regnetx_800mf/pretrained/2022-02-08/yolact_regnetx_800mf.zip>`_
     - `link <https://github.com/dbolya/yolact>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/yolact_regnetx_800mf_20classes.hef>`_

D2S
^^^

.. list-table::
   :widths: 33 7 7 11 9 8 8 8 7
   :header-rows: 1

   * - Network Name
     - mAP
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - FLOPs (G)
     - Pretrained
     - Source
     - Compiled    
   * - yolact_regnetx_600mf_d2s_31classes  
     - 61.685
     - 62.98
     - 512x512x3
     - 22.14
     - 51.62
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/d2s/yolact_regnetx_600mf/pretrained/2022-07-19/yolact_regnetx_600mf_d2s.zip>`_
     - `link <https://github.com/dbolya/yolact>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/yolact_regnetx_600mf_d2s_31classes.hef>`_
 
.. _Depth Estimation:

Depth Estimation
----------------

NYU
^^^

.. list-table::
   :widths: 20 7 12 14 9 8 15 8 7
   :header-rows: 1

   * - Network Name
     - RMSE
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - FLOPs (G)
     - Pretrained
     - Source
     - Compiled       
   * - fast_depth  |star|
     - 0.604
     - 0.61
     - 224x224x3
     - 1.35
     - 0.37
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/DepthEstimation/indoor/fast_depth/pretrained/2021-10-18/fast_depth.zip>`_
     - `link <https://github.com/dwofk/fast-depth>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/fast_depth.hef>`_
 
.. _Facial Landmark:

Facial Landmark
---------------

AFLW2k3d
^^^^^^^^

.. list-table::
   :widths: 24 7 11 14 9 8 12 8 7
   :header-rows: 1

   * - Network Name
     - NME
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - FLOPs (G)
     - Pretrained
     - Source
     - Compiled       
   * - tddfa_mobilenet_v1  |star|
     - 3.68
     - 4.06
     - 120x120x3
     - 3.26
     - 0.18
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceLandmarks3d/tddfa/tddfa_mobilenet_v1/pretrained/2021-11-28/tddfa_mobilenet_v1.zip>`_
     - `link <https://github.com/cleardusk/3DDFA_V2>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/tddfa_mobilenet_v1.hef>`_
 
.. _Person Re-ID:

Person Re-ID
------------

Market1501
^^^^^^^^^^

.. list-table::
   :widths: 28 8 8 14 9 8 10 8 7
   :header-rows: 1

   * - Network Name
     - rank1
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - FLOPs (G)
     - Pretrained
     - Source
     - Compiled    
   * - osnet_x1_0  
     - 94.433
     - 92.24
     - 256x128x3
     - 2.19
     - 0.99
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/PersonReID/osnet_x1_0/2022-05-19/osnet_x1_0.zip>`_
     - `link <https://github.com/KaiyangZhou/deep-person-reid>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/osnet_x1_0.hef>`_    
   * - repvgg_a0_person_reid_2048  
     - 90.021
     - 89.32
     - 256x128x3
     - 9.65
     - 0.89
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/MCPReID/reid/repvgg_a0_person_reid_2048/2022-04-18/repvgg_a0_person_reid_2048.zip>`_
     - `link <https://github.com/DingXiaoH/RepVGG>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/repvgg_a0_person_reid_2048.hef>`_    
   * - repvgg_a0_person_reid_512  
     - 89.9
     - 89.05
     - 256x128x3
     - 7.68
     - 0.89
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/MCPReID/reid/repvgg_a0_person_reid_512/2022-04-18/repvgg_a0_person_reid_512.zip>`_
     - `link <https://github.com/DingXiaoH/RepVGG>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/repvgg_a0_person_reid_512.hef>`_
 
.. _Person Attribute:

Person Attribute
----------------

PETA
^^^^

.. list-table::
   :widths: 24 12 8 14 9 8 10 8 7
   :header-rows: 1

   * - Network Name
     - Mean Accuracy
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - FLOPs (G)
     - Pretrained
     - Source
     - Compiled    
   * - person_attr_resnet_v1_18  
     - 82.504
     - 82.61
     - 224x224x3
     - 11.19
     - 1.82
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/person_attr_resnet_v1_18/pretrained/2022-06-11/person_attr_resnet_v1_18.zip>`_
     - `link <https://github.com/dangweili/pedestrian-attribute-recognition-pytorch>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/person_attr_resnet_v1_18.hef>`_
 
.. _Face Attribute:

Face Attribute
--------------

CELEBA
^^^^^^

.. list-table::
   :widths: 31 9 7 11 9 8 8 8 7
   :header-rows: 1

   * - Network Name
     - Mean Accuracy
     - Quantized
     - Input Resolution (HxWxC)
     - Params (M)
     - FLOPs (G)
     - Pretrained
     - Source
     - Compiled    
   * - face_attr_resnet_v1_18  
     - 81.188
     - 80.79
     - 218x178x3
     - 11.74
     - 1.5
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceAttr/face_attr_resnet_v1_18/2022-06-09/face_attr_resnet_v1_18.zip>`_
     - `link <https://github.com/d-li14/face-attribute-prediction>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/face_attr_resnet_v1_18.hef>`_
 
.. _Hand Landmark Detection:

Hand Landmark Detection
-----------------------

Hand Landmark
^^^^^^^^^^^^^
    
.. list-table::
   :header-rows: 1

   * - Network Name
     - Input Resolution (HxWxC)
     - Params (M)
     - FLOPs (G)
     - Pretrained
     - Source
     - Compiled    
   * - hand_landmark_lite  
     - 224x224x3
     - 1.01
     - 0.15
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HandLandmark/hand_landmark_lite/2022-01-23/hand_landmark_lite.zip>`_
     - `link <https://github.com/google/mediapipe>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/hand_landmark_lite.hef>`_
 
.. _Palm Detection:

Palm Detection
--------------

Palm Detection
^^^^^^^^^^^^^^
    
.. list-table::
   :header-rows: 1

   * - Network Name
     - Input Resolution (HxWxC)
     - Params (M)
     - FLOPs (G)
     - Pretrained
     - Source
     - Compiled    
   * - palm_detection_lite  
     - 192x192x3
     - 1.01
     - 0.31
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-Palm/palm_detection_lite/pretrained/palm_detection_lite.zip>`_
     - `link <https://github.com/google/mediapipe>`_
     - `link <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.3.0/palm_detection_lite.hef>`_
