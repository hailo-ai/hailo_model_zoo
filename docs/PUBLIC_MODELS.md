# Public Pre-Trained Models

Here, we give the full list of publicly pre-trained models supported by the Hailo Model Zoo.

- FLOPs in the table are counted as MAC operations.
- Network available in [**Hailo Benchmark**](https://hailo.ai/developer-zone/benchmarks/) are marked with <html>&#128640;</html>
- Networks available in [**TAPPAS**](https://hailo.ai/developer-zone/tappas-apps-toolkit/) are marked with <html>&#11088;</html>
- Supported tasks:
    - [Classification](#classification)
    - [Object Detection](#object-detection)
    - [Semantic Segmentation](#semantic-segmentation)
    - [Pose Estimation](#pose-estimation)
    - [Face Detection](#face-detection)
    - [Instance Segmentation](#instance-segmentation)
    - [Depth Estimation](#depth-estimation)
    - [Facial Landmark](#facial-landmark)
    - [Person Re-ID](#person-re-id)
    - [Person Attribute](#person-attribute)
    - [Hand Landmark detection](#hand-landmark-detection)
    - [Palm Detection](#palm-detection)
    

<br>

## Classification

### ImageNet
                                                                                                                    
| Network Name | Accuracy (top1) | Quantized | Input Resolution (HxWxC) | Params (M) | FLOPs (G) | Pretrained | Source | Compiled |
| -------------- |----------------| ------------------------ | ---------- | --------- | ---- | -------- | ------- | -------- | 
| efficientnet_l | 80.46          | 79.06  | 300x300x3 | 10.55 | 9.70 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_l/pretrained/2021-07-11/efficientnet_l.zip) | [**link**](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/efficientnet_l.hef) | 
| efficientnet_m<html>&#128640;</html> | 78.91          | 78.28  | 240x240x3 | 6.87 | 3.68 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_m/pretrained/2021-07-11/efficientnet_m.zip) | [**link**](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/efficientnet_m.hef) |
| efficientnet_s | 77.63          | 76.96  | 224x224x3 | 5.41 | 2.36 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_s/pretrained/2021-07-11/efficientnet_s.zip) | [**link**](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/efficientnet_s.hef) |
| efficientnet_lite0 | 74.99          | 74.31  | 224x224x3 | 4.63 | 0.39 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite0/pretrained/2021-07-11/efficientnet_lite0.zip) | [**link**](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/efficientnet_lite0.hef) | 
| efficientnet_lite1 | 76.68          | 76.19  | 240x240x3 | 5.39 | 0.61 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite1/pretrained/2021-07-11/efficientnet_lite1.zip) | [**link**](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/efficientnet_lite1.hef) | 
| efficientnet_lite2 | 77.45          | 76.24  | 260x260x3 | 6.06 | 0.87 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite2/pretrained/2021-07-11/efficientnet_lite2.zip) | [**link**](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/efficientnet_lite2.hef) | 
| efficientnet_lite3 | 79.29          | 78.58  | 280x280x3 | 8.16 | 1.40 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite3/pretrained/2021-07-11/efficientnet_lite3.zip) | [**link**](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/efficientnet_lite3.hef) | 
| efficientnet_lite4 | 80.78          | 80.01  | 300x300x3 | 12.95 | 2.58 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite4/pretrained/2021-07-11/efficientnet_lite4.zip) | [**link**](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/efficientnet_lite4.hef) |   
| hardnet39ds | 73.43          | 71.8  | 224x224x3 | 3.48 | 0.43 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/hardnet39ds/pretrained/2021-07-20/hardnet39ds.zip) | [**link**](https://github.com/PingoLH/Pytorch-HarDNet) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/hardnet39ds.hef) | 
| hardnet68 | 75.47          | 75.04  | 224x224x3 | 17.56 | 4.25 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/hardnet68/pretrained/2021-07-20/hardnet68.zip) | [**link**](https://github.com/PingoLH/Pytorch-HarDNet) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/hardnet68.hef) | 
| inception_v1 | 69.74          | 69.52  | 224x224x3 | 6.62 | 1.50 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/inception_v1/pretrained/2021-07-11/inception_v1.zip) | [**link**](https://github.com/tensorflow/models/tree/v1.13.0/research/slim) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/inception_v1.hef) | 
| mobilenet_v1 | 70.97          | 70.13  | 224x224x3 | 4.22 | 0.57 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v1/pretrained/2021-07-11/mobilenet_v1.zip) | [**link**](https://github.com/tensorflow/models/tree/v1.13.0/research/slim) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/mobilenet_v1.hef) |   
| mobilenet_v2_1.0<html>&#128640;</html> | 71.77          | 71.07  | 224x224x3 | 3.49 | 0.31 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v2_1.0/pretrained/2021-07-11/mobilenet_v2_1.0.zip) | [**link**](https://github.com/tensorflow/models/tree/v1.13.0/research/slim) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/mobilenet_v2_1.0.hef) | 
| mobilenet_v2_1.4 | 74.18          | 73.22  | 224x224x3 | 6.09 | 0.59 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v2_1.4/pretrained/2021-07-11/mobilenet_v2_1.4.zip) | [**link**](https://github.com/tensorflow/models/tree/v1.13.0/research/slim) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/mobilenet_v2_1.4.hef) | 
| mobilenet_v3 | 72.21          | 71.73  | 224x224x3 | 4.07 | 1.00 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v3/pretrained/2021-07-11/mobilenet_v3.zip) | [**link**](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/mobilenet_v3.hef) |   
| mobilenet_v3_large_minimalistic<html>&#128640;</html> | 72.11          | 71.13  | 224x224x3 | 3.91 | 0.21 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v3_large_minimalistic/pretrained/2021-07-11/mobilenet_v3_large_minimalistic.zip) | [**link**](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/mobilenet_v3_large_minimalistic.hef) | 
| regnetx_1.6gf | 77.05          | 76.44  | 224x224x3 | 9.17 | 1.61 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/regnetx_1.6gf/pretrained/2021-07-11/regnetx_1.6gf.zip) | [**link**](https://github.com/facebookresearch/pycls) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/regnetx_1.6gf.hef) |   
| regnetx_800mf<html>&#128640;</html> | 75.156         | 74.51  | 224x224x3 | 7.24 | 0.80 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/regnetx_800mf/pretrained/2021-07-11/regnetx_800mf.zip) | [**link**](https://github.com/facebookresearch/pycls) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/regnetx_800mf.hef) | 
| regnety_200mf | 70.38          | 69.91  | 224x224x3 | 3.15 | 0.20 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/regnety_200mf/pretrained/2021-07-11/regnety_200mf.zip) | [**link**](https://github.com/facebookresearch/pycls) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/regnety_200mf.hef) | 
| resmlp12_relu | 75.26          | 74.06  | 224x224x3 | 15.77 | 3.02 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resmlp12_relu/pretrained/2022-03-03/resmlp12_relu.zip) | [**link**](https://github.com/rwightman/pytorch-image-models/) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/resmlp12_relu.hef) | 
| resnet_v1_18 | 71.26          | 70.54  | 224x224x3 | 11.68 | 1.82 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v1_18/pretrained/2022-04-19/resnet_v1_18.zip) | [**link**](https://github.com/yhhhli/BRECQ) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/resnet_v1_18.hef) | 
| resnet_v1_34 | 72.70          | 71.83  | 224x224x3 | 21.79 | 3.67 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v1_34/pretrained/2021-07-11/resnet_v1_34.zip) | [**link**](https://github.com/tensorflow/models/tree/master/research/slim) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/resnet_v1_34.hef) |    
| resnet_v1_50<html>&#128640;</html><html>&#11088;</html> | 75.124         | 74.69  | 224x224x3 | 25.53 | 3.49 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v1_50/pretrained/2021-07-11/resnet_v1_50.zip) | [**link**](https://github.com/tensorflow/models/tree/master/research/slim) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/resnet_v1_50.hef) | 
| resnet_v2_18 | 69.57          | 68.19  | 224x224x3 | 11.68 | 1.82 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v2_18/pretrained/2021-07-11/resnet_v2_18.zip) | [**link**](https://github.com/onnx/models/tree/master/vision/classification/resnet) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/resnet_v2_18.hef) | 
| resnet_v2_34 | 73.07          | 72.72  | 224x224x3 | 21.79 | 3.67 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v2_34/pretrained/2021-07-11/resnet_v2_34.zip) | [**link**](https://github.com/onnx/models/tree/master/vision/classification/resnet) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/resnet_v2_34.hef) | 
| resnext26_32x4d | 76.18          | 74.86  | 224x224x3 | 15.37 | 2.48 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnext26_32x4d/pretrained/2021-07-11/resnext26_32x4d.zip) | [**link**](https://github.com/osmr/imgclsmob/tree/master/pytorch) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/resnext26_32x4d.hef) | 
| resnext50_32x4d | 79.31          | 78.39  | 224x224x3 | 24.99 | 4.24 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnext50_32x4d/pretrained/2021-07-11/resnext50_32x4d.zip) | [**link**](https://github.com/osmr/imgclsmob/tree/master/pytorch) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/resnext50_32x4d.hef) | 
| shufflenet_g8_w1 | 66.30          | 65.44  | 224x224x3 | 2.46 | 0.18 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/shufflenet_g8_w1/pretrained/2021-07-11/shufflenet_g8_w1.zip) | [**link**](https://github.com/osmr/imgclsmob/tree/master/pytorch) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/shufflenet_g8_w1.hef) |   
| squeezenet_v1.1<html>&#128640;</html> | 59.848         | 58.99  | 224x224x3 | 1.24 | 0.39 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/squeezenet_v1.1/pretrained/2021-07-11/squeezenet_v1.1.zip) | [**link**](https://github.com/osmr/imgclsmob/tree/master/pytorch) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/squeezenet_v1.1.hef) |
<br>
 
## Object Detection

### COCO
                                                                                                     
| Network Name | mAP    | Quantized | Input Resolution (HxWxC) | Params (M) | FLOPs (G) | Pretrained | Source | Compiled |
| -------------- |--------| ------------------------ | ---------- | --------- | ---- | -------- | ------- | -------- | 
| centernet_resnet_v1_18_postprocess | 26.289 | 24.72  | 512x512x3 | 14.22 | 15.63 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/centernet/centernet_resnet_v1_18/pretrained/2021-07-11/centernet_resnet_v1_18.zip) | [**link**](https://cv.gluon.ai/model_zoo/detection.html) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/centernet_resnet_v1_18_postprocess.hef) | 
| centernet_resnet_v1_50_postprocess | 31.778 | 30.08  | 512x512x3 | 30.07 | 28.46 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/centernet/centernet_resnet_v1_50_postprocess/pretrained/2021-07-11/centernet_resnet_v1_50_postprocess.zip) | [**link**](https://cv.gluon.ai/model_zoo/detection.html) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/centernet_resnet_v1_50_postprocess.hef) | 
| nanodet_repvgg | 29.3   | 28.63  | 416x416x3 | 6.74 | 5.64 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/nanodet/nanodet_repvgg/pretrained/2022-02-07/nanodet.zip) | [**link**](https://github.com/RangiLyu/nanodet) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/nanodet_repvgg.hef) | 
| ssd_mobiledet_dsp | 28.9  | 28.17  | 320x320x3 | 7.07 | 2.83 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/ssd/ssd_mobiledet_dsp/pretrained/2021-07-11/ssd_mobiledet_dsp.zip) | [**link**](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/ssd_mobiledet_dsp.hef) |   
| ssd_mobilenet_v1<html>&#128640;</html><html>&#11088;</html> | 23.17  | 22.29  | 300x300x3 | 6.79 | 1.25 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/ssd/ssd_mobilenet_v1/pretrained/2021-07-11/ssd_mobilenet_v1.zip) | [**link**](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/ssd_mobilenet_v1.hef) | 
| ssd_mobilenet_v1_hd | 17.66  | 16.41  | 720x1280x3 | 6.81 | 12.26 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/ssd/ssd_mobilenet_v1_hd/pretrained/2021-07-11/ssd_mobilenet_v1_hd.zip) | [**link**](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/ssd_mobilenet_v1_hd.hef) |   
| ssd_mobilenet_v2<html>&#128640;</html> | 24.15  | 23.28  | 300x300x3 | 4.46 | 0.76 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/ssd/ssd_mobilenet_v2/pretrained/2021-07-11/ssd_mobilenet_v2.zip) | [**link**](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/ssd_mobilenet_v2.hef) | 
| ssd_resnet_v1_18 | 17.664 | 17.41  | 300x300x3 | 16.62 | 3.54 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/ssd/ssd_resnet_v1_18/pretrained/2021-07-11/ssd_resnet_v1_18.zip) | [**link**](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/ssd_resnet_v1_18.hef) |   
| tiny_yolov3<html>&#128640;</html> | 14.36  | 13.45  | 416x416x3 | 8.85 | 2.79 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/tiny_yolov3/pretrained/2021-07-11/tiny_yolov3.zip) | [**link**](https://github.com/Tianxiaomo/pytorch-YOLOv4) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/tiny_yolov3.hef) | 
| tiny_yolov4 | 18.98  | 18.32  | 416x416x3 | 6.05 | 3.46 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/tiny_yolov4/pretrained/2021-07-11/tiny_yolov4.zip) | [**link**](https://github.com/Tianxiaomo/pytorch-YOLOv4) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/tiny_yolov4.hef) |  
| yolov3 | 38.42  | 37.83  | 608x608x3 | 68.79 | 79.17 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov3/pretrained/2021-08-16/yolov3.zip) | [**link**](https://github.com/AlexeyAB/darknet) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/yolov3.hef) | 
| yolov3_416 | 37.73  | 36.39  | 416x416x3 | 61.92 | 32.97 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov3_416/pretrained/2021-08-16/yolov3_416.zip) | [**link**](https://github.com/AlexeyAB/darknet) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/yolov3_416.hef) |   
| yolov3_gluon<html>&#128640;</html><html>&#11088;</html> | 37.28  | 36.43  | 608x608x3 | 68.79 | 79.17 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov3_gluon/pretrained/2021-07-11/yolov3_gluon.zip) | [**link**](https://cv.gluon.ai/model_zoo/detection.html) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/yolov3_gluon.hef) |    
| yolov3_gluon_416<html>&#11088;</html> | 36.274 | 35.27  | 416x416x3 | 61.92 | 32.97 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov3_gluon_416/pretrained/2021-07-11/yolov3_gluon_416.zip) | [**link**](https://cv.gluon.ai/model_zoo/detection.html) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/yolov3_gluon_416.hef) |    
| yolov4_leaky<html>&#128640;</html><html>&#11088;</html> | 42.37 | 41.55  | 512x512x3 | 64.33 | 45.60 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov4/pretrained/2022-03-17/yolov4.zip) | [**link**](https://github.com/AlexeyAB/darknet/wiki/YOLOv4-model-zoo) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/yolov4_leaky.hef) | 
| yolov5l | 46.01  | 43.8  | 640x640x3 | 48.54 | 60.78 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5l_spp/pretrained/2022-02-03/yolov5l.zip) | [**link**](https://github.com/ultralytics/yolov5/releases/tag/v2.0) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/yolov5l.hef) | 
| yolov5m | 42.59  | 40.93  | 640x640x3 | 21.78 | 26.14 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5m_spp/pretrained/2022-01-02/yolov5m.zip) | [**link**](https://github.com/ultralytics/yolov5/releases/tag/v2.0) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/yolov5m.hef) |   
| yolov5m_wo_spp<html>&#128640;</html><html>&#11088;</html> | 42.46  | 40.66  | 640x640x3 | 22.67 | 26.49 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5m/pretrained/2022-04-19/yolov5m_wo_spp.zip) | [**link**](https://github.com/ultralytics/yolov5/releases/tag/v2.0) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/yolov5m_wo_spp_60p.hef) | 
| yolov5s | 35.33  | 33.78  | 640x640x3 | 7.46 | 8.72 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5s_spp/pretrained/2022-01-02/yolov5s.zip) | [**link**](https://github.com/ultralytics/yolov5/releases/tag/v2.0) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/yolov5s.hef) | 
| yolov5s_wo_spp | 34.47  | 33.26  | 640x640x3 | 7.85 | 8.87 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5s/pretrained/2021-07-11/yolov5s.zip) | [**link**](https://github.com/ultralytics/yolov5/releases/tag/v2.0) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/yolov5s_wo_spp.hef) | 
| yolov5xs_wo_spp | 32.78  | 31.43  | 512x512x3 | 7.85 | 5.68 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5xs/pretrained/2021-07-11/yolov5xs.zip) | [**link**](https://github.com/ultralytics/yolov5/releases/tag/v2.0) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/yolov5xs_wo_spp.hef) | 
| yolov5xs_wo_spp_nms | 32.57  | 30.87  | 512x512x3 | 7.85 | 5.68 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolov5xs/pretrained/2022-05-10/yolov5xs_wo_spp_nms.zip) | [**link**](https://github.com/ultralytics/yolov5/releases/tag/v2.0) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/yolov5xs_wo_spp_nms.hef) | 
| yolox_l_leaky<html>&#11088;</html> | 48.68  | 47.08  | 640x640x3 | 54.17 | 77.74 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolox_l_leaky/pretrained/2021-09-23/yolox_l_leaky.zip) | [**link**](https://github.com/Megvii-BaseDetection/YOLOX) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/yolox_l_leaky.hef) | 
| yolox_s_leaky | 38.13  | 37.47  | 640x640x3 | 8.96 | 13.37 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolox_s_leaky/pretrained/2021-09-12/yolox_s_leaky.zip) | [**link**](https://github.com/Megvii-BaseDetection/YOLOX) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/yolox_s_leaky.hef) | 
| yolox_s_wide_leaky | 42.40  | 41.01  | 640x640x3 | 20.12 | 29.73 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolox_s_wide_leaky/pretrained/2021-09-12/yolox_s_wide_leaky.zip) | [**link**](https://github.com/Megvii-BaseDetection/YOLOX) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/yolox_s_wide_leaky.hef) | 
| yolox_tiny_leaky | 30.27  | 29.64  | 416x416x3 | 5.05 | 3.22 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolox_tiny_leaky/pretrained/2021-08-12/yolox_tiny_leaky.zip) | [**link**](https://github.com/Megvii-BaseDetection/YOLOX) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/yolox_tiny_leaky.hef) |

### VisDrone
    
| Network Name | mAP | Quantized | Input Resolution (HxWxC) | Params (M) | FLOPs (G) | Pretrained | Source | Compiled |
| -------------- | ------------------------ | ------------------------ | ---------- | --------- | ---- | -------- | ------- | -------- |  
| ssd_mobilenet_v1_visdrone<html>&#11088;</html> | 2.18 | 2.16  | 300x300x3 | 5.64 | 1.15 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-Visdrone/ssd/ssd_mobilenet_v1_visdrone/pretrained/2021-07-11/ssd_mobilenet_v1_visdrone.zip) | [**link**](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/ssd_mobilenet_v1_visdrone.hef) |
<br>
 
## Semantic Segmentation

### Cityscapes
                
| Network Name | mIoU  | Quantized | Input Resolution (HxWxC) | Params (M) | FLOPs (G) | Pretrained | Source | Compiled |
| -------------- |-------| ------------------------ | ---------- | --------- | ---- | -------- | ------- | -------- | 
| fcn16_resnet_v1_18<html>&#11088;</html> | 66.83 | 66.57  | 1024x1920x3 | 11.19 | 71.26 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Cityscapes/fcn16_resnet_v1_18/pretrained/2022-02-07/fcn16_resnet_v1_18.zip) | [**link**](https://mmsegmentation.readthedocs.io/en/latest) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/fcn16_resnet_v1_18.hef) |   
| fcn8_resnet_v1_18 | 68.75 | 68.51  | 1024x1920x3 | 11.20 | 71.51 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Cityscapes/fcn8_resnet_v1_18/pretrained/2022-02-09/fcn8_resnet_v1_18.zip) | [**link**](https://mmsegmentation.readthedocs.io/en/latest) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/fcn8_resnet_v1_18.hef) | 
| fcn8_resnet_v1_22 | 67.55 | 67.39  | 1920x1024x3 | 15.12 | 150.04 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Cityscapes/fcn8_resnet_v1_22/pretrained/2021-07-11/fcn8_resnet_v1_22.zip) | [**link**](https://cv.gluon.ai/model_zoo/segmentation.html) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/fcn8_resnet_v1_22.hef) | 
| stdc1 | 74.57 | 73.49  | 1024x1920x3 | 8.27 | 63.34 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Cityscapes/stdc1/pretrained/2022-03-17/stdc1.zip) | [**link**](https://mmsegmentation.readthedocs.io/en/latest) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/stdc1.hef) |

### Oxford-IIIT Pet
    
| Network Name | mIoU  | Quantized | Input Resolution (HxWxC) | Params (M) | FLOPs (G) | Pretrained | Source | Compiled |
| -------------- |-------| ------------------------ | ---------- | --------- | ---- | -------- | ------- | -------- |   
| unet_mobilenet_v2<html>&#128640;</html> | 77.32 | 76.42  | 256x256x3 | 10.08 | 14.44 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Oxford_Pet/unet_mobilenet_v2/pretrained/2022-02-03/unet_mobilenet_v2.zip) | [**link**](https://www.tensorflow.org/tutorials/images/segmentation) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/unet_mobilenet_v2.hef) |

### Pascal VOC
        
| Network Name | mIoU   | Quantized | Input Resolution (HxWxC) | Params (M) | FLOPs (G) | Pretrained | Source | Compiled |
| -------------- |--------| ------------------------ | ---------- | --------- | ---- | -------- | ------- | -------- |   
| deeplab_v3_mobilenet_v2<html>&#128640;</html> | 76.05  | 74.8  | 513x513x3 | 2.10 | 8.91 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Pascal/deeplab_v3_mobilenet_v2_dilation/pretrained/2021-09-26/deeplab_v3_mobilenet_v2_dilation.zip) | [**link**](https://github.com/bonlime/keras-deeplab-v3-plus) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/deeplab_v3_mobilenet_v2.hef) | 
| deeplab_v3_mobilenet_v2_wo_dilation | 71.461 | 70.25  | 513x513x3 | 2.10 | 1.64 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Pascal/deeplab_v3_mobilenet_v2/pretrained/2021-08-12/deeplab_v3_mobilenet_v2.zip) | [**link**](https://github.com/tensorflow/models/tree/master/research/deeplab) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/deeplab_v3_mobilenet_v2_wo_dilation.hef) |
<br>
 
## Pose Estimation

### COCO
            
| Network Name | AP     | Quantized | Input Resolution (HxWxC) | Params (M) | FLOPs (G) | Pretrained | Source | Compiled |
| -------------- |--------| ------------------------ | ---------- | --------- | ---- | -------- | ------- | -------- |  
| centerpose_regnetx_1.6gf_fpn | 53.542 | 46.94  | 640x640x3 | 14.28 | 32.38 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/PoseEstimation/centerpose_regnetx_1.6gf_fpn/pretrained/2022-03-23/centerpose_regnetx_1.6gf_fpn.zip) | [**link**](https://github.com/tensorboy/centerpose) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/centerpose_regnetx_1.6gf_fpn.hef) | 
| centerpose_regnetx_800mf | 44.07  | 42.47  | 512x512x3 | 12.31 | 43.06 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/PoseEstimation/centerpose_regnetx_800mf/pretrained/2021-07-11/centerpose_regnetx_800mf.zip) | [**link**](https://github.com/tensorboy/centerpose) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/centerpose_regnetx_800mf.hef) | 
| centerpose_repvgg_a0<html>&#11088;</html> | 39.17  | 36.89  | 416x416x3 | 11.71 | 14.15 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/PoseEstimation/centerpose_repvgg_a0/pretrained/2021-09-26/centerpose_repvgg_a0.zip) | [**link**](https://github.com/tensorboy/centerpose) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/centerpose_repvgg_a0.hef) |
<br>
 
## Face Detection

### WiderFace
        
| Network Name | mAP   | Quantized | Input Resolution (HxWxC) | Params (M) | FLOPs (G) | Pretrained | Source | Compiled |
| -------------- |-------| ------------------------ | ---------- | --------- | ---- | -------- | ------- | -------- | 
| lightface_slim<html>&#11088;</html> | 39.71 | 39.25  | 240x320x3 | 0.26 | 0.08 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/lightface_slim/2021-07-18/lightface_slim.zip) | [**link**](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/lightface_slim.hef) | 
| retinaface_mobilenet_v1<html>&#11088;</html> | 81.27 | 81.17  | 736x1280x3 | 3.49 | 12.57 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/retinaface_mobilenet_v1_hd/2021-07-18/retinaface_mobilenet_v1_hd.zip) | [**link**](https://github.com/biubug6/Pytorch_Retinaface) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/retinaface_mobilenet_v1.hef) |
<br>
 
## Instance Segmentation

### COCO
        
| Network Name | mAP | Quantized | Input Resolution (HxWxC) | Params (M) | FLOPs (G) | Pretrained | Source | Compiled |
| -------------- | ----------------------- | ------------------------ | ---------- | --------- | ---- | -------- | ------- | -------- | 
| yolact_mobilenet_v1 | 17.78 | 17.15  | 512x512x3 | 19.11 | 51.92 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolact_mobilenet_v1/pretrained/2021-01-12/yolact_mobilenet_v1.zip) | [**link**](https://github.com/dbolya/yolact) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/yolact_mobilenet_v1.hef) | 
| yolact_regnetx_800mf_20classes<html>&#11088;</html> | 22.86 | 22.56  | 512x512x3 | 21.97 | 51.47 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/coco/yolact_regnetx_800mf/pretrained/2022-02-08/yolact_regnetx_800mf.zip) | [**link**](https://github.com/dbolya/yolact) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/yolact_regnetx_800mf_20classes.hef) |

### D2S
    
| Network Name | mAP | Quantized | Input Resolution (HxWxC) | Params (M) | FLOPs (G) | Pretrained | Source | Compiled |
| -------------- | ------------------------ | ------------------------ | ---------- | --------- | ---- | -------- | ------- | -------- | 
| yolact_regnetx_600mf_d2s_31classes | 61.68 | 62.98  | 512x512x3 | 22.14 | 51.62 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/InstanceSegmentation/d2s/yolact_regnetx_600mf/pretrained/2020-12-10/yolact_regnetx_600mf.zip) | [**link**](https://github.com/dbolya/yolact) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/yolact_regnetx_600mf_d2s_31classes.hef) |
<br>
 
## Depth Estimation

### NYU
    
| Network Name | RMSE | Quantized | Input Resolution (HxWxC) | Params (M) | FLOPs (G) | Pretrained | Source | Compiled |
| -------------- | ------------------------ | ------------------------ | ---------- | --------- | ---- | -------- | ------- | -------- |   
| fast_depth<html>&#11088;</html> | 0.60 | 0.61  | 224x224x3 | 1.35 | 0.37 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/DepthEstimation/indoor/fast_depth/pretrained/2021-10-18/fast_depth.zip) | [**link**](https://github.com/dwofk/fast-depth) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/fast_depth.hef) |
<br>
 
## Facial Landmark

### AFLW2k3d
    
| Network Name | NME | Quantized | Input Resolution (HxWxC) | Params (M) | FLOPs (G) | Pretrained | Source | Compiled |
| -------------- | ------------------------- | ------------------------ | ---------- | --------- | ---- | -------- | ------- | -------- |   
| tddfa_mobilenet_v1<html>&#11088;</html> | 3.68 | 4.06  | 120x120x3 | 3.26 | 0.18 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceLandmarks3d/tddfa/tddfa_mobilenet_v1/pretrained/2021-11-28/tddfa_mobilenet_v1.zip) | [**link**](https://github.com/cleardusk/3DDFA_V2) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/tddfa_mobilenet_v1.hef) |
<br>
 
## Person Re-ID

 ### market1501
        
| Network Name | rank1 | Quantized | Input Resolution (HxWxC) | Params (M) | FLOPs (G) | Pretrained | Source | Compiled |
| -------------- | ----------------------- | ------------------------ | ---------- | --------- | ---- | -------- | ------- | -------- | 
| osnet_x1_0 | 94.43 | 92.43  | 256x128x3 | 2.19 | 0.99 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/PersonReID/osnet_x1_0/2022-05-19/osnet_x1_0.zip) | [**link**](https://github.com/KaiyangZhou/deep-person-reid) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/osnet_x1_0.hef) | 
| repvgg_a0_person_reid_2048 | 90.02 | 89.32  | 256x128x3 | 9.65 | 0.89 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/MCPReID/reid/repvgg_a0_person_reid_2048/2022-04-18/repvgg_a0_person_reid_2048.zip) | [**link**](https://github.com/DingXiaoH/RepVGG) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/repvgg_a0_person_reid_2048.hef) | 
| repvgg_a0_person_reid_512 | 89.9 | 89.4  | 256x128x3 | 7.68 | 0.89 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/MCPReID/reid/repvgg_a0_person_reid_512/2022-04-18/repvgg_a0_person_reid_512.zip) | [**link**](https://github.com/DingXiaoH/RepVGG) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/repvgg_a0_person_reid_512.hef) |
<br>
 
## Person Attribute

 ### peta
        
| Network Name | Mean Accuracy | Quantized | Input Resolution (HxWxC) | Params (M) | FLOPs (G) | Pretrained | Source | Compiled |
| -------------- | ----------------------- | ------------------------ | ---------- | --------- | ---- | -------- | ------- | -------- | 
| person_attr_resnet_v1_18 | 82.504 | 82.56  | 224x224x3 | 11.19 | 1.82 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/person_attr_resnet_v1_18/pretrained/2022-06-11/person_attr_resnet_v1_18.zip) | [**link**](https://github.com/dangweili/pedestrian-attribute-recognition-pytorch) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/person_attr_resnet_v1_18.hef) |
<br>
 
## Hand Landmark detection
    
| Network Name | Input Resolution (HxWxC) | Params (M) | FLOPs (G) | Pretrained | Source | Compiled |
| -------------- | ---------- | --------- | ---- | -------- | ------- | -------- | 
| hand_landmark_lite | 224x224x3 | 1.01 | 0.15 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HandLandmark/hand_landmark_lite/2022-01-23/hand_landmark_lite.zip) | [**link**](https://github.com/google/mediapipe) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/hand_landmark_lite.hef) |
<br>
 
## Palm Detection
    
| Network Name | Input Resolution (HxWxC) | Params (M) | FLOPs (G) | Pretrained | Source | Compiled |
| -------------- | ---------- | --------- | ---- | -------- | ------- | -------- | 
| palm_detection_lite | 192x192x3 | 1.01 | 0.31 | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-Palm/palm_detection_lite/pretrained/palm_detection_lite.zip) | [**link**](https://github.com/google/mediapipe) | [**link**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.2.0/palm_detection_lite.hef) |
<br>

