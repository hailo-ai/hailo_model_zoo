# License Plate Detection

<p align="center">
  <img src="src/img.jpg" />
</p>

<br>

  Hailo's license detection network (*tiny_yolov4_license_plates*) is based on Tiny-YOLOv4 and was trained in-house using Darknet with a single class. It expects a single vehicle and can work under various weather and lighting conditions, on different vehicle types and numerous camera angles.

  
  ## Model Details
  
  ### Architecture
  * Tiny-YOLOv4 
  * Number of parameters: 5.87M
  * GMACS: 3.4
  * Accuracy<sup>*</sup>: 73.45 mAP
<br>\* Evaluated on internal dataset containing 5000 images

  ### Inputs
  * RGB image with size of 416x416x3
  * Image normalization occurs on-chip

  ### Outputs
  - Two output tensors with sizes of 13x13x18 and 26x26x18.
  - Each output contains 3 anchors that hold the following information:
    - Bounding box coordinates ((x,y) centers, height, width)
    - Box objectness confidence score
    - Class probablity confidence score
  - The above 6 values per anchor are concatenated into the 18 output channels

<br>

---
<br>

### Download
The pre-compiled network can be downloaded from [**here**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/LPR/lp_detector/tiny_yolov4_license_plates/2021-12-23/tiny_yolov4_license_plates.hef).
<br><br>
Use the following command to measure model performance on hailo’s HW:
```
hailortcli benchmark tiny_yolov4_license_plates.hef
```
<br>

---
<br>

## Training on Custom Dataset
A guide for finetuning the pre-trained model on a custom dataset can be found [**here**](./docs/TRAINING_GUIDE)