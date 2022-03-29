# Licesen Plate Recognition
<p align="center">
  <img src="src/img.jpg" />
</p>

<br>

  Hailo's license plate recognition network (*lprnet*) was trained in-house on a synthetic auto-generated dataset to predict registration numbers of license plates under various weather and lighting conditions.


  ## Model Details

  ### Architecture
  * A convolutional network based on [**LPRNet**](https://github.com/sirius-ai/LPRNet_Pytorch), with several modifications:
    - A ResNet like backbone with 4 stages, each containing 2 residual blocks
    - Several kernel shape changes
    - Maximal license plate length of 19 digits
    - More details can be found [**here**](https://github.com/hailo-ai/LPRNet_Pytorch)
  * Number of parameters: 7.14M
  * GMACS: 18.29
  * Accuracy<sup>*</sup>: 53.82%
<br>\* Evaluated on internal dataset containing 1178 images

  ### Inputs
  - RGB liscense plate image with size of 75x300x3
  - Image normalization occurs on-chip

  ### Outputs
  - A tensor with size 5x19x11
    - Post-processing outputs a tensor with size of 1x19x11
    - The 11 channels contain logits scores for 11 classes (10 digits + *blank* class)
  - A Connectionist temporal classification (CTC) greedy decoding outputs the final license plate number prediction

<br>

---
<br>

### Download
The pre-compiled network can be downloaded from [**here**](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/LPR/ocr/lprnet/2022-03-09/lprnet.hef).
<br><br>
Use the following command to measure model performance on hailo’s HW:
```
hailortcli benchmark lprnet.hef
```
<br>

---
<br>

## Training on Custom Dataset
A guide for training the pre-trained model on a custom dataset can be found [**here**](./docs/TRAINING_GUIDE.md)
 - Hailo's LPRNet was trained on a synthetic auto-generated dataset containing 4 million license plate images. Auto-generation of synthetic data for training is cheap, allows one to obtain a large annotated dataset easily and can be adapted quickly for other domains
 -   A notebook for auto-generation of synthetic training data for LPRNet can be found [**here**](./src/lp_autogenerate.ipynb) 
 - For more details on the training data autogeneration, please see the training guide
