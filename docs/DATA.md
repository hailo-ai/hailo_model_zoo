# Datasets

The Hailo Model Zoo works with TFRecord files which store the images and labels of the dataset for evaluation and calibration. The instructions on how to create the TFRecord files are given below. Datasets are stored on the current directory, to change the default location you can use:
```
export HMZ_DATA=/new/path/for/storage/
```

## ImageNet
To evaluate/quantize/compile the classification models of the Hailo Model Zoo you should generate the ImageNet TFRecord files (manual download is required).

1. Download the ImageNet dataset from [**here**](https://www.kaggle.com/c/imagenet-object-localization-challenge/data). The expected dataset structure:

```
imagenet
|_ train
|  |_ n01440764
|  |_ ...
|  |_ n15075141
|_ val
|  |_ n01440764
|  |_ ...
|  |_ n15075141
|_ ...
```

* To avoid downloading the ImageNet training data, you may consider using the validation dataset for calibration (does not apply for finetune).

2. Run the create TFRecord scripts:
```
python hailo_model_zoo/datasets/create_imagenet_tfrecord.py val --img /path/to/imagenet/val/
python hailo_model_zoo/datasets/create_imagenet_tfrecord.py calib --img /path/to/imagenet/train/
```

## COCO2017
To evaluate/quantize/compile the object detection / pose estimation models of the Hailo Model Zoo you should generate the COCO ([**link**](https://cocodataset.org/#home)) TFRecod files. Run the create TFRecord scripts to download the dataset and generate the TFRecord files:

```
python hailo_model_zoo/datasets/create_coco_tfrecord.py val2017
python hailo_model_zoo/datasets/create_coco_tfrecord.py calib2017
```

## Cityscapes
To evaluate/quantize/compile the semantic segmentation models of the Hailo Model Zoo you should generate the Cityscapes TFRecod files (manual download is required).

1. Download the Cityscapes dataset from [**here**](https://www.cityscapes-dataset.com/). The expected dataset structure:

```
Cityscapes
|_ gtFine
|  |_ train
|  |_ test
|  |_ val
|_ leftImg8bit
|  |_ train
|  |_ test
|  |_ val
|  |_ train_extra
|_ ...
```

2. Run the create TFRecord scripts:
```
python hailo_model_zoo/datasets/create_cityscapes_tfrecord.py val --data /path/to/Cityscapes/
python hailo_model_zoo/datasets/create_cityscapes_tfrecord.py calib --data /path/to/Cityscapes/
```
