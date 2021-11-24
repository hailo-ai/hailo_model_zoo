# Datasets

The Hailo Model Zoo works with TFRecord files which store the images and labels of the dataset for evaluation and calibration. The instructions on how to create the TFRecord files are given below. Datasets are stored on the current directory, to change the default location you can use:
```
export HMZ_DATA=/new/path/for/storage/
```
- [ImageNet](#imagenet)
- [COCO2017](#coco2017)
- [Cityscapes](#cityscapes)
- [WIDERFACE](#widerface)
- [VisDrone](#visdrone)
- [Pascal VOC augmented dataset](#pascal-voc-augmented-dataset)
- [D2S augmented dataset](#d2s-augmented-dataset)
- [NYU Depth V2](#nyu-depth-v2)

<br>

## ImageNet
To evaluate/optimize/compile the classification models of the Hailo Model Zoo you should generate the ImageNet TFRecord files (manual download is required).

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

<br>

## COCO2017
To evaluate/optimize/compile the object detection / pose estimation models of the Hailo Model Zoo you should generate the COCO ([**link**](https://cocodataset.org/#home)) TFRecord files. Run the create TFRecord scripts to download the dataset and generate the TFRecord files:

```
python hailo_model_zoo/datasets/create_coco_tfrecord.py val2017
python hailo_model_zoo/datasets/create_coco_tfrecord.py calib2017
```
### Manual Download (Optional)
1. Download COCO ([**here**](https://cocodataset.org/#home)). The expected dataset structure:

Annotations:
```
annotations
├── captions_train2017.json
├── captions_val2017.json
├── instances_train2017.json
├── instances_val2017.json
├── person_keypoints_train2017.json
└── person_keypoints_val2017.json
```
Validation set:
```
val2017
├── 000000000139.jpg
├── 000000000285.jpg
├── 000000000632.jpg
├── 000000000724.jpg
├── 000000000776.jpg
├── 000000000785.jpg
├── 000000000802.jpg
├── 000000000872.jpg
├── 000000000885.jpg
├── ...
```
Training set:
```
train2017
├── 000000000009.jpg
├── 000000000025.jpg
├── 000000000030.jpg
├── 000000000034.jpg
├── 000000000036.jpg
├── 000000000042.jpg
├── 000000000049.jpg
├── 000000000061.jpg
├── 000000000064.jpg
├── ...
```

2. Run the creation scripts:
```
python hailo_model_zoo/datasets/create_coco_tfrecord.py val2017 --img /path/to/val2017 --det /path/to/annotations
python hailo_model_zoo/datasets/create_coco_tfrecord.py calib2017 --img /path/to/train2017 --det /path/to/annotations
```

<br>

## Cityscapes
To evaluate/optimize/compile the semantic segmentation models of the Hailo Model Zoo you should generate the Cityscapes TFRecord files (manual download is required).

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

<br>

## WIDERFACE
To evaluate/optimize/compile the face detection models of the Hailo Model Zoo you should generate the WIDERFACE ([**link**](http://shuoyang1213.me/WIDERFACE/)) TFRecord files. Run the create TFRecord scripts to download the dataset and generate the TFRecord files:

```
python hailo_model_zoo/datasets/create_widerface_tfrecord.py calib
python hailo_model_zoo/datasets/create_widerface_tfrecord.py val
```

### Manual Download (Optional)
1. Download the following from [**here**](http://shuoyang1213.me/WIDERFACE/):
    - WIDER Face Training Images
    - WIDER Face Validation Images
    - Face annotations
2. Download the following from [**here**](https://github.com/biubug6/Pytorch_Retinaface/tree/master/widerface_evaluate/ground_truth)
    - [**wider_hard_val.mat**](https://github.com/biubug6/Pytorch_Retinaface/raw/master/widerface_evaluate/ground_truth/wider_hard_val.mat)

Expected directory structure:
```
widerface/
├── wider_face_split
|   ├── readme.txt
|   ├── wider_face_test_filelist.txt
|   ├── wider_face_test.mat
|   ├── wider_face_train_bbx_gt.txt
|   ├── wider_face_train.mat
|   ├── wider_face_val_bbx_gt.txt
|   ├── wider_face_val.mat
|   └── wider_hard_val.mat
├── WIDER_train
│   └── images
│       ├── 0--Parade
│       ├── 10--People_Marching
│       ├── 11--Meeting
│       ├── ...
└── WIDER_val
    └── images
        ├── 0--Parade
        ├── 10--People_Marching
        ├── 11--Meeting
        ├── ...
```

3. Run the creation scripts
```
python hailo_model_zoo/datasets/create_widerface_tfrecord.py calib --img /path/to/widerface --gt_mat_path /path/to/wider_face_split --hard_mat_path /path/to/wider_face_split 
python hailo_model_zoo/datasets/create_widerface_tfrecord.py val --img /path/to/widerface --gt_mat_path /path/to/wider_face_split --hard_mat_path /path/to/wider_face_split
```

<br>

## VisDrone
To evaluate/optimize/compile the visdrone object detection models of the Hailo Model Zoo you should generate the VisDrone ([**link**](http://aiskyeye.com/download/object-detection-2/)) TFRecord files. Run the create TFRecord scripts to download the dataset and generate the TFRecord files:

```
python hailo_model_zoo/datasets/create_visdrone_tfrecord.py train
python hailo_model_zoo/datasets/create_visdrone_tfrecord.py val
```

### Manual Download (Optional)
1. Download VisDrone ([**here**](http://aiskyeye.com/download/object-detection-2/)). The expected dataset structure:

Training set:
```
VisDrone2019-DET-train/
├── annotations
|   ├── 0000002_00005_d_0000014.txt
│   ├── 0000002_00448_d_0000015.txt
|   ├── ...
└── images
    ├── 0000002_00005_d_0000014.jpg
    ├── 0000002_00448_d_0000015.jpg
    ├── ...
```

Validation set:
```
VisDrone2019-DET-val/
├── annotations
│   ├── 0000001_02999_d_0000005.txt
│   ├── 0000001_03499_d_0000006.txt
|   ├── ...
└── images
    ├── 0000001_02999_d_0000005.jpg
    ├── 0000001_03499_d_0000006.jpg
    ├── ...
```

2. Run the creation scripts:
```
python hailo_model_zoo/datasets/create_visdrone_tfrecord.py train -d /path/to/VisDrone2019-DET-train
python hailo_model_zoo/datasets/create_visdrone_tfrecord.py val -d /path/to/VisDrone2019-DET-val
```

<br>

## Pascal VOC augmented dataset
Run the creation scripts:
```
python hailo_model_zoo/datasets/create_pascal_tfrecord.py calib
python hailo_model_zoo/datasets/create_pascal_tfrecord.py val
```

### Manual Download (Optional)
1. Download the dataset from [**here**](http://home.bharathh.info/pubs/codes/SBD/download.html). Expected dataset structure:
```
benchmark_RELEASE
└── dataset
    ├── cls
    ├── img
    ├── inst
    ├── train.txt
    └── val.txt
```
2. run the creation scripts:
```
python hailo_model_zoo/datasets/create_pascal_tfrecord.py calib --root benchmark_RELEASE/dataset
python hailo_model_zoo/datasets/create_pascal_tfrecord.py val --root benchmark_RELEASE/dataset
```

<br>

## D2S augmented dataset
Run the creation scripts:
```
python hailo_model_zoo/datasets/create_d2s_tfrecord.py calib
python hailo_model_zoo/datasets/create_d2s_tfrecord.py val
```

### Manual Download (Optional)
1. Download the dataset from [**here**](https://www.mydrive.ch/shares/39000/993e79a47832a8ea7208a14d8b277c35/download/420938639-1629953496/d2s_images_v1.tar.xz).
Extract using 'tar -xf d2s_images_v1.1.tar.xz'. Expected dataset structure:
```
└── images
    ├── D2S_000200.jpg
    ├── D2S_000201.jpg
    ├── ...
```
2. Download the annotations from [**here**](https://www.mydrive.ch/shares/39000/993e79a47832a8ea7208a14d8b277c35/download/420938386-1629953481/d2s_annotations_v1.1.tar.xz).
Extract using 'tar -xf d2s_annotations_v1.1.tar.xz'. Expected annotations structure:
```
└── annotations
    ├── D2S_augmented.json
    ├── D2S_validation.json
    ├── ...
```
3. run the creation scripts:
```
python hailo_model_zoo/datasets/create_d2s_tfrecord.py calib --img /path/to/dataset --det /path/to/annotations/D2S_augmented.json
python hailo_model_zoo/datasets/create_d2s_tfrecord.py val --img /path/to/dataset --det /path/to/annotations/D2S_validation.json
```

<br>

## NYU Depth V2
Run the creation scripts:
```
python hailo_model_zoo/datasets/create_nyu_depth_v2_tfrecord.py calib
python hailo_model_zoo/datasets/create_nyu_depth_v2_tfrecord.py val
```

### Manual Download (Optional)
1. Download the dataset from [**here**](http://datasets.lids.mit.edu/fastdepth/data/nyudepthv2.tar.gz).
Extract using 'tar -xf nyudepthv2.tar.gz'. Expected dataset structure:
```
└── train
    └── study_0300
        |── 00626.h5
        ├── 00631.h5
        ├── ...
    ├── ...
└── val
    └── official
        |── 00001.h5
        |── 00002.h5
        |── 00009.h5
        ├── 00014.h5
        ├── ...
```
2. run the creation scripts:
```
python hailo_model_zoo/datasets/create_nyu_depth_v2_tfrecord.py calib --data ./nyu_depth_v2/
python hailo_model_zoo/datasets/create_nyu_depth_v2_tfrecord.py val --data ./nyu_depth_v2/
```