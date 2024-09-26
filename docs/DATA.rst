.. _Datasets:

Datasets
========

| The Hailo Model Zoo works with TFRecord files which store the images and labels of the dataset for evaluation and calibration.
| The instructions on how to create the TFRecord files are given below. By default, datasets are stored in the following path:

.. code-block::

   ~/.hailomz

We recommend to define the data directory path yourself, by setting the ``HMZ_DATA`` environment variable.

.. code-block::

   export HMZ_DATA=/new/path/for/storage/


* `Datasets`_

  * `ImageNet`_
  * `COCO2017`_
  * `Cityscapes`_
  * `WIDERFACE`_
  * `VisDrone`_
  * `Pascal VOC augmented dataset`_
  * `D2S augmented dataset`_
  * `NYU Depth V2`_
  * `AFLW2k3d and 300W-LP`_
  * `Hand Landmark`_
  * `Market1501`_
  * `PETA`_
  * `CelebA`_
  * `LFW`_
  * `BSD100`_
  * `CIFAR100`_
  * `LOL`_
  * `BSD68`_
  * `CBSD68`_
  * `KITTI_STEREO`_
  * `KINETICS400`_
  * `NUSCENES`_

.. _ImageNet:

ImageNet
--------

To evaluate/optimize/compile the classification models of the Hailo Model Zoo you should generate the ImageNet TFRecord files (manual download is required).


#. | Download the ImageNet dataset from `here <https://www.kaggle.com/c/imagenet-object-localization-challenge/data>`_. The expected dataset structure:

   .. code-block::

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


   | \* To avoid downloading the ImageNet training data, you may consider using the validation dataset for calibration (does not apply for finetune).


#. Run the create TFRecord scripts:

   .. code-block::

      python hailo_model_zoo/datasets/create_imagenet_tfrecord.py val --img /path/to/imagenet/val/
      python hailo_model_zoo/datasets/create_imagenet_tfrecord.py calib --img /path/to/imagenet/train/


.. _COCO2017:

COCO2017
--------

| To evaluate/optimize/compile the object detection / pose estimation models of the Hailo Model Zoo you should generate the COCO (\ `link <https://cocodataset.org/#home>`_\ ) TFRecord files.
| Run the create TFRecord scripts to download the dataset and generate the TFRecord files:

.. code-block::

   python hailo_model_zoo/datasets/create_coco_tfrecord.py val2017
   python hailo_model_zoo/datasets/create_coco_tfrecord.py calib2017

| To evaluate/optimize/compile the single person pose estimation models of the Hailo Model Zoo you should generate the single-person COCO TFRecord files.
| Run the create TFRecord scripts to download the dataset and generate the TFRecord files:

.. code-block::

   python hailo_model_zoo/datasets/create_coco_single_person_tfrecord.py val2017
   python hailo_model_zoo/datasets/create_coco_single_person_tfrecord.py calib2017


Manual Download (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^


#. Download COCO (\ `here <https://cocodataset.org/#home>`_\ ). The expected dataset structure:

   Annotations:

   .. code-block::

      annotations
      |_ captions_train2017.json
      |_ captions_val2017.json
      |_ instances_train2017.json
      |_ instances_val2017.json
      |_ person_keypoints_train2017.json
      |_ person_keypoints_val2017.json

   Validation set:

   .. code-block::

      val2017
      |_ 000000000139.jpg
      |_ 000000000285.jpg
      |_ 000000000632.jpg
      |_ 000000000724.jpg
      |_ 000000000776.jpg
      |_ 000000000785.jpg
      |_ 000000000802.jpg
      |_ 000000000872.jpg
      |_ 000000000885.jpg
      |_ ...

   Training set:

   .. code-block::

      train2017
      |_ 000000000009.jpg
      |_ 000000000025.jpg
      |_ 000000000030.jpg
      |_ 000000000034.jpg
      |_ 000000000036.jpg
      |_ 000000000042.jpg
      |_ 000000000049.jpg
      |_ 000000000061.jpg
      |_ 000000000064.jpg
      |_ ...

#. Run the creation scripts:

   .. code-block::

      python hailo_model_zoo/datasets/create_coco_tfrecord.py val2017 --img /path/to/val2017 --det /path/to/annotations
      python hailo_model_zoo/datasets/create_coco_tfrecord.py calib2017 --img /path/to/train2017 --det /path/to/annotations


.. _Cityscapes:

Cityscapes
----------

To evaluate/optimize/compile the semantic segmentation models of the Hailo Model Zoo you should generate the Cityscapes TFRecord files (manual download is required).


#. Download the Cityscapes dataset from `here <https://www.cityscapes-dataset.com/>`_. The expected dataset structure:

   .. code-block::

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


#. Run the create TFRecord scripts:

   .. code-block::

      python hailo_model_zoo/datasets/create_cityscapes_tfrecord.py val --data /path/to/Cityscapes/
      python hailo_model_zoo/datasets/create_cityscapes_tfrecord.py calib --data /path/to/Cityscapes/


.. _WIDERFACE:

WIDERFACE
---------

| To evaluate/optimize/compile the face detection models of the Hailo Model Zoo you should generate the WIDERFACE (\ `link <http://shuoyang1213.me/WIDERFACE/>`_\ ) TFRecord files.
| Run the create TFRecord scripts to download the dataset and generate the TFRecord files:

.. code-block::

   python hailo_model_zoo/datasets/create_widerface_tfrecord.py calib
   python hailo_model_zoo/datasets/create_widerface_tfrecord.py val


Manual Download (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^


#. Download the following from `here <http://shuoyang1213.me/WIDERFACE/>`_\ :

   * WIDER Face Training Images
   * WIDER Face Validation Images
   * Face annotations

#. Download the following from `here <https://github.com/biubug6/Pytorch_Retinaface/tree/master/widerface_evaluate/ground_truth>`_

   * `wider_hard_val.mat <https://github.com/biubug6/Pytorch_Retinaface/raw/master/widerface_evaluate/ground_truth/wider_hard_val.mat>`_

   Expected directory structure:

   .. code-block::

      widerface/
      |_ wider_face_split
      |  |_ readme.txt
      |  |_ wider_face_test_filelist.txt
      |  |_ wider_face_test.mat
      |  |_ wider_face_train_bbx_gt.txt
      |  |_ wider_face_train.mat
      |  |_ wider_face_val_bbx_gt.txt
      |  |_ wider_face_val.mat
      |  |_ wider_hard_val.mat
      |_ WIDER_train
      |  |_ images
      |     |_ 0--Parade
      |     |_ 10--People_Marching
      |     |_ 11--Meeting
      |     |_ ...
      |_ WIDER_val
         |_ images
            |_ 0--Parade
            |_ 10--People_Marching
            |_ 11--Meeting
            |_ ...


#. Run the creation scripts

   .. code-block::

      python hailo_model_zoo/datasets/create_widerface_tfrecord.py calib --img /path/to/widerface --gt_mat_path /path/to/wider_face_split --hard_mat_path /path/to/wider_face_split
      python hailo_model_zoo/datasets/create_widerface_tfrecord.py val --img /path/to/widerface --gt_mat_path /path/to/wider_face_split --hard_mat_path /path/to/wider_face_split


.. _VisDrone:

VisDrone
--------

| To evaluate/optimize/compile the visdrone object detection models of the Hailo Model Zoo you should generate the VisDrone (\ `link <http://aiskyeye.com/download/object-detection-2/>`_\ ) TFRecord files.
| Run the create TFRecord scripts to download the dataset and generate the TFRecord files:

.. code-block::

   python hailo_model_zoo/datasets/create_visdrone_tfrecord.py train
   python hailo_model_zoo/datasets/create_visdrone_tfrecord.py val

Manual Download (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^


#. Download VisDrone (\ `here <http://aiskyeye.com/download/object-detection-2/>`_\ ). The expected dataset structure:

   Training set:

   .. code-block::

      VisDrone2019-DET-train/
      |_ annotations
      |  |_ 0000002_00005_d_0000014.txt
      |  |_ 0000002_00448_d_0000015.txt
      |  |_ ...
      |_ images
         |_ 0000002_00005_d_0000014.jpg
         |_ 0000002_00448_d_0000015.jpg
         |_ ...


   Validation set:

   .. code-block::

      VisDrone2019-DET-val/
      |_ annotations
      |  |_ 0000001_02999_d_0000005.txt
      |  |_ 0000001_03499_d_0000006.txt
      |  |_ ...
      |_ images
         |_ 0000001_02999_d_0000005.jpg
         |_ 0000001_03499_d_0000006.jpg
         |_ ...

#. Run the creation scripts:

   .. code-block::

      python hailo_model_zoo/datasets/create_visdrone_tfrecord.py train -d /path/to/VisDrone2019-DET-train
      python hailo_model_zoo/datasets/create_visdrone_tfrecord.py val -d /path/to/VisDrone2019-DET-val


.. _Pascal VOC augmented dataset:

Pascal VOC augmented dataset
----------------------------

Run the creation scripts:

.. code-block::

   python hailo_model_zoo/datasets/create_pascal_tfrecord.py calib
   python hailo_model_zoo/datasets/create_pascal_tfrecord.py val


Manual Download (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^


#. Download the dataset from `here <http://home.bharathh.info/pubs/codes/SBD/download.html>`_. Expected dataset structure:

   .. code-block::

      benchmark_RELEASE
      |_ dataset
       |_ cls
       |_ img
       |_ inst
       |_ train.txt
       |_ val.txt

#. run the creation scripts:

   .. code-block::

      python hailo_model_zoo/datasets/create_pascal_tfrecord.py calib --root benchmark_RELEASE/dataset
      python hailo_model_zoo/datasets/create_pascal_tfrecord.py val --root benchmark_RELEASE/dataset


.. _D2S augmented dataset:

D2S augmented dataset
---------------------

Run the creation scripts:

.. code-block::

   python hailo_model_zoo/datasets/create_d2s_tfrecord.py calib
   python hailo_model_zoo/datasets/create_d2s_tfrecord.py val

Manual Download (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Download the dataset from `here <https://www.mydrive.ch/shares/39000/993e79a47832a8ea7208a14d8b277c35/download/420938639-1629953496/d2s_images_v1.tar.xz>`_.
   Extract using 'tar -xf d2s_images_v1.1.tar.xz'. Expected dataset structure:

   .. code-block::

      |_ images
       |_ D2S_000200.jpg
       |_ D2S_000201.jpg
       |_ ...

#. Download the annotations from `here <https://www.mydrive.ch/shares/39000/993e79a47832a8ea7208a14d8b277c35/download/420938386-1629953481/d2s_annotations_v1.1.tar.xz>`_.
   Extract using 'tar -xf d2s_annotations_v1.1.tar.xz'. Expected annotations structure:

   .. code-block::

      |_ annotations
       |_ D2S_augmented.json
       |_ D2S_validation.json
       |_ ...

#. run the creation scripts:

   .. code-block::

      python hailo_model_zoo/datasets/create_d2s_tfrecord.py calib --img /path/to/dataset --det /path/to/annotations/D2S_augmented.json
      python hailo_model_zoo/datasets/create_d2s_tfrecord.py val --img /path/to/dataset --det /path/to/annotations/D2S_validation.json


.. _NYU Depth V2:

NYU Depth V2
------------

Run the creation scripts:

.. code-block::

   python hailo_model_zoo/datasets/create_nyu_depth_v2_tfrecord.py calib
   python hailo_model_zoo/datasets/create_nyu_depth_v2_tfrecord.py val

Manual Download (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^


#. Download the dataset from `here <http://datasets.lids.mit.edu/fastdepth/data/nyudepthv2.tar.gz>`_.
   Extract using 'tar -xf nyudepthv2.tar.gz'. Expected dataset structure:

   .. code-block::

      |_ train
       |_ study_0300
           |_ 00626.h5
           |_ 00631.h5
           |_ ...
       |_ ...
      |_ val
       |_ official
           |_ 00001.h5
           |_ 00002.h5
           |_ 00009.h5
           |_ 00014.h5
           |_ ...

#. run the creation scripts:

   .. code-block::

      python hailo_model_zoo/datasets/create_nyu_depth_v2_tfrecord.py calib --data ./nyu_depth_v2/
      python hailo_model_zoo/datasets/create_nyu_depth_v2_tfrecord.py val --data ./nyu_depth_v2/

.. _AFLW2k3d and 300W-LP:

AFLW2k3d and 300W-LP
--------------------

Run the creation scripts:

.. code-block::

   python hailo_model_zoo/datasets/create_300w-lp_tddfa_tfrecord.py
   python hailo_model_zoo/datasets/create_aflw2k3d_tddfa_tfrecord.py

Manual Download (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^


#. Download the augmented_cropped 300W-LP dataset from `here <https://drive.google.com/uc?id=17LfvBZFAeXt0ACPnVckfdrLTMHUpIQqE&export=download>`_ and extract.
   Expected structure:

   .. code-block::

      train_aug_120x120
      |_ AFW_AFW_1051618982_1_0_10.jpg
      |_ AFW_AFW_1051618982_1_0_11.jpg
      |_ AFW_AFW_1051618982_1_0_12.jpg
      |_ AFW_AFW_1051618982_1_0_13.jpg
      |_ AFW_AFW_1051618982_1_0_1.jpg
      |_ AFW_AFW_1051618982_1_0_2.jpg
      |_ AFW_AFW_1051618982_1_0_3.jpg
      |_ AFW_AFW_1051618982_1_0_4.jpg
      |_ ...

#.
   Run

   .. code-block::

      python hailo_model_zoo/datasets/create_300w-lp_tddfa_tfrecord.py --dir /path/to/train_aug_120x120

#. Download the following files:

   * the official dataset from `here <http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip>`_
   * the cropped dataset from `here <https://drive.google.com/open?id=17LfvBZFAeXt0ACPnVckfdrLTMHUpIQqE>`_
   * The following files from `here <https://github.com/cleardusk/3DDFA/tree/master/test.configs>`_

     - AFLW2000-3D.pose.npy
     - AFLW2000-3D.pts68.npy
     - AFLW2000-3D-Reannotated.pts68.npy
     - AFLW2000-3D_crop.roi_box.npy

   The expected structure:

   .. code-block::

      aflw2k3d_tddfa
      |_ AFLW2000-3D_crop.roi_box.npy
      |_ AFLW2000-3D.pose.npy
      |_ AFLW2000-3D.pts68.npy
      |_ AFLW2000-3D-Reannotated.pts68.npy
      |_ test.data
         |_ AFLW2000
         |   |_ Code
         |   |   |_ Mex
         |   |   |_ ModelGeneration
         |   |_ image00002.jpg
         |   |_ image00002.mat
         |   |_ image00004.jpg
         |   |_ image00004.mat
         |   |_ ...
         |_ AFLW2000-3D_crop
         |   |_ image00002.jpg
         |   |_ image00004.jpg
         |   |_ image00006.jpg
         |   |_ image00008.jpg
         |   |_ ...
         |_ AFLW2000-3D_crop.list
         |_ AFLW_GT_crop
         |   |_ ...
         |_ AFLW_GT_crop.list

#. Run the following:

   .. code-block::

      python hailo_model_zoo/datasets/create_aflw2k3d_tddfa_tfrecord.py --dir /path/to/aflw2k3d_tddfa

.. _Hand Landmark:

Hand Landmark
-------------

Run the creation script:

.. code-block::

   python hailo_model_zoo/datasets/create_hand_landmark_tfrecord.py

Manual Download (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^


#. Download the dataset from `here <https://drive.google.com/u/0/uc?id=1KcMYcNJgtK1zZvfl_9sTqnyBUTri2aP2&export=download>`_ and extract.
   Expected structure:

   .. code-block::

      Hands               00  000
      |_ Hand_0011695.jpg
      |_ Hand_0011696.jpg
      |_ Hand_0011697.jpg
      |_ ...

#. Run

   .. code-block::

      python hailo_model_zoo/datasets/create_hand_landmark_tfrecord.py --img /path/to/Hands

.. _Market1501:

Market1501
----------

Run the creation scripts:

.. code-block::

   python hailo_model_zoo/datasets/create_market_tfrecord.py val
   python hailo_model_zoo/datasets/create_market_tfrecord.py calib

Manual Download (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^


#. | Download the dataset from `here <http://zheng-lab.cecs.anu.edu.au/Project/project_reid.html>`_ and extract.
   | Expected structure:

   .. code-block::

      Market-1501-v15.09.15
      |_ bounding_box_test
       |_ 0000_c1s1_000151_01.jpg
       |_ 0000_c1s1_000376_03.jpg
       |_ ...
      |_ bounding_box_train
       |_ 0002_c1s1_000451_03.jpg
       |_ 0002_c1s1_000551_01.jpg
       |_ ...
      |_ gt_bbox
       |_ 0001_c1s1_001051_00.jpg
       |_ 0001_c1s1_002301_00.jpg
       |_ ...
      |_ gt_query
       |_ 0001_c1s1_001051_00_good.mat
       |_ 0001_c1s1_001051_00_junk.mat
       |_ ...
      |_ query
       |_ 0001_c1s1_001051_00.jpg
       |_ 0001_c2s1_000301_00.jpg
       |_ ...

#. Run

   .. code-block::

      python hailo_model_zoo/datasets/create_market_tfrecord.py val --img path/to/Market-1501-v15.09.15/
      python hailo_model_zoo/datasets/create_market_tfrecord.py calib --img path/to/Market-1501-v15.09.15/bounding_box_train/

.. _PETA:

PETA
----
To evaluate/optimize/compile the person attribute models of the
Hailo Model Zoo you should generate the PETA TFRecord files
(manual download is required).

#. Download the PETA dataset from `here <https://github.com/dangweili/pedestrian-attribute-recognition-pytorch>`_.
   The expected dataset structure:

   .. code-block::

      PETA
      |_ images
      |  |_ 00001.png
      |  |_ ...
      |  |_ 19000.png
      |_ PETA.mat

#. Run the create TFRecord scripts:

   .. code-block::

      python hailo_model_zoo/datasets/create_peta_tfrecord.py test --data /path/to/PETA/
      python hailo_model_zoo/datasets/create_peta_tfrecord.py train --data /path/to/PETA/

.. _CelebA:

CelebA
------

To evaluate/optimize/compile the face attribute models of the
Hailo Model Zoo you should generate the CelebA TFRecord files
(manual download is required).


#. Download the CelebA dataset from `here <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_. The expected dataset structure:

   .. code-block::

      Celeba
      |_ img_align_celeba_png
      |  |_ 000001.jpg
      |  |_ ...
      |  |_ 202599.jpg
      |_ list_attr_celeba.txt
      |_ list_eval_partition.txt


#. Run the create TFRecord scripts:

   .. code-block::

      python hailo_model_zoo/datasets/create_celeba_tfrecord.py val --data /path/to/CelebA/
      python hailo_model_zoo/datasets/create_celeba_tfrecord.py train --data /path/to/CelebA/

.. _LFW:

LFW
------

To evaluate/optimize/compile the face recognition models of the
Hailo Model Zoo you should generate the arcface_lfw TFRecord files


Run the creation scripts:

.. code-block::

    python hailo_model_zoo/datasets/create_arcface_lfw_tfrecord.py calib
    python hailo_model_zoo/datasets/create_arcface_lfw_tfrecord.py val

Manual Download (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Download LFW dataset from `here <http://vis-www.cs.umass.edu/lfw/lfw.tgz>`_
#. Download LFW pairs file from `here <http://vis-www.cs.umass.edu/lfw/pairs.txt>`_
#. Run the scripts:

    .. code-block::

        python hailo_model_zoo/datasets/create_arcface_lfw_tfrecord.py calib --tgz /path/to/lfw.tgz --pairs /path/to/pairs.txt
        python hailo_model_zoo/datasets/create_arcface_lfw_tfrecord.py val --tgz /path/to/lfw.tgz --pairs /path/to/pairs.txt

.. _BSD100:

BSD100
------

To evaluate/optimize/compile the super resolution models of the
Hailo Model Zoo you should generate the BSD100 TFRecord files.

Run the creation scripts:

.. code-block::

    python hailo_model_zoo/datasets/create_bsd100_tfrecord.py val
    python hailo_model_zoo/datasets/create_bsd100_tfrecord.py calib

Manual Download (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Download the BSD100 dataset from `here <https://drive.google.com/uc?export=download&id=1oOqJHTu2JIUz0qyEmVuSI_Nye36nioYX>`_ and extract.
   The expected dataset structure:

   .. code-block::

      BSD100
      |_ GTmod12
      |  |_ 101085.png
      |  |_ ...
      |  |_ 97033.png
      |_ GTmod16
      |  |_ ...
      |_ LRbicx8
      |  |_ ...
      |_ LRbicx4
      |  |_ ...
      |_ LRbicx3
      |  |_ ...
      |_ LRbicx2
      |  |_ ...
      |_ LRbicx16
      |  |_ ...
      |_ original
      |  |_ ...


#. Run the scripts:

   .. code-block::

      python hailo_model_zoo/datasets/create_bsd100_tfrecord.py val --lr /path/to/LRbicx4 --hr /path/to/GTmod12
      python hailo_model_zoo/datasets/create_bsd100_tfrecord.py calib --lr /path/to/LRbicx4 --hr /path/to/GTmod12


.. _CLIP_CIFAR100:

CLIP_CIFAR100
------

To evaluate/optimize/compile the CLIP models of the
Hailo Model Zoo you should generate the CIFAR100 TFRecord files.

Run the creation scripts:

.. code-block::

    python hailo_model_zoo/datasets/create_clip_cifar100_tfrecord.py val
    python hailo_model_zoo/datasets/create_clip_cifar100_tfrecord.py calib


.. _LOL:

LOL
------

To evaluate/optimize/compile the low light enhancement models of the
Hailo Model Zoo you should generate the LOL TFRecord files.

Run the creation scripts:

.. code-block::

    python hailo_model_zoo/datasets/create_lol_tfrecord.py val
    python hailo_model_zoo/datasets/create_lol_tfrecord.py calib

Manual Download (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Download the LOL dataset from `here <https://drive.google.com/uc?export=download&id=157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB&authuser=0>`_ and extract.
   The expected dataset structure:

   .. code-block::

      lol_dataset
      |_ eval15
         |_ high
         |   |_ 111.png
         |   |_ 146.png
         |   |_ ...
         |_ low
         |   |_ 111.png
         |   |_ 146.png
         |   |_ ...
      |_ our485
         |_ high
         |   |_ 100.png
         |   |_ 101.png
         |   |_ ...
         |_ low
         |   |_ 100.png
         |   |_ 101.png
         |   |_ ...


#. Run the scripts:

   .. code-block::

      python hailo_model_zoo/datasets/create_lol_tfrecord.py val --ll /path/to/val/lowlight/images --lle /path/to/val/highlight/images
      python hailo_model_zoo/datasets/create_lol_tfrecord.py calib --ll /path/to/train/lowlight/images --lle /path/to/train/highlight/images


.. _BSD68:

BSD68
------

To evaluate/optimize/compile the image denoising models of the
Hailo Model Zoo you should generate the BSD68 TFRecord files.

Run the creation scripts:

.. code-block::

    python hailo_model_zoo/datasets/create_bsd68_tfrecord.py val
    python hailo_model_zoo/datasets/create_bsd68_tfrecord.py calib

Manual Download (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Download the BSD100 dataset from `here <https://drive.google.com/uc?export=download&id=1mwMLt-niNqcQpfN_ZduG9j4k6P_ZkOl0>`_ and extract.
   The expected dataset structure:

   .. code-block::

      test
      |_ BSD68
      |  |_ test001.png
      |  |_ ...
      |  |_ test068.png
      |_ CBSD68
      |  |_ ...
      |_ Kodak
      |  |_ ...
      |_ McMaster
      |  |_ ...
      |_ Set12
      |  |_ ...
      |_ Urban100
      |  |_ ...
      |_ LRbicx16

#. Run the scripts:

   .. code-block::

      python hailo_model_zoo/datasets/create_bsd100_tfrecord.py BSD68 val --data-path <BSD68-extracted-data-folder>
      python hailo_model_zoo/datasets/create_bsd100_tfrecord.py BSD68 calib --data-path <BSD68-extracted-data-folder>


.. _CBSD68:

CBSD68
------

To evaluate/optimize/compile the image denoising models of the
Hailo Model Zoo you should generate the CBSD68 TFRecord files.

Run the creation scripts:

.. code-block::

    python hailo_model_zoo/datasets/create_bsd68_tfrecord.py CBSD68 val
    python hailo_model_zoo/datasets/create_bsd68_tfrecord.py CBSD68 calib

Manual Download (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Download the BSD100 dataset from `here <https://drive.google.com/uc?export=download&id=1mwMLt-niNqcQpfN_ZduG9j4k6P_ZkOl0>`_ and extract.
   The expected dataset structure:

   .. code-block::

      test
      |_ BSD68
      |  |_ ...
      |_ CBSD68
      |  |_ test001.png
      |  |_ ...
      |  |_ test068.png
      |_ Kodak
      |  |_ ...
      |_ McMaster
      |  |_ ...
      |_ Set12
      |  |_ ...
      |_ Urban100
      |  |_ ...
      |_ LRbicx16

#. Run the scripts:

   .. code-block::

      python hailo_model_zoo/datasets/create_bsd100_tfrecord.py CBSD68 val --data-path <CBSD68-extracted-data-folder>
      python hailo_model_zoo/datasets/create_bsd100_tfrecord.py CBSD68 calib --data-path <CBSD68-extracted-data-folder>

.. _KITTI_STEREO:

KITTI_STEREO
------------

To evaluate/optimize/compile the stereo models of the
Hailo Model Zoo you should generate the KITTI Stereo TFRecord files.

Manual Download
^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Download the KITTI Stereo dataset from `here <https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo>`_. One must request access and await approval.
#. Extract the dataset.
   The expected dataset structure:

   .. code-block::

      kitti_stereo
      |_ testing
         |_ image_2
         |   |_ 000000_10.png
         |   |_ 000000_11.png
         |   |_ ...
         |_ image_3
         |   |_ 000000_10.png
         |   |_ 000000_11.png
         |   |_ ...
      |_ training
         |_ image_2
         |   |_ 000000_10.png
         |   |_ 000000_11.png
         |   |_ ...
         |_ disp_occ_0
         |   |_ 000000_10.png
         |   |_ 000001_10.png
         |   |_ 000002_10.png
         |   |_ ...


#. Run the scripts:

   .. code-block::

      python hailo_model_zoo/datasets/create_kitti_stereo_tfrecord.py calib --data <TRAIN_DIR>
      python hailo_model_zoo/datasets/create_kitti_stereo_tfrecord.py val --data <VALIDATION_DIR>

.. _KINETICS400:

KINETICS400
------------

To evaluate/optimize/compile the video classification models of the
Hailo Model Zoo you should generate the KINETICS400 TFRecord files.

Manual Download
^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Download the kinetics400 dataset from `here <https://github.com/cvdfoundation/kinetics-dataset>`_. Follow the instructions to download the dataset.
#. The expected dataset structure:
   .. code-block::

      k400/videos
      |_ test
         |_ abseiling
         |   |_ 0aSqlZT8QmM_000048_000058.mp4
         |   |_ 0xreS8KFbrw_000417_000427.mp4
         |   |_ ...
         |_ air drumming
         |   |_ 013SMb0SX8I_000020_000030.mp4
         |   |_ 013SMb0SX8I_000020_000030.mp4
         |   |_ ...
      |_ train
         |_ abseiling
         |   |_ 0347ZoDXyP0_000095_000105.mp4
         |   |_ 035LtPeUFTE_000085_000095.mp4
         |   |_ ...
         |_ air drumming
         |   |_ 03V2idM7_KY_000003_000013.mp4
         |   |_ 1R7Ds_000003_000013.mp4
         |   |_ 0c1bhfxioqE_000078_000088.mp4
         |   |_ ...
      |_ val
         |_ abseiling
         |   |_ 0wR5jVB-WPk_000417_000427.mp4
         |   |_ 3caPS4FHFF8_000036_000046.mp4
         |   |_ ...
         |_ air drumming
         |   |_ 2cPLjY5AWXU_000001_000011.mp4
         |   |_ 3K0Sw7rbzPU_000114_000124.mp4
         |   |_ 6Tnsmk9C2rg_000048_000058.mp4
         |   |_ ...


#. Run the scripts:

   .. code-block::

      python hailo_model_zoo/datasets/create_kinetics400_tfrecord.py calib --data <path_to_k400/videos>
      python hailo_model_zoo/datasets/create_kinetics400_tfrecord.py val --data <path_to_k400/videos>

.. _NUSCENES:

NUSCENES
------------

#. Download the dataset from `here <https://www.nuscenes.org/download>`_ and extract.
#. The expected dataset structure:

   .. code-block::

      nuscenes
      |_ maps
      |  |_ *.png
      |_ samples
      |  |_ CAM_BACK
      |  |_ |_ *.jpg
      |  |_ CAM_BACK_LEFT
      |  |_ |_ *.jpg
      |  |_ CAM_BACK_RIGHT
      |  |_ |_ *.jpg
      |  |_ CAM_FRONT
      |  |_ |_ *.jpg
      |  |_ CAM_FRONT_LEFT
      |  |_ |_ *.jpg
      |  |_ CAM_FRONT_RIGHT
      |  |_ |_ *.jpg
      |_ sweeps
      |  |_ CAM_BACK
      |  |_ |_ *.jpg
      |  |_ CAM_BACK_LEFT
      |  |_ |_ *.jpg
      |  |_ CAM_BACK_RIGHT
      |  |_ |_ *.jpg
      |  |_ CAM_FRONT
      |  |_ |_ *.jpg
      |  |_ CAM_FRONT_LEFT
      |  |_ |_ *.jpg
      |  |_ CAM_FRONT_RIGHT
      |  |_ |_ *.jpg
      |_ v1.0-trainval
      |  |_ *.json

#. Run the scripts:

   .. code-block::

      python hailo_model_zoo/datasets/create_nuscenes_tfrecord.py calib --ann_file <train_annotation_file.pkl>
      python hailo_model_zoo/datasets/create_nuscenes_tfrecord.py val --ann_file <val_annotation_file.pkl>

   where <\*_annotation_file.pkl> is the train / val .pkl annotation file generated from the PETR training environment.
