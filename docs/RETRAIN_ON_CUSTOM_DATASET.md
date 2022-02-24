# Retrain on Custom Dataset

In this document, we describe the process of re-training a new deep learning network on your custom dataset. Each architecture is served with a compatible Dockerfile, which contains all you need for ramping-up your training environment.

## Object Detection
  <details>
      <summary>YOLOv3</summary>

  * To learn more about yolov3 look [**here**](https://github.com/hailo-ai/darknet)    
  ## Training YOLOv3
  ### Prerequisites
  * docker ([installation instructions](https://docs.docker.com/engine/install/ubuntu/))
  * nvidia-docker2 ([installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
  > **_NOTE:_**  In case you use Hailo Software Suite docker, make sure you are doing all the following instructions outside this docker.
  ### Environement Preparations
  1. Build the docker image:
      ```
      cd model_zoo/training/yolov3
      docker build --build-arg timezone=`cat /etc/timezone` -t yolov3:v0 .
      ```
      - This command will build the docker image with the necessary requirements using the Dockerfile exists in this directory.

  2. Start your docker:
      ```
      docker run --name <you_docker_name> -it --gpus all --ipc=host -v /path/to/local/drive:/path/to/docker/dir yolov3:v0
      ```

  ### Training and exporting to ONNX
  1. Train your model:
    Once the docker is started, you can start training your YOLOv3.
      * Prepare your custom dataset - Follow the full instructions described [**here**](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects):

        * Create `data/obj.data` with paths to your training and validation `.txt` files, which contain the list of the image paths<sup>*</sup>.
          ```
          classes = 80
          train  = data/train.txt
          valid  = data/val.txt
          names = data/coco.names
          backup = backup/
          ```
          \* Tip: specify the paths to the training and validation images in the training and validation `.txt` files relative to `/workspace/darknet/`

          Place your training/validation images and labels in your data folder and make sure you update the number of classes.
        * Labels - each image should have labels in YOLO format with corresponding txt file for each image.

      * Start training - The following command is an example for training the yolov4-leaky model.
      ```
      ./darknet detector train data/obj.data cfg/yolov3.cfg yolov3.weights -map
      ```

      Final trained weights will be available in <code>backup/</code> directory.
  <br>

  2. Export to ONNX:
  In order to export your trained YOLOv3 model to ONNX run the following script:
      ```
      python ../pytorch-YOLOv4/demo_darknet2onnx.py cfg/yolov3.cfg /path/to/trained.weights /path/to/some/image.jpg 1
      ```
      - The ONNX would be available in <code>/workspace/darknet/</code>

  ## Compile the Model using Hailo Model Zoo
  You can generate an HEF file for inference on Hailo-8 from your trained ONNX model.  
  In order to do so you need a working model-zoo environment.
  Choose the corresponding YAML from our networks configuration directory, i.e. <code>hailo_model_zoo/cfg/networks/yolov3_416.yaml</code> (for the default YOLOv3 model).

  Align the corresponding alls, i.e. <code>hailo_model_zoo/cfg/networks/yolov3_416.alls</code> with the size of the calibration set using <code>dataset_size=<number_of_jpgs_in_folder></code> parameter.

  Run compilation using the model zoo:
    ```
    python hailo_model_zoo/main.py compile yolov3_416 --ckpt yolov3_1_416_416.onnx --calib-path /path/to/calibration/imgs/dir/
    ```
    * <code>--ckpt</code> - path to your ONNX file.
    * <code>--calib-path</code> - path to a directory with your calibration images in JPEG format
    The model zoo will take care of adding the input normalization to be part of the model.

  The compiled model can be used in the [**Hailo TAPPAS**](https://hailo.ai/developer-zone/tappas-apps-toolkit/) to generate a full application.  <br>
  </details>


  <details>
      <summary>YOLOv4</summary>

  * To learn more about yolov4 look [**here**](https://github.com/hailo-ai/darknet)    

  ## Training YOLOv4-leaky
  ### Prerequisites
  * docker ([installation instructions](https://docs.docker.com/engine/install/ubuntu/))
  * nvidia-docker2 ([installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
  > **_NOTE:_**  In case you use Hailo Software Suite docker, make sure you are doing all the following instructions outside this docker.
  ### Environement Preparations
  1. Build the docker image:
      ```
      cd model_zoo/training/yolov4
      docker build --build-arg timezone=`cat /etc/timezone` -t yolov4:v0 .
      ```
      - This command will build the docker image with the necessary requirements using the Dockerfile exists in this directory.

  2. Start your docker:
      ```
      docker run -it --gpus all --ipc=host -v /path/to/local/drive:/path/to/docker/dir yolov4:v0
      ```

  ### Training and exporting to ONNX
  1. Train your model:
    Once the docker is started, you can start training your YOLOv4-leaky.
      * Prepare your custom dataset - Follow the full instructions described [**here**](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects):

        * Create `data/obj.data` with paths to your training and validation `.txt` files, which contain the list of the image paths<sup>*</sup>.
          ```
          classes = 80
          train  = data/train.txt
          valid  = data/val.txt
          names = data/coco.names
          backup = backup/
          ```
          \* Tip: specify the paths to the training and validation images in the training and validation `.txt` files relative to `/workspace/darknet/`

          Place your training/validation images and labels in your data folder and make sure you update the number of classes.
        * Labels - each image should have labels in YOLO format with corresponding txt file for each image.

      * Start training - The following command is an example for training the yolov4-leaky model.
      ```
      ./darknet detector train data/obj.data cfg/yolov4-leaky.cfg yolov4-leaky.weights -map
      ```

      Final trained weights will be available in <code>backup/</code> directory.
  <br>

  2. Export to ONNX:
  In order to export your trained YOLOv4 model to ONNX run the following script:
      ```
      python ../pytorch-YOLOv4/demo_darknet2onnx.py cfg/yolov4-leaky.cfg /path/to/trained.weights /path/to/some/image.jpg 1
      ```
      - The ONNX would be available in <code>/workspace/darknet/</code>

  ## Compile the Model using Hailo Model Zoo
  You can generate an HEF file for inference on Hailo-8 from your trained ONNX model.  
  In order to do so you need a working model-zoo environment.
  Choose the corresponding YAML from our networks configuration directory, i.e. <code>hailo_model_zoo/cfg/networks/yolov4_leaky.yaml</code>, and run compilation using the model zoo:  
    ```
    python hailo_model_zoo/main.py compile --ckpt yolov4_1_3_512_512.onnx --calib-path /path/to/calibration/imgs/dir/ --yaml yolov4_leaky.yaml
    ```
    * <code>--ckpt</code> - path to your ONNX file.
    * <code>--calib-path</code> - path to a directory with your calibration images in JPEG format
    * <code>--yaml</code> - path to your configuration YAML file. In case you have made some changes in the model, you might need to update its start/end nodes names / number of classes and so on.  <br>
    The model zoo will take care of adding the input normalization to be part of the model.

  The compiled model can be used in the [**Hailo TAPPAS**](https://hailo.ai/developer-zone/tappas-apps-toolkit/) to generate a full application.  <br>
  </details>


  <details>
      <summary>YOLOv5</summary>

  * To learn more about yolov5 look [**here**](https://github.com/hailo-ai/yolov5)

  ## Training YOLOv5
  ### Prerequisites
  * docker ([installation instructions](https://docs.docker.com/engine/install/ubuntu/))
  * nvidia-docker2 ([installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
  > **_NOTE:_**  In case you use Hailo Software Suite docker, make sure you are doing all the following instructions outside this docker.
  ### Environement Preparations
  1. Build the docker image:
      ```  
      cd model_zoo/training/yolov5
      docker build --build-arg timezone=`cat /etc/timezone` -t yolov5:v0 .
      ```  
      - This command will build the docker image with the necessary requirements using the Dockerfile exists in yolov5 directory.  

  2. Start your docker:
      ```
      docker run -it --gpus all --ipc=host -v /path/to/local/drive:/path/to/docker/dir yolov5:v0
      ```

  ### Training and exporting to ONNX
  1. Train your model:
    Once the docker is started, you can start training your model.
      * Prepare your custom dataset - Follow the steps described [**here**](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) in order to create:
        * <code>dataset.yaml</code> configuration file
        * Labels - each image should have labels in YOLO format with corresponding txt file for each image.  
      * Start training - The following command is an example for training a *yolov5s* model.  
        ```
        python train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov5s.pt --cfg models/yolov5s.yaml
        ```  
        * <code>yolov5s.pt</code> - pretrained weights. You can find the pretrained weights for *yolov5s*, *yolov5m*, *yolov5l*, *yolov5x* in your working directory.
        * <code>models/yolov5s.yaml</code> - configuration file of the yolov5 variant you would like to train. In order to change the number of classes make sure you update this file.
  <br>  

  2. Export to ONNX:
  In order to export your trained YOLOv5 model to ONNX run the following script:
      ```
      python models/export.py --weights /path/to/trained/model.pt --img 640 --batch 1  # export at 640x640 with batch size 1
      ```  

  ## Compile the Model using Hailo Model Zoo
  You can generate an HEF file for inference on Hailo-8 from your trained ONNX model.  
  In order to do so you need a working model-zoo environment.
  Choose the corresponding YAML from our networks configuration directory, i.e. <code>hailo_model_zoo/cfg/networks/yolov5s.yaml</code>, and run compilation using the model zoo:  
    ```
    python hailo_model_zoo/main.py compile --ckpt yolov5s.onnx --calib-path /path/to/calibration/imgs/dir/ --yaml yolov5s.yaml
    ```
    * <code>--ckpt</code> - path to your ONNX file.
    * <code>--calib-path</code> - path to a directory with your calibration images in JPEG format
    * <code>--yaml</code> - path to your configuration YAML file. In case you have made some changes in the model, you might need to update its start/end nodes names / number of classes and so on.  <br>
    The model zoo will take care of adding the input normalization to be part of the model.
  The compiled model can be used in the [**Hailo TAPPAS**](https://hailo.ai/developer-zone/tappas-apps-toolkit/) to generate a full application.  <br>

  </details>

<details>
    <summary>NanoDet</summary>  

  * To learn more about NanoDet look [**here**](https://github.com/hailo-ai/nanodet)

## Training Nanodet
### Prerequisites
* docker ([installation instructions](https://docs.docker.com/engine/install/ubuntu/))
* nvidia-docker2 ([installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
> **_NOTE:_**  In case you use Hailo Software Suite docker, make sure you are doing all the following instructions outside this docker.
### Environement Preparations
1. Build the docker image:
    ```
    cd model_zoo/training/nanodet
    docker build -t nanodet:v0 --build-arg timezone=`cat /etc/timezone` .
    ```
    the following optional arguments can be passed via --build-arg
    - timezone - a string for setting up timezone. E.g. "Asia/Jerusalem"
    - user - username for a local non-root user. Defaults to 'hailo'.
    - group - default group for a local non-root user. Defaults to 'hailo'.
    - uid - user id for a local non-root user.
    - gid - group id for a local non-root user.

2. Start your docker:
    ```
    docker run -it --gpus all -u <username> --ipc=host -v /path/to/local/drive:/path/to/docker/data/dir nanodet:v0
    ```
    username is the same one used when building the image.

### Training and exporting to ONNX
1. Prepare your data: <br>
    Data is expected to be in coco format. More information can be found [here](https://cocodataset.org/#format-data) 

2. Training: <br>
    Configure your model in a .yaml file. We'll use /workspace/nanodet/config/legacy_v0.x_configs/RepVGG/nanodet-RepVGG-A0_416.yml in this guide.
    Modify the path for the dataset in the .yaml configuration file:
    ```
    data:
      train:
        name: CocoDataset
        img_path: <path-to-train-dir>
        ann_path: <path-to-annotations-file>
        ...
      val:
        name: CocoDataset
        img_path: <path-to-validation-dir>
        ann_path: <path-to-annotations-file>
        ...
    ```

    Start training with the following command:
    ```
    cd /workspace/nanodet
    python tools/train.py ./config/legacy_v0.x_configs/RepVGG/nanodet-RepVGG-A0_416.yml
    ```
    In case you want to use the pretrained nanodet-RepVGG-A0_416.ckpt, which was predownloaded into your docker modify your configurationf file:
    ```
    schedule:
      load_model: ./pretrained/nanodet-RepVGG-A0_416.ckpt
    ```
    
    Modifying the batch size and the number of GPUs used for training can be done also in the configuration file:
    ```
    device:
      gpu_ids: [0]
      workers_per_gpu: 1
      batchsize_per_gpu: 128
    ```

3. Exporting to onnx
    After training, run the following command:
    ```
    python tools/export_onnx.py --cfg_path ./config/legacy_v0.x_configs/RepVGG/nanodet-RepVGG-A0_416.yml --model_path /workspace/nanodet/workspace/RepVGG-A0-416/model_last.ckpt
    ```
  > **_NOTE:_**  Your trained model will be found under the following path: /workspace/nanodet/workspace/<backbone-name>/model_last.ckpt, and exported onnx will be written to /workspace/nanodet/nanodet.onnx

## Compile the Model using Hailo Model Zoo
You can generate an HEF file for inference on Hailo-8 from your trained ONNX model.  
In order to do so you need a working model-zoo environment.
Choose the corresponding YAML from our networks configuration directory, i.e. <code>hailo_model_zoo/cfg/networks/nanodet_repvgg.yaml</code>, and run compilation using the model zoo:  
  ```
  python hailo_model_zoo/main.py compile --ckpt nanodet.onnx --calib-path /path/to/calibration/imgs/dir/ --yaml nanodet_repvgg.yaml
  ```
  * <code>--ckpt</code> - path to your ONNX file.
  * <code>--calib-path</code> - path to a directory with your calibration images in JPEG format
  * <code>--yaml</code> - path to your configuration YAML file. In case you have made some changes in the model, you might need to update its start/end nodes names / number of classes and so on.  <br>
  The model zoo will take care of adding the input normalization to be part of the model.

The compiled model can be used in the [**Hailo TAPPAS**](https://hailo.ai/developer-zone/tappas-apps-toolkit/) to generate a full application.  <br>
</details>

<br>

## Pose Estimation
<details>
    <summary>CenterPose</summary>

  * To learn more about CenterPose look [**here**](https://github.com/hailo-ai/centerpose)
## Training Centerpose
### Prerequisites
* docker ([installation instructions](https://docs.docker.com/engine/install/ubuntu/))
* nvidia-docker2 ([installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
> **_NOTE:_**  In case you use Hailo Software Suite docker, make sure you are doing all the following instructions outside this docker.
### Environement Preparations
1. Build the docker image:
    ```
    cd model_zoo/training/centerpose
    docker build -t centerpose:v0 --build-arg timezone=`cat /etc/timezone` .
    ```
	the following optional arguments can be passed via --build-arg
	- timezone - a string for setting up timezone. E.g. "Asia/Jerusalem"
	- user - username for a local non-root user. Defaults to 'hailo'.
	- group - default group for a local non-root user. Defaults to 'hailo'.
	- uid - user id for a local non-root user.
	- gid - group id for a local non-root user.

2. Start your docker:
    ```
    docker run -it --gpus all -u <username> --ipc=host -v /path/to/local/drive:/path/to/docker/dir centerpose:v0
    ```
    username is the same one used when building the image.

### Training and exporting to ONNX
1. Prepare your data: <br>
    Data is expected to be in coco format, and by default should be in /workspace/data/<dataset_name>.

    The expected structure is as follows:
    ```
    /workspace
    |-- data
    `-- |-- coco
        `-- |-- annotations
            |   |-- instances_train2017.json
            |   |-- instances_val2017.json
            |   |-- person_keypoints_train2017.json
            |   |-- person_keypoints_val2017.json
            |   |-- image_info_test-dev2017.json
            |---|-- train2017
            |---|---|-- *.jpg
            |---|-- val2017
            |---|---|-- *.jpg
            |---|-- test2017
            `---|---|-- *.jpg
    ```
    The path for the dataset can be configured in the .yaml file, e.g. centerpose/experiments/regnet_fpn.yaml

2. Training: <br>
    Configure your model in a .yaml file. We'll use /workspace/centerpose/experiments/regnet_fpn.yaml in this guide.

    start training with the following command:
    ```
    cd /workspace/centerpose/tools
    python -m torch.distributed.launch --nproc_per_node 4 train.py --cfg ../experiments/regnet_fpn.yaml
    ```
    Where 4 is the number of GPUs used for training.
    If using a different number, update both this and the used gpus in the .yaml configuration.

3. Exporting to onnx
    After training, run the following command:
    ```
    cd /workspace/tools
    python export.py --cfg ../experiments/regnet_fpn.yaml --TESTMODEL /workspace/out/regnet1_6/model_best.pth
    ```

## Compile the Model using Hailo Model Zoo
You can generate an HEF file for inference on Hailo-8 from your trained ONNX model.  
In order to do so you need a working model-zoo environment.
Choose the corresponding YAML from our networks configuration directory, i.e. <code>hailo_model_zoo/cfg/networks/centerpose_regnetx_1.6gf_fpn.yaml</code>, and run compilation using the model zoo:  
  ```
  python hailo_model_zoo/main.py compile --ckpt coco_pose_regnet1.6_fpn.onnx --calib-path /path/to/calibration/imgs/dir/ --yaml centerpose_regnetx_1.6gf_fpn.yaml
  ```
  * <code>--ckpt</code> - path to your ONNX file.
  * <code>--calib-path</code> - path to a directory with your calibration images in JPEG format
  * <code>--yaml</code> - path to your configuration YAML file. In case you have made some changes in the model, you might need to update its start/end nodes names / number of classes and so on.  <br>
  The model zoo will take care of adding the input normalization to be part of the model.

The compiled model can be used in the [**Hailo TAPPAS**](https://hailo.ai/developer-zone/tappas-apps-toolkit/) to generate a full application.  <br>
</details>
## Semantic Segmentation
<details>
    <summary>FCN</summary>

  * To learn more about FCN look [**here**](https://github.com/hailo-ai/mmsegmentation)
## Training FCN
### Prerequisites
* docker ([installation instructions](https://docs.docker.com/engine/install/ubuntu/))
* nvidia-docker2 ([installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
> **_NOTE:_**  In case you use Hailo Software Suite docker, make sure you are doing all the following instructions outside this docker.
### Environement Preparations
1. Build the docker image:
    ```
    cd model_zoo/training/fcn
    docker build -t fcn:v0 --build-arg timezone=`cat /etc/timezone` .
    ```
    the following optional arguments can be passed via --build-arg
    - timezone - a string for setting up timezone. E.g. "Asia/Jerusalem"
    - user - username for a local non-root user. Defaults to 'hailo'.
    - group - default group for a local non-root user. Defaults to 'hailo'.
    - uid - user id for a local non-root user.
    - gid - group id for a local non-root user.

2. Start your docker:
    ```
    docker run -it --gpus all -u <username> --ipc=host -v /path/to/local/drive:/path/to/docker/dir fcn:v0
    ```
    username is the same one used when building the image.

### Training and exporting to ONNX
1. Prepare your data: <br>
    Data is expected to be in coco format, and by default should be in /workspace/data/<dataset_name>.

    The expected structure is as follows:
    ```
    /workspace
    |-- mmsegmentation
    `-- |-- data
            `-- cityscapes
                |-- gtFine
                |   | -- train
                |   |    | -- aachem
                |   |    | -- | -- *.png
                |   |    ` -- ...
                |   ` -- test
                |        | -- berlin
                |        | -- | -- *.png
                |        ` -- ...
                `-- leftImg8bit
                    | -- train
                    | -- | -- aachem
                    | -- | -- | -- *.png
                    | -- ` -- ...
                    ` -- test
                         | -- berlin
                         | -- | -- *.png
                         ` -- ...
            
    ```

    more information can be found [here](https://github.com/hailo-ai/mmsegmentation/blob/master/docs/en/dataset_prepare.md#cityscapes)

2. Training: <br>
    Configure your model in a .py file. We'll use /workspace/mmsegmentation/configs/fcn/fcn_r18_hailo.py in this guide.

    start training with the following command:
    ```
    cd /workspace/mmsegmentation
    ./tools/dist_train.sh configs/fcn/fcn8_r18_hailo.py 2
    ```
    Where 2 is the number of GPUs used for training.

3. Exporting to onnx
    After training, run the following command:
    ```
    cd /workspace/mmsegmentation
    python ./tools/pytorch2onnx.py configs/fcn/fcn_r18_hailo.py --opset-version 11 --checkpoint ./work_dirs/fcn_r18_hailo/latest.pth --shape 1024 1920 --output-file fcn.onnx
    ```

## Compile the Model using Hailo Model Zoo
You can generate an HEF file for inference on Hailo-8 from your trained ONNX model.  
In order to do so you need a working model-zoo environment.
Choose the corresponding YAML from our networks configuration directory, i.e. <code>hailo_model_zoo/cfg/networks/fcn16_resnet_v1_18.yaml</code>, and run compilation using the model zoo:  
  ```
  python hailo_model_zoo/main.py compile --ckpt fcn.onnx --calib-path /path/to/calibration/imgs/dir/ --yaml fcn16_resnet_v1_18.yaml
  ```
  * <code>--ckpt</code> - path to your ONNX file.
  * <code>--calib-path</code> - path to a directory with your calibration images in JPEG format
  * <code>--yaml</code> - path to your configuration YAML file. In case you have made some changes in the model, you might need to update its start/end nodes names / number of classes and so on.  <br>
  The model zoo will take care of adding the input normalization to be part of the model.

The compiled model can be used in the [**Hailo TAPPAS**](https://hailo.ai/developer-zone/tappas-apps-toolkit/) to generate a full application.  <br>
</details>

## Instance Segmentation
<details>
    <summary>YOLACT</summary>
	
 * To learn more about Yolact look [**here**](https://github.com/hailo-ai/yolact/tree/Model-Zoo-1.5)
## Training Yolact

### Prerequisites
* docker ([installation instructions](https://docs.docker.com/engine/install/ubuntu/))
* nvidia-docker2 ([installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
> **_NOTE:_**  In case you use Hailo Software Suite docker, make sure you are doing all the following instructions outside this docker.
### Environement Preparations
1. Build the docker image:
	```
	cd model_zoo/training/yolact
	docker build --build-arg timezone=`cat /etc/timezone` -t yolact:v0 .
	```
	the following optional arguments can be passed via --build-arg
	- timezone - a string for setting up timezone. E.g. "Asia/Jerusalem"
	- user - username for a local non-root user. Defaults to 'hailo'.
	- group - default group for a local non-root user. Defaults to 'hailo'.
	- uid - user id for a local non-root user.
	- gid - group id for a local non-root user.
	- This command will build the docker image with the necessary requirements using the Dockerfile exists in the yolact directory.
2. Start your docker:
	```
	docker run -it --gpus all --ipc=host -v /path/to/local/data:data/coco/ yolact:v0
	```
	
### Training and exporting to ONNX
1. Prepare your data: <br>
    Data is expected to be in coco format, and by default should be in /workspace/data/<dataset_name>.

    The expected structure is as follows:
    ```
    /workspace
    |-- data
    `-- |-- coco
        `-- |-- annotations
            |   |-- instances_train2017.json
            |   |-- instances_val2017.json
            |   |-- person_keypoints_train2017.json
            |   |-- person_keypoints_val2017.json
            |   |-- image_info_test-dev2017.json
            |---|-- train2017
            |---|---|-- *.jpg
            |---|-- val2017
            |---|---|-- *.jpg
            |---|-- test2017
            `---|---|-- *.jpg
    ```
	For training on custom datasets see [**here**](https://github.com/hailo-ai/yolact/tree/Model-Zoo-1.5#custom-datasets)
2. Train your model:
	Once your dataset is prepared, you can start training your model:
	```
	python train.py --config=yolact_regnetx_800MF_config
	```
	* <code>yolact_regnetx_800MF_config</code> - configuration using the regnetx_800MF backbone. Two other available options are <code>yolact_regnetx_600MF_config</code> and <code>yolact_regnetx_1600MF_config</code>.
3. Export to ONNX:
	In orded to export your trained YOLACT model to ONNX run the following script:
	```
	python export.py --config=yolact_regnetx_800MF_config --trained_model=path/to/trained/model --export_path=path/to/export/model.onnx
	```
	* <code>--config</code> - same configuration used for training.
	* <code>--trained_model</code> - path to the weights produced by the training process.
	* <code>--export_path</code> - path to export the ONNX file to. Include the <code>.onnx</code> extension.
	
## Compile the Model using Hailo Model Zoo
You can generate an HEF file for inference on Hailo-8 from your trained ONNX model.  
In order to do so you need a working model-zoo environment.
Choose the corresponding YAML from our networks configuration directory, i.e. <code>hailo_model_zoo/cfg/networks/yolact.yaml</code>, and run compilation using the model zoo:  
  ```
  python hailo_model_zoo/main.py compile yolact --ckpt yolact.onnx --calib-path /path/to/calibration/imgs/dir/ --yaml yolact_regnetx_800mf_20classes.yaml
  ```
The yolact_regnetx_800mf_20classes is an example yaml where some of the classes were removed. In case you want to remove classes you can remove them in tha yaml file, under *channel_remove*.
</details>
