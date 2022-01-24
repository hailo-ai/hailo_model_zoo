# Retrain on Custom Dataset

In this document, we describe the process of re-training a new deep learning network on your custom dataset. We choose YOLOv3/4/5 as examples due to their popularity and easy-to-use frameworks for training. Learn more about YOLOv3/4 in [**here**](https://github.com/AlexeyAB/darknet) and YOLOv5 in [**here**](https://github.com/ultralytics/yolov5/tree/v2.0).

<details>
    <summary>YOLOv3</summary>

## Training YOLOv3
To train your YOLOv3 network follow these steps (full instructions can be found in [**here**](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects)):
1. Clone the Darknet framework:
    ```
    git clone https://github.com/AlexeyAB/darknet.git; cd darknet
    ```
2. build the framework using <code>make</code> (it is recommended to build with CUDA support by setting <code>GPU=1</code> in the Makefile)
3. Download pretrained weights for YOLOv3 model from [**here**](https://pjreddie.com/media/files/darknet53.conv.74)
4. Create a new cfg file. This file contain the information about your model: input resolution, number of classes and so on. The default cfg file for YOLOv3 can be found in [**here**](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov3.cfg)
5. Add information about your data:
    * Create <code>obj.names</code> and <code>obj.data</code> in <code>build\darknet\x64\data\ </code>
    * Place your jpg images in <code>build\darknet\x64\data\obj\ </code>
    * Generate a txt for each image (in the same directory) containing the annotations in the format of <code>\<object-class> \<x_center> \<y_center> \<width> \<height></code>. For example: for img1.jpg create img1.txt containing:
        ```
        1 0.716797 0.395833 0.216406 0.147222
        0 0.687109 0.379167 0.255469 0.158333
        1 0.420312 0.395833 0.140625 0.166667
        ```
6. Create <code>train.txt</code> in directory <code>build\darknet\x64\data\ </code> with filenames of your images. For example:
    ```
    data/obj/img1.jpg
    data/obj/img2.jpg
    data/obj/img3.jpg
    ```
7. Start training:
    ```
    ./darknet detector train build/darknet/x64/data/obj.data cfg/yolov3.cfg yolov3.conv.74
    ```
8. Final product would be available in <code>build\darknet\x64\backup\ </code>


## Export to ONNX
To export the trained YOLOv3 network to ONNX follow these steps:
1. Clone the following repo:
    ```
    git clone https://github.com/nivosco/pytorch-YOLOv4.git;cd pytorch-YOLOv4
    ```
2. Install onnxruntime:
    ```
    pip install onnxruntime
    ```
3. Run python script to generate the ONNX model (pretrained <code>yolov3.weights</code> can be downloaded from [**here**](https://pjreddie.com/media/files/yolov3.weights)):
    ```
    python demo_darknet2onnx.py cfg/yolov3.cfg yolov3.weights image.jpg 1
    ```

4. (optional) Using your own cfg file might require adding <code>scale_x_y=1.0</code> under each <code>[yolo]</code> block in the cfg file. Check <code>cfg/yolov3.cfg</code> for an example.

## Compile the Model using Hailo Model Zoo
You can generate an HEF file for inference on Hailo-8 from your trained ONNX model.  
In order to do so you need a working model-zoo environment.
Choose the corresponding YAML from our networks configuration directory, i.e. <code>hailo_model_zoo/cfg/networks/yolov3.yaml</code>, and run compilation using the model zoo:  
  ```
  python hailo_model_zoo/main.py compile yolov3 --ckpt yolov3.onnx --calib-path /path/to/calibration/imgs/dir/ --yaml yolov3.yaml
  ```
  * <code>--ckpt</code> - path to your ONNX file.
  * <code>--calib-path</code> - path to a directory with your calibration images in JPEG format
  * <code>--yaml</code> - path to your configuration YAML file. In case you have made some changes in the model, you might need to update its start/end nodes names / number of classes and so on.  <br>
  The model zoo will take care of adding the input normalization to be part of the model.

The compiled model can be used in the [**Hailo TAPPAS**](https://hailo.ai/developer-zone/tappas-apps-toolkit/) to generate a full application.  <br>
</details>

<details>
    <summary>YOLOv4</summary>

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
  python hailo_model_zoo/main.py compile yolov4_leaky --ckpt yolov4_1_3_512_512.onnx --calib-path /path/to/calibration/imgs/dir/ --yaml yolov4_leaky.yaml
  ```
  * <code>--ckpt</code> - path to your ONNX file.
  * <code>--calib-path</code> - path to a directory with your calibration images in JPEG format
  * <code>--yaml</code> - path to your configuration YAML file. In case you have made some changes in the model, you might need to update its start/end nodes names / number of classes and so on.  <br>
  The model zoo will take care of adding the input normalization to be part of the model.

The compiled model can be used in the [**Hailo TAPPAS**](https://hailo.ai/developer-zone/tappas-apps-toolkit/) to generate a full application.  <br>
</details>


<details>
    <summary>YOLOv5</summary>

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
  python hailo_model_zoo/main.py compile yolov5s --ckpt yolov5s.onnx --calib-path /path/to/calibration/imgs/dir/ --yaml yolov5s.yaml
  ```
  * <code>--ckpt</code> - path to your ONNX file.
  * <code>--calib-path</code> - path to a directory with your calibration images in JPEG format
  * <code>--yaml</code> - path to your configuration YAML file. In case you have made some changes in the model, you might need to update its start/end nodes names / number of classes and so on.  <br>
  The model zoo will take care of adding the input normalization to be part of the model.
The compiled model can be used in the [**Hailo TAPPAS**](https://hailo.ai/developer-zone/tappas-apps-toolkit/) to generate a full application.  <br>

</details>

