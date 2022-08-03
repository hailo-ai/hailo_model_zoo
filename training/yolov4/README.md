# YOLOv4-leaky Retraining
* To learn more about yolov4 look [**here**](https://github.com/hailo-ai/darknet)    
---

### Prerequisites
  * docker ([installation instructions](https://docs.docker.com/engine/install/ubuntu/))
  * nvidia-docker2 ([installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
  > **_NOTE:_**  In case you use Hailo Software Suite docker, make sure you are doing all the following instructions outside of this docker.
### Environment Preparations
  1. Build the docker image:
      
      <code stage="docker_build">
      cd <span val="dockerfile_path">hailo_model_zoo/training/yolov4</span>


      docker build --build-arg timezone=\`cat /etc/timezone\` -t yolov4:v0 .
      </code>

      the following optional arguments can be passed via --build-arg:
      
      - `timezone` - a string for setting up timezone. E.g. "Asia/Jerusalem"
      - `user` - username for a local non-root user. Defaults to 'hailo'.
      - `group` - default group for a local non-root user. Defaults to 'hailo'.
      - `uid` - user id for a local non-root user.
      - `gid` - group id for a local non-root user.
      - This command will build the docker image with the necessary requirements using the Dockerfile exists in this directory.

  2. Start your docker:
      
      <code stage="docker_run">
      docker run <span val="replace_none">--name "your_docker_name"</span> -it --gpus all --ipc=host -v <span val="local_vol_path">/path/to/local/data/dir</span>:<span val="docker_vol_path">/path/to/docker/data/dir</span> yolov4:v0
      </code>

      - `docker run` create a new docker container.
      - `--name <your_docker_name>` name for your container.
      - `-it` runs the command interactively.
      - `--gpus all` allows access to all GPUs.
      - `--ipc=host` sets the IPC mode for the container.
      - `-v /path/to/local/data/dir:/path/to/docker/data/dir` maps `/path/to/local/data/dir` from the host to the container. You can use this command multiple times to mount multiple directories.
      - `yolov4:v0` the name of the docker image.

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
      
      <code stage="retrain">
      ./darknet detector train <span val="docker_obj_data_path">data/obj.data</span> cfg/yolov4-leaky.cfg yolov4-leaky.weights -map -clear
      </code>

      Final trained weights will be available in <code>backup/</code> directory.
  <br>

  2. Export to ONNX:
  In order to export your trained YOLOv4 model to ONNX run the following script:
      
       <code stage="export">
      python ../pytorch-YOLOv4/demo_darknet2onnx.py cfg/yolov4-leaky.cfg <span val="docker_path_to_trained_model">/path/to/trained.weights</span> <span val="docker_path_to_image">/path/to/some/image.jpg</span> 1
      </code>

      - The ONNX will be available in <code>/workspace/darknet/</code>
<br>

---

## Compile the Model using Hailo Model Zoo
  You can generate an HEF file for inference on Hailo-8 from your trained ONNX model.
  In order to do so you need a working model-zoo environment.
  Choose the corresponding YAML from our networks configuration directory, i.e. <code>hailo_model_zoo/cfg/networks/yolov4_leaky.yaml</code>, and run compilation using the model zoo:  

  <code stage="compile">
  python <span val="mz_main_path">hailo_model_zoo/main.py</span> compile --ckpt <span val="local_path_to_onnx">yolov4_1_3_512_512.onnx</span> --calib-path <span val="calib_set_path">/path/to/calibration/imgs/</span> --yaml <span val="yaml_file_path">yolov4_leaky.yaml</span>
  </code>

  * <code>--ckpt</code> - path to your  ONNX file.
  * <code>--calib-path</code> - path to a  directory with your calibration images  in JPEG format
  * <code>--yaml</code> - path to your  configuration YAML file. In case you  have made some changes in the model, you  might need to update its start/end nodes  names / number of classes and so on.   <br>
  The model zoo will take care of adding  the input normalization to be part of  the model.

  The compiled model can be used in the [**Hailo TAPPAS**](https://hailo.ai/developer-zone/tappas-apps-toolkit/) to generate a full application.
