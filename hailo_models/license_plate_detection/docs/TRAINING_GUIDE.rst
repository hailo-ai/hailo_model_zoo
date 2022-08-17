Train License Plate Detection on a Custom Dataset
-------------------------------------------------

Here we describe how to finetune Hailo's license plate detection network on your own custom dataset.

Prerequisites
^^^^^^^^^^^^^


* docker (\ `installation instructions <https://docs.docker.com/engine/install/ubuntu/>`_\ )
* nvidia-docker2 (\ `installation instructions <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_\ )


**NOTE:**  In case you are using the Hailo Software Suite docker, make sure to run all of the following instructions outside of that docker.


Environment Preparations
^^^^^^^^^^^^^^^^^^^^^^^^


#. 
   **Build the docker image**

   .. code-block::

       cd hailo_model_zoo/hailo_models/license_plate_detection/
       docker build --build-arg timezone=`cat /etc/timezone` -t license_plate_detection:v0 .


   * This command will build the docker image with the necessary requirements using the Dockerfile exists in this directory.

#. 
   **Start your docker:**

   .. code-block::

       docker run --name <your_docker_name> -it --gpus all --ipc=host -v /path/to/local/drive:/path/to/docker/dir license_plate_detection:v0


   * ``docker run`` create a new docker container.
   * ``--name <your_docker_name>`` name for your container.
   * ``-it`` runs the command interactively.
   * ``--gpus all`` allows access to all GPUs.
   * ``--ipc=host`` sets the IPC mode for the container.
   * ``-v /path/to/local/data/dir:/path/to/docker/data/dir`` maps ``/path/to/local/data/dir`` from the host to the container. You can use this command multiple times to mount multiple directories.
   * ``license_plate_detection:v0`` the name of the docker image.

Finetuning and exporting to ONNX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


#. | **Train the network on your dataset**
   | Once the docker is started, you can train the license plate detector on your custom dataset. We recommend following the instructions for YOLOv4 training that can be found in `here <https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects>`_. The important steps are specified below:


   * 
     Update ``data/obj.data`` with paths to your training and validation ``.txt`` files, which contain the list of the image paths\*.

     .. code-block::

          classes = 1
          train  = data/train.txt
          valid  = data/val.txt
          names = data/obj.names
          backup = backup/


   \* Tip: specify the paths to the training and validation images in the training and validation ``.txt`` files relative to ``/workspace/darknet/``

   * 
     Place your training and validation images and labels in your data folder.

   * 
     Start training on your dataset starting from our pre-trained weights in ``tiny_yolov4_license_plates.weights`` (or download it from `here <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/LPR/lp_detector/tiny_yolov4_license_plates/2021-12-23/tiny_yolov4_license_plates.weights>`_\ )

     .. code-block::

          ./darknet detector train data/obj.data ./cfg/tiny_yolov4_license_plates.cfg tiny_yolov4_license_plates.weights -map

   **NOTE:** If during training you get an error similar to

   .. code-block::

      cuDNN status Error in: file: ./src/convolutional_kernels.cu : () : line: 543 : build time: Jun 21 2022 - 20:09:28

      cuDNN Error: CUDNN_STATUS_BAD_PARAM
      Darknet error location: ./src/dark_cuda.c, cudnn_check_error, line #204
      cuDNN Error: CUDNN_STATUS_BAD_PARAM: Success

   * then please try changing `subdivisions` in the `.cfg` file (e.g., from 16 to 32).
   * For further information, please see discussion `here <https://github.com/AlexeyAB/darknet/issues/7153#issuecomment-965272028>`_.



#. | **Export to ONNX**
   | Export the model to ONNX using the following command:

   .. code-block::

       python ../pytorch-YOLOv4/demo_darknet2onnx.py ./cfg/tiny_yolov4_license_plates.cfg /path/to/trained.weights /path/to/some/image.jpg 1

----

Compile the Model using Hailo Model Zoo
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can generate an HEF file for inference on Hailo-8 from your trained ONNX model. In order to do so you need a working model-zoo environment.
Choose the model YAML from our networks configuration directory, i.e. ``hailo_model_zoo/cfg/networks/tiny_yolov4_license_plates.yaml``\ , and run compilation using the model zoo:

.. code-block::

   hailomz compile --ckpt tiny_yolov4_license_plates.onnx --calib-path /path/to/calibration/imgs/dir/ --yaml tiny_yolov4_license_plates.yaml


* ``--ckpt`` - path to your ONNX file.
* ``--calib-path`` - path to a directory with your calibration images in JPEG format
* ``--yaml`` - path to your configuration YAML file.

The model zoo will take care of adding the input normalization to be part of the model.
