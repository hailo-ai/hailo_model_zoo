=================
YOLOv3 Retraining
=================

* DEPRECATION WARNING: This Docker will be deprecated in next release.
* To learn more about yolov3 look `here <https://github.com/hailo-ai/darknet>`_

-------

Prerequisites
-------------


* docker (\ `installation instructions <https://docs.docker.com/engine/install/ubuntu/>`_\ )
* nvidia-docker2 (\ `installation instructions <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_\ )

**NOTE:**\ In case you use Hailo Software Suite docker, make sure you are doing all the following instructions outside of this docker.

Environment Preparations
------------------------


#. | Build the docker image:

   .. code-block::


      cd hailo_model_zoo/training/yolov3
      docker build --build-arg timezone=`cat /etc/timezone` -t yolov3:v0 .


   | the following optional arguments can be passed via --build-arg:

   * ``timezone`` - a string for setting up timezone. E.g. "Asia/Jerusalem"
   * ``user`` - username for a local non-root user. Defaults to 'hailo'.
   * ``group`` - default group for a local non-root user. Defaults to 'hailo'.
   * ``uid`` - user id for a local non-root user.
   * ``gid`` - group id for a local non-root user.
   * This command will build the docker image with the necessary requirements using the Dockerfile exists in this directory.

#. | Start your docker:

   .. code-block::


      docker run --name "your_docker_name" -it --gpus all --ipc=host -v /path/to/local/data/dir:/path/to/docker/data/dir yolov3:v0


   * ``docker run`` create a new docker container.
   * ``--name <your_docker_name>`` name for your container.
   * ``-it`` runs the command interactively.
   * ``--gpus all`` allows access to all GPUs.
   * ``--ipc=host`` sets the IPC mode for the container.
   * ``-v /path/to/local/data/dir:/path/to/docker/data/dir`` maps ``/path/to/local/data/dir`` from the host to the container. You can use this command multiple times to mount multiple directories.
   * ``yolov3:v0`` the name of the docker image.

Training and exporting to ONNX
------------------------------


#. Train your model: Once the docker is started, you can start training your YOLOv3.

   * Prepare your custom dataset - Follow the full instructions described `here <https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects>`_\ :

     * | Create ``data/obj.data`` with paths to your training and validation ``.txt`` files, which contain the list of the image paths\*.

       .. code-block::

          classes = 80
          train  = data/train.txt
          valid  = data/val.txt
          names = data/coco.names
          backup = backup/

       | \* Tip: specify the paths to the training and validation images in the training and validation ``.txt`` files relative to ``/workspace/darknet/``

       | Place your training/validation images and labels in your data folder and make sure you update the number of classes.

     * Labels - each image should have labels in YOLO format with corresponding txt file for each image.

   * | Start training - The following command is an example for training the yolov3.

   .. code-block::


      ./darknet detector train data/obj.data cfg/yolov3.cfg yolov3.weights -map -clear


   | Final trained weights will be available in ``backup/`` directory.


#. | Export to ONNX:
   | In order to export your trained YOLOv3 model to ONNX run the following script:

   .. code-block::


      python ../pytorch-YOLOv4/demo_darknet2onnx.py cfg/yolov3.cfg /path/to/trained.weights /path/to/some/image.jpg 1


   * The ONNX would be available in ``/workspace/darknet/``


----

Compile the Model using Hailo Model Zoo
---------------------------------------

| You can generate an HEF file for inference on Hailo device from your trained ONNX model.
| In order to do so you need a working model-zoo environment.
| Choose the corresponding YAML from our networks configuration directory, i.e. ``hailo_model_zoo/cfg/networks/yolov3_416.yaml`` (for the default YOLOv3 model).
|
| Align the corresponding alls, i.e. ``hailo_model_zoo/cfg/alls/base/yolov3_416.alls`` with the size of the calibration set using ``dataset_size=<number_of_jpgs_in_folder>`` parameter.
|
| Run compilation using the model zoo:

.. code-block::


  hailomz compile  yolov3_416 --ckpt yolov3_1_416_416.onnx  --calib-path /path/to/calibration/imgs/dir/


* | ``--ckpt`` - path to  your ONNX file.
* | ``--calib-path`` - path to a directory with your calibration images in JPEG/png format
* | The model zoo will take care of adding the input normalization to be part of the model.

.. note::
  - On your desired YOLOv3 YAML, make sure ``preprocessing.input_shape`` fits your model's resolution.
  - For TAPPAS, retrain the model with a resolution of 608x608, and on compilation use ``yolov3_gluon.yaml``.

  More details about YAML files are presented `here <../../docs/YAML.rst>`_.
