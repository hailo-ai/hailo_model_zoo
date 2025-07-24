=================
YOLOv5 Retraining
=================

* To learn more about yolov5 look `here <https://github.com/hailo-ai/yolov5>`_

----------

Prerequisites
-------------

* docker (\ `installation instructions <https://docs.docker.com/engine/install/ubuntu/>`_\ )
* nvidia-docker2 (\ `installation instructions <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_\ )


**NOTE:**\  In case you are using the Hailo Software Suite docker, make sure to run all of the following instructions outside of that docker.

Environment Preparations
------------------------


#. | Build the docker image:

   .. code-block::


      cd hailo_model_zoo/training/yolov5
      docker build --build-arg timezone=`cat /etc/timezone` -t yolov5:v0 .


   | the following optional arguments can be passed via --build-arg:

   * ``timezone`` - a string for setting up timezone. E.g. "Asia/Jerusalem"
   * ``user`` - username for a local non-root user. Defaults to 'hailo'.
   * ``group`` - default group for a local non-root user. Defaults to 'hailo'.
   * ``uid`` - user id for a local non-root user.
   * ``gid`` - group id for a local non-root user.

   | * This command will build the docker image with the necessary requirements using the Dockerfile exists in yolov5 directory.


#. | Start your docker:

   .. code-block::


      docker run --name "your_docker_name" -it --gpus all --ipc=host -v  /path/to/local/data/dir:/path/to/docker/data/dir yolov5:v0


   * ``docker run`` create a new docker container.
   * ``--name <your_docker_name>`` name for your container.
   * ``-it`` runs the command interactively.
   * ``--gpus all`` allows access to all GPUs.
   * ``--ipc=host`` sets the IPC mode for the container.
   * ``-v /path/to/local/data/dir:/path/to/docker/data/dir`` maps ``/path/to/local/data/dir`` from the host to the container. You can use this command multiple times to mount multiple directories.
   * ``yolov5:v0`` the name of the docker image.

Training and exporting to ONNX
------------------------------


#. | Train your model:
   | Once the docker is started, you can start training your model.

   * | Prepare your custom dataset - Follow the steps described `here <https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#1-create-dataset>`_ in order to create:

     * ``dataset.yaml`` configuration file
     * Labels - each image should have labels in YOLO format with corresponding txt file for each image.
     * Make sure to include number of classes field in the yaml, for example: ``nc: 80``

   * | Start training - The following command is an example for training a *yolov5s* model.

     .. code-block::


        python train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov5s.pt --cfg models/yolov5s.yaml


     * ``yolov5s.pt`` - pretrained weights. You can find the pretrained weights for *yolov5s*\ , *yolov5m*\ , *yolov5l*\ , *yolov5x* and *yolov5m_wo_spp* in your working directory.
     * ``models/yolov5s.yaml`` - configuration file of the yolov5 variant you would like to train. In order to change the number of classes make sure you update this file.

     | **NOTE:**\  We recommend to use *yolov5m_wo_spp* for best performance on Hailo-8

#. | Export to ONNX:

   | In order to export your trained YOLOv5 model to ONNX run the following script:

   .. code-block::


      python models/export.py --weights /path/to/trained/model.pt --img 640 --batch 1  # export at 640x640 with batch size 1


----

Compile the Model using Hailo Model Zoo
---------------------------------------

| You can generate an HEF file for inference on Hailo device from your trained ONNX model.
| In order to do so you need a working model-zoo environment.
| Choose the corresponding YAML from our networks configuration directory, i.e. ``hailo_model_zoo/cfg/networks/yolov5s.yaml``\ , and run compilation using the model zoo:

.. code-block::


   hailomz compile --ckpt yolov5s.onnx --calib-path /path/to/calibration/imgs/dir/ --yaml path/to/yolov5s.yaml --start-node-names name1 name2 --end-node-names name1 --classes 80


* | ``--ckpt`` - path to  your ONNX file.
* | ``--calib-path`` - path to a directory with your calibration images in JPEG/png format
* | ``--yaml`` - path to your configuration YAML file.
* | ``--start-node-names`` and ``--end-node-names`` - node names for customizing parsing behavior (optional).
* | ``--classes`` - adjusting the number of classes in post-processing configuration (optional).
* | The model zoo will take care of adding the input normalization to be part of the model.

.. note::
  - Make sure to also update ``preprocessing.input_shape`` field on `yolo.yaml <https://github.com/hailo-ai/hailo_model_zoo/blob/master/hailo_model_zoo/cfg/base/yolo.yaml>`_, if it was changed on retraining.

  More details about YAML files are presented `here <../../docs/YAML.rst>`_.

Anchors Extraction
------------------

| The training flow will automatically try to find more fitting anchors values then the default anchors. In our TAPPAS environment we use the default anchors, but you should be aware that the resulted anchors might be different.
| The model anchors can be retrieved from the trained model using the following snippet:

.. code-block::


   m = torch.load("last.pt")["model"]
   detect = list(m.children())[0][-1]
   print(detect.anchor_grid)

