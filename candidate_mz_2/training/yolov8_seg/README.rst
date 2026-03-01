======================
YOLOv8-seg Retraining
======================

* To learn more about yolov8 look `here <https://github.com/hailo-ai/ultralytics>`_

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

      
      cd hailo_model_zoo/training/yolov8_seg
      docker build --build-arg timezone=`cat /etc/timezone` -t yolov8_seg:v0 .
      

   | the following optional arguments can be passed via --build-arg:

   * ``timezone`` - a string for setting up timezone. E.g. "Asia/Jerusalem"
   * ``user`` - username for a local non-root user. Defaults to 'hailo'.
   * ``group`` - default group for a local non-root user. Defaults to 'hailo'.
   * ``uid`` - user id for a local non-root user.
   * ``gid`` - group id for a local non-root user.

   | * This command will build the docker image with the necessary requirements using the Dockerfile exists in yolov8-seg directory.  


#. | Start your docker:

   .. code-block::

      
      docker run --name "your_docker_name" -it --gpus all --ipc=host -v  /path/to/local/data/dir:/path/to/docker/data/dir yolov8_seg:v0
      

   * ``docker run`` create a new docker container.
   * ``--name <your_docker_name>`` name for your container.
   * ``-it`` runs the command interactively.
   * ``--gpus all`` allows access to all GPUs.
   * ``--ipc=host`` sets the IPC mode for the container.
   * ``-v /path/to/local/data/dir:/path/to/docker/data/dir`` maps ``/path/to/local/data/dir`` from the host to the container. You can use this command multiple times to mount multiple directories.
   * ``yolov8:v0`` the name of the docker image.

Training and exporting to ONNX
------------------------------


#. | Train your model:
   | Once the docker is started, you can start training your model.

   * | Prepare your custom dataset - Follow the steps described `here <https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#1-create-dataset>`_ in order to create:

     * ``dataset.yaml`` configuration file
     * Labels - each image should have labels in YOLO format with corresponding txt file for each image.  
     * Make sure to include number of classes field in the yaml, for example: ``nc: 80``

   * | Start training - The following command is an example for training a *yolov8s-seg* model.  

     .. code-block::
  
        
        yolo segment train data=coco128-seg.yaml model=yolov8s-seg.pt name=retrain_yolov8s_seg epochs=100 batch=16
        

     * ``yolov8s-seg.pt`` - pretrained weights. The pretrained weights for *yolov8n-seg*\ , *yolov8s-seg*\ , *yolov8m-seg*\ , *yolov8l-seg* and *yolov8x-seg* will be downloaded to your working directory when running this command.
     * ``coco128-seg.yaml`` - example file for data.yaml file. Can be found at ultralytics/ultralytics/datasets.
     * ``retrain_yolov8s_seg`` - the new weights will be saved at ultralytics/runs/segment/retrain_yolov8s_seg.
     * ``epochs`` - number of epochs to run. default to 100.
     * ``batch`` - number of images per batch. default to 16.

   **NOTE:**\  more configurable parameters can be found at https://docs.ultralytics.com/modes/train/

#. | Export to ONNX:

   | In order to export your trained YOLOv8-seg model to ONNX run the following script:

   .. code-block::

      
      yolo export model=/path/to/trained/best.pt imgsz=640 format=onnx opset=11  # export at 640x640
      

   **NOTE:**\  more configurable parameters can be found at https://docs.ultralytics.com/modes/export/

----

Compile the Model using Hailo Model Zoo
---------------------------------------

| You can generate an HEF file for inference on Hailo-8 from your trained ONNX model.
| In order to do so you need a working model-zoo environment.
| Choose the corresponding YAML from our networks configuration directory, i.e. ``hailo_model_zoo/cfg/networks/yolov8s-seg.yaml``\ , and run compilation using the model zoo:  

.. code-block::

   
   hailomz compile --ckpt yolov8s-seg.onnx --calib-path /path/to/calibration/imgs/dir/ --yaml path/to/yolov8s-seg.yaml --start-node-names name1 name2 --end-node-names name1
   

* | ``--ckpt`` - path to  your ONNX file.
* | ``--calib-path`` - path to a directory with your calibration images in JPEG/png format
* | ``--yaml`` - path to your configuration YAML file.
* | ``--start-node-names`` and ``--end-node-names`` - node names for customizing parsing behavior (optional).
* | The model zoo will take care of adding the input normalization to be part of the model.

.. note::
  - Make sure to also update ``preprocessing.input_shape`` field on `yolo.yaml <https://github.com/hailo-ai/hailo_model_zoo/blob/master/hailo_model_zoo/cfg/base/yolo.yaml>`_, if it was changed on retraining.
  
  More details about YAML files are presented `here <../../docs/YAML.rst>`_.