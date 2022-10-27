=================
YOLOv5 Retraining
=================

* To learn more about yolov5 look `here <https://github.com/hailo-ai/yolov5>`_

----------

Prerequisites
-------------

* docker (\ `installation instructions <https://docs.docker.com/engine/install/ubuntu/>`_\ )
* nvidia-docker2 (\ `installation instructions <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_\ )


**NOTE:**  In case you are using the Hailo Software Suite docker, make sure to run all of the following instructions outside of that docker.

Environment Preparations
------------------------


#. | Build the docker image:

   .. raw:: html
      :name:validation

      <pre><code stage="docker_build">
      cd <span val="dockerfile_path">hailo_model_zoo/training/yolov5</span>
      docker build --build-arg timezone=`cat /etc/timezone` -t yolov5:v0 .
      </code></pre>

   | the following optional arguments can be passed via --build-arg:

   * ``timezone`` - a string for setting up timezone. E.g. "Asia/Jerusalem"
   * ``user`` - username for a local non-root user. Defaults to 'hailo'.
   * ``group`` - default group for a local non-root user. Defaults to 'hailo'.
   * ``uid`` - user id for a local non-root user.
   * ``gid`` - group id for a local non-root user.

   | * This command will build the docker image with the necessary requirements using the Dockerfile exists in yolov5 directory.  


#. | Start your docker:

   .. raw:: html
      :name:validation

      <code stage="docker_run">
      docker run <span val="replace_none">--name "your_docker_name"</span> -it --gpus all --ipc=host -v <span val="local_vol_path"> /path/to/local/data/dir</span>:<span val="docker_vol_path">/path/to/docker/data/dir</span> yolov5:v0
      </code>

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

   * | Start training - The following command is an example for training a *yolov5s* model.  

     .. raw:: html
        :name:validation
  
        <code stage="retrain">
        python train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov5s.pt --cfg models/yolov5s.yaml
        </code>

     * ``yolov5s.pt`` - pretrained weights. You can find the pretrained weights for *yolov5s*\ , *yolov5m*\ , *yolov5l*\ , *yolov5x* and *yolov5m_wo_spp* in your working directory.
     * ``models/yolov5s.yaml`` - configuration file of the yolov5 variant you would like to train. In order to change the number of classes make sure you update this file.
    
     | **NOTE:**\ We recommend to use *yolov5m_wo_spp* for best performance on Hailo-8

#. | Export to ONNX:

   | In order to export your trained YOLOv5 model to ONNX run the following script:

   .. raw:: html
      :name:validation

      <code stage="export">
      python models/export.py --weights <span val="docker_pretrained_path">/path/to/trained/model.pt</span> --img 640 --batch 1  # export at 640x640 with batch size 1
      </code>

----

Compile the Model using Hailo Model Zoo
---------------------------------------

| You can generate an HEF file for inference on Hailo-8 from your trained ONNX model.
| In order to do so you need a working model-zoo environment.
| Choose the corresponding YAML from our networks configuration directory, i.e. ``hailo_model_zoo/cfg/networks/yolov5s.yaml``\ , and run compilation using the model zoo:  

.. raw:: html
   :name:validation

   <code stage="compile">
   hailomz compile --ckpt <span val="local_path_to_onnx">yolov5s.onnx</span> --calib-path <span val="calib_set_path">/path/to/calibration/imgs/dir/</span> --yaml <span val="yaml_file_path">path/to/yolov5s.yaml</span>
   </code>

* | ``--ckpt`` - path to  your ONNX file.
* | ``--calib-path`` - path to a directory with your calibration images in JPEG/png format
* | ``--yaml`` - path to your configuration YAML file.
* | The model zoo will take care of adding the input normalization to be part of the model.

.. note::
  - Make sure to also update ``preprocessing.input_shape`` field on 
    `yolo.yaml <https://github.com/hailo-ai/hailo_model_zoo/blob/master/hailo_model_zoo/cfg/base/yolo.yaml>`_, if it was changed on retraining.
  
  More details about YAML files are presented `here <../../docs/YAML.rst>`_.