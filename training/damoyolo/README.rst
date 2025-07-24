=====================
DAMO-YOLO Retraining
=====================

* To learn more about DAMO-YOLO visit the `official repository <https://github.com/hailo-ai/DAMO-YOLO>`_

----------

Prerequisites
-------------

* docker (\ `docker installation instructions <https://docs.docker.com/engine/install/ubuntu/>`_\ )
* nvidia-docker2 (\ `nvidia docker installation instructions <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_\ )


**NOTE:**\  In case you are using the Hailo Software Suite docker, make sure to run all of the following instructions outside of that docker.

Environment Preparations
------------------------


#. | Build the docker image:

   .. code-block::

      
      cd hailo_model_zoo/training/damoyolo
      docker build --build-arg timezone=`cat /etc/timezone` -t damoyolo:v0 .
      

   | the following optional arguments can be passed via --build-arg:

   * ``timezone`` - a string for setting up timezone. E.g. "Asia/Jerusalem"
   * ``user`` - username for a local non-root user. Defaults to 'hailo'.
   * ``group`` - default group for a local non-root user. Defaults to 'hailo'.
   * ``uid`` - user id for a local non-root user.
   * ``gid`` - group id for a local non-root user.

   | * This command will build the docker image with the necessary requirements using the Dockerfile exists in damoyolo directory.  


#. | Start your docker:

   .. code-block::

      
      docker run --name "your_docker_name" -it --gpus all --ipc=host -v  /path/to/local/data/dir:/path/to/docker/data/dir damoyolo:v0
      

   * ``docker run`` create a new docker container.
   * ``--name <your_docker_name>`` name for your container.
   * ``-it`` runs the command interactively.
   * ``--gpus all`` allows access to all GPUs.
   * ``--ipc=host`` sets the IPC mode for the container.
   * ``-v /path/to/local/data/dir:/path/to/docker/data/dir`` maps ``/path/to/local/data/dir`` from the host to the container. You can use this command multiple times to mount multiple directories.
   * ``damoyolo:v0`` the name of the docker image.

Training and exporting to ONNX
------------------------------


#. | Train your model:
   | Once the docker is started, you can start training your model.

   * | Prepare your custom dataset (must be coco format) - Follow the steps described `here <https://github.com/tinyvision/DAMO-YOLO/blob/master/assets/CustomDatasetTutorial.md>`_.
   * | Modify ``num_classes`` and ``class_names`` in the configuration file, for example ``damoyolo_tinynasL20_T.py``
   * | Use ``self.train.batch_size`` / ``self.train.total_epochs`` in the configuration file to modify the batch_size and number of epochs
   * | Update the symbolic link to your dataset: ln -sfn /your/coco/like/dataset/path datasets/coco
   * | Start training - The following command is an example for training a *damoyolo_tinynasL20_T* model.

     .. code-block::
  
        
         python tools/train.py -f configs/damoyolo_tinynasL20_T.py
                                 
                                 configs/damoyolo_tinynasL25_S.py
                                 configs/damoyolo_tinynasL35_M.py
                                 
        

     * ``configs/damoyolo_tinynasL20_T.py`` - configuration file of the DAMO-YOLO variant you would like to train. In order to change the number of classes make sure you update ``num_classes`` and ``class_names`` in this file.
    

#. | Export to ONNX:

   | In order to export your trained DAMO-YOLO model to ONNX run the following script:

   .. code-block::

      
      python tools/converter.py -f configs/damoyolo_tinynasL20_T.py -c /path/to/trained/model.pth --batch_size 1 --img_size 640 # export at 640x640 with batch size 1
      

----

Compile the Model using Hailo Model Zoo
---------------------------------------

| You can generate an HEF file for inference on Hailo device from your trained ONNX model.
| In order to do so you need a working model-zoo environment.
| Choose the corresponding YAML from our networks configuration directory, i.e. ``hailo_model_zoo/cfg/networks/damoyolo_tinynasL20_T.yaml``\ , and run compilation using the model zoo:

.. code-block::

   
   hailomz compile --ckpt damoyolo_tinynasL20_T.onnx --calib-path /path/to/calibration/imgs/dir/ --yaml path/to/damoyolo/variant.yaml --start-node-names name1 name2 --end-node-names name1
   

* | ``--ckpt`` - path to  your ONNX file.
* | ``--calib-path`` - path to a directory with your calibration images in JPEG/png format
* | ``--yaml`` - path to your configuration YAML file.
* | ``--start-node-names`` and ``--end-node-names`` - node names for customizing parsing behavior (optional).
* | The model zoo will take care of adding the input normalization to be part of the model.

  
  More details about YAML files are presented `here <../../docs/YAML.rst>`_.
