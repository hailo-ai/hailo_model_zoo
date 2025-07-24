==============
FCN Retraining
==============

* To learn more about FCN look `here <https://github.com/hailo-ai/mmsegmentation>`_

-------

Prerequisites
-------------

* docker (\ `installation instructions <https://docs.docker.com/engine/install/ubuntu/>`_\ )
* nvidia-docker2 (\ `installation instructions <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_\ )

**NOTE:**\  In case you are using the Hailo Software Suite docker, make sure to run all of the following instructions outside of that docker.


Environment Preparations
------------------------

#. | Build the docker image:

   .. code-block::

      
      cd hailo_model_zoo/training/fcn
      docker build -t fcn:v0 --build-arg timezone=`cat /etc/timezone` .
      

   | the following optional arguments can be passed via --build-arg:

   * ``timezone`` - a string for setting up timezone. E.g. "Asia/Jerusalem"
   * ``user`` - username for a local non-root user. Defaults to 'hailo'.
   * ``group`` - default group for a local non-root user. Defaults to 'hailo'.
   * ``uid`` - user id for a local non-root user.
   * ``gid`` - group id for a local non-root user.

#. | Start your docker:
   
   .. code-block::

      
      docker run --name "your_docker_name" -it --gpus all  -u "username" --ipc=host -v /path/to/local/data/dir:/path/to/docker/data/dir  fcn:v0
      

   * ``docker run`` create a new docker container.
   * ``--name <docker-name>`` name for your container.
   * ``-u <username>`` same username as used for building the image.
   * ``-it`` runs the command interactively.
   * ``--gpus all`` allows access to all GPUs.
   * ``--ipc=host`` sets the IPC mode for the container.
   * ``-v /path/to/local/data/dir:/path/to/docker/data/dir`` maps ``/path/to/local/data/dir`` from the host to the container. You can use this command multiple times to mount multiple directories.
   * ``fcn:v0`` the name of the docker image.

Training and exporting to ONNX
------------------------------


#. | Prepare your data:

   | Data is expected to be in coco format, and by default should be in /workspace/data/<dataset_name>.
   | The expected structure is as follows:

   .. code-block::

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

   | more information can be found `here <https://github.com/hailo-ai/mmsegmentation/blob/master/docs/en/dataset_prepare.md#cityscapes>`_


#. | Training:
 
   | Configure your model in a .py file. We'll use /workspace/mmsegmentation/configs/fcn/fcn8_r18_hailo.py in this guide.
   | start training with the following command:

   .. code-block::

      
      cd /workspace/mmsegmentation
      ./tools/dist_train.sh configs/fcn/fcn8_r18_hailo.py 2
      

   | Where 2 is the number of GPUs used for training.

#. | Exporting to onnx

   | After training, run the following command:

   .. code-block::

      
      cd /workspace/mmsegmentation
      python ./tools/pytorch2onnx.py configs/fcn/fcn8_r18_hailo.py --checkpoint ./work_dirs/fcn8_r18_hailo/iter_59520.pth --shape 1024 1920 --out_name fcn.onnx
      


----

Compile the Model using Hailo Model Zoo
---------------------------------------

| You can generate an HEF file for inference on Hailo device from your trained ONNX model.
| In order to do so you need a working model-zoo environment.
| Choose the corresponding YAML from our networks configuration directory, i.e. ``hailo_model_zoo/cfg/networks/fcn8_resnet_v1_18.yaml``\ , and run compilation using the model zoo:  

.. code-block::

   
   hailomz compile --ckpt fcn.onnx --calib-path /path/to/calibration/imgs/dir/ --yaml path/to/fcn8_resnet_v1_18.yaml --start-node-names name1 name2 --end-node-names name1
   


* | ``--ckpt`` - path to  your ONNX file.
* | ``--calib-path`` - path to a directory with your calibration images in JPEG/png format
* | ``--yaml`` - path to your configuration YAML file.
* | ``--start-node-names`` and ``--end-node-names`` - node names for customizing parsing behavior (optional).
* | The model zoo will take care of adding the input normalization to be part of the model.

.. note::
  More details about YAML files are presented `here <../../docs/YAML.rst>`_.
