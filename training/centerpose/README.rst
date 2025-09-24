=====================
Centerpose Retraining
=====================

* To learn more about CenterPose look `here <https://github.com/hailo-ai/centerpose>`_

----

Prerequisites
-------------


* docker (\ `installation instructions <https://docs.docker.com/engine/install/ubuntu/>`_\ )
* nvidia-docker2 (\ `installation instructions <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_\ )

**NOTE:**\  In case you are using the Hailo Software Suite docker, make sure to run all of the following instructions outside of that docker.

Environment Preparations
------------------------


#. | Build the docker image:
 
   .. code-block::

      
      cd hailo_model_zoo/training/centerpose
      docker build -t centerpose:v0 --build-arg timezone=`cat /etc/timezone` .
      

   | the following optional arguments can be passed via --build-arg:
 
   * ``timezone`` - a string for setting up timezone. E.g. "Asia/Jerusalem"
   * ``user`` - username for a local non-root user. Defaults to 'hailo'.
   * ``group`` - default group for a local non-root user. Defaults to 'hailo'.
   * ``uid`` - user id for a local non-root user.
   * ``gid`` - group id for a local non-root user.
  
 
#. | Start your docker:

   .. code-block::

      
      docker run --name "your_docker_name" -it --gpus all -u "username" --ipc=host -v /path/to/local/data/dir:/path/to/docker/data/dir  centerpose:v0
      

   * ``docker run`` create a new docker container.
   * ``--name <your_docker_name>`` name for your container.
   * ``-it`` runs the command interactively.
   * ``--gpus all`` allows access to all GPUs.
   * ``--ipc=host`` sets the IPC mode for the container.
   * ``-v /path/to/local/data/dir:/path/to/docker/data/dir`` maps ``/path/to/local/data/dir`` from the host to the container. You can use this command multiple times to mount multiple directories.
   * ``centerpose:v0`` the name of the docker image.

Training and exporting to ONNX
------------------------------

#. | Prepare your data:

   | Data is expected to be in coco format, and by default should be in /workspace/data/<dataset_name>.
   | The expected structure is as follows:

   .. code-block::

       /workspace
       |-- data
       `-- |-- coco
           `-- |-- annotations
               |   |-- instances_train2017.json
               |   |-- instances_val2017.json
               |   |-- person_keypoints_train2017.json
               |   |-- person_keypoints_val2017.json
               |   |-- image_info_test-dev2017.json
           `-- |-- images
               |---|-- train2017
               |---|---|-- *.jpg
               |---|-- val2017
               |---|---|-- *.jpg
               |---|-- test2017
               `---|---|-- *.jpg

   | The path for the dataset can be configured in the .yaml file, e.g. centerpose/experiments/regnet_fpn.yaml

#. | Training:

   | Configure your model in a .yaml file. We'll use /workspace/centerpose/experiments/regnet_fpn.yaml in this guide.
   | start training with the following command:
   
   .. code-block::

      
      cd /workspace/centerpose/tools
      python -m torch.distributed.launch --nproc_per_node 4 train.py --cfg ../experiments/regnet_fpn.yaml
      
  
   | Where 4 is the number of GPUs used for training.
   | If using a different number, update both this and the used gpus in the .yaml configuration.

#. | Exporting to onnx After training, run the following command:

   .. code-block::

      
      cd /workspace/centerpose/tools
      python export.py --cfg ../experiments/regnet_fpn.yaml --TESTMODEL /workspace/out/regnet1_6/model_best.pth
      

----

Compile the Model using Hailo Model Zoo
---------------------------------------

You can generate an HEF file for inference on Hailo-8 from your trained ONNX model.
In order to do so you need a working model-zoo environment.
Choose the corresponding YAML from our networks configuration directory, i.e. ``hailo_model_zoo/cfg/networks/centerpose_regnetx_1.6gf_fpn.yaml``\ , and run compilation using the model zoo:  

.. code-block::

   
   hailomz compile --ckpt coco_pose_regnet1.6_fpn.onnx --calib-path /path/to/calibration/imgs/dir/ --yaml path/to/centerpose_regnetx_1.6gf_fpn.yaml --start-node-names name1 name2 --end-node-names name1
   

* | ``--ckpt`` - path to  your ONNX file.
* | ``--calib-path`` - path to a directory with your calibration images in JPEG/png format
* | ``--yaml`` - path to your configuration YAML file.
* | ``--start-node-names`` and ``--end-node-names`` - node names for customizing parsing behavior (optional).
* | The model zoo will take care of adding the input normalization to be part of the model.

.. note::
  More details about YAML files are presented `here <../../docs/YAML.rst>`_.