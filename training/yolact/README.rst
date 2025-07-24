=================
YOLACT Retraining
=================

* To learn more about Yolact look `here <https://github.com/hailo-ai/yolact/tree/Model-Zoo-1.5>`_

----------

Prerequisites
-------------

* docker (\ `installation instructions <https://docs.docker.com/engine/install/ubuntu/>`_\ )
* nvidia-docker2 (\ `installation instructions <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_\ )

**NOTE:**\  In case you use Hailo Software Suite docker, make sure you are doing all the following instructions outside of this docker.


Environment Preparations
------------------------

#. | Build the docker image:

   .. code-block::


      cd hailo_model_zoo/training/yolact
      docker build --build-arg timezone=`cat /etc/timezone` -t yolact:v0 .




   | the following optional arguments can be passed via --build-arg:

   * ``timezone`` - a string for setting up timezone. E.g. "Asia/Jerusalem"
   * ``user`` - username for a local non-root user. Defaults to 'hailo'.
   * ``group`` - default group for a local non-root user. Defaults to 'hailo'.
   * ``uid`` - user id for a local non-root user.
   * ``gid`` - group id for a local non-root user.
   * This command will build the docker image with the necessary requirements using the Dockerfile exists in the yolact directory.


#. | Start your docker:

   .. code-block::


      docker run --name "your_docker_name" -it --gpus all --ipc=host -v /path/to/local/data/dir:/path/to/docker/data/dir  yolact:v0


   * ``docker run`` create a new docker container.
   * ``--name <your_docker_name>`` name for your container.
   * ``-it`` runs the command interactively.
   * ``--gpus all`` allows access to all GPUs.
   * ``--ipc=host`` sets the IPC mode for the container.
   * ``-v /path/to/local/data/dir:/path/to/docker/data/dir`` maps ``/path/to/local/data/dir`` from the host to the container. You can use this command multiple times to mount multiple directories.
   * ``yolact:v0`` the name of the docker image.

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
               |-- images
                   |-- train2017
                       |-- *.jpg
                   |-- val2017
                       |-- *.jpg
                   |-- test2017
                   `   |-- *.jpg

   | For training on custom datasets see `here <https://github.com/hailo-ai/yolact/tree/Model-Zoo-1.5#custom-datasets>`_

#. | Train your model:

   | Once your dataset is prepared, create a soft link to it under the yolact/data work directory, then you can start training your model:

   .. code-block::


      cd /workspace/yolact
      ln -s /workspace/data/coco data/coco
      python train.py --config=yolact_regnetx_800MF_config


   * ``yolact_regnetx_800MF_config`` - configuration using the regnetx_800MF backbone.

#. | Export to ONNX: In order to export your trained YOLACT model to ONNX run the following script:

   .. code-block::


      python export.py --config=yolact_regnetx_800MF_config --trained_model=path/to/trained/model --export_path=path/to/export/model.onnx


   * ``--config`` - same configuration used for training.
   * ``--trained_model`` - path to the weights produced by the training process.
   * ``--export_path`` - path to export the ONNX file to. Include the ``.onnx`` extension.

----

Compile the Model using Hailo Model Zoo
---------------------------------------

You can generate an HEF file for inference on Hailo device from your trained ONNX model.
In order to do so you need a working model-zoo environment.
Choose the corresponding YAML from our networks configuration directory, i.e. ``hailo_model_zoo/cfg/networks/yolact.yaml``\ , and run compilation using the model zoo:

.. code-block::


   hailomz compile yolact --ckpt yolact.onnx --calib-path /path/to/calibration/imgs/dir/ --yaml path/to/yolact_regnetx_800mf_20classes.yaml --start-node-names name1 name2 --end-node-names name1


* | ``--ckpt`` - path to  your ONNX file.
* | ``--calib-path`` - path to a directory with your calibration images in JPEG/png format
* | ``--yaml`` - path to your configuration YAML file.
* | ``--start-node-names`` and ``--end-node-names`` - node names for customizing parsing behavior (optional).
* | The model zoo will take care of adding the input normalization to be part of the model.

.. note::
  - The `yolact_regnetx_800mf_20classes.yaml<https://github.com/hailo-ai/hailo_model_zoo/blob/master/hailo_model_zoo/cfg/networks/yolact_regnetx_800mf_20classes.yaml>`_
    is an example yaml where some of the classes (out of 80) were removed. If you wish to change the number of classes, the easiest way is to retrain with the exact number
    of classes, erase the ``channels_remove`` section (lines 18 to 437).

  More details about YAML files are presented `here <../../docs/YAML.rst>`_.
