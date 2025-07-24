================
YOLOX Retraining
================

* To learn more about yolox look `here <https://github.com/hailo-ai/YOLOX>`_

----------------------------------------------------------------------------------------

Prerequisites
-------------


* docker (\ `installation instructions <https://docs.docker.com/engine/install/ubuntu/>`_\ )
* nvidia-docker2 (\ `installation instructions <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_\ )

     **NOTE:**\  In case you use Hailo Software Suite docker, make sure you are doing all the following instructions outside of this docker.


Environment Preparations
------------------------

#. | Build the docker image:

   .. code-block::


      cd hailo_model_zoo/training/yolox
      docker build --build-arg timezone=`cat /etc/timezone` -t yolox:v0 .


   | the following optional arguments can be   passed via --build-arg:

   * ``timezone`` - a string for setting up   timezone. E.g. "Asia/Jerusalem"
   * ``user`` - username for a local non-root   user. Defaults to 'hailo'.
   * ``group`` - default group for a local   non-root user. Defaults to 'hailo'.
   * ``uid`` - user id for a local non-root user.
   * ``gid`` - group id for a local non-root user.

#. | Start your docker:

   .. code-block::


      docker run --name "your_docker_name" -it --gpus all -u "username" --ipc=host -v /path/to/local/data/dir:/path/to/docker/data/dir yolox:v0


   * ``docker run`` create a new docker container.
   * ``--name <your_docker_name>`` name for your container.
   * ``-it`` runs the command interactively.
   * ``--gpus all`` allows access to all GPUs.
   * ``--ipc=host`` sets the IPC mode for the container.
   * ``-v /path/to/local/data/dir:/path/to/docker/data/dir`` maps ``/path/to/local/data/dir`` from the host to the container. You can use this command multiple times to mount multiple directories.
   * ``yolox:v0`` the name of the docker image.

Training and exporting to ONNX
------------------------------

#. | Prepare your data:

   | You can use coco format, which is already supported for training on your own custom dataset. More information can be found `here <https://github.com/hailo-ai/YOLOX/blob/main/docs/train_custom_data.md>`_

#. | Training:

   | Start training with the following command:

   .. code-block::


      python tools/train.py -f exps/default/yolox_s_leaky.py -d 8 -b 64 -c yolox_s.pth

                              exps/default/yolox_m_leaky.py
                              exps/default/yolox_l_leaky.py
                              exps/default/yolox_x_leaky.py
                              exps/default/yolox_s_wide_leaky.py





   * -f: experiment description file
   * -d: number of gpu devices
   * -b: total batch size, the recommended number for -b is num-gpu * 8
   * -c: path to pretrained weights which can be found in your working directory

     .. code-block::

        |_ yolox_s.pth
        |_ yolox_m.pth
        |_ yolox_l.pth
        |_ yolox_x.pth

#. | Exporting to onnx:

   | After finishing training run the following command:

   .. code-block::


      python tools/export_onnx.py --output-name yolox_s_leaky.onnx -f ./exps/default/yolox_s_leaky.py -c YOLOX_outputs/yolox_s_leaky/best_ckpt.pth



 **NOTE:**\  Your trained model will be found under the following path: ``/workspace/YOLOX/YOLOX_outputs/yolox_s_leaky/``\ , and the exported onnx will be written to ``/workspace/YOLOX/yolox_s_leaky.onnx``


----

Compile the Model using Hailo Model Zoo
---------------------------------------

You can generate an HEF file for inference on Hailo device from your trained ONNX model.
In order to do so you need a working model-zoo environment.
Choose the corresponding YAML from our networks configuration directory, i.e. ``hailo_model_zoo/cfg/networks/yolox_s_leaky.yaml``\ , and run compilation using the model zoo:

.. code-block::


   hailomz compile --ckpt yolox_s_leaky.onnx --calib-path /path/to/calibration/imgs/dir/ --yaml path/to/yolox_s_leaky.yaml --start-node-names name1 name2 --end-node-names name1 --classes 80


* | ``--ckpt`` - path to  your ONNX file.
* | ``--calib-path`` - path to a directory with your calibration images in JPEG/png format
* | ``--yaml`` - path to your configuration YAML file.
* | ``--start-node-names`` and ``--end-node-names`` - node names for customizing parsing behavior (optional).
* | ``--classes`` - adjusting the number of classes in post-processing configuration (optional).
* | The model zoo will take care of adding the input normalization to be part of the model.

.. note::
  More details about YAML files are presented `here <../../docs/YAML.rst>`_.
