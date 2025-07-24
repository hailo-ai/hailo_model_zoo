========================
YOLOX-hailo Retraining
========================

* To learn more about yolox-hailo look `here <https://github.com/hailo-ai/YOLOX/tree/yolox-hailo-model>`_

----------------------------------------------------------------------------------------

Prerequisites
-------------


* docker (\ `installation instructions <https://docs.docker.com/engine/install/ubuntu/>`_\ )
* nvidia-docker2 (\ `nvidia installation instructions <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_\ )

     **NOTE:**\  In case you use Hailo Software Suite docker, make sure you are doing all the following instructions outside of this docker.


Environment Preparations
------------------------

#. | Build the docker image:

   .. code-block::


      cd hailo_model_zoo/training/yolox_hailo
      docker build --build-arg timezone=`cat /etc/timezone` -t yolox_hailo:v0 .


   | The following optional arguments can be passed via --build-arg:

   * ``timezone`` - a string for setting up   timezone. E.g. "Asia/Jerusalem"
   * ``user`` - username for a local non-root   user. Defaults to 'hailo'.
   * ``group`` - default group for a local   non-root user. Defaults to 'hailo'.
   * ``uid`` - user id for a local non-root user.
   * ``gid`` - group id for a local non-root user.

#. | Start your docker:

   .. code-block::


      docker run --name "your_docker_name" -it --gpus all -u "username" --ipc=host -v /path/to/local/data/dir:/path/to/docker/data/dir yolox_hailo:v0


   * ``docker run`` create a new docker container.
   * ``--name <your_docker_name>`` name for your container.
   * ``-it`` runs the command interactively.
   * ``--gpus all`` allows access to all GPUs.
   * ``--ipc=host`` sets the IPC mode for the container.
   * ``-v /path/to/local/data/dir:/path/to/docker/data/dir`` maps ``/path/to/local/data/dir`` from the host to the container. You can use this command multiple times to mount multiple directories.
   * ``yolox_hailo:v0`` the name of the docker image.

Training and exporting to ONNX
------------------------------

#. | Prepare your data:

   | You can use coco format, which is already supported for training on your own custom dataset. More information can be found `in this link <https://github.com/hailo-ai/YOLOX/blob/main/docs/train_custom_data.md>`_

#. | Training:

   | Start training with the following command:

   .. code-block::


      python tools/train.py -n yolox_hailo -d 1 -b 8 -expn train1 --fp16


   * -f: experiment description file
   * -d: number of gpu devices
   * -b: total batch size, the recommended number for -b is num-gpu * 8


#. | Exporting to onnx:

   | After finishing training run the following command:

   .. code-block::


      python tools/export_onnx.py -n yolox_hailo --output-name yolox_hailo.onnx -o 11 -c yolox_hailo_outputs/train1/best_ckpt.pth



 **NOTE:**\  Your trained model will be found under the following path: ``/workspace/YOLOX/yolox_hailo_outputs/train1/``\ , and the exported onnx will be written to ``/workspace/YOLOX/yolox_hailo.onnx``


----

Compile the Model using Hailo Model Zoo
---------------------------------------

You can generate an HEF file for inference on Hailo device from your trained ONNX model.
In order to do so you need a working model-zoo environment.
Choose the corresponding YAML from our networks configuration directory, i.e. ``hailo_model_zoo/cfg/networks/yolox_hailo_pp.yaml``\ , and run compilation using the model zoo:

.. code-block::


   hailomz compile --ckpt yolox_hailo.onnx --calib-path /path/to/calibration/imgs/dir/ --yaml path/to/yolox_hailo_pp_pruned50.yaml --start-node-names name1 name2 --end-node-names name1


* | ``--ckpt`` - path to  your ONNX file.
* | ``--calib-path`` - path to a directory with your calibration images in JPEG/png format
* | ``--yaml`` - path to your configuration YAML file.
* | ``--start-node-names`` and ``--end-node-names`` - node names for customizing parsing behavior (optional).
* | The model zoo will take care of adding the input normalization to be part of the model.

.. note::
  More details about YAML files are `presented here <../../docs/YAML.rst>`_.
