==================
Nanodet Retraining
==================

* To learn more about NanoDet look `here <https://github.com/hailo-ai/nanodet>`_

---------

Prerequisites
-------------

* docker (\ `installation instructions <https://docs.docker.com/engine/install/ubuntu/>`_\ )
* nvidia-docker2 (\ `installation instructions <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_\ )

**NOTE:**\  In case you are using the Hailo Software Suite docker, make sure to run all of the following instructions outside of that docker.


Environment Preparations
------------------------

#. | Build the docker image:

   .. code-block::


      cd hailo_model_zoo/training/nanodet
      docker build -t nanodet:v0 --build-arg timezone=`cat /etc/timezone` .


   | the following optional arguments can be passed via --build-arg:

   * ``timezone`` - a string for setting up timezone. E.g. "Asia/Jerusalem"
   * ``user`` - username for a local non-root user. Defaults to 'hailo'.
   * ``group`` - default group for a local non-root user. Defaults to 'hailo'.
   * ``uid`` - user id for a local non-root user.
   * ``gid`` - group id for a local non-root user.


#. | Start your docker:

   .. code-block::


      docker run --name "your_docker_name" -it --gpus all -u "username" --ipc=host -v /path/to/local/data/dir:/path/to/docker/data/dir  nanodet:v0


   * ``docker run`` create a new docker container.
   * ``--name <your_docker_name>`` name for your container.
   * ``-u <username>`` same username as used for building the image.
   * ``-it`` runs the command interactively.
   * ``--gpus all`` allows access to all GPUs.
   * ``--ipc=host`` sets the IPC mode for the container.
   * ``-v /path/to/local/data/dir:/path/to/docker/data/dir`` maps ``/path/to/local/data/dir`` from the host to the container. You can use this command multiple times to mount multiple directories.
   * ``nanodet:v0`` the name of the docker image.


Training and exporting to ONNX
------------------------------

#. | Prepare your data:

   | Data is expected to be in coco format. More information can be found `here <https://cocodataset.org/#format-data>`_

#. | Training:

   | Configure your model in a .yaml file. We'll use /workspace/nanodet/config/legacy_v0.x_configs/RepVGG/nanodet-RepVGG-A0_416.yml in this guide.
   | Modify the path for the dataset in the .yaml configuration file:

   .. code-block::

       data:
         train:
           name: CocoDataset
           img_path: <path-to-train-dir>
           ann_path: <path-to-annotations-file>
           ...
         val:
           name: CocoDataset
           img_path: <path-to-validation-dir>
           ann_path: <path-to-annotations-file>
           ...

   | Start training with the following commands:

   .. code-block::


      cd /workspace/nanodet
      ln -s /workspace/data/coco/ /coco
      python tools/train.py ./config/legacy_v0.x_configs/RepVGG/nanodet-RepVGG-A0_416.yml


   | In case you want to use the pretrained nanodet-RepVGG-A0_416.ckpt, which was predownloaded into your docker modify your configurationf file:

   .. code-block::

       schedule:
         load_model: ./pretrained/nanodet-RepVGG-A0_416.ckpt

   | Modifying the batch size and the number of GPUs used for training can be done also in the configuration file:

   .. code-block::

       device:
         gpu_ids: [0]
         workers_per_gpu: 1
         batchsize_per_gpu: 128

#. | Exporting to onnx

   | After training, install the ONNX and ONNXruntime packages, then export the ONNX model:

   .. code-block::


      python tools/export_onnx.py --cfg_path ./config/legacy_v0.x_configs/RepVGG/nanodet-RepVGG-A0_416.yml --model_path /workspace/nanodet/workspace/RepVGG-A0-416/model_last.ckpt


**NOTE:**\  Your trained model will be found under the following path: /workspace/nanodet/workspace/<backbone-name> /model_last.ckpt, and exported onnx will be written to /workspace/nanodet/nanodet.onnx


----

Compile the Model using Hailo Model Zoo
---------------------------------------

| You can generate an HEF file for inference on Hailo device from your trained ONNX model.
| In order to do so you need a working model-zoo environment.
| Choose the corresponding YAML from our networks configuration directory, i.e. ``hailo_model_zoo/cfg/networks/nanodet_repvgg.yaml``\ , and run compilation using the model zoo:

.. code-block::


   hailomz compile --ckpt nanodet.onnx --calib-path /path/to/calibration/imgs/dir/ --yaml path/to/nanodet_repvgg.yaml --start-node-names name1 name2 --end-node-names name1 --classes 80


* | ``--ckpt`` - path to  your ONNX file.
* | ``--calib-path`` - path to a directory with your calibration images in JPEG/png format
* | ``--yaml`` - path to your configuration YAML file.
* | ``--start-node-names`` and ``--end-node-names`` - node names for customizing parsing behavior (optional).
* | ``--classes`` - adjusting the number of classes in post-processing configuration (optional).
* | The model zoo will take care of adding the input normalization to be part of the model.

.. note::
  - On your desired YAML file, change ``preprocessing.input_shape`` if changed on retraining.

  More details about YAML files are presented `here <../../docs/YAML.rst>`_.
