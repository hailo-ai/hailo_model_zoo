==================
Nanodet Retraining
==================

* To learn more about NanoDet look `here <https://github.com/hailo-ai/nanodet>`_

---------

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
      cd <span val="dockerfile_path">hailo_model_zoo/training/nanodet</span>
      docker build -t nanodet:v0 --build-arg timezone=`cat /etc/timezone` .
      </code></pre>

   | the following optional arguments can be passed via --build-arg:

   * ``timezone`` - a string for setting up timezone. E.g. "Asia/Jerusalem"
   * ``user`` - username for a local non-root user. Defaults to 'hailo'.
   * ``group`` - default group for a local non-root user. Defaults to 'hailo'.co 
   * ``uid`` - user id for a local non-root user.
   * ``gid`` - group id for a local non-root user.
  

#. | Start your docker:

   .. raw:: html
      :name:validation

      <code stage="docker_run">
      docker run <span val="replace_none">--name "your_docker_name"</span> -it --gpus all <span val="replace_none">-u "username"</span> --ipc=host -v <span val="local_vol_path">/path/to/local/data/dir</span>:<span val="docker_vol_path">/path/to/docker/data/dir</span>  nanodet:v0
      </code>

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

   .. raw:: html
      :name:validation

      <pre><code stage="retrain">
      <span val="replace_none">cd /workspace/nanodet</span>
      ln -s /workspace/data/coco/ /coco
      python tools/train.py ./config/legacy_v0.x_configs/RepVGG/nanodet-RepVGG-A0_416.yml
      </code></pre>
   
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

   .. raw:: html
      :name:validation

      <pre><code stage="export">
      pip install onnx onnxruntime
      python tools/export_onnx.py --cfg_path ./config/legacy_v0.x_configs/RepVGG/nanodet-RepVGG-A0_416.yml --model_path /workspace/nanodet/workspace/RepVGG-A0-416/model_last.ckpt
      </code></pre>

**NOTE:**  Your trained model will be found under the following path: /workspace/nanodet/workspace/<backbone-name> /model_last.ckpt, and exported onnx will be written to /workspace/nanodet/nanodet.onnx
 

----

Compile the Model using Hailo Model Zoo
---------------------------------------

| You can generate an HEF file for inference on Hailo-8 from your trained ONNX model.
| In order to do so you need a working model-zoo environment.
| Choose the corresponding YAML from our networks configuration directory, i.e. ``hailo_model_zoo/cfg/networks/nanodet_repvgg.yaml``\ , and run compilation using the model zoo:  

.. raw:: html
   :name:validation

   <code stage="compile">
   hailomz compile --ckpt <span val="local_path_to_onnx">nanodet.onnx</span> --calib-path <span val="calib_set_path">/path/to/calibration/imgs/dir/</span> --yaml <span val="yaml_file_path">path/to/nanodet_repvgg.yaml</span>
   </code>

* | ``--ckpt`` - path to  your ONNX file.
* | ``--calib-path`` - path to a directory with your calibration images in JPEG/png format
* | ``--yaml`` - path to your configuration YAML file.
* | The model zoo will take care of adding the input normalization to be part of the model.

.. note::
  - On your desired YAML file, change ``preprocessing.input_shape`` if changed on retraining.
  
  More details about YAML files are presented `here <../../docs/YAML.rst>`_.
