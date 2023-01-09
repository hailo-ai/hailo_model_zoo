================
Arface Retraining
================

* To learn more about arcface look `here <https://github.com/hailo-ai/insightface/tree/master/recognition/arcface_torch>`_

----------------------------------------------------------------------------------------

Prerequisites
-------------


* docker (\ `installation instructions <https://docs.docker.com/engine/install/ubuntu/>`_\ )
* nvidia-docker2 (\ `installation instructions <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_\ )

     **NOTE:**\  In case you use Hailo Software Suite docker, make sure you are doing all the following instructions outside of this docker.


Environment Preparations
------------------------

#. | Build the docker image:

   .. raw:: html
      :name:validation

      <pre><code stage="docker_build">
      cd <span val="dockerfile_path">hailo_model_zoo/training/arcface</span>
      docker build --build-arg timezone=`cat /etc/timezone` -t arcface:v0 .
      </code></pre>

   | the following optional arguments can be   passed via --build-arg:

   * ``timezone`` - a string for setting up   timezone. E.g. "Asia/Jerusalem"
   * ``user`` - username for a local non-root   user. Defaults to 'hailo'.
   * ``group`` - default group for a local   non-root user. Defaults to 'hailo'.
   * ``uid`` - user id for a local non-root user.
   * ``gid`` - group id for a local non-root user.

#. | Start your docker:

   .. raw:: html
      :name:validation

      <code stage="docker_run">
      docker run <span val="replace_none">--name "your_docker_name"</span> -it --gpus all <span val="replace_none">-u "username"</span> --ipc=host -v <span val="local_vol_path">/path/to/local/data/dir</span>:<span val="docker_vol_path">/path/to/docker/data/dir</span> arcface:v0
      </code>

   * ``docker run`` create a new docker container.
   * ``--name <your_docker_name>`` name for your container.
   * ``-it`` runs the command interactively.
   * ``--gpus all`` allows access to all GPUs.
   * ``--ipc=host`` sets the IPC mode for the container.
   * ``-v /path/to/local/data/dir:/path/to/docker/data/dir`` maps ``/path/to/local/data/dir`` from the host to the container. You can use this command multiple times to mount multiple directories.
   * ``arcface:v0`` the name of the docker image.

Training and exporting to ONNX
------------------------------

#. | Prepare your data:

   | For more information on obtraining datasets `here <https://github.com/hailo-ai/insightface/tree/develop/recognition/arcface_torch#download-datasets-or-prepare-datasets>`_
   | The repository supports the following formats:

   #. | ImageFolder dataset - each class (person) has its own directory
      | Validation data is packed .bin files

      .. code-block::

         data_dir/
         ├── agedb_30.bin
         ├── cfp_fp.bin
         ├── lfw.bin
         ├── person0/
         ├── person1/
         ├── ...
         └── personlast/

   #. | MxNetRecord - train.rec and train.idx files. This is the format of insightface datasets.
      | Validation data is packed .bin files
   
      .. code-block::

         data_dir/
         ├── agedb_30.bin
         ├── cfp_fp.bin
         ├── lfw.bin
         ├── train.idx
         └── train.rec

#. | Training:

   | Start training with the following command:

   .. raw:: html
      :name:validation

      <code stage="retrain">
      python -m torch.distributed.launch --nproc_per_node=<span val="gpu_num">2</span> --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12581 train_v2.py <span val="cfg">/path/to/config</span>
      </code>


   * nproc_per_node: number of gpu devices

#. | Exporting to onnx:

   | After finishing training run the following command:

   .. raw:: html
      :name:validation

      <code stage="export">
      python torch2onnx.py <span val="model_path">/path/to/model.pt</span> --network <span val="arch">mbf</span> --output <span val="model_onnx">/path/to/model.onnx</span> --simplify true
      </code>



----

Compile the Model using Hailo Model Zoo
---------------------------------------

You can generate an HEF file for inference on Hailo-8 from your trained ONNX model.
In order to do so you need a working model-zoo environment.
Choose the corresponding YAML from our networks configuration directory, i.e. ``hailo_model_zoo/cfg/networks/arcface_mobilefacenet.yaml``\ , and run compilation using the model zoo:

.. raw:: html
   :name:validation

   <code stage="compile">
   hailomz compile --ckpt <span val="local_path_to_onnx">arcface_s_leaky.onnx</span> --calib-path <span val="calib_set_path">/path/to/calibration/imgs/dir/</span> --yaml <span val="yaml_file_path">/path/to/arcface_mobilefacenet.yaml</span>
   </code>


* | ``--ckpt`` - path to  your ONNX file.
* | ``--calib-path`` - path to a directory with your calibration images in JPEG/png format
* | ``--yaml`` - path to your configuration YAML file.
* | The model zoo will take care of adding the input normalization to be part of the model.

.. note::
  More details about YAML files are presented `here <../../docs/YAML.rst>`_.
