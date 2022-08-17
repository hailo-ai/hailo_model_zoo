Train Person-ReID on a Custom Dataset
-------------------------------------

Here we describe how to finetune Hailo's person-reid network with your own custom dataset.

Prerequisites
^^^^^^^^^^^^^


* docker (\ `installation instructions <https://docs.docker.com/engine/install/ubuntu/>`_\ )
* nvidia-docker2 (\ `installation instructions <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_\ )

**NOTE:**  In case you are using the Hailo Software Suite docker, make sure to run all of the following instructions outside of that docker.

Environment Preparations
^^^^^^^^^^^^^^^^^^^^^^^^

#. 
   **Build the docker image:**

   .. raw:: html
      :name:validation

      <code stage="docker_build">
      cd <span val="dockerfile_path">hailo_model_zoo/hailo_models/reid/</span>

      docker build  --build-arg timezone=`cat /etc/timezone` -t person_reid:v0 .
      </code>

   | the following optional arguments can be passed via --build-arg:


   * | ``timezone`` - a string for setting up timezone. E.g. "Asia/Jerusalem"
   * | ``user`` - username for a local non-root user. Defaults to 'hailo'.
   * | ``group`` - default group for a local non-root user. Defaults to 'hailo'.
   * | ``uid`` - user id for a local non-root user.
   * | ``gid`` - group id for a local non-root user.

     | This command will build the docker image with the necessary requirements using the Dockerfile that exists in this directory.

#. 
   **Start your docker:**

   .. raw:: html
      :name:validation

      <code stage="docker_run">
      docker run <span val="replace_none">--name "your_docker_name"</span> -it --gpus all --ipc=host -v <span val="local_vol_path">/path/to/local/drive</span>:<span
      val="docker_vol_path">/path/to/docker/dir</span> person_reid:v0
      </code>


   * ``docker run`` create a new docker container.
   * ``--name <your_docker_name>`` name for your container.
   * ``-it`` runs the command interactively.
   * ``--gpus all`` allows access to all GPUs.
   * ``--ipc=host`` sets the IPC mode for the container.
   * ``-v /path/to/local/data/dir:/path/to/docker/data/dir`` maps ``/path/to/local/data/dir`` from the host to the container. You can use this command multiple times to mount multiple directories.
   * ``personface_detection:v0`` the name of the docker image.

Finetuning and exporting to ONNX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


#. | **Train the network on your dataset**
   | Once the docker is started, you can train the person-reid model on your custom dataset. We recommend following the instructions from the torchreid repo that can be found in `here <https://kaiyangzhou.github.io/deep-person-reid/user_guide.html#use-your-own-dataset>`_.


   * 
     Insert your dataset as described in `use-your-own-dataset <https://kaiyangzhou.github.io/deep-person-reid/user_guide.html#use-your-own-dataset>`_.

   * 
     Start training on your dataset starting from our pre-trained weights in ``models/repvgg_a0_person_reid_512.pth`` or ``models/repvgg_a0_person_reid_2048.pth`` (you can also download it from `512-dim <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/MCPReID/reid/repvgg_a0_person_reid_512/2022-04-18/repvgg_a0_person_reid_512.pth>`_ & `2048-dim <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/MCPReID/reid/repvgg_a0_person_reid_2048/2022-04-18/repvgg_a0_person_reid_2048.pth>`_\ ) - to do so, you can edit the added yaml ``configs/repvgg_a0_hailo_pre_train.yaml`` and take a look at the examples in `torchreid <https://github.com/KaiyangZhou/deep-person-reid>`_.

     .. raw:: html
        :name:validation

         <code stage="retrain">
         python scripts/main.py  --config-file configs/repvgg_a0_hailo_pre_train.yaml
         </code>

#. | **Export to ONNX**
   | Export the model to ONNX using the following command:

   .. raw:: html
      :name:validation

      <code stage="export">
      python scripts/export.py --model_name <span val="model_name"><model_name></span> --weights <span val="weights">/path/to/model/pth</span>
      </code>

----

Compile the Model using Hailo Model Zoo
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| In case you exported to onnx based on one of our provided RepVGG models, you can generate an HEF file for inference on Hailo-8 from your trained ONNX model. In order to do so you need a working model-zoo environment.
| Choose the model YAML from our networks configuration directory, i.e. ``hailo_model_zoo/cfg/networks/repvgg_a0_person_reid_512.yaml`` (or 2048), and run compilation using the model zoo:

.. raw:: html
   :name:validation

   <code stage="compile">
   hailomz compile --ckpt <span val="local_path_to_onnx">repvgg_a0_person_reid_512.onnx</span> --calib-path <span val="calib_set_path">/path/to/calibration/imgs/dir/</span> --yaml <span val="yaml_file_path">repvgg_a0_person_reid_512.yaml</span>
   </code>


* | ``--ckpt`` - path to your ONNX file.
* | ``--calib-path`` - path to a directory with your calibration images in JPEG format
* | ``--yaml`` - path to your configuration YAML file.

| The model zoo will take care of adding the input normalization to be part of the model.
