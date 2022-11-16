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

   .. raw:: html
      :name:validation

      <pre><code stage="docker_build">
      cd <span val="dockerfile_path">hailo_model_zoo/training/yolact</span>
      docker build --build-arg timezone=`cat /etc/timezone` -t yolact:v0 .
      </code></pre>



   | the following optional arguments can be passed via --build-arg:

   * ``timezone`` - a string for setting up timezone. E.g. "Asia/Jerusalem"
   * ``user`` - username for a local non-root user. Defaults to 'hailo'.
   * ``group`` - default group for a local non-root user. Defaults to 'hailo'.
   * ``uid`` - user id for a local non-root user.
   * ``gid`` - group id for a local non-root user.
   * This command will build the docker image with the necessary requirements using the Dockerfile exists in the yolact directory.


#. | Start your docker:

   .. raw:: html
      :name:validation

      <code stage="docker_run">
      docker run <span val="replace_none">--name "your_docker_name"</span> -it --gpus all --ipc=host -v <span val="local_vol_path">/path/to/local/data/dir</span>:<span val="docker_vol_path">/path/to/docker/data/dir</span>  yolact:v0
      </code>

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
               |---|-- train2017
               |---|---|-- *.jpg
               |---|-- val2017
               |---|---|-- *.jpg
               |---|-- test2017
               `---|---|-- *.jpg

   | For training on custom datasets see `here <https://github.com/hailo-ai/yolact/tree/Model-Zoo-1.5#custom-datasets>`_

#. | Train your model:

   | Once your dataset is prepared, create a soft link to it under the yolact/data work directory, then you can start training your model:

   .. raw:: html
      :name:validation

      <pre><code stage="retrain">
      cd /workspace/yolact
      ln -s /workspace/data/coco data/coco
      python train.py --config=yolact_regnetx_800MF_config
      </code></pre>

   * ``yolact_regnetx_800MF_config`` - configuration using the regnetx_800MF backbone.

#. | Export to ONNX: In orded to export your trained YOLACT model to ONNX run the following script:
    
   .. raw:: html
      :name:validation

      <code stage="export">
      python export.py --config=yolact_regnetx_800MF_config --trained_model=<span val="docker_path_to_trained_model">path/to/trained/model</span> --export_path=<span val="docker_path_to_onnx">path/to/export/model.onnx</span>
      </code>

   * ``--config`` - same configuration used for training.
   * ``--trained_model`` - path to the weights produced by the training process.
   * ``--export_path`` - path to export the ONNX file to. Include the ``.onnx`` extension.

----

Compile the Model using Hailo Model Zoo
---------------------------------------

You can generate an HEF file for inference on Hailo-8 from your trained ONNX model.
In order to do so you need a working model-zoo environment.
Choose the corresponding YAML from our networks configuration directory, i.e. ``hailo_model_zoo/cfg/networks/yolact.yaml``\ , and run compilation using the model zoo:  

.. raw:: html
   :name:validation

   <code stage="compile">
   hailomz compile <span val="replace_none">yolact</span> --ckpt <span val="local_path_to_onnx">yolact.onnx</span> --calib-path <span val="calib_set_path">/path/to/calibration/imgs/dir/</span> --yaml <span val="yaml_file_path">path/to/yolact_regnetx_800mf_20classes.yaml</span>
   </code>

* | ``--ckpt`` - path to  your ONNX file.
* | ``--calib-path`` - path to a directory with your calibration images in JPEG/png format
* | ``--yaml`` - path to your configuration YAML file.
* | The model zoo will take care of adding the input normalization to be part of the model.

.. note::
  - The `yolact_regnetx_800mf_20classes.yaml<https://github.com/hailo-ai/hailo_model_zoo/blob/master/hailo_model_zoo/cfg/networks/yolact_regnetx_800mf_20classes.yaml>`_ 
    is an example yaml where some of the classes (out of 80) were removed. If you wish to change the number of classes, the easiest way is to retrain with the exact number
    of classes, erase the ``channels_remove`` section (lines 18 to 437), and update ``parser.start_node_shape`` to fit your input resolution
  
  More details about YAML files are presented `here <../../docs/YAML.rst>`_.