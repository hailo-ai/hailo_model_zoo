===============
MSPN Retraining
===============

Prerequisites
-------------


* docker (\ `installation instructions <https://docs.docker.com/engine/install/ubuntu/>`_\ )
* nvidia-docker2 (\ `installation instructions <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_\ )

**NOTE:**  In case you are using the Hailo Software Suite docker, make sure to run all of the following instructions outside of that docker.


Environment Preparations
------------------------

#. Build the docker image:

   .. raw:: html
      :name:validation

      <pre><code stage="docker_build">
      cd <span val="dockerfile_path">hailo_model_zoo/training/mspn</span>
      docker build -t mspn:v0 --build-arg timezone=`cat /etc/timezone` .
      </code></pre>

   | the following optional arguments can be passed via --build-arg:

   * ``timezone`` - a string for setting up timezone. E.g. "Asia/Jerusalem"
   * ``user`` - username for a local non-root user. Defaults to 'hailo'.
   * ``group`` - default group for a local non-root user. Defaults to 'hailo'.
   * ``uid`` - user id for a local non-root user.
   * ``gid`` - group id for a local non-root user.


#. Start your docker:

   .. raw:: html
      :name:validation

      <code stage="docker_run">
      docker run <span val="replace_none">--name "your_docker_name"</span> -it --gpus all <span val="replace_none">-u "username"</span> --ipc=host -v <span val="local_vol_path">/path/to/local/data/dir</span>:<span val="docker_vol_path">/path/to/docker/data/dir</span>  mspn:v0
      </code>

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

           `---|-- train2017
           |---|---|-- *.jpg
           `---|-- val2017
           |---|---|-- *.jpg
           `---|-- test2017
           `---|---|-- *.jpg

   The path for the dataset can be configured in the .py config file, e.g. ``configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/regnetx_800mf_256x192.py``

#. Training:

   Configure your model in a .py config file. We will use ``/workspace/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/regnetx_800mf_256x192.py`` in this guide.
   Start training with the following command:

   .. raw:: html
      :name:validation

      <pre><code stage="retrain">
      cd /workspace/mmpose
      ./tools/dist_train.sh ./configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/regnetx_800mf_256x192.py <span val="gpu_num">4</span> --work-dir exp0
      </code></pre>

   Where 4 is the number of GPUs used for training. In this example, the trained model will be saved under ``exp0`` directory.

#. Export to onnx

   In orded to export your trained model to ONNX run the following script:

   .. raw:: html
      :name:validation

      <pre><code stage="export">
      cd /workspace/mmpose
      python tools/deployment/pytorch2onnx.py ./configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/regnetx_800mf_256x192.py <span val="docker_trained_path">exp0/best_AP_epoch_310.pth</span> --output-file mspn_regnetx_800mf.onnx
      </code></pre>

   where ``exp0/best_AP_epoch_310.pth`` should be replaced by the trained model file path.     

----

Compile the Model using Hailo Model Zoo
---------------------------------------

| You can generate an HEF file for inference on Hailo-8 from your trained ONNX model.
| In order to do so you need a working model-zoo environment.
| Choose the corresponding YAML from our networks configuration directory, i.e. ``hailo_model_zoo/cfg/networks/mspn_regnetx_800mf.yaml``\ , and run compilation using the model zoo:  

.. raw:: html
   :name:validation

   <code stage="compile">
   hailomz compile --ckpt <span val="local_path_to_onnx">mspn_regnetx_800mf.onnx</span> --calib-path <span val="calib_set_path">/path/to/calibration/imgs/dir/</span> --yaml <span val="yaml_file_path">path/to/mspn_regnetx_800mf.yaml</span>
   </code>


* | ``--ckpt`` - path to  your ONNX file.
* | ``--calib-path`` - path to a directory with your calibration images in JPEG/png format
* | ``--yaml`` - path to your configuration YAML file.
* | The model zoo will take care of adding the input normalization to be part of the model.

.. note::
  More details about YAML files are presented `here <../../docs/YAML.rst>`_.