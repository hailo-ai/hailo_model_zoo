===============
PETR Retraining
===============

Prerequisites
-------------


* docker (\ `installation instructions <https://docs.docker.com/engine/install/ubuntu/>`_\ )
* nvidia-docker2 (\ `installation instructions <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_\ )

**NOTE:**\  In case you are using the Hailo Software Suite docker, make sure to run all of the following instructions outside of that docker.


Environment Preparations
------------------------

#. Build the docker image:

   .. code-block::


      cd hailo_model_zoo/training/petr
      docker build -t petr:v2 --build-arg timezone=`cat /etc/timezone` --build-arg user="username" .


   | the following optional arguments can be passed via --build-arg:

   * ``timezone`` - a string for setting up timezone. E.g. "Asia/Jerusalem"
   * ``user`` - username for a local non-root user. Defaults to 'hailo'.
   * ``group`` - default group for a local non-root user. Defaults to 'hailo'.
   * ``uid`` - user id for a local non-root user.
   * ``gid`` - group id for a local non-root user.


#. Start your docker:

   .. code-block::


      docker run --name "your_docker_name" -it --gpus all --shm-size 32gb -u "username" --ipc=host -v /path/to/local/data/dir:/path/to/docker/data/dir  petr:v2


   * ``docker run`` create a new docker container.
   * ``--name <your_docker_name>`` name for your container.
   * ``-it`` runs the command interactively.
   * ``--gpus all`` allows access to all GPUs.
   * ``--shm-size`` container shared memory size
   * ``--ipc=host`` sets the IPC mode for the container.
   * ``-v /path/to/local/data/dir:/path/to/docker/data/dir`` maps ``/path/to/local/data/dir`` from the host to the container. You can use this command multiple times to mount multiple directories.
   * ``petr:v2`` the name of the docker image.

   .. code-block::

      docker start "your_docker_name"
      docker exec -it "your_docker_name" /bin/bash --login


Training and exporting to ONNX
------------------------------

#. | Prepare your data:

   | Data is expected to be in NuScenes format. For more information on obtaining datasets see `here <https://github.com/open-mmlab/mmdetection3d/blob/1.0/docs/en/data_preparation.md>`_
   | The expected structure is as follows:

   .. code-block::

       /workspace
       |-- PETR
       `-- |-- data
           `-- |-- nuscenes
               |   |-- maps
               |   |-- samples
               |   |-- sweeps
               |   |-- v1.0-trainval
               |   |-- mmdet3d_nuscenes_30f_infos_val.pkl
               |   |-- mmdet3d_nuscenes_30f_infos_train.pkl


   The path for the dataset can be configured via the config file, e.g. ``projects/configs/petrv2/petrv2_fcos3d_repvgg_b0x32_BN_q_304_decoder_3_UN_800x320.py``. It is recommended to generate a symlink of the dataset to ``/workspace/PETR/data/``.
   In order to generate the .pkl train / val annotation files, use the scripts (more info can be found `here <https://github.com/open-mmlab/mmdetection3d/blob/1.0/docs/en/data_preparation.md#nuscenes>`_):

   .. code-block::


      python tools/create_data.py nuscenes --root-path &lt;data_path&gt; --out-dir &lt;data_path&gt; --extra-tag nuscenes
      python tools/generate_sweep_pkl.py


#. Training:

   Configure your model in a .py config file. We will use ``projects/configs/petrv2/petrv2_fcos3d_repvgg_b0x32_BN_q_304_decoder_3_UN_800x320.py`` as the config file in this guide.
   Start training with the following command:

   .. code-block::


      cd /workspace/PETR
      ./tools/dist_train.sh projects/configs/petrv2/petrv2_fcos3d_repvgg_b0x32_BN_q_304_decoder_3_UN_800x320.py 4 --work-dir work_dirs/petrv2_exp0/


   Where 4 is the number of GPUs used for training. In this example, the trained model will be saved under ``work_dirs/petrv2_exp0/latest.pth`` directory.

#. Export to onnx

   Run the following script to export the backbone part of the model:

   .. code-block::


      cd /workspace/PETR
      python tools/export_onnx.py &lt;cfg.py&gt; &lt;trained.pth&gt; --split backbone --out petrv2_backbone.onnx


      Run the following script to export the transformer part of the model:


      python tools/export_onnx.py &lt;cfg.py&gt; &lt;trained.pth&gt; --split transformer --out petrv2_transformer.onnx --reshape-cfg tools/onnx_reshape_cfg_repvgg_b0x32_BN2D_decoder_3_q_304_UN_800x320.json


   * | ``cfg.py`` - model config file path e.g., ``projects/configs/petrv2/petrv2_fcos3d_repvgg_b0x32_BN_q_304_decoder_3_UN_800x320.py``
   * | ``trained.pth`` - the trained model file path e.g., ``work_dirs/petrv2_exp0/latest.pth``
   * | ``--split`` - backbone or transformer export
   * | ``--out`` - output onnx file path
   * | ``--reshape-cfg`` - .json file with node names and config info for further reshape of the transformer export e.g., ``tools/onnx_reshape_cfg_repvgg_b0x32_BN2D_decoder_3_q_304_UN_800x320.json`` for the model we use here

   .. **NOTE:**\  Exporting the transformer also produces the ``reference_points.npy`` postprocessing configuration file.

#. Generate 3D positional embedding data

   Run the following script to generate the 3D coordinates positional embeddings (.npy files) for the transformer model:

   .. raw:: html


      cd /workspace/PETR
      python tools/gen_coords3d_pe.py &lt;cfg.py&gt; &lt;trained.pth&gt;


----

Compile the Model using Hailo Model Zoo
---------------------------------------

You can generate an HEF file for inference on Hailo device from your trained ONNX model.
In order to do so you need a working model-zoo environment.
Choose the corresponding YAMLs from our networks configuration directory, i.e. ``hailo_model_zoo/cfg/networks/petrv2_repvggB0_transformer_pp_800x320.yaml``\ and run parsing, optimization and compilation using the model zoo.


#. Backbone

   .. code-block::


      hailomz compile --ckpt petrv2_backbone.onnx --calib-path /path/to/calibration/imgs/dir/ --yaml path/to/petrv2_repvggB0_backbone_pp_800x320.yaml --start-node-names name1 name2 --end-node-names name1



   * | ``--ckpt`` - path to your ONNX file.
   * | ``--calib-path`` - path to a directory with your calibration images in JPEG/png format
   * | ``--yaml`` - path to your configuration YAML file.
   * | ``--start-node-names`` and ``--end-node-names`` - node names for customizing parsing behavior (optional).
   * | The model zoo will take care of adding the input normalization to be part of the model.


#. Transformer

   .. code-block::


      hailomz compile --ckpt petrv2_transformer.onnx --calib-path /path/to/calibration/tfrecord --yaml path/to/petrv2_repvggB0_transformer_pp_800x320.yaml --start-node-names name1 name2 --end-node-names name1


   * | ``--ckpt`` - path to your ONNX file.
   * | ``--calib-set-path`` - path to transformer calibration set in tfrecord format
   * | ``--yaml`` - path to your configuration YAML file
   * | ``--start-node-names`` and ``--end-node-names`` - node names for customizing parsing behavior (optional).


.. note::
  More details about YAML files are presented `here <../../docs/YAML.rst>`_.
