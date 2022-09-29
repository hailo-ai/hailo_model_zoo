Train Person-Face Detection on a Custom Dataset
-----------------------------------------------

Here we describe how to finetune Hailo's person-face detection network with your own custom dataset.

Prerequisites
^^^^^^^^^^^^^


* docker (\ `installation instructions <https://docs.docker.com/engine/install/ubuntu/>`_\ )
* nvidia-docker2 (\ `installation instructions <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_\ )

**NOTE:**  In case you are using the Hailo Software Suite docker, make sure to run all of the following instructions outside of that docker.


Environment Preparations
^^^^^^^^^^^^^^^^^^^^^^^^


#. **Build the docker image:**

   .. raw:: html
      :name:validation

      <code stage="docker_build">
      cd <span val="dockerfile_path">hailo_model_zoo/hailo_models/personface_detection/</span>

      docker build  --build-arg timezone=\`cat /etc/timezone\` -t personface_detection:v0 .
      </code>

   * This command will build the docker image with the necessary requirements using the Dockerfile that exists in this directory.

#. **Start your docker:**

   .. raw:: html
      :name:validation

      <code stage="docker_run">
      docker run <span val="replace_none">--name "your_docker_name"</span> -it --gpus all --ipc=host -v <span val="local_vol_path"> /path/to/local/data/dir</span>:<span val="docker_vol_path">/path/to/docker/data/dir</span> personface_detection:v0
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
   | Once the docker is started, you can train the person-face detector on your custom dataset. We recommend following the instructions for YOLOV5 training that can be found in `here <https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#11-create-datasetyaml>`_. The important steps are specified below:


   * Update the dataset config file ``data/personface_data.yaml`` with the paths to your training and validation images files.

     .. code-block::

         #update your data paths
         train: /path/to/personface/images/train/
         val: /path/to/personface/images/val
         # number of classes
         nc: 2
         # class names
         names: ['person', 'face']

   * Start training on your dataset starting from our pre-trained weights in ``weights/yolov5s_personface.pt`` (you can also download it from `here <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HailoNets/MCPReID/personface_detector/yolov5s_personface/2022-04-01/yolov5s_personface.pt>`_

   .. raw:: html
      :name:validation

      <code stage="retrain">
      python train.py --data ./data/personface_data.yaml --cfg ./models/yolov5s_personface.yaml --weights ./weights/yolov5s_personface.pt --epochs <span val="epochs">300</span> --batch <span val="batch_size">128</span><span val="replace_none"> --device 1,2,3,4</span>
      </code>

#. 
   **Export to ONNX**
   Export the model to ONNX using the following command:

   .. raw:: html
      :name:validation

      <code stage="export">
      python models/export.py --weights <span val="docker_trained_path">./runs/exp<#>/weights/best.pt</span> --img-size 640 --batch-size 1
      </code>

   * | The best model's weights will be saved under the following path: ``./runs/exp<#>/weights/best.pt``
     | , where <#> is the experiment number.
   * | Export at 640x640 with batch size 1

----

Compile the Model using Hailo Model Zoo
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| You can generate an HEF file for inference on Hailo-8 from your trained ONNX model. In order to do so you need a working model-zoo environment.
| Choose the model YAML from our networks configuration directory, i.e. ``hailo_model_zoo/cfg/networks/yolov5s_personface.yaml``\ , and run compilation using the model zoo:

.. raw:: html
   :name:validation

   <code stage="compile">
   hailomz compile --ckpt <span val="local_path_to_onnx">yolov5s_personface.onnx</span> --calib-path <span val="calib_set_path">/path/to/calibration/imgs/dir/</span> --yaml <span val="yaml_file_path">path/to/yolov5s_personface.yaml</span>
   </code>

* | ``--ckpt`` - path to  your ONNX file.
* | ``--calib-path`` - path to a directory with your calibration images in JPEG/png format
* | ``--yaml`` - path to your configuration YAML file.
* | The model zoo will take care of adding the input normalization to be part of the model.

.. note::
  - Since it’s an Hailo model, calibration set must be manually supplied. 
  - On `yolo.yaml <https://github.com/hailo-ai/hailo_model_zoo/blob/master/hailo_model_zoo/cfg/base/yolo.yaml>`_,
    change ``preprocessing.input_shape`` if changed on retraining
  
  More details about YAML files are presented `here <../../../docs/YAML.rst>`_.