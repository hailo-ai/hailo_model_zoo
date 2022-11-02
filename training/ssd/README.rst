===========================
SSD-Mobilenet-V1 Retraining
===========================

* To learn more about ssd look `here <https://github.com/hailo-ai/models/tree/master/research/object_detection>`_

-------

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
      cd <span val="dockerfile_path">hailo_model_zoo/training/ssd</span>
      docker build -t tf1od:v0 --build-arg timezone=`cat /etc/timezone` .
      </code></pre>

   | the following optional arguments can be passed via --build-arg:
   
   * ``timezone`` - a string for setting up timezone. E.g. "Asia/Jerusalem"
   * ``user`` - username for a local non-root user. Defaults to 'hailo'.
   * ``group`` - default group for a local non-root user. Defaults to 'hailo'.
   * ``uid`` - user id for a local non-root user.
   * ``gid`` - group id for a local non-root user.
  

#. | Start your docker:

   .. raw:: html
      :name:validation

      <code stage="docker_run">
      docker run <span val="replace_none">--name "docker-name"</span> -it --gpus all <span val="replace_none">-u "username"</span> --ipc=host -v <span val="local_vol_path">/path/to/local/data/dir</span>:<span val="docker_vol_path">/path/to/docker/data/dir</span> tf1od:v0
      </code>

   * ``docker run`` create a new docker container.
   * ``--name <docker-name>`` name for your container.
   * ``-u <username>`` same username as used for building the image.
   * ``-it`` runs the command interactively.
   * ``--gpus all`` allows access to all GPUs.
   * ``--ipc=host`` sets the IPC mode for the container.
   * ``-v /path/to/local/data/dir:/path/to/docker/data/dir`` maps ``/path/to/local/data/dir`` from the host to the container. You can use this command multiple times to mount multiple directories.
   * ``tf1od:v0`` the name of the docker image.

Training and exporting
----------------------


#. | Prepare your data:
   | Data is required to be in a specific TFRecord format, accompanied by a label map file. Infomation about the dataset structure can be found `here <https://github.com/hailo-ai/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md>`_.   In addition you can find useful datasets conversion utils in the following path ``object_detection/dataset_tools/``

#. | Training:
   | Configure your model by modifying the config file ``/home/tensorflow/models/research/pipeline.config`` found in your working directory.
   | You should first update your dataset information, update the input_path and label_map_path to correspond the data prepared in the previous stage.

   .. code-block::

       train_input_reader {
         label_map_path: "PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt"
         tf_record_input_reader {
           input_path: "PATH_TO_BE_CONFIGURED/<you-own-dataset>_train.record-?????-of-00100"
         }
       }
       eval_input_reader {
         label_map_path: "PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt"
         shuffle: false
         num_readers: 1
         tf_record_input_reader {
           input_path: "PATH_TO_BE_CONFIGURED/<your-own-dataset>_val.record-?????-of-00010"
         }

   | The pretrained weights were already downloaded for you and configured into your config file.

   .. code-block::

         fine_tune_checkpoint: "/home/tensorflow/models/research/ssd_mobilenet_v1_coco_2018_01_28/model.ckpt"
         from_detection_checkpoint: true

   | Start training with the following command:

   .. raw:: html
      :name:validation

      <code stage="retrain">
      python object_detection/model_main.py --pipeline_config_path=/home/tensorflow/models/research/pipeline.config --model_dir=ssd_mobilenet_v1_training --num_train_steps=<span val="iterations">200000</span> --sample_1_of_n_eval_examples=3 --alsologtostderr
      </code>

   * ``--pipeline_config_path`` - path to your training configuration file.
   * ``--model_dir`` - output training directory.
   * ``--num_train_steps`` - exists also in the configuration file but can be overwritten as cli argument.
   * ``--sample_1_of_n_eval_examples`` - sample of one every n eval input examples, where n is provided. Modifying training hyper parameters (batch size, learning rate, optimizer etc...) can be done in the ``train_config`` section:
     
     .. code-block::

        train_config {
        batch_size: 24
        data_augmentation_options {
          random_horizontal_flip {
          }
        }
        data_augmentation_options {
          ssd_random_crop {
          }
        }
        optimizer {
          rms_prop_optimizer {
            learning_rate {
              exponential_decay_learning_rate {
                initial_learning_rate: 0.00400000018999
                decay_steps: 800720
                decay_factor: 0.949999988079
              }
            }
            momentum_optimizer_value: 0.899999976158
            decay: 0.899999976158
            epsilon: 1.0
          }
        }

#. | Exporting the model After training, run the following command:


   .. raw:: html
      :name:validation

      <code stage="export">
      python object_detection/export_inference_graph.py --input_type image_tensor --input_shape -1,300,300,3 --pipeline_config_path pipeline.config --trained_checkpoint_prefix ./ssd_mobilenet_v1_training/model.ckpt-<span val="iterations">"iteration-number"</span> --output_directory ./ssd_mobilenet_v1
      </code>

   | Exported ``model.ckpt`` files will be found in the given output directory.


----

Compile the Model using Hailo Model Zoo
---------------------------------------

You can generate an HEF file for inference on Hailo-8 from your trained checkpoint.
In order to do so you need a working model-zoo environment.
Choose the corresponding YAML from our networks configuration directory, i.e. ``hailo_model_zoo/cfg/networks/ssd_mobilenet_v1.yaml``\ , and run compilation using the model zoo:


.. raw:: html
  :name:validation

  <code stage="compile">
  hailomz compile --ckpt  <span val="local_path_to_onnx">ssd_mobilenet_v1.ckpt</span> --calib-path <span val="calib_set_path">/path/to/calibration/imgs/dir/</span> --yaml <span val="yaml_file_path">path/to/ssd_mobilenet_v1.yaml</span>
  </code>

* | ``--ckpt`` - path to  your ckpt file.
* | ``--calib-path`` - path to a directory with your calibration images in JPEG/png format
* | ``--yaml`` - path to your configuration YAML file.
* | The model zoo will take care of adding the input normalization to be part of the model.

.. note::
  - This model uses on-chip NMS capabilities. Therefore, the on-chip NMS parameters should be updated if they have changed:

    - Set ModelZoo data folder location: ``export HMZ_DATA=/local/workspace/data``
    - Use ``hailomz parse ssd_mobilenet_v1`` so the model is downloaded
    - ``cd $HMZ_DATA/models_files/ObjectDetection/Detection-COCO/ssd/ssd_mobilenet_v1/pretrained/``
    - Update the ``mobilenet_ssd_nms_postprocess_config.json`` file to match the updated NMS post-process configuration.
    - Run the command from above.
  
  More details about YAML files are presented `here <../../docs/YAML.rst>`_.