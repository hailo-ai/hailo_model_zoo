=================
ViT Retraining
=================

This docker is based upon https://github.com/huggingface/pytorch-image-models and allows retraining and reproducing of:

1) Original ViT results - training, evaluation and exporting to ONNX
2) Hailo's ViT versions with BN replacing LN all across the network - training, evaluation and exporting to ONNX


Environment Preparations
------------------------


#. | Build the docker image:

   .. code-block::


       cd hailo_model_zoo/training/vit
       cd hailo_model_zoo/training/vit
       docker build --build-arg timezone=`cat /etc/timezone` -t vit:v0 .


   | the following optional arguments can be passed via --build-arg:

   * ``timezone`` - a string for setting up timezone. E.g. "Asia/Jerusalem"
   * ``user`` - username for a local non-root user. Defaults to 'hailo'.
   * ``group`` - default group for a local non-root user. Defaults to 'hailo'.
   * ``uid`` - user id for a local non-root user.
   * ``gid`` - group id for a local non-root user.

   | * This command will build the docker image with the necessary requirements using the Dockerfile exists in vit directory.



#. | Start your docker:

   .. code-block::



      docker run --name "your_docker_name" -it --gpus all --ipc=host -v  /path/to/local/data/dir:/path/to/docker/data/dir vit:v0


   * ``docker run`` create a new docker container.
   * ``--name <your_docker_name>`` name for your container.
   * ``-it`` runs the command interactively.
   * ``--gpus all`` allows access to all GPUs.
   * ``--ipc=host`` sets the IPC mode for the container.
   * ``-v /path/to/local/data/dir:/path/to/docker/data/dir`` maps ``/path/to/local/data/dir`` from the host to the container. You can use this command multiple times to mount multiple directories.
   * ``vit:v0`` the name of the docker image.

Training and exporting to ONNX
------------------------------


#. | Train your model:
   | Once the docker is started, you can start training your model.

   * | Prepare your custom dataset - Follow the steps described `here <https://timm.fast.ai/training>`

   * | Start training - The following command is an example for training a *vit_tiny_un_patch16_224* model.

     .. code-block::


        python3 -m torch.distributed.launch --nproc_per_node=1 train.py ../data/imagenet_10000/ --model vit_tiny_un_patch16_224 --output output --experiment retrain --initial-checkpoint vit_tiny_un_patch16_224.pth.tar  --epochs 1 --workers 6 --batch-size=64 --drop-path 0.1 --model-ema --model-ema-decay 0.99996 --opt adamw --opt-eps 1e-8 --weight-decay 0.05 --lr 0.00001 --aa rand-m9-mstd0.5-inc1 --train-interpolation bicubic --use-ra-sampler --reprob 0.25 --mixup 0.8 --cutmix 1.0


      * ``vit_tiny_un_patch16_224.pth.tar`` - pretrained weights.
      * ``--model-ema`` - use exponential moving average weights.


#. | Export to ONNX:

   | In order to export your trained ViT model to ONNX run the following script:

   .. code-block::


      python export.py --model vit_tiny_un_patch16_224 --checkpoint=/path/to/trained/best.pt --use-ema


      * ``--use-ema`` - optional to use if --model-ema was used during training.

----

Compile the Model using Hailo Model Zoo
---------------------------------------

| You can generate an HEF file for inference on Hailo device from your trained ONNX model.
| In order to do so you need a working model-zoo environment.
| Choose the corresponding YAML from our networks configuration directory, i.e. ``hailo_model_zoo/cfg/networks/vit_tiny.yaml``\ , and run compilation using the model zoo:


   hailomz compile --ckpt vit_tiny_un_patch16_224.onnx --calib-path /path/to/calibration/imgs/dir/ --yaml path/to/vit_tiny_un_patch16_224.yaml --start-node-names name1 name2 --end-node-names name1



* | ``--ckpt`` - path to  your ONNX file.
* | ``--calib-path`` - path to a directory with your calibration images in JPEG/png format
* | ``--yaml`` - path to your configuration YAML file.
* | ``--start-node-names`` and ``--end-node-names`` - node names for customizing parsing behavior (optional).
* | The model zoo will take care of adding the input normalization to be part of the model.

  More details about YAML files are presented `here <../../docs/YAML.rst>`_.
