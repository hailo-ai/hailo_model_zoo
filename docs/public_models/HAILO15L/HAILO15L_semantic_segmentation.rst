


Public Models
=============

* System host: Intel® Core™ i5-9400 CPU @ 2.90GHz
* Hailo Dataflow Compiler Version v5.4.0
* Measurement conditions: Measuring from the SoC, room temperature

|

Semantic Segmentation
=====================

|

Link Legend
-----------

|

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - **Key / Icon**
     - **Description**
   * - ⭐
     - Networks used by `Hailo-apps <https://github.com/hailo-ai/hailo-apps-infra>`_.
   * - **S**
     - Source – Link to the model's open-source repository.
   * - **PT**
     - Pretrained – Download the pretrained model file (ZIP format).
   * - **HEF, NV12, RGBX**
     - Compiled Models – Links to models in various formats:
       - **HEF:** RGB format
       - **NV12:** NV12 format
       - **RGBX:** RGBX format
   * - **PR**
     - Profiler Report – Download the model's performance profiling report.

|

Pascal Voc
----------

|








.. list-table::
   :header-rows: 1
   :widths: 31 9 7 11 9 8 8 8 9

   * - Network Name
     - float mIoU
     - Hardware mIoU
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
   
   
   
   
   
   
   

   * - deeplab_v3_mobilenet_v2
     - 76.0
     - 74.6
     - 24.0
     - 28.3
     - | `S <https://github.com/bonlime/keras-deeplab-v3-plus>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Pascal/deeplab_v3_mobilenet_v2_dilation/pretrained/2023-08-22/deeplab_v3_mobilenet_v2_dilation.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15l/deeplab_v3_mobilenet_v2_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15l/deeplab_v3_mobilenet_v2.hef>`_
     - 513x513x3
     - 2.10
     - 17.65
   
   
   
   
   
   
   

   * - deeplab_v3_mobilenet_v2_wo_dilation
     - 71.5
     - 71.1
     - 94.6
     - 121
     - | `S <https://github.com/tensorflow/models/tree/master/research/deeplab>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Pascal/deeplab_v3_mobilenet_v2/pretrained/2025-01-20/deeplab_v3_mobilenet_v2_wo_dilation_sim.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15l/deeplab_v3_mobilenet_v2_wo_dilation_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15l/deeplab_v3_mobilenet_v2_wo_dilation.hef>`_
     - 513x513x3
     - 2.10
     - 3.21

|

Cityscapes
----------

|








.. list-table::
   :header-rows: 1
   :widths: 31 9 7 11 9 8 8 8 9

   * - Network Name
     - float mIoU
     - Hardware mIoU
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
   
   
   
   
   
   
   

   * - fcn8_resnet_v1_18
     - 69.3
     - 69.2
     - 14.0
     - 14.5
     - | `S <https://mmsegmentation.readthedocs.io/en/latest>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Cityscapes/fcn8_resnet_v1_18/pretrained/2023-06-22/fcn8_resnet_v1_18.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15l/fcn8_resnet_v1_18_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15l/fcn8_resnet_v1_18.hef>`_
     - 1024x1920x3
     - 11.20
     - 142.82

|

Oxford-Iiit Pet
---------------

|








.. list-table::
   :header-rows: 1
   :widths: 31 9 7 11 9 8 8 8 9

   * - Network Name
     - float mIoU
     - Hardware mIoU
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
   
   
   
   
   
   
   

   * - unet_mobilenet_v2
     - 77.3
     - 77.1
     - 146
     - 224
     - | `S <https://www.tensorflow.org/tutorials/images/segmentation>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Oxford_Pet/unet_mobilenet_v2/pretrained/2025-01-15/unet_mobilenet_v2.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15l/unet_mobilenet_v2_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15l/unet_mobilenet_v2.hef>`_
     - 256x256x3
     - 10.08
     - 28.88