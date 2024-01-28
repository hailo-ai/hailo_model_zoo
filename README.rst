Hailo Model Zoo
===============

.. |python| image:: https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue.svg
   :target: https://www.python.org/downloads/release/python-380/
   :alt: Python 3.8
   :width: 150
   :height: 20


.. |tensorflow| image:: https://img.shields.io/badge/Tensorflow-2.12.0-blue.svg
   :target: https://github.com/tensorflow/tensorflow/releases/tag/v2.12.0
   :alt: Tensorflow
   :width: 110
   :height: 20


.. |cuda| image:: https://img.shields.io/badge/CUDA-11.8-blue.svg
   :target: https://developer.nvidia.com/cuda-toolkit
   :alt: Cuda
   :width: 80
   :height: 20


.. |compiler| image:: https://img.shields.io/badge/Hailo%20Dataflow%20Compiler-3.26.0-brightgreen.svg
   :target: https://hailo.ai/company-overview/contact-us/
   :alt: Hailo Dataflow Compiler
   :width: 180
   :height: 20


.. |runtime| image:: https://img.shields.io/badge/HailoRT%20(optional)-4.16.0-brightgreen.svg
   :target: https://hailo.ai/company-overview/contact-us/
   :alt: HailoRT
   :width: 170
   :height: 20


.. |license| image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://github.com/hailo-ai/hailo_model_zoo/blob/master/LICENSE
   :alt: License: MIT
   :width: 80
   :height: 20


.. image:: docs/images/logo.png

|python| |tensorflow| |cuda| |compiler| |runtime| |license|


The Hailo Model Zoo provides pre-trained models for high-performance deep learning applications. Using the Hailo Model Zoo you can measure the full precision accuracy of each model, the quantized accuracy using the Hailo Emulator and measure the accuracy on the Hailo-8 device. Finally, you will be able to generate the Hailo Executable Format (HEF) binary file to speed-up development and generate high quality applications accelerated with Hailo-8. The Hailo Model Zoo also provides re-training instructions to train the models on custom datasets and models that were trained for specific use-cases on internal datasets.

Models
------

| Hailo provides different pre-trained models in ONNX / TF formats and pre-compiled HEF (Hailo Executable Format) binary file to execute on the Hailo-8 device.
| The models are divided to:


* | PUBLIC MODELS - which were trained on publicly available datasets.

  PUBLIC MODELS HAILO8 - Task Type

     `PUBLIC MODELS_HAILO8 Classification <docs/PUBLIC_MODELS_HAILO8_Classification.rst>`_

     `PUBLIC MODELS_HAILO8 Object_Detection <docs/PUBLIC_MODELS_HAILO8_Object_Detection.rst>`_

     `PUBLIC MODELS_HAILO8 Semantic_Segmentation <docs/PUBLIC_MODELS_HAILO8_Semantic_Segmentation.rst>`_

     `PUBLIC MODELS_HAILO8 Pose_Estimation <docs/PUBLIC_MODELS_HAILO8_Pose_Estimation.rst>`_

     `PUBLIC MODELS_HAILO8 Single_Person_Pose_Estimation <docs/PUBLIC_MODELS_HAILO8_Single_Person_Pose_Estimation.rst>`_

     `PUBLIC MODELS_HAILO8 Face_Detection <docs/PUBLIC_MODELS_HAILO8_Face_Detection.rst>`_

     `PUBLIC MODELS_HAILO8 Instance_Segmentation <docs/PUBLIC_MODELS_HAILO8_Instance_Segmentation.rst>`_

     `PUBLIC MODELS_HAILO8 Depth_Estimation <docs/PUBLIC_MODELS_HAILO8_Depth_Estimation.rst>`_

     `PUBLIC MODELS_HAILO8 Facial_Landmark_Detectio <docs/PUBLIC_MODELS_HAILO8_Facial_Landmark_Detection.rst>`_

     `PUBLIC MODELS_HAILO8 Person_Re-ID <docs/PUBLIC_MODELS_HAILO8_Person_Re-ID.rst>`_

     `PUBLIC MODELS_HAILO8 Super_Resolution <docs/PUBLIC_MODELS_HAILO8_Super_Resolution.rst>`_

     `PUBLIC MODELS_HAILO8 Face_Recognition <docs/PUBLIC_MODELS_HAILO8_Face_Recognition.rst>`_

     `PUBLIC MODELS_HAILO8 Person_Attribute <docs/PUBLIC_MODELS_HAILO8_Person_Attribute.rst>`_

     `PUBLIC MODELS_HAILO8 Face_Attribute <docs/PUBLIC_MODELS_HAILO8_Face_Attribute.rst>`_

     `PUBLIC MODELS_HAILO8 Zero-shot_Classification <docs/PUBLIC_MODELS_HAILO8_Zero-shot_Classification.rst>`_

     `PUBLIC MODELS_HAILO8 Stereo_Depth_Estimation <docs/PUBLIC_MODELS_HAILO8_Stereo_Depth_Estimation.rst>`_

     `PUBLIC MODELS_HAILO8 Low_Light_Enhancement <docs/PUBLIC_MODELS_HAILO8_Low_Light_Enhancement.rst>`_

     `PUBLIC MODELS_HAILO8 Image_Denoising <docs/PUBLIC_MODELS_HAILO8_Image_Denoising.rst>`_

     `PUBLIC MODELS_HAILO8 Hand_Landmark detection <docs/PUBLIC_MODELS_HAILO8_Hand_Landmark detection.rst>`_

  PUBLIC MODELS HAILO8L - Task Type

    `PUBLIC MODELS_HAILO8L Classification <docs/PUBLIC_MODELS_HAILO8L_Classification.rst>`_

    `PUBLIC MODELS_HAILO8L Object_Detection <docs/PUBLIC_MODELS_HAILO8L_Object_Detection.rst>`_

    `PUBLIC MODELS_HAILO8L Semantic_Segmentation <docs/PUBLIC_MODELS_HAILO8L_Semantic_Segmentation.rst>`_

    `PUBLIC MODELS_HAILO8L Pose_Estimation <docs/PUBLIC_MODELS_HAILO8L_Pose_Estimation.rst>`_

    `PUBLIC MODELS_HAILO8L Single_Person_Pose_Estimation <docs/PUBLIC_MODELS_HAILO8L_Single_Person_Pose_Estimation.rst>`_

    `PUBLIC MODELS_HAILO8L Face_Detection <docs/PUBLIC_MODELS_HAILO8L_Face_Detection.rst>`_

    `PUBLIC MODELS_HAILO8L Instance_Segmentation <docs/PUBLIC_MODELS_HAILO8L_Instance_Segmentation.rst>`_

    `PUBLIC MODELS_HAILO8L Depth_Estimation <docs/PUBLIC_MODELS_HAILO8L_Depth_Estimation.rst>`_

    `PUBLIC MODELS_HAILO8L Facial_Landmark_Detection <docs/PUBLIC_MODELS_HAILO8L_Facial_Landmark_Detection.rst>`_

    `PUBLIC MODELS_HAILO8L Person_Re-ID <docs/PUBLIC_MODELS_HAILO8L_Person_Re-ID.rst>`_

    `PUBLIC MODELS_HAILO8L Super_Resolution <docs/PUBLIC_MODELS_HAILO8L_Super_Resolution.rst>`_

    `PUBLIC MODELS_HAILO8L Face_Recognition <docs/PUBLIC_MODELS_HAILO8L_Face_Recognition.rst>`_

    `PUBLIC MODELS_HAILO8L Person_Attribute <docs/PUBLIC_MODELS_HAILO8L_Person_Attribute.rst>`_

    `PUBLIC MODELS_HAILO8L Face_Attribute <docs/PUBLIC_MODELS_HAILO8L_Face_Attribute.rst>`_

    `PUBLIC MODELS_HAILO8L Zero-shot_Classification <docs/PUBLIC_MODELS_HAILO8L_Zero-shot_Classification.rst>`_

    `PUBLIC MODELS_HAILO8L Low_Light_Enhancement <docs/PUBLIC_MODELS_HAILO8L_Low_Light_Enhancement.rst>`_

    `PUBLIC MODELS_HAILO8L Image_Denoising <docs/PUBLIC_MODELS_HAILO8L_Image_Denoising.rst>`_

    `PUBLIC MODELS_HAILO8L Hand_Landmark detection <docs/PUBLIC_MODELS_HAILO8L_Hand_Landmark detection.rst>`_

  PUBLIC MODELS HAILO15H - Task Type

    `PUBLIC MODELS_HAILO15H Classification <docs/PUBLIC_MODELS_HAILO15H_Classification.rst>`_

    `PUBLIC MODELS_HAILO15H Object_Detection <docs/PUBLIC_MODELS_HAILO15H_Object_Detection.rst>`_

    `PUBLIC MODELS_HAILO15H Semantic_Segmentation <docs/PUBLIC_MODELS_HAILO15H_Semantic_Segmentation.rst>`_

    `PUBLIC MODELS_HAILO15H Pose_Estimation <docs/PUBLIC_MODELS_HAILO15H_Pose_Estimation.rst>`_

    `PUBLIC MODELS_HAILO15H Single_Person_Pose_Estimation <docs/PUBLIC_MODELS_HAILO15H_Single_Person_Pose_Estimation.rst>`_

    `PUBLIC MODELS_HAILO15H Face_Detection <docs/PUBLIC_MODELS_HAILO15H_Face_Detection.rst>`_

    `PUBLIC MODELS_HAILO15H Instance_Segmentation <docs/PUBLIC_MODELS_HAILO15H_Instance_Segmentation.rst>`_

    `PUBLIC MODELS_HAILO15H Depth_Estimation <docs/PUBLIC_MODELS_HAILO15H_Depth_Estimation.rst>`_

    `PUBLIC MODELS_HAILO15H Facial_Landmark_Detection <docs/PUBLIC_MODELS_HAILO15H_Facial_Landmark_Detection.rst>`_

    `PUBLIC MODELS_HAILO15H Person_Re-ID <docs/PUBLIC_MODELS_HAILO15H_Person_Re-ID.rst>`_

    `PUBLIC MODELS_HAILO15H Super_Resolution <docs/PUBLIC_MODELS_HAILO15H_Super_Resolution.rst>`_

    `PUBLIC MODELS_HAILO15H Face_Recognition <docs/PUBLIC_MODELS_HAILO15H_Face_Recognition.rst>`_

    `PUBLIC MODELS_HAILO15H Person_Attribute <docs/PUBLIC_MODELS_HAILO15H_Person_Attribute.rst>`_

    `PUBLIC MODELS_HAILO15H Face_Attribute <docs/PUBLIC_MODELS_HAILO15H_Face_Attribute.rst>`_

    `PUBLIC MODELS_HAILO15H Low_Light_Enhancement <docs/PUBLIC_MODELS_HAILO15H_Low_Light_Enhancement.rst>`_

    `PUBLIC MODELS_HAILO15H Image_Denoising <docs/PUBLIC_MODELS_HAILO15H_Image_Denoising.rst>`_

    `PUBLIC MODELS_HAILO15H Hand_Landmark detection <docs/PUBLIC_MODELS_HAILO15H_Hand_Landmark detection.rst>`_

  PUBLIC MODELS HAILO15M - Task Type

    `PUBLIC MODELS_HAILO15M Classification <docs/PUBLIC_MODELS_HAILO15M_Classification.rst>`_

    `PUBLIC MODELS_HAILO15M Object_Detection <docs/PUBLIC_MODELS_HAILO15M_Object_Detection.rst>`_

    `PUBLIC MODELS_HAILO15M Semantic_Segmentation <docs/PUBLIC_MODELS_HAILO15M_Semantic_Segmentation.rst>`_

    `PUBLIC MODELS_HAILO15M Pose_Estimation <docs/PUBLIC_MODELS_HAILO15M_Pose_Estimation.rst>`_

    `PUBLIC MODELS_HAILO15M Single_Person_Pose_Estimation <docs/PUBLIC_MODELS_HAILO15M_Single_Person_Pose_Estimation.rst>`_

    `PUBLIC MODELS_HAILO15M Face_Detection <docs/PUBLIC_MODELS_HAILO15M_Face_Detection.rst>`_

    `PUBLIC MODELS_HAILO15M Instance_Segmentation <docs/PUBLIC_MODELS_HAILO15M_Instance_Segmentation.rst>`_

    `PUBLIC MODELS_HAILO15M Depth_Estimation <docs/PUBLIC_MODELS_HAILO15M_Depth_Estimation.rst>`_

    `PUBLIC MODELS_HAILO15M Facial_Landmark_Detection <docs/PUBLIC_MODELS_HAILO15M_Facial_Landmark_Detection.rst>`_

    `PUBLIC MODELS_HAILO15M Person_Re-ID <docs/PUBLIC_MODELS_HAILO15M_Person_Re-ID.rst>`_

    `PUBLIC MODELS_HAILO15M Super_Resolution <docs/PUBLIC_MODELS_HAILO15M_Super_Resolution.rst>`_

    `PUBLIC MODELS_HAILO15M Face_Recognition <docs/PUBLIC_MODELS_HAILO15M_Face_Recognition.rst>`_

    `PUBLIC MODELS_HAILO15M Person_Attribute <docs/PUBLIC_MODELS_HAILO15M_Person_Attribute.rst>`_

    `PUBLIC MODELS_HAILO15M Face_Attribute <docs/PUBLIC_MODELS_HAILO15M_Face_Attribute.rst>`_

    `PUBLIC MODELS_HAILO15M Low_Light_Enhancement <docs/PUBLIC_MODELS_HAILO15M_Low_Light_Enhancement.rst>`_

    `PUBLIC MODELS_HAILO15M Image_Denoising <docs/PUBLIC_MODELS_HAILO15M_Image_Denoising.rst>`_

    `PUBLIC MODELS_HAILO15M Hand_Landmark detection <docs/PUBLIC_MODELS_HAILO15M_Hand_Landmark detection.rst>`_

* | `HAILO MODELS <docs/HAILO_MODELS.rst>`_ which were trained in-house for specific use-cases on internal datasets.
  | Each Hailo Model is accompanied with retraining instructions.


Retraining
----------

Hailo also provides `RETRAINING INSTRUCTIONS <docs/RETRAIN_ON_CUSTOM_DATASET.rst>`_ to train a network from the Hailo Model Zoo with custom dataset.

Benchmarks
----------

| List of Hailo's benchmarks can be found in `hailo.ai <https://hailo.ai/developer-zone/benchmarks/>`_.
| In order to reproduce the measurements please refer to the following `page <docs/BENCHMARKS.rst>`_.


Quick Start Guide
------------------


* Install Hailo Dataflow Compiler and enter the virtualenv. In case you are not Hailo customer please contact `hailo.ai <https://hailo.ai/company-overview/contact-us/>`_
* Install HailoRT (optional). Required only if you want to run on Hailo-8. In case you are not Hailo customer please contact `hailo.ai <https://hailo.ai/company-overview/contact-us/>`_
* Clone the Hailo Model Zoo


  .. code-block::

      git clone https://github.com/hailo-ai/hailo_model_zoo.git

* Run the setup script


  .. code-block::

     cd hailo_model_zoo; pip install -e .

* Run the Hailo Model Zoo. For example, print the information of the MobileNet-v1 model:


  .. code-block::

     hailomz info mobilenet_v1

Getting Started
^^^^^^^^^^^^^^^

For full functionality please see the `INSTALLATION GUIDE <docs/GETTING_STARTED.rst>`_ page (full install instructions and usage examples). The Hailo Model Zoo is using the Hailo Dataflow Compiler for parsing, model optimization, emulation and compilation of the deep learning models. Full functionality includes:


* | Parse: model translation of the input model into Hailo's internal representation.
* | Profiler: generate profiler report of the model. The report contains information about your model and expected performance on the Hailo hardware.
* | Optimize: optimize the deep learning model for inference and generate a numeric translation of the input model into a compressed integer representation.
  | For further information please see our `OPTIMIZATION <docs/OPTIMIZATION.rst>`_ page.
* | Compile: run the Hailo compiler to generate the Hailo Executable Format file (HEF) which can be executed on the Hailo hardware.
* | Evaluate: infer the model using the Hailo Emulator or the Hailo hardware and produce the model accuracy.

For further information about the Hailo Dataflow Compiler please contact `hailo.ai <https://hailo.ai/company-overview/contact-us/>`_.


.. figure:: docs/images/usage_flow.svg


License
-------

The Hailo Model Zoo is released under the MIT license. Please see the `LICENSE <https://github.com/hailo-ai/hailo_model_zoo/blob/master/LICENSE>`_ file for more information.

Contact
-------

Please visit `hailo.ai <https://hailo.ai/>`_ for support / requests / issues.

Changelog
---------

For further information please see our `CHANGELOG <docs/CHANGELOG.rst>`_ page.
