Hailo Model Zoo
===============

.. |python| image:: https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg
   :target: https://www.python.org/downloads/release/python-3123/
   :alt: Python 3.12
   :width: 150
   :height: 20


.. |tensorflow| image:: https://img.shields.io/badge/Tensorflow-2.19.1-blue.svg
   :target: https://github.com/tensorflow/tensorflow/releases/tag/v2.19.1
   :alt: Tensorflow
   :width: 110
   :height: 20


.. |cuda| image:: https://img.shields.io/badge/CUDA-12.5.1-blue.svg
   :target: https://developer.nvidia.com/cuda-toolkit
   :alt: Cuda
   :width: 80
   :height: 20


.. |compiler| image:: https://img.shields.io/badge/Hailo%20Dataflow%20Compiler-5.5.0-brightgreen.svg
   :target: https://hailo.ai/company-overview/contact-us/
   :alt: Hailo Dataflow Compiler
   :width: 180
   :height: 20


.. |runtime| image:: https://img.shields.io/badge/HailoRT%20(optional)-5.5.0-brightgreen.svg
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

The Hailo Model Zoo provides pre-trained models for high-performance deep learning applications. Using the Hailo Model Zoo, you can measure the full precision accuracy of each model, the quantized accuracy using the Hailo Emulator, and measure the accuracy on Hailo's device. Finally, you will be able to generate the Hailo Executable Format (HEF) binary file to speed up development and generate high-quality applications accelerated with Hailo's technology. The Hailo Model Zoo also provides re-training instructions to train the models on custom datasets and models that were trained for specific use cases on internal datasets.

Models
Hailo provides different pre-trained models in ONNX / TF formats and pre-compiled HEF (Hailo Executable Format) binary file to execute on the Hailo devices.

The models are divided to:

* Public models - which were trained on publicly available datasets.

    * For Hailo-15H - `Classification <docs/public_models/HAILO15H/HAILO15H_classification.rst>`_, `Object Detection <docs/public_models/HAILO15H/HAILO15H_object_detection.rst>`_, `Segmentation <docs/public_models/HAILO15H/HAILO15H_semantic_segmentation.rst>`_, `other tasks <docs/PUBLIC_MODELS.rst>`_

    * For Hailo-10H - `Classification <docs/public_models/HAILO10H/HAILO10H_classification.rst>`_, `Object Detection <docs/public_models/HAILO10H/HAILO10H_object_detection.rst>`_, `Segmentation <docs/public_models/HAILO10H/HAILO10H_semantic_segmentation.rst>`_, `other tasks <docs/PUBLIC_MODELS.rst>`_

    * For Hailo-15L - `Classification <docs/public_models/HAILO15L/HAILO15L_classification.rst>`_, `Object Detection <docs/public_models/HAILO15L/HAILO15L_object_detection.rst>`_, `Segmentation <docs/public_models/HAILO15L/HAILO15L_semantic_segmentation.rst>`_, `other tasks <docs/PUBLIC_MODELS.rst>`_

    **⚙️  Compatibility Notice:** The Hailo-8 and Hailo-8L devices are supported on the **Hailo Model Zoo v2.x** branch, in combination with the **Hailo Dataflow Compiler v3.x** branch. The `master` branch is intended for **Hailo-10** and **Hailo-15** devices only.

    * For Hailo-8 - `Classification <docs/public_models/HAILO8/HAILO8_classification.rst>`_, `Object Detection <docs/public_models/HAILO8/HAILO8_object_detection.rst>`_, `Segmentation <docs/public_models/HAILO8/HAILO8_semantic_segmentation.rst>`_, `other tasks <docs/PUBLIC_MODELS.rst>`_

    * For Hailo-8L - `Classification <docs/public_models/HAILO8L/HAILO8L_classification.rst>`_, `Object Detection <docs/public_models/HAILO8L/HAILO8L_object_detection.rst>`_, `Segmentation <docs/public_models/HAILO8L/HAILO8L_semantic_segmentation.rst>`_, `other tasks <docs/PUBLIC_MODELS.rst>`_



* | `HAILO MODELS <docs/HAILO_MODELS.rst>`_ which were trained in-house for specific use-cases on internal datasets.
  | Each Hailo Model is accompanied with retraining instructions.

What's new
----------

**Core stack updates** This release aligns with the latest version of the Hailo software stack, ensuring compatibility and performance improvements.

* Upgraded to Dataflow Compiler v5.5.0

* Upgraded to HailoRT v5.5.0

**New model support** We have expanded the Model Zoo with new tasks and architectures for object tracking, audio classification, sentence embeddings, OCR, depth estimation, super-resolution and real-time semantic segmentation:

* `STARK-Lightning <https://arxiv.org/abs/2103.17154>`_ - stark_lightning_repvgg_a0 - Single-object tracking, the first tracker in the Model Zoo
* `ByteTrack <https://github.com/ifzhang/ByteTrack>`_ - bytetrack_tiny_mot17, bytetrack_s_mot17, bytetrack_nano_mot17 (Hailo-15H) - Multi-object pedestrian tracking
* `YamNet <https://github.com/tensorflow/models/tree/master/research/audioset/yamnet>`_ - yamnet - AudioSet audio event classification, the first audio network in the Model Zoo
* `all-MiniLM-L6-v2 <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>`_ - all_minilm_l6_v2, all_minilm_l6_v2_v2a - Sentence embeddings, the first embedding model in the Model Zoo
* `MiVOLO <https://github.com/WildChlamydia/MiVOLO>`_ - mivolo_d1 - Age and gender estimation
* `DINOv2 <https://arxiv.org/abs/2304.07193>`_ - dinov2_vits14_lc - Self-supervised Vision Transformer backbone with a linear ImageNet head
* `ZipDepth <https://arxiv.org/abs/2607.08771>`_ - zipdepth_256x832, zipdepth_384x384 (Hailo-15H) - Lightweight zero-shot monocular depth estimation
* `PaddleOCR-v6 <https://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html>`_ - paddle_ocr_v6 detection and recognition models - the complete PP-OCRv6 OCR pipeline
* `SPAN <https://github.com/hongyuanyu/SPAN>`_ / `SPANF <https://github.com/Amazingren/NTIRE2025_ESR>`_ - span_x2, span_x4, spanf_x4 - Efficient super-resolution, the NTIRE 2024 and 2025 winners
* Real-time Cityscapes semantic segmentation - `PIDNet <https://github.com/XuJiacong/PIDNet>`_ (pidnet_s, pidnet_l), `PP-LiteSeg <https://arxiv.org/abs/2204.02681>`_ (pp_liteseg_t, pp_liteseg_b), `BiSeNetV2 <https://github.com/CoinCheung/BiSeNet>`_ (bisenetv2), `DDRNet <https://github.com/ydhongHIT/DDRNet>`_ (ddrnet_23_slim) and `YOLO26-sem <https://arxiv.org/abs/2606.03748>`_ (yolo26n_sem, yolo26s_sem, yolo26m_sem, yolo26l_sem)
* `CABiNet <https://github.com/dronefreak/CABiNet>`_ - cabinet_mnv3l_uavid - Aerial (UAV) semantic segmentation

**New variants and wider hardware support** Existing families gained new variants, and the CLIP-style encoders are now available on Hailo-15L:

* `YOLOv11-seg <https://docs.ultralytics.com/models/yolo11/>`_ - yolov11n_seg, yolov11s_seg, yolov11m_seg, yolov11l_seg, yolov11x_seg - Instance segmentation
* `YOLO26 <https://arxiv.org/abs/2606.03748>`_ - yolo26s_768_448, yolo26s_768_576 and yolo26x_seg (Hailo-15H)
* `YOLO26-sem <https://arxiv.org/abs/2606.03748>`_ aerial - yolo26n_sem and yolo26s_sem fine-tuned on UAVid and AeroScapes
* `YOLO-World <https://arxiv.org/abs/2401.17270>`_ - yolo_world_v2m - Open-vocabulary detection
* `SCRFD <https://arxiv.org/abs/2105.04714>`_ - scrfd_2.5g_384_640, scrfd_10g_384_640 - 384x640 face detectors
* `CLIP <https://arxiv.org/abs/2103.00020>`_, `SigLIP <https://arxiv.org/abs/2303.15343>`_, `SigLIP2 <https://arxiv.org/abs/2502.14786>`_ and `TinyCLIP <https://arxiv.org/pdf/2309.12314>`_ image and text encoders are now also released for Hailo-15L


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
* Install HailoRT (optional). Required only if you want to run on Hailo's. In case you are not Hailo customer please contact `hailo.ai <https://hailo.ai/company-overview/contact-us/>`_
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


Claude Code Skills
------------------

The Hailo Model Zoo ships three `Claude Code <https://claude.com/claude-code>`_ skills that walk you through the full DFC flow on a fresh model — auto-matching the Model Zoo's parser configs and ALLS optimization scripts when the model name is known, and falling back to interactive guidance when it is not:

* ``/hailo-parse <path-to-model> [hw_arch]`` — parses an ONNX or TensorFlow model into a Hailo HAR, looking up start/end nodes from `cfg/networks/ <hailo_model_zoo/cfg/networks>`_ when available.
* ``/hailo-optimize <parsed.har> [hw_arch]`` — applies a model script and quantizes the network, auto-loading a matching ALLS from `cfg/alls/ <hailo_model_zoo/cfg/alls>`_, validating calibration-set normalization, using GPU when available, and gating on the post-optimize SNR.
* ``/hailo-compile <optimized.har> [hw_arch]`` — compiles the optimized HAR into a deployable HEF and provides triage for common compilation failures.

The skills are auto-discovered when you launch Claude Code from inside the cloned Model Zoo (they live under ``.claude/skills/``). They use only the public ``hailo_sdk_client`` API from the DFC wheel, so the DFC remains a black box. See the DFC User Guide on the `developer-zone <https://hailo.ai/developer-zone/>`_ for the full DFC reference.


License
-------

The Hailo Model Zoo is released under the MIT license. Please see the `LICENSE <https://github.com/hailo-ai/hailo_model_zoo/blob/master/LICENSE>`_ file for more information.

Support & Issues
----------------

⚠️ **Issue reporting is disabled in this repository.**

For bug reports, feature requests, or discussions, please visit our `Hailo Community Forum <https://community.hailo.ai/>`_.

Changelog
---------

For further information please see our `CHANGELOG <docs/CHANGELOG.rst>`_ page.

----
