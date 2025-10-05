
Public Pre-Trained Models
=========================

.. |rocket| image:: ../../images/rocket.png
  :width: 18

.. |star| image:: ../../images/star.png
  :width: 18

Here, we give the full list of publicly pre-trained models supported by the Hailo Model Zoo.

* Benchmark Networks are marked with |rocket|
* Networks available in `TAPPAS <https://github.com/hailo-ai/tappas>`_ are marked with |star|
* Benchmark and TAPPAS  networks run in performance mode
* All models were compiled using Hailo Dataflow Compiler v3.33.0

Link Legend

The following shortcuts are used in the table below to indicate available resources for each model:

* S – Source: Link to the model’s open-source code repository.
* PT – Pretrained: Download the pretrained model file (compressed in ZIP format).
* H, NV, X – Compiled Models: Links to the compiled model in various formats:
            * H: regular HEF with RGB format
            * NV: HEF with NV12 format
            * X: HEF with RGBX format

* PR – Profiler Report: Download the model’s performance profiling report.
.. list-table::
   :widths: 31 9 7 11 9 8 8 8 9
   :header-rows: 1

   * - Network Name
     - float mAP
     - Hardware mAP
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
   * - clip_resnet_50_text_encoder
     - 88.8
     - 83.6
     - 21
     - 51
     - `S <https://huggingface.co/timm/resnet50x4_clip.openai>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_resnet_50x4/text_encoder/pretrained/2024-09-16/clip_text_encoder_resnet50x4.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8l/clip_resnet_50x4_text_encoder.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8l/clip_resnet_50x4_text_encoder_profiler_results_compiled.html>`_
     - 1x77x512
     - 37.8
     - 6
   * - clip_resnet_50x4_text_encoder
     - 91.2
     - 88.3
     - 16
     - 37
     - `S <https://huggingface.co/timm/resnet50x4_clip.openai>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_resnet_50x4/text_encoder/pretrained/2024-09-16/clip_text_encoder_resnet50x4.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8l/clip_resnet_50x4_text_encoder.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8l/clip_resnet_50x4_text_encoder_profiler_results_compiled.html>`_
     - 1x77x640
     - 59.1
     - 9.3
   * - clip_vit_b_16_text_encoder
     - 90.9
     - 90
     - 22
     - 52
     - `S <https://huggingface.co/openai/clip-vit-base-patch16>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_base_patch16_224/text_encoder/pretrained/2024-12-04/clip_text_encoder_vitb_16_sim.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8l/clip_vit_b_16_text_encoder.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8l/clip_vit_b_16_text_encoder_profiler_results_compiled.html>`_
     - 1x77x512
     - 37.8
     - 6
   * - clip_vit_b_32_text_encoder
     - 90.6
     - 88.7
     - 26
     - 66
     - `S <https://huggingface.co/openai/clip-vit-base-patch32>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ZeroShotClassification/clip/clip_vit_base_patch32_224/text_encoder/pretrained/2024-12-04/clip_text_encoder_vitb_32_sim.zip>`_ `H <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8l/clip_vit_b_32_text_encoder.hef>`_ `PR <https://hailo-model-zoo.s3.eu-west2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8l/clip_vit_b_32_text_encoder_profiler_results_compiled.html>`_
     - 1x77x512
     - 37.8
     - 6
