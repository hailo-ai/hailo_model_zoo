


Public Models
=============

* System host: Intel® Core™ i5-9400 CPU @ 2.90GHz
* Hailo Dataflow Compiler Version v5.4.0
* Measurement conditions: Measuring from the SoC, room temperature

|

Classification
==============

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

Imagenet
--------

|








.. list-table::
   :header-rows: 1
   :widths: 31 9 7 11 9 8 8 8 9

   * - Network Name
     - float Accuracy (top1)
     - Hardware Accuracy (top1)
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
     - Links
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
   
   
   
   
   
   
   

   * - cas_vit_m
     - 81.2
     - 81.1
     - 81.0
     - 207
     - | `S <https://github.com/Tianfang-Zhang/CAS-ViT>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/cas_vit_m/pretrained/2024-09-03/cas_vit_m.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/cas_vit_m_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/cas_vit_m.hef>`_
     - 384x384x3
     - 12.42
     - 10.89
   
   
   
   
   
   
   

   * - cas_vit_s
     - 79.9
     - 79.8
     - 111
     - 295
     - | `S <https://github.com/Tianfang-Zhang/CAS-ViT>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/cas_vit_s/pretrained/2024-08-13/cas_vit_s.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/cas_vit_s_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/cas_vit_s.hef>`_
     - 384x384x3
     - 5.5
     - 5.4
   
   
   
   
   
   
   

   * - cas_vit_t
     - 81.9
     - 81.6
     - 63.0
     - 135
     - | `S <https://github.com/Tianfang-Zhang/CAS-ViT>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/cas_vit_t/pretrained/2024-09-03/cas_vit_t.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/cas_vit_t_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/cas_vit_t.hef>`_
     - 384x384x3
     - 21.76
     - 20.85
   
   
   
   
   
   
   

   * - davit_tiny
     - 82.7
     - 82.3
     - 20.9
     - 47.2
     - | `S <https://huggingface.co/timm/davit_tiny.msft_in1k>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/davit_tiny/pretrained/2024-10-01/davit_tiny.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/davit_tiny_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/davit_tiny.hef>`_
     - 224x224x3
     - 28.36
     - 9.1
   
   
   
   
   
   
   

   * - deit_base
     - 80.9
     - 79.9
     - 52.4
     - 141
     - | `S <https://github.com/facebookresearch/deit>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/deit_base/pretrained/2024-05-21/deit_base.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/deit_base_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/deit_base.hef>`_
     - 224x224x3
     - 80.26
     - 35.22
   
   
   
   
   
   
   

   * - deit_small
     - 78.2
     - 77.5
     - 127
     - 426
     - | `S <https://github.com/facebookresearch/deit>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/deit_small/pretrained/2024-05-21/deit_small.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/deit_small_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/deit_small.hef>`_
     - 224x224x3
     - 20.52
     - 9.4
   
   
   
   
   
   
   

   * - deit_tiny
     - 69.1
     - 68.7
     - 147
     - 513
     - | `S <https://github.com/facebookresearch/deit>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/deit_tiny/pretrained/2024-05-21/deit_tiny.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/deit_tiny_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/deit_tiny.hef>`_
     - 224x224x3
     - 5.3
     - 2.57
   
   
   
   
   
   
   

   * - efficientformer_l1
     - 79.1
     - 76.6
     - 86.8
     - 164
     - | `S <https://github.com/snap-research/EfficientFormer/tree/main>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientformer_l1/pretrained/2024-08-11/efficientformer_l1.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/efficientformer_l1_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/efficientformer_l1.hef>`_
     - 224x224x3
     - 12.3
     - 2.6
   
   
   
   
   
   
   

   * - efficientnet_l
     - 80.5
     - 79.3
     - 126
     - 234
     - | `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_l/pretrained/2023-07-18/efficientnet_l.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/efficientnet_l_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/efficientnet_l.hef>`_
     - 300x300x3
     - 10.55
     - 19.4
   
   
   
   
   
   
   

   * - efficientnet_lite0
     - 75.0
     - 73.8
     - 2216
     - 2216
     - | `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite0/pretrained/2023-07-18/efficientnet_lite0.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/efficientnet_lite0_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/efficientnet_lite0.hef>`_
     - 224x224x3
     - 4.63
     - 0.78
   
   
   
   
   
   
   

   * - efficientnet_lite1
     - 76.7
     - 76.3
     - 962
     - 962
     - | `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite1/pretrained/2023-07-18/efficientnet_lite1.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/efficientnet_lite1_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/efficientnet_lite1.hef>`_
     - 240x240x3
     - 5.39
     - 1.22
   
   
   
   
   
   
   

   * - efficientnet_lite2
     - 77.5
     - 76.7
     - 208
     - 498
     - | `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite2/pretrained/2023-07-18/efficientnet_lite2.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/efficientnet_lite2_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/efficientnet_lite2.hef>`_
     - 260x260x3
     - 6.06
     - 1.74
   
   
   
   
   
   
   

   * - efficientnet_lite3
     - 79.3
     - 78.6
     - 165
     - 371
     - | `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite3/pretrained/2023-07-18/efficientnet_lite3.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/efficientnet_lite3_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/efficientnet_lite3.hef>`_
     - 280x280x3
     - 8.16
     - 2.8
   
   
   
   
   
   
   

   * - efficientnet_lite4
     - 80.8
     - 80.1
     - 138
     - 331
     - | `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite4/pretrained/2023-07-18/efficientnet_lite4.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/efficientnet_lite4_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/efficientnet_lite4.hef>`_
     - 300x300x3
     - 12.95
     - 5.10
   
   
   
   
   
   
   

   * - efficientnet_m
     - 78.9
     - 78.6
     - 550
     - 550
     - | `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_m/pretrained/2023-07-18/efficientnet_m.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/efficientnet_m_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/efficientnet_m.hef>`_
     - 240x240x3
     - 6.87
     - 7.32
   
   
   
   
   
   
   

   * - efficientnet_s
     - 77.6
     - 76.8
     - 903
     - 903
     - | `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_s/pretrained/2023-07-18/efficientnet_s.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/efficientnet_s_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/efficientnet_s.hef>`_
     - 224x224x3
     - 5.41
     - 4.72
   
   
   
   
   
   
   

   * - fastvit_sa12⭐
     - 79.8
     - 76.8
     - 284
     - 915
     - | `S <https://github.com/apple/ml-fastvit/tree/main>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/fastvit_sa12/pretrained/2023-08-21/fastvit_sa12.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/fastvit_sa12_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/fastvit_sa12.hef>`_
     - 224x224x3
     - 11.99
     - 3.59
   
   
   
   
   
   
   

   * - hardnet39ds⭐
     - 73.4
     - 73.0
     - 567
     - 1655
     - | `S <https://github.com/PingoLH/Pytorch-HarDNet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/hardnet39ds/pretrained/2021-07-20/hardnet39ds.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/hardnet39ds_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/hardnet39ds.hef>`_
     - 224x224x3
     - 3.48
     - 0.86
   
   
   
   
   
   
   

   * - hardnet68⭐
     - 75.5
     - 75.2
     - 220
     - 564
     - | `S <https://github.com/PingoLH/Pytorch-HarDNet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/hardnet68/pretrained/2021-07-20/hardnet68.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/hardnet68_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/hardnet68.hef>`_
     - 224x224x3
     - 17.56
     - 8.5
   
   
   
   
   
   
   

   * - inception_v1
     - 69.7
     - 69.5
     - 1308
     - 1308
     - | `S <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/inception_v1/pretrained/2023-07-18/inception_v1.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/inception_v1_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/inception_v1.hef>`_
     - 224x224x3
     - 6.62
     - 3
   
   
   
   
   
   
   

   * - levit128
     - 78.4
     - 76.4
     - 219
     - 905
     - | `S <https://github.com/facebookresearch/LeViT>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/levit_128/pretrained/2024-07-10/LeViT_128_simp.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/levit128_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/levit128.hef>`_
     - 224x224x3
     - 9.2
     - 0.8
   
   
   
   
   
   
   

   * - levit192
     - 79.7
     - 77.6
     - 230
     - 1000
     - | `S <https://github.com/facebookresearch/LeViT>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/levit_192/pretrained/2024-07-10/LeViT_192_simp.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/levit192_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/levit192.hef>`_
     - 224x224x3
     - 10.9
     - 1.3
   
   
   
   
   
   
   

   * - levit256
     - 81.4
     - 79.2
     - 174
     - 692
     - | `S <https://github.com/facebookresearch/LeViT>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/levit_256/2024-05-13/levit-256.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/levit256_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/levit256.hef>`_
     - 224x224x3
     - 18.9
     - 2.3
   
   
   
   
   
   
   

   * - levit384
     - 82.3
     - 79.2
     - 113
     - 431
     - | `S <https://github.com/facebookresearch/LeViT>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/levit_384/pretrained/2024-07-10/LeViT_384_simp.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/levit384_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/levit384.hef>`_
     - 224x224x3
     - 39.1
     - 4.7
   
   
   
   
   
   
   

   * - mobilenet_v1
     - 71.0
     - 70.4
     - 4160
     - 4159
     - | `S <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v1/pretrained/2023-07-18/mobilenet_v1.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/mobilenet_v1_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/mobilenet_v1.hef>`_
     - 224x224x3
     - 4.22
     - 1.14
   
   
   
   
   
   
   

   * - mobilenet_v2_1.0
     - 71.8
     - 71.0
     - 3456
     - 3455
     - | `S <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v2_1.0/pretrained/2025-01-15/mobilenet_v2_1.0.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/mobilenet_v2_1.0_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/mobilenet_v2_1.0.hef>`_
     - 224x224x3
     - 3.49
     - 0.62
   
   
   
   
   
   
   

   * - mobilenet_v2_1.4
     - 74.2
     - 73.2
     - 1818
     - 1818
     - | `S <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v2_1.4/pretrained/2021-07-11/mobilenet_v2_1.4.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/mobilenet_v2_1.4_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/mobilenet_v2_1.4.hef>`_
     - 224x224x3
     - 6.09
     - 1.18
   
   
   
   
   
   
   

   * - mobilenet_v3
     - 72.2
     - 71.8
     - 3283
     - 3283
     - | `S <https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v3/pretrained/2023-07-18/mobilenet_v3.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/mobilenet_v3_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/mobilenet_v3.hef>`_
     - 224x224x3
     - 4.07
     - 2
   
   
   
   
   
   
   

   * - nextvit_base
     - 83.2
     - 83.3
     - 48.5
     - 207
     - | `S <https://github.com/bytedance/Next-ViT>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/nextvit_base/pretrained/2024-08-22/nextvit_base_timm_sim.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/nextvit_base_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/nextvit_base.hef>`_
     - 224x224x3
     - 44.8
     - 16.6
   
   
   
   
   
   
   

   * - nextvit_small
     - 82.5
     - 82.5
     - 63.9
     - 279
     - | `S <https://github.com/bytedance/Next-ViT>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/nextvit_small/pretrained/2024-08-22/nextvit_small_timm_sim.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/nextvit_small_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/nextvit_small.hef>`_
     - 224x224x3
     - 31.7
     - 11.6
   
   
   
   
   
   
   

   * - poolformer_s12
     - 74.0
     - 73.1
     - 74.9
     - 285
     - | `S <https://github.com/sail-sg/poolformer>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/poolformer_s12/pretrained/2024-07-14/poolformer_s12.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/poolformer_s12_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/poolformer_s12.hef>`_
     - 224x224x3
     - 11.93
     - 3.67
   
   
   
   
   
   
   

   * - regnetx_1.6gf
     - 77.0
     - 76.7
     - 2742
     - 2742
     - | `S <https://github.com/facebookresearch/pycls>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/regnetx_1.6gf/pretrained/2021-07-11/regnetx_1.6gf.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/regnetx_1.6gf_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/regnetx_1.6gf.hef>`_
     - 224x224x3
     - 9.17
     - 3.22
   
   
   
   
   
   
   

   * - regnetx_800mf
     - 75.2
     - 74.9
     - 4779
     - 4778
     - | `S <https://github.com/facebookresearch/pycls>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/regnetx_800mf/pretrained/2021-07-11/regnetx_800mf.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/regnetx_800mf_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/regnetx_800mf.hef>`_
     - 224x224x3
     - 7.24
     - 1.6
   
   
   
   
   
   
   

   * - repghost_1_0x
     - 73.0
     - 72.2
     - 352
     - 1262
     - | `S <https://github.com/ChengpengChen/RepGhost>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/repghost/repghostnet_1_0x/pretrained/2023-04-03/repghostnet_1_0x.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/repghost_1_0x_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/repghost_1_0x.hef>`_
     - 224x224x3
     - 4.1
     - 0.28
   
   
   
   
   
   
   

   * - repghost_2_0x
     - 77.2
     - 76.9
     - 210
     - 781
     - | `S <https://github.com/ChengpengChen/RepGhost>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/repghost/repghostnet_2_0x/pretrained/2023-04-03/repghostnet_2_0x.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/repghost_2_0x_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/repghost_2_0x.hef>`_
     - 224x224x3
     - 9.8
     - 1.04
   
   
   
   
   
   
   

   * - repvgg_a1
     - 74.4
     - 72.4
     - 2019
     - 2019
     - | `S <https://github.com/DingXiaoH/RepVGG>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/repvgg/repvgg_a1/pretrained/2022-10-02/RepVGG-A1.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/repvgg_a1_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/repvgg_a1.hef>`_
     - 224x224x3
     - 12.79
     - 4.7
   
   
   
   
   
   
   

   * - repvgg_a2
     - 76.5
     - 74.4
     - 300
     - 644
     - | `S <https://github.com/DingXiaoH/RepVGG>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/repvgg/repvgg_a2/pretrained/2022-10-02/RepVGG-A2.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/repvgg_a2_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/repvgg_a2.hef>`_
     - 224x224x3
     - 25.5
     - 10.2
   
   
   
   
   
   
   

   * - resmlp12_relu
     - 75.3
     - 75.0
     - 150
     - 688
     - | `S <https://github.com/rwightman/pytorch-image-models/>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resmlp12_relu/pretrained/2022-03-03/resmlp12_relu.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/resmlp12_relu_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/resmlp12_relu.hef>`_
     - 224x224x3
     - 15.77
     - 6.04
   
   
   
   
   
   
   

   * - resnet_v1_18⭐
     - 71.3
     - 70.8
     - 2709
     - 2709
     - | `S <https://github.com/yhhhli/BRECQ>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v1_18/pretrained/2022-04-19/resnet_v1_18.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/resnet_v1_18_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/resnet_v1_18.hef>`_
     - 224x224x3
     - 11.68
     - 3.64
   
   
   
   
   
   
   

   * - resnet_v1_34⭐
     - 72.7
     - 72.2
     - 364
     - 1042
     - | `S <https://github.com/tensorflow/models/tree/master/research/slim>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v1_34/pretrained/2025-01-15/resnet_v1_34.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/resnet_v1_34_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/resnet_v1_34.hef>`_
     - 224x224x3
     - 21.79
     - 7.34
   
   
   
   
   
   
   

   * - resnet_v1_50⭐
     - 75.2
     - 74.7
     - 332
     - 1047
     - | `S <https://github.com/tensorflow/models/tree/master/research/slim>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v1_50/pretrained/2025-01-15/resnet_v1_50.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/resnet_v1_50_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/resnet_v1_50.hef>`_
     - 224x224x3
     - 25.53
     - 6.98
   
   
   
   
   
   
   

   * - resnext26_32x4d
     - 76.2
     - 75.9
     - 861
     - 861
     - | `S <https://github.com/osmr/imgclsmob/tree/master/pytorch>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnext26_32x4d/pretrained/2023-09-18/resnext26_32x4d.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/resnext26_32x4d_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/resnext26_32x4d.hef>`_
     - 224x224x3
     - 15.37
     - 4.96
   
   
   
   
   
   
   

   * - resnext50_32x4d
     - 79.3
     - 78.3
     - 252
     - 640
     - | `S <https://github.com/osmr/imgclsmob/tree/master/pytorch>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnext50_32x4d/pretrained/2023-07-18/resnext50_32x4d.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/resnext50_32x4d_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/resnext50_32x4d.hef>`_
     - 224x224x3
     - 24.99
     - 8.48
   
   
   
   
   
   
   

   * - squeezenet_v1.1
     - 59.8
     - 59.3
     - 4310
     - 4310
     - | `S <https://github.com/osmr/imgclsmob/tree/master/pytorch>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/squeezenet_v1.1/pretrained/2023-07-18/squeezenet_v1.1.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/squeezenet_v1.1_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/squeezenet_v1.1.hef>`_
     - 224x224x3
     - 1.24
     - 0.78
   
   
   
   
   
   
   

   * - swin_small
     - 83.1
     - 80.0
     - 27.1
     - 89.4
     - | `S <https://huggingface.co/microsoft/swin-small-patch4-window7-224>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/swin_small/pretrained/2024-08-01/swin_small_classifier.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/swin_small_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/swin_small.hef>`_
     - 224x224x3
     - 50
     - 17.6
   
   
   
   
   
   
   

   * - swin_tiny
     - 81.3
     - 79.4
     - 49.8
     - 148
     - | `S <https://huggingface.co/microsoft/swin-tiny-patch4-window7-224>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/swin_tiny/pretrained/2024-08-01/swin_tiny_classifier.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/swin_tiny_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/swin_tiny.hef>`_
     - 224x224x3
     - 29
     - 9.1
   
   
   
   
   
   
   

   * - vit_base
     - 84.5
     - 83.6
     - 58.1
     - 178
     - | `S <https://github.com/rwightman/pytorch-image-models>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_base/pretrained/2024-04-03/vit_base_patch16_224_ops17.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/vit_base_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/vit_base.hef>`_
     - 224x224x3
     - 86.5
     - 35.188
   
   
   
   
   
   
   

   * - vit_base_bn
     - 80.0
     - 79.1
     - 72.2
     - 209
     - | `S <https://github.com/rwightman/pytorch-image-models>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_base_bn/pretrained/2023-01-25/vit_base.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/vit_base_bn_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/vit_base_bn.hef>`_
     - 224x224x3
     - 86.5
     - 35.188
   
   
   
   
   
   
   

   * - vit_large
     - 83.2
     - 82.5
     - 18.9
     - 51.6
     - | `S <github.com/google-research/vision_transformer>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_large/pretrained/2024-05-19/vit_large.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/vit_large_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/vit_large.hef>`_
     - 224x224x3
     - 304.2
     - 123.4
   
   
   
   
   
   
   

   * - vit_small
     - 81.5
     - 80.5
     - 117
     - 416
     - | `S <https://github.com/rwightman/pytorch-image-models>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_small/pretrained/2024-04-03/vit_small_patch16_224_ops17.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/vit_small_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/vit_small.hef>`_
     - 224x224x3
     - 21.12
     - 8.62
   
   
   
   
   
   
   

   * - vit_small_bn
     - 78.1
     - 77.3
     - 166
     - 664
     - | `S <https://github.com/rwightman/pytorch-image-models>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_small_bn/pretrained/2022-08-08/vit_small.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/vit_small_bn_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/vit_small_bn.hef>`_
     - 224x224x3
     - 21.12
     - 8.62
   
   
   
   
   
   
   

   * - vit_tiny
     - 75.5
     - 74.7
     - 148
     - 513
     - | `S <https://github.com/rwightman/pytorch-image-models>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_tiny/pretrained/2024-04-03/vit_tiny_patch16_224_ops17.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/vit_tiny_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/vit_tiny.hef>`_
     - 224x224x3
     - 5.73
     - 2.2
   
   
   
   
   
   
   

   * - vit_tiny_bn
     - 69.0
     - 67.4
     - 333
     - 1498
     - | `S <https://github.com/rwightman/pytorch-image-models>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_tiny_bn/pretrained/2023-08-29/vit_tiny_bn.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/vit_tiny_bn_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo15h/vit_tiny_bn.hef>`_
     - 224x224x3
     - 5.73
     - 2.2