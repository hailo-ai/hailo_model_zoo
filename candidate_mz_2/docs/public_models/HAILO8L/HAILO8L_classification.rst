


Public Models
=============

All models were compiled using Hailo Dataflow Compiler v2.18.0.

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
     - 80.9
     - 39.0
     - 103
     - | `S <https://github.com/Tianfang-Zhang/CAS-ViT>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/cas_vit_m/pretrained/2024-09-03/cas_vit_m.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/cas_vit_m_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/cas_vit_m.hef>`_
     - 384x384x3
     - 12.42
     - 10.89
   
   
   
   
   
   
   

   * - cas_vit_s
     - 79.9
     - 79.7
     - 55.3
     - 153
     - | `S <https://github.com/Tianfang-Zhang/CAS-ViT>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/cas_vit_s/pretrained/2024-08-13/cas_vit_s.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/cas_vit_s_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/cas_vit_s.hef>`_
     - 384x384x3
     - 5.5
     - 5.4
   
   
   
   
   
   
   

   * - cas_vit_t
     - 81.9
     - 81.6
     - 27.3
     - 63.8
     - | `S <https://github.com/Tianfang-Zhang/CAS-ViT>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/cas_vit_t/pretrained/2024-09-03/cas_vit_t.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/cas_vit_t_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/cas_vit_t.hef>`_
     - 384x384x3
     - 21.76
     - 20.85
   
   
   
   
   
   
   

   * - davit_tiny
     - 82.7
     - 82.2
     - 12.5
     - 26.1
     - | `S <https://huggingface.co/timm/davit_tiny.msft_in1k>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/davit_tiny/pretrained/2024-10-01/davit_tiny.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/davit_tiny_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/davit_tiny.hef>`_
     - 224x224x3
     - 28.36
     - 9.1
   
   
   
   
   
   
   

   * - deit_base
     - 80.9
     - 80.4
     - 21.6
     - 59.5
     - | `S <https://github.com/facebookresearch/deit>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/deit_base/pretrained/2024-05-21/deit_base.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/deit_base_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/deit_base.hef>`_
     - 224x224x3
     - 80.26
     - 35.22
   
   
   
   
   
   
   

   * - deit_small
     - 78.2
     - 77.0
     - 54.0
     - 203
     - | `S <https://github.com/facebookresearch/deit>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/deit_small/pretrained/2024-05-21/deit_small.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/deit_small_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/deit_small.hef>`_
     - 224x224x3
     - 20.52
     - 9.4
   
   
   
   
   
   
   

   * - deit_tiny
     - 69.1
     - 68.5
     - 68.6
     - 258
     - | `S <https://github.com/facebookresearch/deit>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/deit_tiny/pretrained/2024-05-21/deit_tiny.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/deit_tiny_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/deit_tiny.hef>`_
     - 224x224x3
     - 5.3
     - 2.57
   
   
   
   
   
   
   

   * - efficientformer_l1
     - 79.1
     - 76.1
     - 80.1
     - 205
     - | `S <https://github.com/snap-research/EfficientFormer/tree/main>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientformer_l1/pretrained/2024-08-11/efficientformer_l1.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/efficientformer_l1_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/efficientformer_l1.hef>`_
     - 224x224x3
     - 12.3
     - 2.6
   
   
   
   
   
   
   

   * - efficientnet_l
     - 80.5
     - 79.3
     - 78.6
     - 162
     - | `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_l/pretrained/2023-07-18/efficientnet_l.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/efficientnet_l_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/efficientnet_l.hef>`_
     - 300x300x3
     - 10.55
     - 19.4
   
   
   
   
   
   
   

   * - efficientnet_lite0
     - 75.0
     - 73.9
     - 1057
     - 1057
     - | `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite0/pretrained/2023-07-18/efficientnet_lite0.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/efficientnet_lite0_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/efficientnet_lite0.hef>`_
     - 224x224x3
     - 4.63
     - 0.78
   
   
   
   
   
   
   

   * - efficientnet_lite1
     - 76.7
     - 76.3
     - 474
     - 474
     - | `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite1/pretrained/2023-07-18/efficientnet_lite1.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/efficientnet_lite1_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/efficientnet_lite1.hef>`_
     - 240x240x3
     - 5.39
     - 1.22
   
   
   
   
   
   
   

   * - efficientnet_lite2
     - 77.5
     - 76.7
     - 109
     - 274
     - | `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite2/pretrained/2023-07-18/efficientnet_lite2.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/efficientnet_lite2_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/efficientnet_lite2.hef>`_
     - 260x260x3
     - 6.06
     - 1.74
   
   
   
   
   
   
   

   * - efficientnet_lite3
     - 79.3
     - 78.6
     - 111
     - 305
     - | `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite3/pretrained/2023-07-18/efficientnet_lite3.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/efficientnet_lite3_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/efficientnet_lite3.hef>`_
     - 280x280x3
     - 8.16
     - 2.8
   
   
   
   
   
   
   

   * - efficientnet_lite4
     - 80.8
     - 80.1
     - 73.5
     - 196
     - | `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_lite4/pretrained/2023-07-18/efficientnet_lite4.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/efficientnet_lite4_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/efficientnet_lite4.hef>`_
     - 300x300x3
     - 12.95
     - 5.10
   
   
   
   
   
   
   

   * - efficientnet_m
     - 78.9
     - 78.5
     - 155
     - 408
     - | `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_m/pretrained/2023-07-18/efficientnet_m.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/efficientnet_m_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/efficientnet_m.hef>`_
     - 240x240x3
     - 6.87
     - 7.32
   
   
   
   
   
   
   

   * - efficientnet_s
     - 77.6
     - 77.2
     - 548
     - 548
     - | `S <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/efficientnet_s/pretrained/2023-07-18/efficientnet_s.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/efficientnet_s_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/efficientnet_s.hef>`_
     - 224x224x3
     - 5.41
     - 4.72
   
   
   
   
   
   
   

   * - fastvit_sa12⭐
     - 79.8
     - 76.7
     - 138
     - 535
     - | `S <https://github.com/apple/ml-fastvit/tree/main>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/fastvit_sa12/pretrained/2023-08-21/fastvit_sa12.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/fastvit_sa12_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/fastvit_sa12.hef>`_
     - 224x224x3
     - 11.99
     - 3.59
   
   
   
   
   
   
   

   * - hardnet39ds⭐
     - 73.4
     - 73.1
     - 277
     - 933
     - | `S <https://github.com/PingoLH/Pytorch-HarDNet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/hardnet39ds/pretrained/2021-07-20/hardnet39ds.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/hardnet39ds_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/hardnet39ds.hef>`_
     - 224x224x3
     - 3.48
     - 0.86
   
   
   
   
   
   
   

   * - hardnet68⭐
     - 75.5
     - 75.2
     - 115
     - 298
     - | `S <https://github.com/PingoLH/Pytorch-HarDNet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/hardnet68/pretrained/2021-07-20/hardnet68.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/hardnet68_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/hardnet68.hef>`_
     - 224x224x3
     - 17.56
     - 8.5
   
   
   
   
   
   
   

   * - inception_v1
     - 69.7
     - 69.5
     - 313
     - 882
     - | `S <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/inception_v1/pretrained/2023-07-18/inception_v1.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/inception_v1_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/inception_v1.hef>`_
     - 224x224x3
     - 6.62
     - 3
   
   
   
   
   
   
   

   * - levit128
     - 78.4
     - 76.0
     - 103
     - 494
     - | `S <https://github.com/facebookresearch/LeViT>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/levit_128/pretrained/2024-07-10/LeViT_128_simp.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/levit128_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/levit128.hef>`_
     - 224x224x3
     - 9.2
     - 0.8
   
   
   
   
   
   
   

   * - levit192
     - 79.7
     - 77.2
     - 98.6
     - 494
     - | `S <https://github.com/facebookresearch/LeViT>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/levit_192/pretrained/2024-07-10/LeViT_192_simp.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/levit192_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/levit192.hef>`_
     - 224x224x3
     - 10.9
     - 1.3
   
   
   
   
   
   
   

   * - levit256
     - 81.4
     - 79.1
     - 79.5
     - 389
     - | `S <https://github.com/facebookresearch/LeViT>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/levit_256/2024-05-13/levit-256.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/levit256_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/levit256.hef>`_
     - 224x224x3
     - 18.9
     - 2.3
   
   
   
   
   
   
   

   * - levit384
     - 82.3
     - 78.6
     - 51.4
     - 233
     - | `S <https://github.com/facebookresearch/LeViT>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/levit_384/pretrained/2024-07-10/LeViT_384_simp.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/levit384_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/levit384.hef>`_
     - 224x224x3
     - 39.1
     - 4.7
   
   
   
   
   
   
   

   * - mobilenet_v1
     - 71.0
     - 70.3
     - 2565
     - 2565
     - | `S <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v1/pretrained/2023-07-18/mobilenet_v1.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/mobilenet_v1_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/mobilenet_v1.hef>`_
     - 224x224x3
     - 4.22
     - 1.14
   
   
   
   
   
   
   

   * - mobilenet_v2_1.0
     - 71.8
     - 70.9
     - 1738
     - 1738
     - | `S <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v2_1.0/pretrained/2025-01-15/mobilenet_v2_1.0.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/mobilenet_v2_1.0_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/mobilenet_v2_1.0.hef>`_
     - 224x224x3
     - 3.49
     - 0.62
   
   
   
   
   
   
   

   * - mobilenet_v2_1.4
     - 74.2
     - 73.3
     - 1098
     - 1098
     - | `S <https://github.com/tensorflow/models/tree/v1.13.0/research/slim>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v2_1.4/pretrained/2021-07-11/mobilenet_v2_1.4.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/mobilenet_v2_1.4_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/mobilenet_v2_1.4.hef>`_
     - 224x224x3
     - 6.09
     - 1.18
   
   
   
   
   
   
   

   * - mobilenet_v3
     - 72.2
     - 71.8
     - 1936
     - 1936
     - | `S <https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/mobilenet_v3/pretrained/2023-07-18/mobilenet_v3.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/mobilenet_v3_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/mobilenet_v3.hef>`_
     - 224x224x3
     - 4.07
     - 2
   
   
   
   
   
   
   

   * - regnetx_1.6gf
     - 77.0
     - 76.8
     - 1339
     - 1339
     - | `S <https://github.com/facebookresearch/pycls>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/regnetx_1.6gf/pretrained/2021-07-11/regnetx_1.6gf.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/regnetx_1.6gf_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/regnetx_1.6gf.hef>`_
     - 224x224x3
     - 9.17
     - 3.22
   
   
   
   
   
   
   

   * - regnetx_800mf
     - 75.2
     - 74.9
     - 2450
     - 2450
     - | `S <https://github.com/facebookresearch/pycls>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/regnetx_800mf/pretrained/2021-07-11/regnetx_800mf.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/regnetx_800mf_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/regnetx_800mf.hef>`_
     - 224x224x3
     - 7.24
     - 1.6
   
   
   
   
   
   
   

   * - repghost_1_0x
     - 73.0
     - 72.2
     - 189
     - 804
     - | `S <https://github.com/ChengpengChen/RepGhost>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/repghost/repghostnet_1_0x/pretrained/2023-04-03/repghostnet_1_0x.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/repghost_1_0x_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/repghost_1_0x.hef>`_
     - 224x224x3
     - 4.1
     - 0.28
   
   
   
   
   
   
   

   * - repghost_2_0x
     - 77.2
     - 76.9
     - 112
     - 427
     - | `S <https://github.com/ChengpengChen/RepGhost>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/repghost/repghostnet_2_0x/pretrained/2023-04-03/repghostnet_2_0x.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/repghost_2_0x_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/repghost_2_0x.hef>`_
     - 224x224x3
     - 9.8
     - 1.04
   
   
   
   
   
   
   

   * - repvgg_a1
     - 74.4
     - 72.4
     - 1346
     - 1346
     - | `S <https://github.com/DingXiaoH/RepVGG>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/repvgg/repvgg_a1/pretrained/2022-10-02/RepVGG-A1.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/repvgg_a1_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/repvgg_a1.hef>`_
     - 224x224x3
     - 12.79
     - 4.7
   
   
   
   
   
   
   

   * - repvgg_a2
     - 76.5
     - 74.3
     - 145
     - 335
     - | `S <https://github.com/DingXiaoH/RepVGG>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/repvgg/repvgg_a2/pretrained/2022-10-02/RepVGG-A2.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/repvgg_a2_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/repvgg_a2.hef>`_
     - 224x224x3
     - 25.5
     - 10.2
   
   
   
   
   
   
   

   * - resmlp12_relu
     - 75.3
     - 75.0
     - 47.9
     - 200
     - | `S <https://github.com/rwightman/pytorch-image-models/>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resmlp12_relu/pretrained/2022-03-03/resmlp12_relu.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/resmlp12_relu_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/resmlp12_relu.hef>`_
     - 224x224x3
     - 15.77
     - 6.04
   
   
   
   
   
   
   

   * - resnet_v1_18⭐
     - 71.3
     - 71.1
     - 1281
     - 1281
     - | `S <https://github.com/yhhhli/BRECQ>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v1_18/pretrained/2022-04-19/resnet_v1_18.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/resnet_v1_18_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/resnet_v1_18.hef>`_
     - 224x224x3
     - 11.68
     - 3.64
   
   
   
   
   
   
   

   * - resnet_v1_34⭐
     - 72.7
     - 72.2
     - 160
     - 483
     - | `S <https://github.com/tensorflow/models/tree/master/research/slim>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v1_34/pretrained/2025-01-15/resnet_v1_34.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/resnet_v1_34_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/resnet_v1_34.hef>`_
     - 224x224x3
     - 21.79
     - 7.34
   
   
   
   
   
   
   

   * - resnet_v1_50⭐
     - 75.2
     - 74.7
     - 138
     - 502
     - | `S <https://github.com/tensorflow/models/tree/master/research/slim>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnet_v1_50/pretrained/2025-01-15/resnet_v1_50.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/resnet_v1_50_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/resnet_v1_50.hef>`_
     - 224x224x3
     - 25.53
     - 6.98
   
   
   
   
   
   
   

   * - resnext26_32x4d
     - 76.2
     - 75.9
     - 200
     - 565
     - | `S <https://github.com/osmr/imgclsmob/tree/master/pytorch>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnext26_32x4d/pretrained/2023-09-18/resnext26_32x4d.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/resnext26_32x4d_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/resnext26_32x4d.hef>`_
     - 224x224x3
     - 15.37
     - 4.96
   
   
   
   
   
   
   

   * - resnext50_32x4d
     - 79.3
     - 78.4
     - 126
     - 348
     - | `S <https://github.com/osmr/imgclsmob/tree/master/pytorch>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/resnext50_32x4d/pretrained/2023-07-18/resnext50_32x4d.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/resnext50_32x4d_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/resnext50_32x4d.hef>`_
     - 224x224x3
     - 24.99
     - 8.48
   
   
   
   
   
   
   

   * - squeezenet_v1.1
     - 59.8
     - 59.4
     - 2514
     - 2514
     - | `S <https://github.com/osmr/imgclsmob/tree/master/pytorch>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/squeezenet_v1.1/pretrained/2023-07-18/squeezenet_v1.1.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/squeezenet_v1.1_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/squeezenet_v1.1.hef>`_
     - 224x224x3
     - 1.24
     - 0.78
   
   
   
   
   
   
   

   * - swin_small
     - 83.1
     - 79.9
     - 12.9
     - 31.2
     - | `S <https://huggingface.co/microsoft/swin-small-patch4-window7-224>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/swin_small/pretrained/2024-08-01/swin_small_classifier.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/swin_small_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/swin_small.hef>`_
     - 224x224x3
     - 50
     - 17.6
   
   
   
   
   
   
   

   * - swin_tiny
     - 81.3
     - 79.3
     - 23.0
     - 55.1
     - | `S <https://huggingface.co/microsoft/swin-tiny-patch4-window7-224>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/swin_tiny/pretrained/2024-08-01/swin_tiny_classifier.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/swin_tiny_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/swin_tiny.hef>`_
     - 224x224x3
     - 29
     - 9.1
   
   
   
   
   
   
   

   * - vit_base
     - 84.5
     - 83.2
     - 21.9
     - 60.9
     - | `S <https://github.com/rwightman/pytorch-image-models>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_base/pretrained/2024-04-03/vit_base_patch16_224_ops17.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/vit_base_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/vit_base.hef>`_
     - 224x224x3
     - 86.5
     - 35.188
   
   
   
   
   
   
   

   * - vit_base_bn
     - 80.0
     - 78.8
     - 35.6
     - 121
     - | `S <https://github.com/rwightman/pytorch-image-models>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_base_bn/pretrained/2023-01-25/vit_base.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/vit_base_bn_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/vit_base_bn.hef>`_
     - 224x224x3
     - 86.5
     - 35.188
   
   
   
   
   
   
   

   * - vit_small
     - 81.5
     - 80.0
     - 54.7
     - 205
     - | `S <https://github.com/rwightman/pytorch-image-models>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_small/pretrained/2024-04-03/vit_small_patch16_224_ops17.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/vit_small_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/vit_small.hef>`_
     - 224x224x3
     - 21.12
     - 8.62
   
   
   
   
   
   
   

   * - vit_small_bn
     - 78.1
     - 77.1
     - 98.0
     - 414
     - | `S <https://github.com/rwightman/pytorch-image-models>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_small_bn/pretrained/2022-08-08/vit_small.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/vit_small_bn_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/vit_small_bn.hef>`_
     - 224x224x3
     - 21.12
     - 8.62
   
   
   
   
   
   
   

   * - vit_tiny
     - 75.5
     - 73.6
     - 68.5
     - 258
     - | `S <https://github.com/rwightman/pytorch-image-models>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_tiny/pretrained/2024-04-03/vit_tiny_patch16_224_ops17.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/vit_tiny_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/vit_tiny.hef>`_
     - 224x224x3
     - 5.73
     - 2.2
   
   
   
   
   
   
   

   * - vit_tiny_bn
     - 69.0
     - 67.2
     - 193
     - 951
     - | `S <https://github.com/rwightman/pytorch-image-models>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Classification/vit_tiny_bn/pretrained/2023-08-29/vit_tiny_bn.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/vit_tiny_bn_profiler_results_compiled.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/vit_tiny_bn.hef>`_
     - 224x224x3
     - 5.73
     - 2.2