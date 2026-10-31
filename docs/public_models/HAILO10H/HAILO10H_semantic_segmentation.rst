


Public Models
=============

* System host: Intel® Core™ i5-9400 CPU @ 2.90GHz
* Hailo Dataflow Compiler Version v5.5.0
* Measurement conditions: PCIe Gen 3 x 4 lanes, room temperature

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
   
   
   
   
   
   
   

   * - bisenetv2
     - 73.0
     - 72.5
     - 36.7
     - 41.9
     - | `S <https://github.com/CoinCheung/BiSeNet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Cityscapes/bisenetv2/pretrained/2026-09-17/bisenetv2.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/bisenetv2_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/bisenetv2.hef>`_
     - 1024x2048x3
     - 3.36
     - 196.84
   
   
   
   
   
   
   

   * - ddrnet_23_slim
     - 76.7
     - 76.4
     - 40.4
     - 49.2
     - | `S <https://github.com/ydhongHIT/DDRNet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Cityscapes/ddrnet_23_slim/pretrained/2026-09-17/ddrnet_23_slim.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/ddrnet_23_slim_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/ddrnet_23_slim.hef>`_
     - 1024x2048x3
     - 5.70
     - 72.60
   
   
   
   
   
   
   

   * - fcn8_resnet_v1_18⭐
     - 69.4
     - 69.2
     - 31.1
     - 38.2
     - | `S <https://mmsegmentation.readthedocs.io/en/latest>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Cityscapes/fcn8_resnet_v1_18/pretrained/2023-06-22/fcn8_resnet_v1_18.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/fcn8_resnet_v1_18_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/fcn8_resnet_v1_18.hef>`_
     - 1024x1920x3
     - 11.20
     - 142.82
   
   
   
   
   
   
   

   * - pidnet_l
     - 80.3
     - 80.3
     - 14.5
     - 14.7
     - | `S <https://github.com/XuJiacong/PIDNet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Cityscapes/pidnet_l/pretrained/2026-09-16/pidnet_l.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/pidnet_l_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/pidnet_l.hef>`_
     - 1024x2048x3
     - 36.94
     - 550.05
   
   
   
   
   
   
   

   * - pidnet_s
     - 77.8
     - 77.7
     - 52.2
     - 63.3
     - | `S <https://github.com/XuJiacong/PIDNet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Cityscapes/pidnet_s/pretrained/2026-09-16/pidnet_s.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/pidnet_s_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/pidnet_s.hef>`_
     - 1024x2048x3
     - 7.63
     - 94.48
   
   
   
   
   
   
   

   * - pp_liteseg_b
     - 78.4
     - 78.3
     - 26.1
     - 34.8
     - | `S <https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.9/configs/pp_liteseg>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Cityscapes/pp_liteseg_b/pretrained/2026-09-23/pp_liteseg_b.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/pp_liteseg_b_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/pp_liteseg_b.hef>`_
     - 1024x2048x3
     - 12.23
     - 153.83
   
   
   
   
   
   
   

   * - pp_liteseg_t
     - 76.4
     - 76.0
     - 32.7
     - 41.7
     - | `S <https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.9/configs/pp_liteseg>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Cityscapes/pp_liteseg_t/pretrained/2026-09-23/pp_liteseg_t.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/pp_liteseg_t_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/pp_liteseg_t.hef>`_
     - 1024x2048x3
     - 8.02
     - 91.38
   
   
   
   
   
   
   

   * - segformer_b0_bn⭐
     - 69.8
     - 68.2
     - 17.2
     - 24.8
     - | `S <https://github.com/NVlabs/SegFormer>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Cityscapes/segformer_b0_512x1024_bn/pretrained/2023-09-04/segformer_b0_512x1024_bn.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/segformer_b0_bn_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/segformer_b0_bn.hef>`_
     - 512x1024x3
     - 3.72
     - 35.76
   
   
   
   
   
   
   

   * - stdc1⭐
     - 74.6
     - 73.7
     - 32.1
     - 40.9
     - | `S <https://mmsegmentation.readthedocs.io/en/latest>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Cityscapes/stdc1/pretrained/2023-06-12/stdc1.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/stdc1_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/stdc1.hef>`_
     - 1024x1920x3
     - 8.27
     - 126.47
   
   
   
   
   
   
   

   * - yolo26l_sem
     - 82.0
     - 81.7
     - 13.7
     - 
     - | `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Cityscapes/yolo26l_sem/pretrained/2026-05-28/yolo26l-sem.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/yolo26l_sem_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/yolo26l_sem.hef>`_
     - 1024x2048x3
     - 17.85
     - 388.3
   
   
   
   
   
   
   

   * - yolo26m_sem
     - 81.3
     - 81.0
     - 20.8
     - 
     - | `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Cityscapes/yolo26m_sem/pretrained/2026-05-28/yolo26m-sem.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/yolo26m_sem_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/yolo26m_sem.hef>`_
     - 1024x2048x3
     - 14.30
     - 305.6
   
   
   
   
   
   
   

   * - yolo26n_sem
     - 77.4
     - 76.0
     - 106
     - 152
     - | `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Cityscapes/yolo26n_sem/pretrained/2026-05-28/yolo26n-sem.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/yolo26n_sem_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/yolo26n_sem.hef>`_
     - 1024x2048x3
     - 1.63
     - 23.9
   
   
   
   
   
   
   

   * - yolo26s_sem
     - 80.0
     - 79.6
     - 57.2
     - 80.0
     - | `S <https://github.com/ultralytics/ultralytics>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Cityscapes/yolo26s_sem/pretrained/2026-05-28/yolo26s-sem.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/yolo26s_sem_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/yolo26s_sem.hef>`_
     - 1024x2048x3
     - 6.49
     - 91.1

|

Uavid
-----

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
   
   
   
   
   
   
   

   * - cabinet_mnv3l_uavid
     - 54.4
     - 52.6
     - 32.6
     - 73.4
     - | `S <https://github.com/dronefreak/CABiNet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/UAVid/cabinet_mnv3l_uavid/pretrained/2026-09-03/cabinet_mnv3l_uavid.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/cabinet_mnv3l_uavid_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/cabinet_mnv3l_uavid.hef>`_
     - 576x1024x3
     - 6.36
     - 27.20
   
   
   
   
   
   
   

   * - yolo26n_sem_uavid
     - 58.1
     - 57.9
     - 307
     - 613
     - | `S <https://github.com/dronefreak/CABiNet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/UAVid/yolo26n_sem_uavid/pretrained/2026-09-03/yolo26n_sem_uavid.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/yolo26n_sem_uavid_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/yolo26n_sem_uavid.hef>`_
     - 576x1024x3
     - 1.63
     - 6.4
   
   
   
   
   
   
   

   * - yolo26s_sem_uavid
     - 61.5
     - 61.3
     - 144
     - 259
     - | `S <https://github.com/dronefreak/CABiNet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/UAVid/yolo26s_sem_uavid/pretrained/2026-09-03/yolo26s_sem_uavid.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/yolo26s_sem_uavid_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/yolo26s_sem_uavid.hef>`_
     - 576x1024x3
     - 6.50
     - 24.9

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
     - 80.0
     - 79.5
     - | `S <https://github.com/bonlime/keras-deeplab-v3-plus>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Pascal/deeplab_v3_mobilenet_v2_dilation/pretrained/2023-08-22/deeplab_v3_mobilenet_v2_dilation.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/deeplab_v3_mobilenet_v2_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/deeplab_v3_mobilenet_v2.hef>`_
     - 513x513x3
     - 2.10
     - 17.65
   
   
   
   
   
   
   

   * - deeplab_v3_mobilenet_v2_wo_dilation⭐
     - 71.5
     - 71.3
     - 373
     - 372
     - | `S <https://github.com/tensorflow/models/tree/master/research/deeplab>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Pascal/deeplab_v3_mobilenet_v2/pretrained/2025-01-20/deeplab_v3_mobilenet_v2_wo_dilation_sim.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/deeplab_v3_mobilenet_v2_wo_dilation_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/deeplab_v3_mobilenet_v2_wo_dilation.hef>`_
     - 513x513x3
     - 2.10
     - 3.21

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
     - 693
     - 693
     - | `S <https://www.tensorflow.org/tutorials/images/segmentation>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/Oxford_Pet/unet_mobilenet_v2/pretrained/2025-01-15/unet_mobilenet_v2.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/unet_mobilenet_v2_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/unet_mobilenet_v2.hef>`_
     - 256x256x3
     - 10.08
     - 28.88

|

Aeroscapes
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
   
   
   
   
   
   
   

   * - yolo26n_sem_aeroscapes
     - 65.6
     - 65.8
     - 307
     - 464
     - | `S <https://github.com/dronefreak/CABiNet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/AeroScapes/yolo26n_sem_aeroscapes/pretrained/2026-09-03/yolo26n_sem_aeroscapes.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/yolo26n_sem_aeroscapes_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/yolo26n_sem_aeroscapes.hef>`_
     - 576x1024x3
     - 1.63
     - 6.4
   
   
   
   
   
   
   

   * - yolo26s_sem_aeroscapes
     - 66.5
     - 66.1
     - 143
     - 259
     - | `S <https://github.com/dronefreak/CABiNet>`_ `PT <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/Segmentation/AeroScapes/yolo26s_sem_aeroscapes/pretrained/2026-09-03/yolo26s_sem_aeroscapes.zip>`_ `PR <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/yolo26s_sem_aeroscapes_profiler_results_compiled_runtime_data.html>`_  `HEF <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/yolo26s_sem_aeroscapes.hef>`_
     - 576x1024x3
     - 6.50
     - 24.9