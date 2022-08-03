============
Hailo Models
============

Here, we give the full list of models trained in-house for specific use-cases.
Each model is accompanied with its own README, retraining docker and retraining guide.


* FLOPs in the table are counted as MAC operations.
* Supported tasks:

  * `Object Detection`_
  * `Person & Face Detection`_
  * `License Plate Recognition`_
  * `Person Re-Identification`_

**Important:**
Retraining is not available inside the docker version of Hailo Software Suite. In case you use it, clone the hailo_model_zoo outside of the docker, and perform the retraining there:
``git clone https://github.com/hailo-ai/hailo_model_zoo.git``


.. _Object Detection:

.. _Person & Face Detection:

1. **Object Detection**

.. list-table::
   :header-rows: 1

   * - Network Name
     - mAP*
     - Input Resolution (HxWxC)
     - Params (M)
     - FLOPs (G)
   * - `yolov5m_vehicles <../hailo_models/vehicle_detection/README.rst>`_
     - 46.5
     - 640x640x3
     - 21.47
     - 25.63
   * - `tiny_yolov4_license_plates <../hailo_models/license_plate_detection/README.rst>`_
     - 73.45
     - 416x416x3
     - 5.87
     - 3.4
   * - `yolov5s_personface <../hailo_models/personface_detection/README.rst>`_
     - 47.5
     - 640x640x3
     - 7.25
     - 8.38


.. _License Plate Recognition:

2. **License Plate Recognition**

.. list-table::
   :header-rows: 1

   * - Network Name
     - Accuracy*
     - Input Resolution (HxWxC)
     - Params (M)
     - FLOPs (G)
   * - `lprnet <../hailo_models/license_plate_recognition/README.rst>`_
     - 99.96
     - 75x300x3
     - 7.14
     - 18.29

\* Evaluated on internal dataset

.. _Person Re-Identification:

3. **Person Re-ID**

.. list-table::
   :header-rows: 1

   * - Network Name
     - Accuracy*
     - Input Resolution (HxWxC)
     - Params (M)
     - FLOPs (G)
   * - `repvgg_a0_person_reid_512 <../hailo_models/reid/README.rst>`_
     - 89.9
     - 256x128x3
     - 7.68
     - 0.89
   * - `repvgg_a0_person_reid_2048 <../hailo_models/reid/README.rst>`_
     - 90.02
     - 256x128x3
     - 9.65
     - 0.89

\* Evaluated on Market-1501