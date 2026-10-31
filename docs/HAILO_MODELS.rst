============
Hailo Models
============

Here, we give the full list of models trained in-house for specific use-cases.
Each model is accompanied with its own README. Most are also accompanied by a retraining docker and a
retraining guide; models marked as *pre-compiled only* are distributed as HEFs, without a retraining flow.
The compiled models listed here target **Hailo-10H**; for **Hailo-8**, switch to the relevant branch
`HAILO_MODELS.rst <https://github.com/hailo-ai/hailo_model_zoo/blob/v5.5/docs/HAILO_MODELS.rst>`_.


* FLOPs in the table are counted as MAC operations.

  * `Person & Face Detection`_
  * `License Plate Recognition`_
  * `Person Re-Identification`_
  * `Multi-Class Detection`_

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

.. _Multi-Class Detection:

4. **Multi-Class Detection** (*pre-compiled only*)

Four classes in a single network: person, vehicle, face and license plate.
Released as pre-compiled HEFs for **Hailo-15H**, **Hailo-15L** and **Hailo-10H** — no retraining flow is
available for these models. The mAP below is the full-precision reference; see the model's README for the
quantized accuracy and throughput of each device.

.. list-table::
   :header-rows: 1

   * - Network Name
     - mAP*
     - Input Resolution (HxWxC)
     - Params (M)
     - FLOPs (G)
   * - `hailo_yolov8n_384_640 <../hailo_models/multiclass_detection/README.rst>`_
     - 44.4
     - 384x640x3
     - 3.0
     - 2.4
   * - `hailo_yolov8n_480_640 <../hailo_models/multiclass_detection/README.rst>`_
     - 47.8
     - 480x640x3
     - 3.0
     - 3.0
   * - `hailo_yolov8s_384_640 <../hailo_models/multiclass_detection/README.rst>`_
     - 50.8
     - 384x640x3
     - 11.13
     - 8.5
   * - `hailo_yolov8s_480_640 <../hailo_models/multiclass_detection/README.rst>`_
     - 53.4
     - 480x640x3
     - 11.13
     - 10.65
   * - `hailo_yolov8m_384_640 <../hailo_models/multiclass_detection/README.rst>`_
     - 53.9
     - 384x640x3
     - 25.9
     - 23.6
   * - `hailo_yolov8m_480_640 <../hailo_models/multiclass_detection/README.rst>`_
     - 56.2
     - 480x640x3
     - 25.9
     - 29.5

\* Evaluated on a 4-class subset of COCO val2017. Not comparable to 80-class COCO mAP.
