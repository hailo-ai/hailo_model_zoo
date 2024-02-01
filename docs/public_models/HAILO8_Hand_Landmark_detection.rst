
Public Pre-Trained Models
=========================

.. |rocket| image:: ../images/rocket.png
  :width: 18

.. |star| image:: ../images/star.png
  :width: 18
.. _Hand Landmark detection:

Hand Landmark detection
-----------------------

Hand Landmark
^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Network Name
     - Input Resolution (HxWxC)
     - Params (M)
     - OPS (G)
     - Pretrained
     - Source
     - Compiled
     - FPS (Batch Size=1)
     - FPS (Batch Size=8)
   * - hand_landmark_lite
     - 224x224x3
     - 1.01
     - 0.3
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/HandLandmark/hand_landmark_lite/2023-07-18/hand_landmark_lite.zip>`_
     - `link <https://github.com/google/mediapipe>`_
     - `download <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.10.0/hailo8/hand_landmark_lite.hef>`_
     - 2,672.09
     - 2,671.97
