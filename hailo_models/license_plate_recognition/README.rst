=========================
Licesen Plate Recognition
=========================

.. image:: src/img.jpg
   :align: center

Hailo's license plate recognition network (\ *lprnet*\ ) was trained in-house on a synthetic auto-generated dataset to predict registration numbers of license plates under various weather and lighting conditions.

Model Details
-------------

Architecture
^^^^^^^^^^^^


* A convolutional network based on `\ LPRNet <https://github.com/sirius-ai/LPRNet_Pytorch>`_\ , with several modifications:

  * A ResNet like backbone with 4 stages, each containing 2 residual blocks
  * Several kernel shape changes
  * Maximal license plate length of 19 digits
  * More details can be found `here <https://github.com/hailo-ai/LPRNet_Pytorch>`_

* | Number of parameters: 7.14M
* | GMACS: 18.29
* | Accuracy\* : 99.96%
  | \* Evaluated on internal dataset containing 1178 images


Inputs
^^^^^^

* RGB license plate image with size of 75x300x3
*
  Image normalization occurs on-chip

Outputs
^^^^^^^

* A tensor with size 5x19x11

  * Post-processing outputs a tensor with size of 1x19x11
  * The 11 channels contain logits scores for 11 classes (10 digits + *blank* class)

* A Connectionist temporal classification (CTC) greedy decoding outputs the final license plate number prediction


----


Download
^^^^^^^^

| The pre-compiled network can be downloaded from `here <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/lprnet.hef>`_.
|
| Use the following command to measure model performance on hailo’s HW:

.. code-block::

   hailortcli benchmark lprnet.hef

----

Training on Custom Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^

* Hailo's LPRNet was trained on a synthetic auto-generated dataset containing 4 million license plate images. Auto-generation of synthetic data for training is cheap, allows one to obtain a large annotated dataset easily and can be adapted quickly for other domains
* A notebook for auto-generation of synthetic training data for LPRNet can be found `here <./src/lp_autogenerate.ipynb>`_
* For more details on the training data autogeneration, please see the training guide

.. include:: docs/TRAINING_GUIDE.rst

.. raw:: html

  A guide for training the pre-trained model on a custom dataset can be found <a href="./docs/TRAINING_GUIDE.rst">here</a>

