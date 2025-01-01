Cascade
===============

| This document provides instructions and basic usage examples for inferring cascade networks.

General
-------

* *cascade* allows to generate a sequence of models by cascading them one after the other to generate complex deep learning pipeline. For example, 3D object detection in birds-eye-view pipeline such as PETRv2.

* Currently cascade eval API supports PETRv2 only, see ``petrv2_repvggB0.yaml`` for further configurations.

* The user needs existing hars/hefs: both ``petrv2_repvggB0_backbone_pp_800x320`` & ``petrv2_repvggB0_transformer_pp_800x320``.

Evaluation
----------

To evaluate a cascade model on different targets, use the cascade flag:

.. code-block::

    hailomz cascade eval petrv2
    hailomz cascade eval petrv2 --override target=emulator
    hailomz cascade eval petrv2 --override target=hardware

To explore other options use:

.. code-block::

   hailomz cascade eval --help
