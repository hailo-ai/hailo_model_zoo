Benchmarks
==========

| To measure FPS, power and latency of the Hailo Model Zoo networks, use the HailoRT command line interface.
| For more information, please refer to the HailoRT documentation at `hailo.ai <https://hailo.ai/>`_.

Example
-------

| The HailoRT command line interface works with the Hailo Executable File (HEF) of the model. 
| To generate the HEF file use the following command:

.. code-block::

   hailomz compile <model_name>

| Only after first building the HEF, will it be possible to measure the performance of the model by using the HailoRT command line interface.
| Example for measuring performance of resnet_v1_50:

.. code-block::

   hailortcli benchmark resnet_v1_50.hef

Example output:

.. code-block::

   =======
   Summary
   =======
   FPS     (hw_only)                 = 1328.83
           (streaming)               = 1328.8
   Latency (hw)                      = 2.93646 ms
   Power in streaming mode (average) = 3.19395 W
                           (max)     = 3.20456 W


Using Datasets from the Hailo Model Zoo
---------------------------------------

To use datasets from the Hailo Model Zoo, you can use the command:

.. code-block::

   python hailo_model_zoo/tools/conversion_tool.py /path/to/tfrecord_file resnet_v1_50

which will generate a bin file with serialized images. This bin file can be used with HailoRT:

.. code-block::

   hailortcli benchmark resnet_v1_50.hef --input-files tfrecord_file.bin
