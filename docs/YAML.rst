
.. _yaml_description:

Ha    * | **normaliz  * | **padding_color** *(['integer', '  * | **bgr2rgb** *(boolean)*\ - ``evaluation`` and ``postprocessing`` properties aren't needed for compilation as they are used by the Model Zoo for model evaluation 
  (which isn't supported yet for retrained models). Also the ``info`` field is just used for description.
  
  - Only in the YOLOv4 family, the ``evaluation.classes`` and ``postprocessing.anchors.sizes`` fields are used for compilation,
    that's these values should be updated even if just for compilation of the BGR to RGB layer.
    | In some training frameworks, the models are trained on BGR inputs. When we want to feed RGB images to the network (whether in the MZ or in the user application), 
      we need to transform the images from RGB to BGR. The MZ automatically inserts this layer into the on-chip model.
    | We have already set the "bgr2rgb" flag in the yaml files that correspond to the relevant retraining dockers. Default: ``False``.'])*\ : In the training environments, the input images to the model have used this color to indicate "padding" around resized images. Default: ``114`` for YOLO architectures, ``0`` for others._in_net** *(boolean)*\ : Whether or not the network includes an on-chip normalization layer. If so, the normalization layer will appear in the .alls file that is used. Default: ``False``.
      | Example alls command: ``normalization1 = normalization([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])``
      | If the alls doesn't include the required normalization, then the MZ (and the user application) will apply normalization before feeding inputs to the network Model Zoo YAML Description
================================

Properties
----------


* | **network**

  * | **network_name** *(['string', 'null'])*\ : The network name. Default: ``None``.

* | **paths**

  * | **network_path** *(array)*\ : Path to the network files, can be ONNX / TF Frozen graph (pb) / TF CKPT files / TF2 saved model.
  * | **alls_script** *(['string', 'null'])*\ : Path to model script in the alls directory.

* | **parser**

  * | **nodes** *(['array', 'null'])*\ : List of [start node, [end nodes]] for parsing.
    | For example: ["resnet_v1_50/conv1/Pad", "resnet_v1_50/predictions/Softmax"].
  * | **normalization_params**\ : Add normalization on chip.
    | For example: normalization_params: { "normalize_in_net": true, "mean_list": [123.68, 116.78, 103.94], "std_list": [58.395, 57.12, 57.375] }.

    * | **normalize_in_net** *(boolean)*\ : Whether or not the network includes an on-chip normalization layer. If so, the normalization layer will appear on the .alls file that is used. Default: ``False``.
      | Example alls command: ``normalization1 = normalization([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])``
      | If the alls doesn’t include the required normalization, then the MZ (and the user application) will apply normalization before feeding inputs to the network
    * | **mean_list** *(['array', 'null'])*\ : Used only in case normalize_in_net=false. The MZ automatically performs normalization to the calibration set, so the network receives already-normalized
        input (saves the user the need to normalize the dataset). Default: ``None``.
    * | **std_list** *(['array', 'null'])*\ : Used only in case normalize_in_net=false: The MZ automatically performs normalization to the calibration set, so the network receives already-normalized
        input (saves the user the need to normalize the dataset). Default: ``None``.

  * | **start_node_shapes** *(['array', 'null'])*\ : Dict for setting input shape of supported models that do not explicitly use it.
    | For example, models with input shape of [?, ?, ?, 3] can be set with {"Preprocessor/sub:0": [1, 512, 512, 3]}. Default: ``None``.

* | **preprocessing**

  * | **network_type** *(['string', 'null'])*\ : The type of the network. Default: ``classification``.
  * | **meta_arch** *(['string', 'null'])*\ : The network preprocessing meta-architecture.
    | For example:  mobilenet_ssd, efficientnet, etc. Default: ``None``.
  * | **padding_color** *(['integer', 'null'])*\ : On the training environments, the input images to the model have used this color to indicate “padding” around resized images. Default: ``114`` for YOLO architectures, ``0`` for others.

* | **quantization**

  * | **calib_set** *(['array', 'null'])*\ : List containing the calibration set path.
    | For example: ['models_files/imagenet/2021-06-20/imagenet_calib.tfrecord']. Default: ``None``.
  * | **calib_set_name** *(['string', 'null'])*\ : Name of the dataset used for calibration. By default (null) uses the evaluation dataset name. Default: ``None``.

* | **postprocessing**

  * | **meta_arch** *(['string', 'null'])*\ : Postprocessing meta architecture name.
    | For example: yolo_v3, yolo_v4, etc. Default: ``None``.
  * | **postprocess_config_file** *(['string', 'null'])*\ : Path to a file with the postprocessing configuration (for example, for offloading NMS to the Hailo-8). Default: ``None``.
  * | **hpp** *(boolean)*\ : Activate the Hailo postprocessing on host. Default: ``False``.
  * | **device_pre_post_layers**\ : Whether to use postprocessing on chip or do it on the host.

    * | **bilinear** *(boolean)*\ : Activate the bilinear PPU layer. Default: ``False``.
    * | **argmax** *(boolean)*\ : Activate the Argmax PPU layer. Default: ``False``.
    * | **softmax** *(boolean)*\ : Activate the Softmax PPU layer. Default: ``False``.
    * | **nms** *(boolean)*\ : Activate the NMS PPU layer and the relevant decoding layers. Default: ``False``.

* | **evaluation**

  * | **dataset_name** *(['string', 'null'])*\ : Name of the dataset to be used in evaluation. Default: ``None``.
  * | **data_set** *(['string', 'null'])*\ : Path to TFrecord dataset for evaluation. Default: ``None``.
  * | **classes** *(['integer', 'null'])*\ : Number of classes in the model. Default: ``1000``.
  * | **labels_offset** *(['integer', 'null'])*\ : Offset of labels. Default: ``0``.
  * | **network_type** *(['string', 'null'])*\ : The type of the network used for evaluation.
    | Use this field if evaluation type is different than preprocessing type. Default: ``None``.

* | **hn_editor**

  * | **yuv2rgb** *(boolean)*\ : Add YUV to RGB layer. Default: ``False``.
  * | **flip** *(boolean)*\ : Rotate input by 90 degrees. Default: ``False``.
  * | **input_resize**\ : Add resize bilinear layer at the start of the network.

    * | **enabled** *(boolean)*\ : Whether this is enabled or disabled. Default: ``False``.
    * | **input_shape** *(array)*\ : List of input shape to resize from [H, W].

  * | **bgr2rgb** *(boolean)*\ : Add BGR to RGB layer.
    | On some training frameworks, the models are trained on BGR inputs. When we want to feed RGB images to the network (whether on the MZ or on the user application), 
      we need to transform the images from RGB to BGR. The MZ automatically inserts this layer to the on-chip model.
    | We have already set the “bgr2rgb” flag on the yaml files that correspond to the relevant retraining dockers. Default: ``False``.


YAML hierarchies
----------------

- The MZ uses hierarchical .yaml infrastructure for configuration. For example, for yolov5m_vehicles:
    - Network yaml is `networks/yolov5m_vehicles.yaml <https://github.com/hailo-ai/hailo_model_zoo/blob/master/hailo_model_zoo/cfg/networks/yolov5m_vehicles.yaml>`_
    - It includes at the top the lines:
     
      .. code::

         base:
         - base/yolov5.yaml
    - Meaning it inherits from `base/yolov5.yaml <https://github.com/hailo-ai/hailo_model_zoo/blob/master/hailo_model_zoo/cfg/base/yolov5.yaml>`_
    - Which inherits from `base/yolo.yaml <https://github.com/hailo-ai/hailo_model_zoo/blob/master/hailo_model_zoo/cfg/base/yolo.yaml>`_
    - Which inherits from `base/base.yaml <https://github.com/hailo-ai/hailo_model_zoo/blob/master/hailo_model_zoo/cfg/base/base.yaml>`_
- Each property in the child hierarchies replaces the properties in the parent ones. For example, if `preprocessing.input_shape`
  is defined both in `base/yolov5.yaml` and `base/base.yaml`, the one from `base/yolov5.yaml` will be used
- Therefore, if we want to change some property, we can just update the last child file that is using that property


Notes for Retraining
--------------------

- ``evaluation`` and ``postprocessing`` properties aren’t needed for compilation as they are used by the Model-Zoo for model evaluation 
  (which isn’t supported yet for retrained models). Also ``info`` field is just used for description.
  
  - Only on YOLOv4 family, the ``evaluation.classes`` and ``postprocessing.anchors.sizes`` fields are used for compilation,
    that’s why you should update those values even if just for compilation
- You might want to update those default values in some advanced scenarios:

  - preprocessing.padding_color
    
    - Change these values only if you have used a different value for training your model
  - parser.normalization_params.normalize_in_net

    - If you have manually changed the normalization values in the retraining docker, and `normalize_in_net=true`, remember to update the corresponding alls command
  - parser.normalization_params.mean_list
    
    - Update these values if `normalize_in_net=false` and you have manually changed the normalization values in the retraining docker
  - parser.normalization_params.std_list
    
    - Update those values if `normalize_in_net=false` and you have manually changed the normalization values on the retraining docker
