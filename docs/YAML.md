# Hailo Model Zoo YAML Description

## Properties

- **`network`**
  - **`network_name`** *(['string', 'null'])*: The network name. Default: `None`.
- **`paths`**
  - **`network_path`** *(array)*: Path to the network files, can be ONNX / TF Frozen graph (pb) / TF CKPT files / TF2 saved model.
  - **`alls_script`** *(['string', 'null'])*: Path to model script in the alls directory.
- **`parser`**
  - **`nodes`** *(['array', 'null'])*: List of [start node, [end nodes]] for parsing.
  For example: ["resnet_v1_50/conv1/Pad", "resnet_v1_50/predictions/Softmax"].
  - **`normalization_params`**: Add normalization on chip.
  For example: normalization_params: { "normalize_in_net": true, "mean_list": [123.68, 116.78, 103.94], "std_list": [1.0, 1.0, 1.0] }.
    - **`normalize_in_net`** *(boolean)*: Whether to run normalization on chip. Default: `False`.
    - **`mean_list`** *(['array', 'null'])*: List of means used for normalization. Default: `None`.
    - **`std_list`** *(['array', 'null'])*: List of STD used for normalization. Default: `None`.
  - **`start_node_shape`** *(['array', 'null'])*: Dict for setting input shape of supported models that does not explicitly use it.
  For example, models with input shape of [?, ?, ?, 3] can be set with {"Preprocessor/sub:0": [1, 512, 512, 3]}. Default: `None`.
- **`preprocessing`**
  - **`network_type`** *(['string', 'null'])*: The type of the network. Default: `classification`.
  - **`meta_arch`** *(['string', 'null'])*: The network preprocessing meta-architecture.
   For example:  mobilenet_ssd, efficientnet, etc. Default: `None`.
- **`quantization`**
  - **`calib_set`** *(['array', 'null'])*: List contains the calibration set path.
  For example: ['models_files/imagenet/2021-06-20/imagenet_calib.tfrecord']. Default: `None`.
  - **`calib_set_name`** *(['string', 'null'])*: Name of the dataset used for calibration. By default (null) uses the evaluation dataset name. Default: `None`.
- **`postprocessing`**
  - **`meta_arch`** *(['string', 'null'])*: Postprocessing meta architecture name.
  For example: yolo_v3, yolo_v4, etc. Default: `None`.
  - **`postprocess_config_json`** *(['string', 'null'])*: Path to JSON file with the postprocessing configuration (for offloading NMS to the Hailo-8). Default: `None`.
  - **`device_pre_post_layers`**: Whether to use postprocessing on chip or do it on the host.
    - **`bilinear`** *(boolean)*: Activate the bilinear PPU layer. Default: `False`.
    - **`argmax`** *(boolean)*: Activate the Argmax PPU layer. Default: `False`.
    - **`softmax`** *(boolean)*: Activate the Softmax PPU layer. Default: `False`.
    - **`nms`** *(boolean)*: Activate the NMS PPU layer and the relevant decoding layers. Default: `False`.
- **`evaluation`**
  - **`dataset_name`** *(['string', 'null'])*: Name of the dataset to be used in evaluation. Default: `None`.
  - **`data_set`** *(['string', 'null'])*: Path to TFrecord dataset for evaluation. Default: `None`.
  - **`classes`** *(['integer', 'null'])*: Number of classes in the model. Default: `1000`.
  - **`labels_offset`** *(['integer', 'null'])*: Offset of labels. Default: `0`.
  - **`network_type`** *(['string', 'null'])*: The type of the network used for evaluation.
  Use this field if evaluation type is different than preprocessing type. Default: `None`.
- **`hn_editor`**
  - **`yuv2rgb`** *(boolean)*: Add YUV to RGB layer. Default: `False`.
  - **`flip`** *(boolean)*: Rotate input by 90 degrees. Default: `False`.
  - **`input_resize`**: Add resize bilinear layer at the start of the network.
    - **`enabled`** *(boolean)*: Whether this is enabled or disabled. Default: `False`.
    - **`input_shape`** *(array)*: List of input shape to resize from [H, W].
  - **`bgr2rgb`** *(boolean)*: Add BGR to RGB layer. Default: `False`.
