from hailo_sdk_client.exposed_definitions import Dims
from hailo_sdk_client.tools.tf_proto_helper import TF_OPTIONAL_EXTENSIONS
from hailo_sdk_common.hailo_nn.exceptions import UnsupportedModelError

_DIM_LETTER_TO_DIMS = {
    "N": Dims.BATCH,
    "G": Dims.GROUPS,
    "H": Dims.HEIGHT,
    "W": Dims.WIDTH,
    "C": Dims.CHANNELS,
}


def _parse_input_format(input_format_entries):
    """Convert YAML entries like ['masked_fill=NGWC'] into {'masked_fill': [Dims.BATCH, ...]}."""
    if not input_format_entries:
        return None
    result = {}
    for entry in input_format_entries:
        name, _, fmt = entry.partition("=")
        result[name.strip()] = [_DIM_LETTER_TO_DIMS[c] for c in fmt.strip()]
    return result


def get_normalize_in_net(network):
    normalization_params = network["parser"].get("normalization_params", {})
    return normalization_params.get("normalize_in_net", False)


def get_normalization_params(network_info):
    normalize_in_net = get_normalize_in_net(network_info)
    normalization_params = network_info.parser.get("normalization_params")
    if normalization_params:
        mean_list = network_info.parser.normalization_params.mean_list
        std_list = network_info.parser.normalization_params.std_list
    else:
        mean_list, std_list = None, None
    return normalize_in_net, mean_list, std_list


def translate_model(runner, network_info, ckpt_path):
    model_name = network_info.network.network_name
    start_node, end_node = network_info.parser.nodes[0:2]

    if isinstance(end_node, str):
        end_node = [end_node]
    if isinstance(start_node, str):
        start_node = [start_node]

    net_input_format = _parse_input_format(network_info.parser.get("input_format"))

    ckpt_path = str(ckpt_path)
    if ckpt_path.endswith(".onnx"):
        kwargs = {}
        if net_input_format:
            kwargs["net_input_format"] = net_input_format
        runner.translate_onnx_model(
            ckpt_path,
            model_name,
            start_node_names=start_node,
            end_node_names=end_node,
            **kwargs,
        )
    elif ckpt_path.endswith(TF_OPTIONAL_EXTENSIONS):
        kwargs = {}
        if net_input_format:
            kwargs["net_input_format"] = net_input_format
        runner.translate_tf_model(
            ckpt_path,
            model_name,
            start_node_names=start_node,
            end_node_names=end_node,
            **kwargs,
        )
    else:
        raise UnsupportedModelError(
            "Failed to analyze the model, it appears the model provided is in an unsupported format. "
            "If you are using a TF1.x model (such as .ckpt or .pb), or a TF2.x model "
            "(such as .h5 or saved_model.pb), please refer to the user guide for details on how to "
            "convert to TensorFlow Lite format.",
        )
    return model_name
