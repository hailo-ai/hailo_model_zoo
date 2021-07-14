from hailo_sdk_common.preprocessing import Normalization


def get_normalize_in_net(network):
    normalization_params = network['parser'].get('normalization_params', {})
    return normalization_params.get('normalize_in_net', False)


def get_normalization_params(network_info):
    normalize_in_net = get_normalize_in_net(network_info)
    normalization_params = network_info.parser.get('normalization_params')
    if normalization_params:
        mean_list = network_info.parser.normalization_params.mean_list
        std_list = network_info.parser.normalization_params.std_list
    else:
        mean_list, std_list = None, None
    return normalize_in_net, mean_list, std_list


def translate_model(runner, network_info, ckpt_path, *, tensor_shapes=None):
    model_name = network_info.network.network_name
    start_node, end_node = network_info.parser.nodes[0:2]
    if type(end_node) == str:
        end_node = [end_node]

    normalize_in_net, mean_list, std_list = get_normalization_params(network_info)
    normalization_obj = None
    if normalize_in_net:
        normalization_obj = Normalization(mean=mean_list, std=std_list)

    ckpt_path = str(ckpt_path)
    if ckpt_path.endswith('.onnx'):
        runner.translate_onnx_model(ckpt_path, model_name,
                                    integrated_preprocess=normalization_obj,
                                    start_node_name=start_node,
                                    end_node_names=end_node,
                                    net_input_shape=tensor_shapes)
    else:
        runner.translate_tf_model(ckpt_path, model_name,
                                  integrated_preprocess=normalization_obj,
                                  start_node_name=start_node,
                                  end_node_names=end_node,
                                  tensor_shapes=tensor_shapes)
    return model_name
