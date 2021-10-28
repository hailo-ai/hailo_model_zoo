from hailo_sdk_client import ClientRunner, JoinAction
from hailo_sdk_client.runner.client_runner import InvalidArgumentsException
from hailo_model_zoo.utils.parse_utils import translate_model
from hailo_model_zoo.utils.path_resolver import resolve_model_path
from hailo_sdk_client.tools.hn_modifications import transpose_hn_height_width


def _apply_scope(layer_name, scope):
    return layer_name if '/' in layer_name else f'{scope}/{layer_name}'


def _make_tensor_shapes(hn, ports, start_node_name):
    tensor_shapes = {}
    for port_source_layer, port_dest_layer in ports.items():
        shape = hn.get_layer_by_name(port_source_layer).output_shape
        # hn lists batch dimension as -1 which isn't supported by the parser
        shape[0] = 1

        # FUTURE maybe work with original names? This should be port_original_dest_layer_name
        tensor_name = "{}:0".format(start_node_name)
        tensor_shapes[tensor_name] = shape

    return tensor_shapes


def _translate_model(runner, network_info, tensor_shapes):
    try:
        network_path = network_info.paths.network_path
        ckpt_path = resolve_model_path(network_path)
        translate_model(runner, network_info, ckpt_path, tensor_shapes=tensor_shapes)
    except InvalidArgumentsException:
        # translation failed, probably due to network already have tensor shape. Falling back to without it.
        translate_model(runner, network_info, ckpt_path)


def _adjust_output_order(runner, chained_runner, original_output_order):
    new_hn = runner.get_hn_model()
    params = runner.get_params()
    chained_runner_outputs = chained_runner.get_hn_model().net_params.output_layers_order
    runner_outputs = new_hn.net_params.output_layers_order
    missing_outputs = set(original_output_order) - set(runner_outputs)
    if not missing_outputs:
        return
    # Heuristic: replace the first output with all the new ones
    insert_index = min([original_output_order.index(missing_output) for missing_output in missing_outputs])
    adjusted_runner_outputs = runner_outputs[:insert_index] + \
        runner_outputs[-len(chained_runner_outputs):] + \
        runner_outputs[insert_index:-len(chained_runner_outputs)]

    new_hn.net_params.output_layers_order = adjusted_runner_outputs
    runner.set_hn(new_hn)
    runner.load_params(params)


def integrate_postprocessing(runner, integrated_postprocessing_info):
    for chain in integrated_postprocessing_info.chains:
        hn = runner.get_hn_model()
        ports = chain.ports
        start_node_name, _ = chain.parser.nodes
        original_output_order = hn.net_params.output_layers_order

        # if the network has scopes, we need to fix the ports
        if hn.net_params.net_scopes:
            scope = hn.net_params.net_scopes[0]
            fixed_ports = {_apply_scope(source, scope): dest for source, dest in ports.items()}
        else:
            fixed_ports = ports
        tensor_shapes = _make_tensor_shapes(hn, fixed_ports, start_node_name)

        # If the first network is transposed, the chained network must also be trasnposed.
        # So we first parse it with original input shapes and then "flip" it
        if hn.net_params.transposed_net:
            for k, v in tensor_shapes.items():
                v[1], v[2] = v[2], v[1]
                tensor_shapes[k] = v
        chained_runner = ClientRunner()
        _translate_model(chained_runner, chain, tensor_shapes=tensor_shapes)

        if hn.net_params.transposed_net:
            transpose_hn_height_width(chained_runner)

        chained_name = chained_runner.get_hn()['name']
        scope = hn.net_params.net_scopes[0] if hn.net_params.net_scopes else hn.name

        scoped_ports = {_apply_scope(source_layer, scope): f'{chained_name}/{dest_layer}'
                        for source_layer, dest_layer in ports.items()}
        # Fuse the two networks
        runner.join(chained_runner, join_action=JoinAction.CUSTOM, join_action_info=scoped_ports)

        # Override the default output order
        _adjust_output_order(runner, chained_runner, [_apply_scope(layer, scope) for layer in original_output_order])
