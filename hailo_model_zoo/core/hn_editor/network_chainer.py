from hailo_sdk_client.exposed_definitions import JoinAction, JoinOutputLayersOrder
from hailo_sdk_client import ClientRunner
from hailo_sdk_client.runner.client_runner import InvalidArgumentsException
from hailo_model_zoo.utils.parse_utils import translate_model
from hailo_model_zoo.utils.path_resolver import resolve_model_path, resolve_alls_path


def _apply_scope(layer_name, scope):
    return layer_name if '/' in layer_name else f'{scope}/{layer_name}'


def _make_tensor_shapes(hn, ports, start_node_name, is_onnx):
    tensor_shapes = {}
    for port_source_layer, _ in ports.items():
        shape = hn.get_layer_by_name(port_source_layer).output_shape
        # hn lists batch dimension as -1 which isn't supported by the parser
        shape[0] = 1
        # FUTURE maybe work with original names? This should be port_original_dest_layer_name
        tensor_name = "{}".format(start_node_name)
        if is_onnx:
            tensor_shapes = None
        else:
            tensor_name += ":0"
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


def integrate_postprocessing(runner, integrated_postprocessing_info):
    for chain in integrated_postprocessing_info.chains:
        hn = runner.get_hn_model()
        ports = chain.ports
        start_node_name, _ = chain.parser.nodes

        # if the network has scopes, we need to fix the ports
        if hn.net_params.net_scopes:
            scope = hn.net_params.net_scopes[0]
            fixed_ports = {_apply_scope(source, scope): dest for source, dest in ports.items()}
        else:
            fixed_ports = ports
        tensor_shapes = _make_tensor_shapes(hn, fixed_ports, start_node_name,
                                            integrated_postprocessing_info.
                                            chains[0]['paths']['network_path'][0].endswith('.onnx'))

        chained_runner = ClientRunner()
        _translate_model(chained_runner, chain, tensor_shapes=tensor_shapes)

        if chain.paths.alls_script is not None:
            model_script = resolve_alls_path(chain.paths.alls_script)
            chained_runner.load_model_script(model_script)

        chained_name = chained_runner.get_hn()['name']
        scope = hn.net_params.net_scopes[0] if hn.net_params.net_scopes else hn.name

        scoped_ports = {_apply_scope(source_layer, scope): f'{chained_name}/{dest_layer}'
                        for source_layer, dest_layer in ports.items()}

        # Make sure the new outputs are in the same order so postprocessing won't break
        scoped_ports['output_layers_order'] = JoinOutputLayersOrder.NEW_OUTPUTS_IN_PLACE

        # Fuse the two networks
        runner.join(chained_runner, join_action=JoinAction.CUSTOM, join_action_info=scoped_ports)
