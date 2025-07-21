from hailo_sdk_client import ClientRunner
from hailo_sdk_client.exposed_definitions import JoinAction, JoinOutputLayersOrder

from hailo_model_zoo.utils.parse_utils import translate_model
from hailo_model_zoo.utils.path_resolver import resolve_model_path


def _apply_scope(layer_name, scope):
    return layer_name if "/" in layer_name else f"{scope}/{layer_name}"


def _translate_model(runner, network_info):
    network_path = network_info.paths.network_path
    ckpt_path = resolve_model_path(network_path)
    translate_model(runner, network_info, ckpt_path)


def integrate_postprocessing(runner, integrated_postprocessing_info, network_info):
    for chain in integrated_postprocessing_info.chains:
        hn = runner.get_hn_model()
        ports = chain.ports

        # if the network has scopes, we need to fix the ports
        if hn.net_params.net_scopes:
            scope = hn.net_params.net_scopes[0]

        chained_runner = ClientRunner()
        _translate_model(chained_runner, chain)

        chained_name = chained_runner.get_hn()["name"]
        scope = hn.net_params.net_scopes[0] if hn.net_params.net_scopes else hn.name

        scoped_ports = {
            _apply_scope(source_layer, scope): f"{chained_name}/{dest_layer}"
            for source_layer, dest_layer in ports.items()
        }

        # Make sure the new outputs are in the same order so postprocessing won't break
        scoped_ports["output_layers_order"] = JoinOutputLayersOrder.NEW_OUTPUTS_IN_PLACE

        # Fuse the two networks
        runner.join(chained_runner, join_action=JoinAction.CUSTOM, join_action_info=scoped_ports)
