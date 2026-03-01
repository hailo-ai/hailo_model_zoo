from hailo_sdk_common.hailo_nn.hn_definitions import ActivationTypes


def change_activations(runner, activation_changes):
    network = runner.get_hn_model()
    params = runner.get_params()

    for change in activation_changes.changes:
        layer_name = change.layer_name
        layer = network.get_layer_by_name(layer_name)
        layer.activation = ActivationTypes[change.activation]

    runner.set_hn(network)
    runner.load_params(params)
