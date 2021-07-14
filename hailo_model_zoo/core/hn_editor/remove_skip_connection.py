import copy


def _add_layers(db, dup_layer, connect_layer, params):
    model_name = db['name']
    new_layer = db['layers'][dup_layer].copy()
    new_layer['output'] = [connect_layer]
    new_layer['input'] = []
    new_layer_name = dup_layer + '_new'
    db['layers'][connect_layer]['input'].append(new_layer_name)
    db['layers'][new_layer_name] = new_layer
    if new_layer['type'] != 'input_layer':
        params_keys = list(params.keys()).copy()
        for k in params_keys:
            if model_name + '/' + dup_layer + '/' in k:
                weight = params[k].copy()
                params[k.replace(dup_layer, new_layer_name)] = weight
        for layer in db['layers'][dup_layer]['input']:
            _add_layers(db, layer, new_layer_name, params)


def skip_connection(runner, remove_info):
    """Function gets runner and updates its HN and NPZ
    The layers are duplicated and a new input to the network is added to remove the skip connection

    Args:
        runner: ~hailo_sdk_client.runner.client_runner.ClientRunner
        remove_info: dict containing two fields.
            layer_name: list of lists of layers. Each list contain start/end layer to be removed
    """
    db = runner.get_hn()
    tmp_params = dict(copy.deepcopy(runner.get_params()))
    input_layer = [x for x in db['layers'] if 'input' in x]
    assert len(input_layer) == 1, "Only one input layer is supported"
    for layer in remove_info:
        inp_layer, out_layer = layer
        assert inp_layer in db['layers'], "Input layer does not exist in the HN"
        assert out_layer in db['layers'], "Output layer does not exist in the HN"

        # remove the skip connection
        db['layers'][inp_layer]['output'].remove(out_layer)
        db['layers'][out_layer]['input'].remove(inp_layer)

        # add duplication of all the layers between input_layer and inp_layer
        _add_layers(db, inp_layer, out_layer, tmp_params)

    runner.set_hn(db)
    runner.load_params(tmp_params)
